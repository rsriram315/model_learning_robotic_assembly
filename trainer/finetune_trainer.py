import torch
import numpy as np
from model import MLP, MCDropout
from trainer.base_trainer import BaseTrainer
from utils import MetricTracker, ensure_dir, prepare_device
from utils.geodesic_loss import GeodesicLoss


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self,
                 dataloader,
                 dataset_stats,
                 dataset_cfg,
                 trainer_cfg,
                 optim_cfg,
                 model_cfg,
                 resume_path=None,
                 valid_dataloader=None,
                 lr_scheduler=None):

        ensure_dir(trainer_cfg["ckpts_dir"])
        self.dataloader = dataloader

        self.valid_dataloader = valid_dataloader
        self.do_validation = self.valid_dataloader is not None

        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(self.dataloader.batch_size))

        # prepare for (multi-device) GPU training
        self.device, self.device_ids = prepare_device(model_cfg["n_gpu"])

        model = self._build_model(model_cfg, dataset_stats)
        optimizer = self._build_optim(model, optim_cfg)
        if trainer_cfg["criterion"] == "Geodesic_MSE":
            criterion = (torch.nn.MSELoss(reduction='mean'), GeodesicLoss(reduction='mean'))
        elif trainer_cfg["criterion"] == "MSE":
            criterion = torch.nn.MSELoss(reduction='mean')

        metric_fns = []
        self.train_metrics = \
            MetricTracker('loss', *[m.__name__ for m in metric_fns])
        self.valid_metrics = \
            MetricTracker('loss', *[m.__name__ for m in metric_fns])

        super().__init__(model,
                         criterion,
                         metric_fns,
                         optimizer,
                         dataset_stats,
                         dataset_cfg,
                         trainer_cfg,
                         resume_path)
        

    def _build_model(self, model_cfg, ds_stats):
        # build model architecture, then print to console
        if model_cfg["name"] == "MLP":
            model = MLP(model_cfg["input_dims"],
                        model_cfg["output_dims"],
                        self.device)
        elif model_cfg["name"] == "MCDropout":
            model = MCDropout(model_cfg["input_dims"],
                              model_cfg["output_dims"],
                              self.device,
                              ds_stats)

        print(model)
        model = model.to(self.device)

        if len(self.device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        return model

    def _build_optim(self, model, optim_cfg):
        # build optimizer, learning rate scheduler.
        # delete every lines containing lr_scheduler for disabling scheduler.
        trainable_params = filter(lambda p: p.requires_grad,
                                  model.parameters())
        optimizer = torch.optim.Adam(trainable_params,
                                     lr=optim_cfg["lr"],
                                     weight_decay=optim_cfg["weight_decay"],
                                     amsgrad=optim_cfg["amsgrad"])
        return optimizer

    def _train_epoch(self, epoch):
        """
        training logic for an epoch

        Args:
            epoch: integer, current training epoch.
        Returns:
            a log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        tb_step = (epoch - 1) * len(self.dataloader)

        for batch_idx, (state_action, target) in enumerate(self.dataloader):
 
            state_action, target = \
                (state_action.to(self.device, non_blocking=True),
                 target.to(self.device, non_blocking=True))

            # intialize the gradient to None first
            self.optimizer.zero_grad(set_to_none=True)
            loss = 0
            if self.dataset_cfg["multi_horizon_training"]:
                assert self.dataset_cfg["training_horizon"] > 1, " value of config 'training_horizon' should be greateer than 1 for multi horizon training"
                if self.trainer_cfg["criterion"] == "Geodesic_MSE":
                    input = state_action[:,0].clone().detach()
                    self.criterion_1, self.criterion_2 = self.criterion
                    for horizon in range(target.shape[1]):
                        output = self.model(input)
                        horizon_loss = 1 * self.criterion_1(output[:, :3], target[:, horizon, :3])
                        # print("MSE loss:", horizon_loss)
                        horizon_loss += 1 * self.criterion_2(output[:, 3:].reshape(-1,3,3), target[:, horizon, 3:].reshape(-1,3,3))
                        # print("geodesic loss:", 1 * self.criterion_2(output[:, 6:].reshape(-1,3,3), target[:, horizon, 6:].reshape(-1,3,3)))
                        loss += horizon_loss
                        # print("total loss :", loss)
                        if horizon+1 < target.shape[1]:
                            input = torch.hstack((output, state_action[:, horizon+1, 12:]))
                
                elif self.trainer_cfg["criterion"] == "MSE":
                    input = state_action[:,0].clone().detach()
                    for horizon in range(target.shape[1]):
                        output = self.model(input)
                        # print("\n output shape", output.shape)
                        horizon_loss = self.criterion(output, target[:,horizon, :])
                        # print("\nhorizon loss", horizon_loss)
                        loss += horizon_loss
                        if horizon+1 < target.shape[1]:
                            input = torch.hstack((output, state_action[:, horizon+1, 12:]))
            
                loss = loss / target.shape[1]

            else:
                if self.trainer_cfg["criterion"] == "Geodesic_MSE":
                    output = self.model(state_action)
                    self.criterion_1, self.criterion_2 = self.criterion
                    loss = 1 * self.criterion_1(output[:, :3], target[:, :3])
                    loss += 1 * self.criterion_2(output[:, 3:].reshape(-1,3,3), target[:, 3:].reshape(-1,3,3))

                elif self.trainer_cfg["criterion"] == "MSE":
                    output = self.model(state_action)
                    loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            # logging
            self.train_metrics.update('loss', loss.item())
            self.train_tb_writer.add_scalar("loss", loss.item(), tb_step)
            tb_step += 1

            # for met in self.metric_fns:
            #     self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.write_log(f'Train Epoch: {epoch} '
                               f'{self._progress(batch_idx)} '
                               f'Loss: {self.train_metrics.avg("loss"):.6f}')

        log = self.train_metrics.result()

        return log

    def _valid_epoch(self):
        """
        validate after training an epoch

        Params:
            epoch: integer, current training epoch.
        Return:
            A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for _, (state_action, target) in enumerate(self.valid_dataloader):
                state_action, target = state_action.to(self.device), target.to(self.device)
                loss = 0
                if self.dataset_cfg["multi_horizon_training"]:
                    assert self.dataset_cfg["training_horizon"] > 1, " value of config 'training_horizon' should be greateer than 1 for multi horizon training"
                    if self.trainer_cfg["criterion"] == "Geodesic_MSE":
                        input = state_action[:,0].clone().detach()
                        self.criterion_1, self.criterion_2 = self.criterion
                        for horizon in range(target.shape[1]):
                            output = self.model(input)
                            val_loss = 1 * self.criterion_1(output[:, :3], target[:, horizon, :3])
                            val_loss += 1 * self.criterion_2(output[:, 3:].reshape(-1,3,3), target[:, horizon, 3:].reshape(-1,3,3))
                            loss += val_loss
                            if horizon+1 < target.shape[1]:
                                input = torch.hstack((output, state_action[:, horizon+1, 12:]))
                    
                    elif self.trainer_cfg["criterion"] == "MSE":
                        input = state_action[:,0].clone().detach()
                        for horizon in range(target.shape[1]):
                            output = self.model(input)
                            val_loss = self.criterion(output, target[:,horizon, :])
                            loss += val_loss
                            if horizon+1 < target.shape[1]:
                                input = torch.hstack((output, state_action[:, horizon+1, 12:]))
                    loss = loss / target.shape[1]

                else :
                    if self.trainer_cfg["criterion"] == "Geodesic_MSE":
                        self.criterion_1, self.criterion_2 = self.criterion
                        output = self.model(state_action)
                        loss = 1 * self.criterion_1(output[:, :3], target[:, :3])
                        loss += 1 * self.criterion_2(output[:, 3:].reshape(-1,3,3), target[:, 3:].reshape(-1,3,3))
                    
                    elif self.trainer_cfg["criterion"] == "MSE":
                        output = self.model(state_action)
                        loss = self.criterion(output, target) 

                self.valid_metrics.update('loss', loss.item())
                # for met in self.metric_fns:
                #     self.valid_metrics.update(met.__name__,
                #                               met(output, target))
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        current = batch_idx * self.dataloader.batch_size
        total = self.dataloader.n_samples

        return f'[{current}/{total} ({(100.0 * current / total):.0f}%)]'
