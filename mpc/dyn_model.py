# flake8: noqa
from operator import contains
import numpy as np
import torch
from pathlib import Path
from copy import deepcopy
from model import MLP
from utils import prepare_device
from dataloaders.data_processor import Normalization, recover_rotation

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True


class Dyn_Model:
    """
    This class implements: init, do_forward_sim
    """

    def __init__(self, cfg, param_cb=None):
        self.cfg = deepcopy(cfg)
        self.param_cb = param_cb
        self.model, self.cfg = self._build_model(self.cfg)
        self.norm = Normalization(self.cfg["dataset"]["stats"])


    def _build_model(self, cfg):
        self.device, self.device_ids = prepare_device(
                                        self.cfg["model"]["n_gpu"])

        if self.cfg["eval"]["ckpt_pth"] is None:
            ckpt_pth = self._find_ckpt(self.cfg["eval"]["ckpt_dir"])
        else:
            ckpt_pth = self.cfg["eval"]["ckpt_pth"]

        ckpt = torch.load(ckpt_pth, map_location=self.device)
        if self.param_cb:
            if 'model' in ckpt.keys():
                self.param_cb(dict(model_used_during_training=ckpt['model']))
            else:
                self.param_cb(dict(model_used_during_training=None))
        cfg["dataset"]["stats"] = ckpt["dataset_stats"]
        model_cfg = cfg["model"]

        # build model architecture, then print to console
        model = MLP(model_cfg["input_dims"],
                    model_cfg["output_dims"],
                    self.device)
        print(model)
        

        # load model checkpoint
        model.load_state_dict(ckpt["state_dict"])
        print(f'... Load checkpoint: {ckpt_pth}')

        model = model.to(self.device)
        if len(self.device_ids) > 1:
            model = torch.nn.DataParallel(model, self.device_ids)
        model.eval()
        return model, cfg

    def _find_ckpt(self, ckpt_dir):
        ckpt_dir = Path(ckpt_dir)
        ckpt_pths = [pth for pth in list(ckpt_dir.glob("*.pth"))]
        return ckpt_pths[0]

    def do_forward_sim(self, curr_state, actions_to_perform):
        """
        perform multistep prediction
        """
        # 1. we need to first normalize the curr_state, and
        #    replace it with the z in random action.
        # 2. the predicted states need to recover because we
        #    need to calculate the loss.
        # 3. the action should also be recovered.
        state_list = []
        norm_state_list = []
        num_sample_seq, horizon, _ = actions_to_perform.shape

        # populate the state to dim of [num_traj, action_dim],
        # in favor of the parallel computation
        curr_state_per_seq = np.vstack(
            [curr_state for _ in range(num_sample_seq)])
        curr_state_per_seq = self.norm.normalize(curr_state_per_seq[:, None, :])


        for h in range(horizon):
            with torch.no_grad():
                # change only the z direction only for prototyping
                # curr_action_per_seq = np.copy(curr_state_per_seq)
                curr_action_per_seq = actions_to_perform[:, h]

                curr_state_action = np.hstack((curr_state_per_seq,
                                               curr_action_per_seq))

                curr_state_action = torch.tensor(curr_state_action,
                                                 dtype=torch.float32).to(self.device)
                # print("curr state action", curr_state_action[:5])
                # run through NN to get predictions (diff)
                pred_state_diff_K = self.model(curr_state_action)
                # print("model prediction", pred_state_diff_K[:5])

                # predictions are the diff, we need to recover it
                pred_state_K, recover_pred_state_K = \
                    self._recover_data(pred_state_diff_K.cpu().numpy(),
                                       curr_state_per_seq)
                # pred_state_K is normalized rollout state, while
                # recover_pred_state_K is unnormalized
                curr_state_per_seq = np.copy(pred_state_K)
                # save current state
                state_list.append(np.copy(recover_pred_state_K))
                norm_state_list.append(np.copy(pred_state_K))

                # next_state = self.norm.inv_normalize(actions_to_perform[:, None, h], is_action=True)
                # state_list.append(next_state)
                # next_state_norm  =actions_to_perform[:, h]
                # norm_state_list.append(np.copy(next_state_norm))

        return np.array(state_list), np.array(norm_state_list)

    def _recover_data(self, ro_pred, ro_state):
        curr_ro_state = self.norm.inv_normalize(ro_state[:, None, :])
        # recover predicted diff of states
        recover_ro_output = self.norm.inv_normalize(ro_pred[:, None, :], is_res=True)
        recover_ro_output[:, :6] += curr_ro_state[:, :6]
        recover_ro_output = recover_rotation(np.copy(recover_ro_output),
                                             np.copy(curr_ro_state))
        new_ro_state = self.norm.normalize(recover_ro_output[:, None ,:])

        return new_ro_state, recover_ro_output
