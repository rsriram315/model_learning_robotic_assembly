import matplotlib.pyplot as plt
import numpy as np
import torch
from copy import deepcopy
from model import MLP
from dataloaders import Normalization, Standardization
from pathlib import Path
from utils import prepare_device
from dataloaders import DemoDataset


class Visualize:
    def __init__(self, cfg):
        self.cfg = cfg
        self.vis_cfg = cfg["visualization"]

        # create dirs
        self.vis_dir = Path("saved/visualizations")
        for name in ["loss", "axis", "trajectory"]:
            Path(self.vis_dir / name / 'train').mkdir(parents=True,
                                                      exist_ok=True)
            Path(self.vis_dir / name / 'test').mkdir(parents=True,
                                                     exist_ok=True)

        self.device, self.device_ids = prepare_device(cfg["model"]["n_gpu"])
        self.demo_fnames, self.train_demo_fnames, self.test_demo_fnames =\
            self._find_demos(self.cfg["dataset"])

        self.learn_residual = cfg["dataset"]["learn_residual"]
        self.ds_stats = None
        self.norm = None

    def visualize(self):
        model, ds_stats = self._build_model(self.cfg)

        if self.cfg["dataset"]["preprocess"]["normalize"]:
            self.norm = Normalization(ds_stats)
        elif self.cfg["dataset"]["preprocess"]["standardize"]:
            self.norm = Standardization(ds_stats)

        for fname in self.demo_fnames:
            if fname in self.train_demo_fnames:
                suffix_fname = Path('train') / Path(fname).stem
            elif fname in self.test_demo_fnames:
                suffix_fname = Path('test') / Path(fname).stem

            dataset = self._read_single_demo(deepcopy(self.cfg["dataset"]),
                                             [fname], ds_stats)
            losses_per_demo, preds_per_demo = self._evaluate(model, dataset)
            time = dataset.sample_time

            # feature only
            state = dataset.states_actions[:, 0]
            state = self.norm.inverse_normalize(state, is_target=True)

            if self.vis_cfg["loss"]:
                loss_fname = self.vis_dir / "loss" / suffix_fname
                self._vis_loss(losses_per_demo, time, loss_fname)

            if self.vis_cfg["axis"]:
                axis_fname = self.vis_dir / "axis" / suffix_fname
                self._vis_axis(preds_per_demo, state, time, axis_fname)

            if self.vis_cfg["trajectory"]:
                traj_fname = self.vis_dir / "trajectory" / suffix_fname
                self._vis_trajectory(preds_per_demo[:, :3], state[:, :3],
                                     traj_fname)

            print(f"... Generated visualization for {fname}")

    def _vis_loss(self, loss, time, fname):
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.set_title(f"sum of the loss: {sum(loss)}")
        ax.scatter(time, loss, s=2)
        ax.set_xlabel("time")
        ax.set_ylabel("loss")
        plt.savefig(fname)
        plt.close(fig)

    def _vis_axis(pred, state, time, fname):
        fig, axs = plt.subplots(4, 3, figsize=(25, 15), sharex='all')

        pred = np.array(pred)

        # the predicted states should start with the prediction for t=1 not t=0
        size = 1
        # features = ['pos', 'force', 'rot_cosine', 'rot_sine']
        features = ['pos', 'force', 'matrix R row 1', 'matrix R row 2',
                    'matrix R row 3']
        axis = ['x', 'y', 'z']

        for r, feature in enumerate(features):
            for c, ax in enumerate(axis):
                idx = c + 3 * r
                axs[r, c].scatter(time[1:],
                                  state[1:, idx],
                                  s=size,
                                  c='tab:blue',
                                  label="ground truth")
                axs[r, c].scatter(time[1:],
                                  pred[:-1, idx],
                                  s=size,
                                  c='tab:orange',
                                  label="predictions")
                axs[r, c].set_title(f'{feature} {ax} axis')
                axs[r, c].legend()
                if c == 0:
                    axs[r, c].set_ylabel('coordinate')
                if r == 3:
                    axs[r, c].set_xlabel('time')
        plt.savefig(fname)
        plt.close(fig)

    def _vis_trajectory(self, pred, state, fname):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        max_size = 10
        size_ls = np.arange(0, max_size, max_size / len(state[1:, 1]))

        ax.scatter3D(state[1:, 1],
                     state[1:, 0],
                     state[1:, 2],
                     label='state trajectory',
                     s=size_ls,
                     c='tab:blue')
        ax.scatter3D(pred[:-1, 1],
                     pred[:-1, 0],
                     pred[:-1, 2],
                     label='predicted trajectory',
                     s=size_ls,
                     c='tab:orange')

        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        ax.set_zlabel('Z')
        ax.legend()

        plt.savefig(fname)
        plt.close(fig)

    def _evaluate(self, model, dataset):
        # get function handles of loss and metrics
        criterion = torch.nn.MSELoss()

        losses = []
        preds = []

        with torch.no_grad():
            for i in range(len(dataset)):
                state_action, target = dataset.__getitem__(i)

                # batch size = 1
                state_action = state_action[None, ...]
                target = target[None, ...]

                state_action = torch.tensor(state_action).to('cuda')
                target = torch.tensor(target).to('cuda')

                output = model(state_action)
                loss = criterion(output, target)

                if self.learn_residual:
                    new_res = \
                        self.norm.residual_inv_normalize(output.cpu().numpy())
                    new_state = self.norm.inverse_normalize(
                                    state_action.cpu().numpy()[:9],
                                    is_target=True)
                    new_output = new_res + new_state
                else:
                    new_output = \
                        self.norm.inverse_normalize(output.cpu().numpy(),
                                                    is_target=True)

                preds.append(new_output[0])
                losses.append(loss.item())
        return np.array(losses), np.array(preds)

    def _read_single_demo(self, ds_cfg, fname, stats):
        ds_cfg["fnames"] = fname
        ds_cfg["sample_freq"] = 100
        ds_cfg["stats"] = stats

        dataset = DemoDataset(ds_cfg)
        return dataset

    def _build_model(self, cfg):
        if cfg["eval"]["ckpt_pth"] is None:
            ckpt_pth = self._find_ckpt(cfg["eval"]["ckpt_dir"])
        else:
            ckpt_pth = cfg["eval"]["ckpt_pth"]

        model_cfg = cfg["model"]

        # build model architecture, then print to console
        model = MLP(model_cfg["input_dims"], model_cfg["output_dims"])

        ckpt = torch.load(ckpt_pth)
        model.load_state_dict(ckpt["state_dict"])

        model = model.to(self.device)
        if len(self.device_ids) > 1:
            model = torch.nn.DataParallel(model, self.device_ids)
        model.eval()

        # restore stats for data
        dataset_stats = ckpt["dataset_stats"]
        return model, dataset_stats

    def _find_ckpt(self, ckpt_dir):
        ckpt_dir = Path(ckpt_dir)
        ckpt_pths = [pth for pth in list(ckpt_dir.glob("*.pth"))]
        return ckpt_pths[0]

    def _find_demos(self, ds_cfg):
        # for demos for test set
        if len(ds_cfg["fnames"]) == 0:
            ds_root = Path(ds_cfg["root"])
            demos = [pth.name for pth in list(ds_root.glob("*.h5"))]
        else:
            demos = ds_cfg["fnames"]

        num_train_demo = int(len(demos) * 0.8)
        train_demos = (np.random.RandomState(
                        ds_cfg["seed"]).permutation(demos)[:num_train_demo])
        test_demos = (np.random.RandomState(
                        ds_cfg["seed"]).permutation(demos)[num_train_demo:])
        ds_cfg["fnames"] = np.hstack((train_demos, test_demos))
        return ds_cfg["fnames"], train_demos, test_demos