import matplotlib.pyplot as plt
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import DataLoader
from model import MLP
from dataloaders import Normalization, Standardization
from pathlib import Path
from utils import prepare_device
from dataloaders import DemoDataset
from scipy.spatial.transform import Rotation as R


class Visualize:
    def __init__(self, cfg, vis_dir="saved/visualizations"):
        self.cfg = deepcopy(cfg)
        self.vis_cfg = self.cfg["visualization"]
        self.rot_repr = self.cfg["dataset"]["rotation_representation"]

        # create dirs
        self.vis_dir = Path(vis_dir)
        for name in ["loss", "axis", "trajectory"]:
            Path(self.vis_dir / name / 'train').mkdir(parents=True,
                                                      exist_ok=True)
            Path(self.vis_dir / name / 'test').mkdir(parents=True,
                                                     exist_ok=True)

        self.device, self.device_ids = prepare_device(
                                        self.cfg["model"]["n_gpu"])
        self.demo_fnames, self.train_demo_fnames, self.test_demo_fnames =\
            self._find_demos(self.cfg["dataset"])

        self.learn_residual = self.cfg["dataset"]["learn_residual"]
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

            # read dataset
            dataset = self._read_single_demo(self.cfg["dataset"],
                                             [fname], ds_stats)
            time = dataset.sample_time
            losses_per_demo, preds_per_demo, target_per_demo = \
                self._evaluate(model, dataset)

            if self.vis_cfg["loss"]:
                loss_fname = self.vis_dir / "loss" / suffix_fname
                self._vis_loss(losses_per_demo, time, loss_fname)

            if self.vis_cfg["axis"]:
                axis_fname = self.vis_dir / "axis" / suffix_fname
                self._vis_axis(preds_per_demo, target_per_demo, time,
                               axis_fname)

            if self.vis_cfg["trajectory"]:
                traj_fname = self.vis_dir / "trajectory" / suffix_fname
                self._vis_trajectory(preds_per_demo[:, :3],
                                     target_per_demo[:, :3],
                                     traj_fname)

            print(f"... Generated visualization for {fname}")

    def _vis_loss(self, loss, time, fname):
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.set_title(f"sum of the loss: {sum(loss)}")
        ax.scatter(time, loss, s=2)
        ax.set_xlabel("time")
        ax.set_ylabel("loss")

        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close(fig)

    def _vis_axis(self, pred, target, time, fname):
        size = 1
        # features = ['pos', 'force', 'rot_cosine', 'rot_sine', 'euler angles']
        features = ['pos', 'force', 'matrix R row 1', 'matrix R row 2',
                    'matrix R row 3', 'euler angles']
        axis = ['x', 'y', 'z']

        rows = len(features)
        cols = len(axis)
        fig, axs = plt.subplots(rows, cols, figsize=(25, 20), sharex='all')

        for r, feature in enumerate(features):
            for c, ax in enumerate(axis):
                idx = c + 3 * r
                axs[r, c].scatter(time,
                                  target[:, idx],
                                  s=size,
                                  c='tab:blue',
                                  label="ground truth")
                axs[r, c].scatter(time,
                                  pred[:, idx],
                                  s=size,
                                  c='tab:orange',
                                  label="predictions")
                axs[r, c].set_title(f'{feature} {ax} axis')
                axs[r, c].legend()
                if c == 0:
                    axs[r, c].set_ylabel('coordinate')
                if r == rows - 1:
                    axs[r, c].set_xlabel('time')
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close(fig)

    def _vis_trajectory(self, pred, state, fname):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        max_size = 100
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

        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close(fig)

    def _evaluate(self, model, dataset):
        bs = 512  # batch size
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)
        # get function handles of loss and metrics
        criterion = torch.nn.MSELoss(reduction='none')

        losses = []
        preds = []
        targets = []

        with torch.no_grad():
            for _, (state_action, target) in enumerate(dataloader):
                state_action, target = (state_action.to('cuda'),
                                        target.to('cuda'))

                output = model(state_action)
                loss = criterion(output, target)
                loss = torch.sum(loss, dim=1)

                if self.learn_residual:
                    recover_res = \
                        self.norm.res_inv_normalize(output.cpu().numpy())
                    target_res = self.norm.res_inv_normalize(target.cpu()
                                                                   .numpy())
                    recover_state = self.norm.inv_normalize(
                                state_action.cpu().numpy()[:, None, :15])
                    recover_output = recover_res + recover_state
                    recover_target = target_res + recover_state
                else:
                    recover_output = \
                        self.norm.inv_normalize(
                            output.cpu().numpy()[:, None, :])
                    recover_target = \
                        self.norm.inv_normalize(
                            target.cpu().numpy()[:, None, :])

                if self.rot_repr == "6D":
                    # output euler angles
                    pred_angle = \
                        (R.from_matrix(recover_output[:, 6:].reshape(-1, 3, 3))
                          .as_euler('xyz', degrees=True))
                    target_angle = \
                        (R.from_matrix(recover_target[:, 6:].reshape(-1, 3, 3))
                          .as_euler('xyz', degrees=True))

                elif self.rot_repr == "euler_cos_sin":
                    # sine, cosine
                    pred_angle_x = np.arctan2(recover_output[:, 9],
                                              recover_output[:, 6]) * 180 / np.pi
                    pred_angle_y = np.arctan2(recover_output[:, 10],
                                              recover_output[:, 7]) * 180 / np.pi
                    pred_angle_z = np.arctan2(recover_output[:, 11],
                                              recover_output[:, 8]) * 180 / np.pi
                    pred_angle = np.hstack((pred_angle_x[..., None],
                                            pred_angle_y[..., None],
                                            pred_angle_z[..., None]))

                    target_angle_x = np.arctan2(recover_target[:, 9],
                                                recover_target[:, 6]) * 180 / np.pi
                    target_angle_y = np.arctan2(recover_target[:, 10],
                                                recover_target[:, 7]) * 180 / np.pi
                    target_angle_z = np.arctan2(recover_target[:, 11],
                                                recover_target[:, 8]) * 180 / np.pi
                    target_angle = np.hstack((target_angle_x[..., None],
                                              target_angle_y[..., None],
                                              target_angle_z[..., None]))

                losses.extend(loss.cpu().numpy())
                preds.extend(np.hstack((recover_output, pred_angle)))
                targets.extend(np.hstack((recover_target, target_angle)))

        return np.array(losses), np.array(preds), np.array(targets)

    def _read_single_demo(self, ds_cfg, fname, stats):
        ds_cfg = deepcopy(ds_cfg)
        ds_cfg["fnames"] = fname
        sample_freq = ds_cfg["sample_freq"]
        sl_factor = ds_cfg["sl_factor"]
        ds_cfg["sample_freq"] = int(sample_freq / sl_factor)
        ds_cfg["sl_factor"] = 1
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
        ds_stats = ckpt["dataset_stats"]
        return model, ds_stats

    def _find_ckpt(self, ckpt_dir):
        ckpt_dir = Path(ckpt_dir)
        ckpt_pths = [pth for pth in list(ckpt_dir.glob("*.pth"))]
        return ckpt_pths[0]

    def _find_demos(self, ds_cfg):
        # for demos for test set
        ds_cfg = deepcopy(ds_cfg)
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
