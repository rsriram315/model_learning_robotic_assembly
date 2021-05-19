import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from copy import deepcopy
from torch.utils.data import DataLoader
from model import MLP
from utils import prepare_device
from dataloaders import DemoDataset, Normalization,\
                        recover_rotation, add_euler_angle


class BaseVisualize:
    def __init__(self, cfg, vis_dir="saved/visualizations"):
        self.cfg = deepcopy(cfg)
        self.vis_cfg = self.cfg["visualization"]

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
        self.norm = None

    def visualize(self):
        cfg = deepcopy(self.cfg)
        model, cfg = self._build_model(cfg)
        self.norm = Normalization(cfg["dataset"]["stats"])

        for fname in self.demo_fnames:
            if fname in self.train_demo_fnames:
                suffix_fname = Path('train') / Path(fname).stem
            elif fname in self.test_demo_fnames:
                suffix_fname = Path('test') / Path(fname).stem

            # read dataset
            dataset = self._read_single_demo(cfg["dataset"], [fname])
            time = dataset.sample_time
            losses_per_demo, preds_per_demo, state_per_demo =\
                self._evaluate(model, dataset)
            preds_per_demo, target_per_demo = \
                self._recover_data(preds_per_demo, state_per_demo)

            if self.vis_cfg["loss"]:
                loss_fname = self.vis_dir / "loss" / suffix_fname
                self._vis_loss(losses_per_demo, time, loss_fname)

            if self.vis_cfg["axis"]:
                axis_fname = self.vis_dir / "axis" / suffix_fname
                self._vis_axis(preds_per_demo[:-1, :],
                               target_per_demo[1:, :],
                               time[1:],
                               axis_fname)

            if self.vis_cfg["trajectory"]:
                traj_fname = self.vis_dir / "trajectory" / suffix_fname
                self._vis_trajectory(preds_per_demo[:-1, :3],
                                     target_per_demo[1:, :3],
                                     traj_fname)

            print(f"... Generated visualization for {fname}\n")

    def _vis_loss(self, loss, time, fname):
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.set_title(f"overall average loss: {np.mean(loss)}")
        ax.scatter(time, loss, s=2)
        ax.set_xlabel("time")
        ax.set_ylabel("average loss")

        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close(fig)

    def _vis_axis(self, pred, target, time, fname, plot_mat=False):
        size = 1
        if plot_mat:
            features = ['pos', 'force', 'matrix R row 1', 'matrix R row 2',
                        'matrix R row 3', 'euler angles']
            figsize = (25, 20)
        else:
            features = ['pos', 'force', 'euler angles']
            figsize = (20, 10)
        axis = ['x', 'y', 'z']

        rows = len(features)
        cols = len(axis)
        fig, axs = plt.subplots(rows, cols, figsize=figsize, sharex='all')

        for r, feature in enumerate(features):
            # not plotting matrix, skip the matrix
            if not plot_mat:
                r = 5 if r > 2 else r
                feature = features[r]
            for c, ax in enumerate(axis):
                idx = c + 3 * r
                axs[r, c].scatter(time,
                                  target[:, idx],
                                  s=size,
                                  c='tab:blue',
                                  label="ground truth")

                axs[r, c].scatter(time[:],
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
        max_size = 10
        size_ls = np.arange(0, max_size, max_size / len(state[1:, 1]))

        ax.scatter3D(state[:, 1],
                     state[:, 0],
                     state[:, 2],
                     label='state trajectory',
                     s=size_ls,
                     c='tab:blue')
        ax.scatter3D(pred[:, 1],
                     pred[:, 0],
                     pred[:, 2],
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

    def _recover_data(self, pred, state):
        curr_state = self.norm.inv_normalize(state)
        recover_output = self.norm.inv_normalize(pred, is_res=True)
        # recover pos and forces
        recover_output[:, :6] += curr_state[:, :6]

        # recover rotation matrix and add euler angles
        recover_target = np.copy(curr_state)  # target is just the next state
        recover_target = add_euler_angle(recover_target)
        recover_output = recover_rotation(recover_output, curr_state)
        recover_output = add_euler_angle(recover_output)
        return recover_output, recover_target

    def _evaluate(self, model, dataset):
        bs = 1024  # batch size
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)
        # get function handles of loss and metrics
        criterion = torch.nn.MSELoss(reduction='none')

        losses = []
        preds = []
        states = []

        with torch.no_grad():
            for _, (state_action, target) in enumerate(dataloader):
                states.extend(state_action[:, None, :15].numpy())
                # targets.extend(target[:, None, :].numpy())

                state_action, target = (state_action.to('cuda'),
                                        target.to('cuda'))
                output = model(state_action)
                loss = criterion(output, target)
                loss = torch.mean(loss, axis=1)

                losses.extend(loss.cpu().numpy())
                preds.extend(output.cpu().numpy()[:, None, :])

        return (np.array(losses), np.array(preds), np.array(states))

    def _read_single_demo(self, ds_cfg, fname):
        ds_cfg = deepcopy(ds_cfg)
        ds_cfg["fnames"] = fname
        sample_freq = ds_cfg["sample_freq"]
        sl_factor = ds_cfg["sl_factor"]
        ds_cfg["sample_freq"] = int(sample_freq / sl_factor)
        ds_cfg["sl_factor"] = 1

        dataset = DemoDataset(ds_cfg)
        return dataset

    def _build_model(self, cfg):
        if cfg["eval"]["ckpt_pth"] is None:
            ckpt_pth = self._find_ckpt(cfg["eval"]["ckpt_dir"])
        else:
            ckpt_pth = cfg["eval"]["ckpt_pth"]

        model_cfg = cfg["model"]

        ckpt = torch.load(ckpt_pth)
        cfg["dataset"]["stats"] = ckpt["dataset_stats"]

        # build model architecture, then print to console
        model = MLP(model_cfg["input_dims"],
                    model_cfg["output_dims"])
        model.load_state_dict(ckpt["state_dict"])

        model = model.to(self.device)
        if len(self.device_ids) > 1:
            model = torch.nn.DataParallel(model, self.device_ids)
        model.eval()
        return model, cfg

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
