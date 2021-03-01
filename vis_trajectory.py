import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from model import MLP
from dataloaders import DemoDataset
from utils import read_json


def main():
    # create dirs
    vis_dir = Path("visualization")
    for name in ["loss", "axis", "trajectory"]:
        Path(vis_dir / name).mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path("saved/ckpts/")
    # load the best(last) model
    ckpt_pth = list(ckpt_dir.glob("*.pth"))[0]

    demos_dir = Path("data/")
    demos_fnames = list(demos_dir.glob("*.h5"))

    model = build_model(ckpt_pth)

    for fname in demos_fnames:
        dataset = read_demo([fname.name])
        loss_ls, pred_pos = evaluate(model, dataset)

        time = dataset.sample_time
        # scale to the prediction magnitude
        state_pos = dataset.states_actions[:, 0] * 10
        # action_pos = dataset.states_actions[:, 1] * 10

        loss_fname = vis_dir / "loss" / fname.stem
        axis_fname = vis_dir / "axis" / fname.stem
        traj_fname = vis_dir / "trajectory" / fname.stem

        vis_loss(loss_ls, time, loss_fname)
        vis_axis(pred_pos, state_pos, time, axis_fname)
        vis_trajectory(pred_pos, state_pos, time, traj_fname)
        print(f"Generated visualization for {fname}")


def vis_loss(loss, time, fname):
    # # normalize loss
    # loss_ls = loss_ls / np.amax(loss_ls)
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.set_title(f"sum of the loss: {sum(loss)}")
    ax.scatter(time, loss, s=2)
    ax.set_xlabel("time")
    ax.set_ylabel("loss")
    plt.savefig(fname)


def vis_axis(pred, state, time, fname, subsample=3):

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    pred = np.array(pred)

    # the predicted states should start with the prediction for t=1 not t=0
    for t in range(1, len(time), subsample):
        size = 2

        ax1.scatter(time[t], pred[t-1, 0], s=size, c='tab:orange')
        ax1.scatter(time[t], state[t, 0], s=size, c='tab:blue')

        ax1.set_title('X axis')
        ax1.set_xlabel('time')
        ax1.set_ylabel('coordinate')

        ax2.scatter(time[t], pred[t-1, 1], s=size, c='tab:orange')
        ax2.scatter(time[t], state[t-1, 1], s=size, c='tab:blue')

        ax2.set_title('Y axis')
        ax2.set_xlabel('time')

        ax3.scatter(time[t], pred[t-1, 2], s=size, c='tab:orange')
        ax3.scatter(time[t], state[t, 2], s=size, c='tab:blue')
        ax3.set_title('Z axis')
        ax3.set_xlabel('time')

    plt.savefig(fname)


def vis_trajectory(pred, state, time, fname, subsample=3):

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    pred = np.array(pred)

    for t in range(1, len(time) - 1, subsample):
        size = t/100
        ax.scatter(pred[t-1, 1], pred[t-1, 0], pred[t-1, 2],
                   s=size, c='tab:orange')
        ax.scatter(state[t, 1], state[t, 0], state[t, 2],
                   s=size, c='tab:blue')
        # ax.scatter(action[t,1], action[t,0], action[t,2],
        #            s=size, c='tab:green')

    ax.scatter(pred[-1, 1], pred[-1, 0], pred[-1, 2],
               label='predicted trajectory', s=size, c='tab:orange')
    ax.scatter(state[-1, 1], state[-1, 0], state[-1, 2],
               label='state trajectory', s=size, c='tab:blue')
    # ax.scatter(action_pos[-1,1], action_pos[-1,0], action_pos[-1,2],
    #            label='action trajectory', s=size, c='tab:green')

    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.set_zlabel('Z')
    ax.legend()
    plt.savefig(fname)


def build_model(ckpt_pth):
    # build model architecture, then print to console
    model = MLP(input_dims=6, output_dims=3)

    ckpt = torch.load(ckpt_pth)
    model.load_state_dict(ckpt["state_dict"])
    print(f"load model from {ckpt_pth}")

    # device = torch.device('cuda')
    model = model.to('cuda')
    model.eval()

    return model


def evaluate(model, dataset):
    # get function handles of loss and metrics
    criterion = torch.nn.MSELoss()

    loss_ls = []
    pred_pos = []

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            state_action, target = dataset.__getitem__(i)
            state_action = torch.tensor(state_action).to('cuda')
            target = torch.tensor(target).to('cuda')

            output = model(state_action)

            loss = criterion(output, target)

            pred_pos.append(output.cpu().numpy()[:3])
            loss_ls.append(loss.item())

    return loss_ls, pred_pos


def read_demo(fname):
    # setup dataloader instances
    cfg = read_json("config.json")
    ds_cfg = cfg["dataset"]
    ds_cfg["params"]["fnames"] = fname
    ds_cfg["params"]["sample_freq"] = 100
    # dataset_cfg["params"]["process"]["normalize"] = false

    dataset = DemoDataset(**ds_cfg["params"])

    return dataset


if __name__ == '__main__':
    main()
