import argparse
import torch
from tqdm import tqdm
from functools import partial
from model import MLP
from dataloaders import DemoDataset, DemoDataLoader
from utils import read_json
from logger import write_log


def test(cfg_path):
    # setup dataloader instances
    cfg = read_json(cfg_path)
    dataset_cfg = cfg["dataset"]
    dataloader_cfg = cfg["dataloader"]
    test_cfg = cfg["test"]

    write_test_log = partial(write_log, test_cfg["log_file"])

    dataset = DemoDataset(**dataset_cfg["params"])
    test_dataset = dataset.split_train_test(train=False,
                                            seed=dataset_cfg["seed"])
    dataloader = DemoDataLoader(test_dataset, dataloader_cfg)

    # build model architecture, then print to console
    model = MLP(input_dims=12, output_dims=6)
    # get function handles of loss and metrics
    criterion = torch.nn.MSELoss()
    metrics = []

    write_test_log(f'... Loading checkpoint: {test_cfg["ckpt_pth"]}')
    ckpt = torch.load(test_cfg["ckpt_pth"])
    if cfg['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(ckpt["state_dict"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metrics))

    with torch.no_grad():
        for i, (state_action, target) in enumerate(tqdm(dataloader)):            
            state_action, target = state_action.to(device), target.to(device)
            output = model(state_action)

            loss = criterion(output, target)
            batch_size = state_action.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metrics):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(dataloader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() /
        n_samples for i, met in enumerate(metrics)
    })
    print(f"total test samples is {n_samples}")
    write_test_log(f'Loss: {log["loss"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    args = parser.parse_args()
    test(args.config)
