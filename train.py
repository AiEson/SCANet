import monai
import torch
import argparse

from tqdm import tqdm
import yaml
from utils.losses import ComboLoss
from utils.utils import *
import torch.backends.cudnn as cudnn
from torch.optim import AdamW, SGD, Adam
from torch.utils.data import DataLoader
import models.my_smp as smp
from datasets.whu import WHUDataset
from datasets.mass import MASSDataset
from torch.utils.data import DataLoader
import lightning as L
from monai.inferers import SlidingWindowInferer
import torch.multiprocessing
from torch.nn import init
torch.multiprocessing.set_sharing_strategy("file_system")


parser = argparse.ArgumentParser(
    description="SegRSNet: A powerful and efficient attention network for remote sensing image extraction"
)

parser.add_argument("--config", type=str, required=True)
parser.add_argument("--train-id-path", type=str, required=True)
parser.add_argument("--test-id-path", type=str, required=True)
parser.add_argument("--val_id_path", type=str, required=True)
parser.add_argument("--save-path", type=str, required=True)
parser.add_argument("--local-rank", default=0, type=int)
parser.add_argument("--port", default=None, type=int)
parser.add_argument("--exp_name", default=None, type=str)
parser.add_argument("--decoder_name", default=None, type=str)
parser.add_argument("--encoder_name", default=None, type=str)
parser.add_argument("--dataset", default=None, type=str)
parser.add_argument("--data_root", default=None, type=str)



def main():
    torch.set_float32_matmul_precision('high')
    fabric = L.Fabric(precision="bf16-mixed")
    fabric.launch()
    L.seed_everything(42)
    
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True
    model = None

    if args.decoder_name == "unet" or args.decoder_name is None:
        model = smp.Unet(
            encoder_name=args.encoder_name,
            classes=cfg["nclass"],
            encoder_weights=None,
            in_channels=3,
            encoder_depth=5,
        )
    elif args.decoder_name == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name=args.encoder_name,
            classes=cfg["nclass"],
            encoder_weights=None,
            in_channels=3,
        )
    print("Using {}".format(args.decoder_name))
    logger.info("Total params: {:.1f}M\n".format(count_params(model)))
    optimizer = AdamW(model.parameters(),
        cfg["lr"],
        weight_decay=1e-5
    )

    model = torch.compile(model, mode="reduce-overhead")
    
    # init model with kaiming_normal
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    if os.path.exists(os.path.join(args.save_path, "best.pth")):
        model.load_state_dict(
            torch.load(os.path.join(args.save_path, "best.pth"))["model"], strict=False
        )
        print("---------------- Load best model ----------------")

    model, optimizer = fabric.setup(model, optimizer)
    
    criterion_combo = ComboLoss({"bce": 2, "dice": 2})

    data_root = cfg["data_root"] if args.data_root is None else args.data_root

    if args.dataset == "whu":
        train_set = WHUDataset(
            dataset_root=data_root,
            ids_filepath=args.train_id_path,
            test_mode=False,
            size=cfg["crop_size"],
        )

        val_set = WHUDataset(
            dataset_root=data_root, ids_filepath=args.test_id_path, test_mode=True
        )
    elif args.dataset == "mass":
        train_set = MASSDataset(
            dataset_root=data_root,
            ids_filepath=args.train_id_path,
            test_mode=False,
            size=cfg["crop_size"],
        )

        val_set = MASSDataset(
            dataset_root=data_root, ids_filepath=args.test_id_path, test_mode=True
        )
    
    trainloader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=16,
        drop_last=True,
        shuffle=False,
        prefetch_factor=2,
    )
    trainloader = fabric.setup_dataloaders(trainloader)
    valloader = DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=16,
        drop_last=False,
        shuffle=False,
    )
    valloader = fabric.setup_dataloaders(valloader)
    loader_len = len(trainloader)
    total_iters = loader_len * cfg["epochs"]
    previous_best = 0.0
    epoch = -1

    with tqdm(total=loader_len * cfg["epochs"]) as pbar:
        for epoch in range(epoch + 1, cfg["epochs"]):
            logger.info(
                "\n\n===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.5f}\n".format(
                    epoch, optimizer.param_groups[0]["lr"], previous_best
                )
            )

            total_loss = AverageMeter()
            loader_len = len(trainloader)
            model.train()

            for i, (img, mask, cutmix_box) in enumerate(trainloader):
                pbar.update(1)
                output = model(img)
                loss = criterion_combo(output, mask)
                optimizer.zero_grad()
                fabric.backward(loss)
                optimizer.step()
                total_loss.update(loss.item())
                iters = epoch * loader_len + i
                lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
                optimizer.param_groups[0]["lr"] = lr

            logger.info("\nIters: {:}, Total loss: {:.3f}\n".format(i, total_loss.avg))
            if (epoch % 2 == 0):
                model.eval()
                dice_class = [0] * 1
                iou_class = [0] * 1
                with torch.no_grad():
                    for i, (img, mask) in tqdm(enumerate(valloader)):
                        pred = model(img)
                        pred = (pred.sigmoid() > 0.5).long()

                        cls = 1
                        inter = ((pred == cls) * (mask == cls)).sum().item()
                        union = (pred == cls).sum().item() + (mask == cls).sum().item()
                        dice_class[cls - 1] += (
                            2.0 * inter / union if union != 0 else 1.0
                        )
                        iou_class[cls - 1] += cacl_iou(pred == cls, mask == cls)
                dice_class = [dice / len(valloader) for dice in dice_class]
                iou_class = [iou / len(valloader) for iou in iou_class]
                mean_dice = sum(dice_class) / len(dice_class)
                mean_iou = sum(iou_class) / len(iou_class)

                logger.info(
                    "\n***** Evaluation ***** >>>> MeanIOU: {:.4f}, MeanDice: {:.4f}\n".format(
                        mean_iou, mean_dice
                    )
                )

                is_best = mean_iou > previous_best
                previous_best = max(mean_iou, previous_best)
                checkpoint = {
                    "model": model.state_dict(),
                    "encoder_name": args.encoder_name,
                }
                torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))
                if is_best:
                    logger.info(
                        f"\n\n***** Best MeanScore: {mean_iou:.4f} Saved! ***** \n\n"
                    )
                    torch.save(checkpoint, os.path.join(args.save_path, "best.pth"))


if __name__ == "__main__":
    main()
