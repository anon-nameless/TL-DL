import os
import datetime
import random
import torch
import rasterio
import warnings
import numpy as np
from Models import AMGUnet
from my_dataset import LandslideDataset
from utils import *  # noqa


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def creat_model(name, **kwargs):
    if name == "AMGUnet":
        model = AMGUnet(3, 1, mgb=True)
    return model


def data_transform(img, target):
    mean = (93.818, 99.977, 92.004)
    std = (46.138, 43.456, 41.677)
    return get_transform(img, target, mean=mean, std=std)   # noqa


def main():
    results_file = f"{model_name}_results_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}.txt"
    best_dice = 10000.
    # dataset
    train_dataset = LandslideDataset(train_root, transforms=data_transform)
    val_dataset = LandslideDataset(val_root, transforms=data_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=3,
                                               shuffle=True,
                                               pin_memory=False,
                                               drop_last=True,
                                               )
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             num_workers=3,
                                             pin_memory=False,
                                             )
    # model
    model = creat_model(model_name)
    model.to(device)

    # optimizer & lr_scheduler
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params_to_optimize,
        **optim_params
    )
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer,
        total_iters=epoch_set+20, power=3)

    if resume:
        resume_dict = torch.load(resume)
        model.load_state_dict(resume_dict["model"])
        optimizer.load_state_dict(resume_dict["optimizer"])

    for epoch in range(0, epoch_set):
        # train_one_epoch
        loss_dice, aux_ce, mem, epoch_time = train_one_epoch(  # noqa
            model, optimizer, train_loader, device, epoch
        )  # loss_dict, mem, epoch_time
        # val_one_epoch
        confmat, val_time, dice = val_one_epoch(  # noqa
            model, val_loader, device
        )  # confmat, epoch_time, dice
        lr_scheduler.step()
        lr = lr_scheduler.get_last_lr()

        # record & save_weights
        save = {"model": model.state_dict(),
                "optimizer": optimizer.state_dict()}
        if save_best is True:
            if dice < best_dice and epoch_set/4 < epoch:
                best_dice = dice  # noqa
                torch.save(save,
                           f"save_weights/best_{model_name}_model.pth")
        else:
            torch.save(save,
                       f"save_weights/{model_name}_model.pth")

        with open(os.path.join("save_weights", results_file), "a") as file:
            head = f"epoch: {epoch}, lr: {lr}"
            train_info = f"TRAIN: loss_dice: {loss_dice:4f}, aux_ce: {aux_ce:4f}, " +\
                f"mem: {mem:.2f} GB, train_time: {epoch_time:.1f} min. "
            tn, fp, fn, tp = confmat.ravel()
            val_info = f"VALIDATE: tn:{tn.item():,}, fp:{fp.item():,.0f}, " +\
                f"fn:{fn.item():,.0f}, tp:{tp.item():,.0f}, " +\
                f"dice: {dice:.4f}, val_time: {val_time:.1f} min. "
            file.write(head + "\n" + train_info + "\n" + val_info + "\n\n")


if __name__ == "__main__":
    #  settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1

    train_root = "the path to CAS" 
    val_root = "the path to a dataset" 
    model_name = "AMGUnet"
    optim_params = dict(lr=0.01, momentum=0.9, weight_decay=0.001)
    epoch_set = 2
    resume = None
    save_best = True

    if not os.path.exists("save_weights"):
        os.mkdir("save_weights")
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    set_random_seed(0)
    warnings.simplefilter("ignore",
                          category=rasterio.errors.NotGeoreferencedWarning)
    try:
        main()
    finally:
        if os.name == "posix":
            result_zip = f"result_{datetime.datetime.now().strftime('%Y%m%d')}.zip"
            os.system(f"zip -q -r {result_zip} save_weights")

