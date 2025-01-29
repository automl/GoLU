"""
The following code has been taken from - https://github.com/pytorch/vision/blob/main/references/segmentation/train.py
"""


import datetime
import os
import time
import warnings

from tasks.semantic_segmentation import presets
import torch
import torch.utils.data
import torchvision
from tasks.semantic_segmentation import utils
from tasks.semantic_segmentation.coco_utils import get_coco
from torch import nn
from torch.optim.lr_scheduler import PolynomialLR
from torchvision.transforms import functional as F, InterpolationMode

from tasks.utils.utils import set_seed, load_json, save_results, str_to_bool
from tasks.utils.plots import save_segmentation_plots
from golu.activation_utils import replace_activation_by_torch_module

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import _deeplabv3_resnet
from torchvision.models import resnet50


def get_dataset(args, is_train):
    def sbd(*args, **kwargs):
        kwargs.pop("use_v2")
        return torchvision.datasets.SBDataset(*args, mode="segmentation", **kwargs)

    def voc(*args, **kwargs):
        kwargs.pop("use_v2")
        return torchvision.datasets.VOCSegmentation(*args, **kwargs)

    paths = {
        "voc": (args.data_path, voc, 21),
        "voc_aug": (args.data_path, sbd, 21),
        "coco": (args.data_path, get_coco, 21),
    }
    p, ds_fn, num_classes = paths[args.dataset]

    image_set = "train" if is_train else "val"
    ds = ds_fn(p, image_set=image_set, transforms=get_transform(is_train, args), use_v2=args.use_v2)
    return ds, num_classes


def get_transform(is_train, args):
    if is_train:
        return presets.SegmentationPresetTrain(base_size=520, crop_size=480, backend=args.backend, use_v2=args.use_v2)
    else:
        return presets.SegmentationPresetEval(base_size=520, backend=args.backend, use_v2=args.use_v2)


def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses["out"]

    return losses["out"] + 0.5 * losses["aux"]


def evaluate(model, criterion, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    num_processed_samples = 0
    total_loss = 0.0
    
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target)
            total_loss += loss.item() * image.size(0)

            confmat.update(target.view(image.size(0), -1), output["out"].argmax(1).view(image.size(0), -1))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            num_processed_samples += image.size(0)
            
            torch.cuda.empty_cache()

        confmat.reduce_from_all_processes()

    num_processed_samples = utils.reduce_across_processes(num_processed_samples).item()
    total_loss = utils.reduce_across_processes(total_loss).item()
    total_loss /= num_processed_samples
    acc_global, acc, iu = confmat.compute()
    mean_iou = iu.mean().item()
    
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    return total_loss, mean_iou * 100, acc_global.item() * 100


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, num_classes, scaler=None):
    model.train()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch + 1}]"
    total_loss = 0.0
    num_processed_samples = 0
    
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        
        total_loss += loss.item() * image.size(0)
        num_processed_samples += image.size(0)
        confmat.update(target.view(image.size(0), -1), output["out"].argmax(1).view(image.size(0), -1))
        
        torch.cuda.empty_cache()
    
    metric_logger.synchronize_between_processes()

    total_loss = utils.reduce_across_processes(total_loss).item()
    num_processed_samples = utils.reduce_across_processes(num_processed_samples).item()
    total_loss /= num_processed_samples

    acc_global, acc, iu = confmat.compute()
    mean_iou = iu.mean().item()

    return total_loss, mean_iou * 100, acc_global.item() * 100


def main(args):
    
    model_name = args.model
    dataset = args.dataset
    activation = args.activation
    seed = int(args.seed)
    output_dir = f'{args.output_dir}/{dataset}/{model_name}/{activation}'
    backbone_path = f'{args.backbone_checkpoint_path}/{activation}/checkpoint_{seed}.pth'
    checkpoint_path = f'{output_dir}/chekpoint_{seed}.pth'
    opt_trace_path = f'{output_dir}/opt_trace_{seed}.json'
    results_path = f'{output_dir}/results_{seed}.json'
    plots_path = f'{output_dir}/plot_{seed}.png'
    
    if args.backend.lower() != "pil" and not args.use_v2:
        # TODO: Support tensor backend in V1?
        raise ValueError("Use --use-v2 if you want to use the tv_tensor or tensor backend.")
    if args.use_v2 and args.dataset != "coco":
        raise ValueError("v2 is only support supported for coco dataset for now.")
    
    if output_dir:
        utils.mkdir(output_dir)
    
    set_seed(seed, args.device)

    utils.init_distributed_mode(args)
    print(args)
    
    print(f'Dataset - {dataset} | Model - {model_name} | Activation - {activation} | Seed - {seed}')
    
    opt_trace = {
        'epochs': [],
        'train_loss': [],
        'train_miou': [],
        'train_pixel_accuracy': [],
        'test_loss': [],
        'test_miou': [],
        'test_pixel_accuracy': [],
        'train_time': 0,
        'inference_time': 0
    }
    if os.path.exists(opt_trace_path):
        opt_trace = load_json(path=opt_trace_path)
        print(f'Loaded optimization trace from path - {opt_trace_path}')

    device = torch.device(args.device)
    
    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    dataset, num_classes = get_dataset(args, is_train=True)
    dataset_test, _ = get_dataset(args, is_train=False)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )
    
    print(f"# Train Points - {len(dataset)} | # Test Points - {len(dataset_test)}")

    backbone = resnet50(weights=None, replace_stride_with_dilation=[False, True, True])
    backbone = replace_activation_by_torch_module(
        module=backbone, old_activation=nn.ReLU, new_activation=activation
    )
    if args.distributed:
        backbone = nn.SyncBatchNorm.convert_sync_batchnorm(backbone)
    state_dict = torch.load(backbone_path, map_location=device)
    backbone.load_state_dict(state_dict['model'])
    model = _deeplabv3_resnet(backbone=backbone, num_classes=num_classes, aux=args.aux_loss)    
    model = replace_activation_by_torch_module(
        module=model, old_activation=nn.ReLU, new_activation=activation
    )
    
    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    total_parameters = 0
    trainable_parameters = 0
    for param in model.parameters():
        total_parameters += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(f'# of Total Parameters - {total_parameters / 1000000:.4f} Million')
    print(f'# of Trainable Parameters - {trainable_parameters / 1000000:.4f} Million')
    print(f"The model looks like - \n{model}")

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]
    if args.aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})
    optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    iters_per_epoch = len(data_loader)
    main_lr_scheduler = PolynomialLR(
        optimizer, total_iters=iters_per_epoch * (args.epochs - args.lr_warmup_epochs), power=0.9
    )

    if args.lr_warmup_epochs > 0:
        warmup_iters = iters_per_epoch * args.lr_warmup_epochs
        args.lr_warmup_method = args.lr_warmup_method.lower()
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_iters]
        )
    else:
        lr_scheduler = main_lr_scheduler

    start_epoch = 0
    if opt_trace['epochs'] != []:
        checkpoint = torch.load(checkpoint_path)
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"]
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
        checkpoint = None  # Free memory

    for epoch in range(start_epoch, args.epochs):

        if epoch == 0:
            print(f'Training starts at - {datetime.datetime.now()}')

        # Train engine -------------------------------------------------------------------------------------------------
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_start_time = time.time()
        train_loss, train_miou, train_pixel_accuracy = train_one_epoch(
            model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq, num_classes, scaler
        )
        opt_trace['train_time'] += (time.time() - train_start_time) / 60  # Convert from seconds to mins
        opt_trace['epochs'].append(epoch + 1)
        opt_trace['train_loss'].append(train_loss)
        opt_trace['train_miou'].append(train_miou)
        opt_trace['train_pixel_accuracy'].append(train_pixel_accuracy)
        
        # Inference engine ---------------------------------------------------------------------------------------------
        inference_start_time = time.time()
        test_loss, test_miou, test_pixel_accuracy = evaluate(
            model, criterion, data_loader_test, device=device, num_classes=num_classes
        )
        opt_trace['inference_time'] += (time.time() - inference_start_time) / 60  # Convert from seconds to mins
        opt_trace['test_loss'].append(test_loss)
        opt_trace['test_miou'].append(test_miou)
        opt_trace['test_pixel_accuracy'].append(test_pixel_accuracy)
        
        train_time = opt_trace['train_time']
        inference_time = opt_trace['inference_time']

        print(
            f'Epoch [{epoch + 1}/{args.epochs}] : Train Loss = {train_loss:.4f} | Train mIoU = {train_miou:.4f} | Train PA - {train_pixel_accuracy:.4f} | Test Loss = {test_loss:.4f} | Test mIoU = {test_miou:.4f} | Test PA - {test_pixel_accuracy:.4f} | Train Time = {train_time:.2f} mins | Inference Time - {inference_time:.2f} mins'
        )

        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch + 1,
            "args": args,
        }
        if args.amp:
            checkpoint["scaler"] = scaler.state_dict()
        utils.save_on_master(checkpoint, checkpoint_path)
        if utils.is_main_process():
            save_results(opt_trace, opt_trace_path)
            print("Saved checkpoint!")

    results = dict()
    if utils.is_main_process():
        results['train_loss'] = opt_trace['train_loss'][-1]
        results['train_miou'] = opt_trace['train_miou'][-1]
        results['train_pixel_accuracy'] = opt_trace['train_pixel_accuracy'][-1]
        results['test_loss'] = opt_trace['test_loss'][-1]
        results['test_miou'] = opt_trace['test_miou'][-1]
        results['test_pixel_accuracy'] = opt_trace['test_pixel_accuracy'][-1]
        results['train_time'] = opt_trace['train_time']
        results['inference_time'] = opt_trace['inference_time']
        save_results(results, results_path)
        save_segmentation_plots(
            model_name=model_name, activation=activation, seed=seed, opt_trace=opt_trace, path=plots_path
        )
    
    print("Training completed!")
    
    if args.distributed:
        torch.distributed.destroy_process_group()


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--output-dir", default="./results", type=str, help="path to save outputs")
    parser.add_argument("--activation", default="ReLU", type=str, help="activation to train model with")
    parser.add_argument("--seed", default=0, type=int, help="the seed to train with")
    parser.add_argument("--data-path", default="", type=str, help="dataset path")
    parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
    parser.add_argument("--model", default="deeplabv3_resnet50", type=str, help="model name")
    parser.add_argument("--aux-loss", default="True", type=str_to_bool, help="auxiliary loss")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to run")

    parser.add_argument(
        "-j", "--workers", default=8, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", default="linear", type=str, help="the warmup method (default: linear)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    # Mixed precision training parameters
    parser.add_argument("--amp", default="False", type=str_to_bool, help="Use torch.cuda.amp for mixed precision training")

    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", default="False", type=str_to_bool, help="Use V2 transforms")
    parser.add_argument("--backbone-checkpoint-path", default="./results/imagenet_1k/resnet50", type=str, help="Path to ResNet50 checkpoints")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
