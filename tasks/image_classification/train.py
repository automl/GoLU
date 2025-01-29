"""
The following code has been taken from - https://github.com/pytorch/vision/blob/main/references/classification/train.py
"""

import datetime
import os
import time
import datetime
import warnings
import logging

import torch
import torch.utils.data
import torchvision
import torchvision.transforms
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
import torch.distributed as dist
# import intel_extension_for_pytorch as ipex
# import oneccl_bindings_for_pytorch as torch_ccl

import tasks.image_classification.presets as presets
import tasks.image_classification.utils as utils
from tasks.image_classification.sampler import RASampler
from tasks.image_classification.transforms import get_mixup_cutmix
from tasks.utils.plots import save_classification_large_plots
from tasks.utils.utils import set_worker_sharing_strategy, load_json, save_results
from golu.activation_utils import replace_activation_by_torch_module


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None, gacs=1):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    
    optimizer.zero_grad()
    accumulated_loss = 0.0
    accumulated_acc1 = 0.0
    num_datapoints = 0
    
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        if 'cuda' in device:
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                output = model(image)
                loss = criterion(output, target) / gacs
        else:
            with torch.xpu.amp.autocast(enabled=scaler is not None):
                output = model(image)
                loss = criterion(output, target) / gacs

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (i + 1) % gacs == 0:
            if scaler is not None:
                if args.clip_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
            optimizer.zero_grad()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item() * gacs, lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

        accumulated_loss += loss.item() * gacs * image.size(0)
        accumulated_acc1 += acc1.item() * image.size(0)
        num_datapoints += image.size(0)

    accumulated_loss = torch.tensor(accumulated_loss, device=device)
    accumulated_acc1 = torch.tensor(accumulated_acc1, device=device)
    num_datapoints = torch.tensor(num_datapoints, device=device)
    dist.all_reduce(accumulated_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(accumulated_acc1, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_datapoints, op=dist.ReduceOp.SUM)
    avg_loss = accumulated_loss.item() / num_datapoints.item()
    avg_acc1 = accumulated_acc1.item() / num_datapoints.item()

    return avg_loss, avg_acc1 


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    accumulated_loss = 0.0
    accumulated_acc1 = 0.0
    num_datapoints = 0
    
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_datapoints += batch_size
            
            accumulated_loss += loss.item() * batch_size
            accumulated_acc1 += acc1.item() * batch_size
    
    accumulated_loss = torch.tensor(accumulated_loss, device=device)
    accumulated_acc1 = torch.tensor(accumulated_acc1, device=device)
    num_datapoints = torch.tensor(num_datapoints, device=device)
    dist.all_reduce(accumulated_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(accumulated_acc1, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_datapoints, op=dist.ReduceOp.SUM)
    avg_loss = accumulated_loss.item() / num_datapoints.item()
    avg_acc1 = accumulated_acc1.item() / num_datapoints.item()
            
    # gather the stats from all processes

    if 'cuda' in device:
        metric_logger.synchronize_between_processes()
        if utils.is_main_process():
            logging.info(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    
    return avg_loss, avg_acc1


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, args):
    # Data loading code
    if utils.is_main_process():
        logging.info("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    if utils.is_main_process():
        logging.info("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        if utils.is_main_process():
            logging.info(f"Loading dataset_train from {cache_path}")
        # TODO: this could probably be weights_only=True
        dataset, _ = torch.load(cache_path, weights_only=False)
    else:
        # We need a default value for the variables below because args may come
        # from train_quantization.py which doesn't define them.
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = getattr(args, "ra_magnitude", None)
        augmix_severity = getattr(args, "augmix_severity", None)
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
                ra_magnitude=ra_magnitude,
                augmix_severity=augmix_severity,
                backend=args.backend,
                use_v2=args.use_v2,
            ),
        )
        if args.cache_dataset:
            if utils.is_main_process():
                logging.info(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    if utils.is_main_process():
        logging.info(f"Took {time.time() - st} seconds")

    if utils.is_main_process():
        logging.info("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        if utils.is_main_process():
            logging.info(f"Loading dataset_test from {cache_path}")
        # TODO: this could probably be weights_only=True
        dataset_test, _ = torch.load(cache_path, weights_only=False)
    else:
        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            preprocessing = weights.transforms(antialias=True)
            if args.backend == "tensor":
                preprocessing = torchvision.transforms.Compose([torchvision.transforms.PILToTensor(), preprocessing])

        else:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size,
                resize_size=val_resize_size,
                interpolation=interpolation,
                backend=args.backend,
                use_v2=args.use_v2,
            )

        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            preprocessing,
        )
        if args.cache_dataset:
            if utils.is_main_process():
                logging.info(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    if utils.is_main_process():
        logging.info("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):
    
    model_name = args.model
    dataset = args.dataset
    activation = args.activation
    seed = int(args.seed)
    output_dir = f'{args.output_dir}/{dataset}/{model_name}/{activation}'
    checkpoint_path = f'{output_dir}/checkpoint_{seed}.pth'
    opt_trace_path = f'{output_dir}/opt_trace_{seed}.json'
    results_path = f'{output_dir}/results_{seed}.json'
    plots_path = f'{output_dir}/plot_{seed}.png'
    
    if args.output_dir:
        utils.mkdir(output_dir)

    utils.init_distributed_mode(args)
    if utils.is_main_process():
        logging.info(args)
    
    torch.manual_seed(seed)
    if 'cuda' in args.device:
        torch.cuda.manual_seed(seed)
        if torch.cuda.device_count() > 1:
            torch.cuda.manual_seed_all(seed)
    elif 'xpu' in args.device:
        torch.xpu.manual_seed(seed)
        if torch.xpu.device_count() > 1:
            torch.xpu.manual_seed_all(seed)

    device = args.device
    
    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    
    if utils.is_main_process():
        logging.info(f'Dataset - {dataset} | Model - {model_name} | Activation - {activation} | Seed - {seed}')

    opt_trace = {
        'epochs': [],
        'train_loss': [],
        'train_top1_acc': [],
        'test_loss': [],
        'test_top1_acc': [],
        'train_time': 0,
        'inference_time': 0
    }
    if os.path.exists(opt_trace_path):
        opt_trace = load_json(path=opt_trace_path)
        if utils.is_main_process():
            logging.info(f'Loaded optimization trace from path - {opt_trace_path}')

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    num_classes = len(dataset.classes)
    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_classes=num_classes, use_v2=args.use_v2
    )
    if mixup_cutmix is not None:

        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))

    else:
        collate_fn = default_collate
        
    batch_size_per_gpu = int(args.batch_size / (args.world_size * args.gradient_accumulation_steps))
    if utils.is_main_process():
        logging.info(f'Provided Batch Size = {args.batch_size} | World Size = {args.world_size} | Gradient Accumulation Steps = {args.gradient_accumulation_steps} | Batch Size / GPU = {batch_size_per_gpu}')

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
        worker_init_fn=set_worker_sharing_strategy
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size_per_gpu, sampler=test_sampler, num_workers=args.workers, pin_memory=True,
        worker_init_fn=set_worker_sharing_strategy, persistent_workers=True
    )
    
    if utils.is_main_process():
        logging.info(f"# Train Points - {len(dataset)} | # Test Points - {len(dataset_test)}")

    if utils.is_main_process():
        logging.info("Creating model")
    model = torchvision.models.get_model(model_name, weights=args.weights, num_classes=num_classes)
    if 'vit' in model_name:
        model = replace_activation_by_torch_module(model, nn.GELU, activation)
    else:
        model = replace_activation_by_torch_module(model, nn.ReLU, activation)
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    total_parameters = 0
    trainable_parameters = 0
    for param in model.parameters():
        total_parameters += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    if utils.is_main_process():
        logging.info(f'# of Total Parameters - {total_parameters / 1000000:.4f} Million')
        logging.info(f'# of Trainable Parameters - {trainable_parameters / 1000000:.4f} Million')
        logging.info(f"The model looks like - \n{model}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp and 'cuda' in device else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"]
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        else:
            evaluate(model, criterion, data_loader_test, device=device)
        return

    if utils.is_main_process():
        logging.info("Start training")
    for epoch in range(start_epoch, args.epochs):
        
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_start_time = time.time()
        train_loss, train_top1_acc = train_one_epoch(
            model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler, args.gradient_accumulation_steps
        )
        opt_trace['train_time'] += (time.time() - train_start_time) / 60
        opt_trace['epochs'].append(epoch + 1)
        opt_trace['train_loss'].append(train_loss)
        opt_trace['train_top1_acc'].append(train_top1_acc)
        
        lr_scheduler.step()
        
        inference_start_time = time.time()
        if model_name in ['vit_b_32', 'vit_b_16', 'swin_v2_t']:
            test_loss, test_top1_acc = evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        else:
            test_loss, test_top1_acc = evaluate(model, criterion, data_loader_test, device=device)
        opt_trace['inference_time'] += (time.time() - inference_start_time) / 60
        opt_trace['test_loss'].append(test_loss)
        opt_trace['test_top1_acc'].append(test_top1_acc)

        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch + 1
        }
        if model_ema:
            checkpoint["model_ema"] = model_ema.state_dict()
        if scaler:
            checkpoint["scaler"] = scaler.state_dict()
        utils.save_on_master(checkpoint, checkpoint_path)
        if utils.is_main_process():
            save_results(opt_trace, opt_trace_path)
            if utils.is_main_process():
                logging.info("Saved checkpoint!")
        
        train_time = opt_trace['train_time']
        inference_time = opt_trace['inference_time']
        if utils.is_main_process():
            logging.info(f"{datetime.datetime.now()} Epoch [{epoch + 1}/{args.epochs}] - Train Loss - {train_loss:.4f} | Train Top-1 Acc - {train_top1_acc:.4f} | Test Loss - {test_loss:.4f} | Test Top-1 Acc - {test_top1_acc:.4f} | Train Time - {train_time:.4f} | Inference Time - {inference_time:.4f}")

    results = dict()
    results['train_loss'] = opt_trace['train_loss'][-1]
    results['train_top1_acc'] = opt_trace['train_top1_acc'][-1]
    results['test_loss'] = opt_trace['test_loss'][-1]
    results['test_top1_acc'] = opt_trace['test_top1_acc'][-1]
    results['train_time'] = opt_trace['train_time']
    results['inference_time'] = opt_trace['inference_time']
    
    if utils.is_main_process():
        save_results(results, results_path)
        save_classification_large_plots(
            model_name=model_name, activation=activation, seed=seed, opt_trace=opt_trace, path=plots_path
        )
        logging.info(f"Training completed!")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--output-dir", default="./results", type=str, help="path to save outputs")
    parser.add_argument("--data-path", default="", type=str, help="dataset path")
    parser.add_argument("--dataset", default="imagenet_1k", type=str, help="dataset name")
    parser.add_argument("--model", default="", type=str, help="model name")
    parser.add_argument("--activation", default="", type=str, help="activation name")
    parser.add_argument("--seed", default=0, type=int, help="seed number")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu or xpu Default: cuda)")
    parser.add_argument(
        "-gacs", "--gradient-accumulation-steps", default=1, type=int, help="steps for gradient accumulation"
    )
    parser.add_argument(
        "-b", "--batch-size", default=256, type=int, help="images as whole - further handled by gacs and num gpus"
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=8, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
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
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=100, type=int, help="print frequency")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)