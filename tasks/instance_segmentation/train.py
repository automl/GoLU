r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.

The following code has been taken from https://github.com/pytorch/vision/blob/main/references/detection/train.py
"""

import datetime
import os
import time

from tasks.instance_segmentation import presets
import torch
from torch import nn
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
from tasks.instance_segmentation import utils
from tasks.instance_segmentation.coco_utils import get_coco
from tasks.instance_segmentation.engine import evaluate, train_one_epoch
from tasks.instance_segmentation.group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler
from torchvision.transforms import InterpolationMode
from tasks.instance_segmentation.transforms import SimpleCopyPaste
# import intel_extension_for_pytorch as ipex
# import oneccl_bindings_for_pytorch as torch_ccl

import torchvision.ops.misc as misc_nn_ops
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models.detection.faster_rcnn import _resnet_fpn_extractor
from torchvision.models import resnet50
from torchvision.models.detection import MaskRCNN

from tasks.utils.utils import str_to_bool, load_instances, load_json, set_seed, save_results, set_worker_sharing_strategy
from golu.activation_utils import replace_activation_by_torch_module
from tasks.utils.plots import save_instance_segmentation_plots


def copypaste_collate_fn(batch):
    copypaste = SimpleCopyPaste(blending=True, resize_interpolation=InterpolationMode.BILINEAR)
    return copypaste(*utils.collate_fn(batch))


def get_dataset(is_train, args):
    image_set = "train" if is_train else "val"
    num_classes, mode = {"coco": (91, "instances"), "coco_kp": (2, "person_keypoints")}[args.dataset]
    with_masks = "mask" in args.model
    ds = get_coco(
        root=args.data_path,
        image_set=image_set,
        transforms=get_transform(is_train, args),
        mode=mode,
        use_v2=args.use_v2,
        with_masks=with_masks,
    )
    return ds, num_classes


def get_transform(is_train, args):
    if is_train:
        return presets.DetectionPresetTrain(
            data_augmentation=args.data_augmentation, backend=args.backend, use_v2=args.use_v2
        )
    else:
        return presets.DetectionPresetEval(backend=args.backend, use_v2=args.use_v2)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--output-dir", default="./results", type=str, help="path to save outputs")
    parser.add_argument("--activation", default="ReLU", type=str, help="activation to train model with")
    parser.add_argument("--seed", default=0, type=int, help="the seed to train with")
    parser.add_argument("--data-path", default="", type=str, help="dataset path")
    parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
    parser.add_argument("--model", default="maskrcnn_resnet50_fpn", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=26, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=8, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument(
        "--lr",
        default=0.02,
        type=float,
        help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
    )
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
        "--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    parser.add_argument(
        "--trainable-backbone-layers", default=None, type=int, help="number of trainable layers of backbone"
    )
    parser.add_argument(
        "--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)"
    )
    parser.add_argument(
        "--sync-bn",
        default="True",
        help="Use sync batch norm",
        type=str_to_bool
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    # Mixed precision training parameters
    parser.add_argument("--amp", default="False", type=str_to_bool ,help="Use torch.cuda.amp for mixed precision training")

    # Use CopyPaste augmentation training parameter
    parser.add_argument(
        "--use-copypaste",
        default="False",
        type=str_to_bool,
        help="Use CopyPaste data augmentation. Works only with data-augmentation='lsj'.",
    )

    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", default="False", type=str_to_bool, help="Use V2 transforms")
    parser.add_argument("--backbone-checkpoint-path", default="./results/imagenet_1k/resnet50", type=str, help="Path to ResNet50 checkpoints")

    return parser


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

    if args.backend.lower() == "tv_tensor" and not args.use_v2:
        raise ValueError("Use --use-v2 if you want to use the tv_tensor backend.")
    if args.dataset not in ("coco", "coco_kp"):
        raise ValueError(f"Dataset should be coco or coco_kp, got {args.dataset}")
    if "keypoint" in args.model and args.dataset != "coco_kp":
        raise ValueError("Oops, if you want Keypoint detection, set --dataset coco_kp")
    if args.dataset == "coco_kp" and args.use_v2:
        raise ValueError("KeyPoint detection doesn't support V2 transforms yet")

    if args.output_dir:
        utils.mkdir(output_dir)
    
    set_seed(seed, args.device)
    
    utils.init_distributed_mode(args)
    print(args)
    
    print(f'Dataset - {dataset} | Model - {model_name} | Activation - {activation} | Seed - {seed}')

    opt_trace = {
        'epochs': [],
        'train_loss': [],
        'test_box_map': [],
        'test_mask_map': [],
        'train_time': 0,
        'inference_time': 0
    }
    if os.path.exists(opt_trace_path):
        opt_trace = load_json(path=opt_trace_path)
        print(f'Loaded optimization trace from path - {opt_trace_path}')

    device = args.device
    
    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    # Data loading code
    print("Loading data")

    dataset, num_classes = get_dataset(is_train=True, args=args)
    dataset_test, _ = get_dataset(is_train=False, args=args)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    train_collate_fn = utils.collate_fn
    if args.use_copypaste:
        if args.data_augmentation != "lsj":
            raise RuntimeError("SimpleCopyPaste algorithm currently only supports the 'lsj' data augmentation policies")

        train_collate_fn = copypaste_collate_fn

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers, pin_memory=True, persistent_workers=True,
        collate_fn=train_collate_fn, worker_init_fn=set_worker_sharing_strategy
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, pin_memory=True,
        persistent_workers=True, collate_fn=utils.collate_fn, worker_init_fn=set_worker_sharing_strategy
    )
    
    print(f"# Train Points - {len(dataset)} | # Test Points - {len(dataset_test)}")

    print("Creating model")
    kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}
    if args.data_augmentation in ["multiscale", "lsj"]:
        kwargs["_skip_resize"] = True
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh

    # Load the backbone weights from locally stored path ---------------------------------------------------------------
    if model_name == 'maskrcnn_resnet50_fpn':
        is_trained = True
        trainable_backbone_layers = _validate_trainable_layers(is_trained, kwargs["trainable_backbone_layers"], 5, 3)
        backbone = resnet50(weights=None)
        backbone = replace_activation_by_torch_module(
            module=backbone, old_activation=nn.ReLU, new_activation=activation
        )
        state_dict = torch.load(backbone_path, map_location=device)
        backbone.load_state_dict(state_dict['model'])
        backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
        model = MaskRCNN(backbone, num_classes=num_classes, **kwargs)
        model = replace_activation_by_torch_module(
            module=model, old_activation=nn.ReLU, new_activation=activation
        )
    
    model.to(device)
    
    # Skip this as the layers are frozen
    if args.distributed and args.sync_bn:
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
    
    if args.norm_weight_decay is None:
        parameters = [p for p in model.parameters() if p.requires_grad]
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [args.norm_weight_decay, args.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")

    if args.amp and 'cuda' in args.device:
        scaler = torch.cuda.amp.GradScaler()
    elif args.amp and 'xpu' in args.device:
        scaler = torch.xpu.amp.GradScaler()
    else:
        scaler = None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )

    start_epoch = 0
    if opt_trace['epochs'] != []:
        checkpoint = torch.load(checkpoint_path)
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"]
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
        checkpoint = None

    for epoch in range(start_epoch, args.epochs):
        
        if epoch == 0:
            print(f'Training starts at - {datetime.datetime.now()}')
        
        # Train engine -------------------------------------------------------------------------------------------------
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_start_time = time.time()
        train_loss = train_one_epoch(
            model, optimizer, data_loader, device, epoch, args.print_freq, scaler
        )
        opt_trace['train_time'] += (time.time() - train_start_time) / 60  # Convert from seconds to mins
        opt_trace['epochs'].append(epoch + 1)
        opt_trace['train_loss'].append(train_loss)
        
        lr_scheduler.step()
        
        # evaluate after every epoch
        inference_start_time = time.time()
        test_box_map, test_mask_map = evaluate(
            model, data_loader_test, device=device
        )
        opt_trace['inference_time'] += (time.time() - inference_start_time) / 60  # Convert from seconds to mins
        opt_trace['test_box_map'].append(test_box_map)
        opt_trace['test_mask_map'].append(test_mask_map)
        
        train_time = opt_trace['train_time']
        inference_time = opt_trace['inference_time']

        print(
            f'Epoch [{epoch + 1}/{args.epochs}] : Train Loss = {train_loss:.4f} | Test Box MAP = {test_box_map:.4f} | Test Mask MAP = {test_mask_map:.4f} | Train Time = {train_time:.2f} mins | Inference Time - {inference_time:.2f} mins'
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
        results['test_box_map'] = opt_trace['test_box_map'][-1]
        results['test_mask_map'] = opt_trace['test_mask_map'][-1]
        results['train_time'] = opt_trace['train_time']
        results['inference_time'] = opt_trace['inference_time']
        save_results(results, results_path)
        save_instance_segmentation_plots(
            model_name=model_name, activation=activation, seed=seed, opt_trace=opt_trace, path=plots_path
        )

    print(f"Training completed!")
    
    if args.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
