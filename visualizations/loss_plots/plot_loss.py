import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tasks.utils.utils import load_json, create_directory

root_path = './visualizations/loss_plots/saved_plots__'
create_directory(root_path)

seeds = [1, 2, 3]
original_name = {'GoLU': 'GoLU'}

# ResNet-50 on ImageNet-1K ---------------------------------------------------------------------------------------------
activations = ['GoLU', 'GELU', 'ReLU']
colors = sns.color_palette("Set1", len(activations))
resnet50_path = './results/imagenet_1k/resnet50'

epochs = dict.fromkeys(activations, None)
test_loss = dict.fromkeys(activations, None)
test_top1_acc = dict.fromkeys(activations, None)

for activation in activations:
    loss = []
    top1_acc = []
    for seed in seeds:
        file_path = f'{resnet50_path}/{original_name.get(activation, activation)}/opt_trace_{seed}.json'
        opt_trace = load_json(file_path)
        loss.append(opt_trace['test_loss'][30:])
        top1_acc.append(opt_trace['test_top1_acc'][30:])
        if seed == 1:
            epochs[activation] = opt_trace['epochs'][30:]
    test_loss[activation] = np.mean(loss, axis=0)
    test_top1_acc[activation] = np.mean(top1_acc, axis=0)
    

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for activation, color in zip(activations, colors):
    axes[0].plot(epochs[activation], test_loss[activation], label=activation, color=color, linewidth=2.5)
    axes[1].plot(epochs[activation], test_top1_acc[activation], label=activation, color=color, linewidth=2.5)

axes[0].set_title('ResNet-50 ImageNet-1K Inference Loss', fontsize=20)
axes[0].set_xlabel('Epochs', fontsize=20)
axes[0].set_ylabel('Loss', fontsize=20)

axes[1].set_title('ResNet-50 ImageNet-1K Inference Top-1 Accuracy', fontsize=20)
axes[1].set_xlabel('Epochs', fontsize=20)
axes[1].set_ylabel('Top-1 Accuracy (in %age)', fontsize=20)

for ax in axes:
    ax.grid(True, color='#afbab2', linewidth=1.5)

axes[0].legend(fontsize=20)
axes[1].legend(fontsize=20)

plt.tight_layout()
plt.savefig(f'{root_path}/resnet50_plots.png')




# ViT-B/32 on ImageNet-1K ----------------------------------------------------------------------------------------------
activations = ['GoLU', 'GELU', 'ReLU']
colors = sns.color_palette("Set1", len(activations))
vitb32_path = './results/imagenet_1k/vit_b_32'

epochs = dict.fromkeys(activations, None)
test_loss = dict.fromkeys(activations, None)
test_top1_acc = dict.fromkeys(activations, None)

for activation in activations:
    loss = []
    top1_acc = []
    for seed in seeds:
        file_path = f'{vitb32_path}/{original_name.get(activation, activation)}/opt_trace_{seed}.json'
        opt_trace = load_json(file_path)
        loss.append(opt_trace['test_loss'][150:])
        top1_acc.append(opt_trace['test_top1_acc'][150:])
        if seed == 1:
            epochs[activation] = opt_trace['epochs'][150:]
    test_loss[activation] = np.mean(loss, axis=0)
    test_top1_acc[activation] = np.mean(top1_acc, axis=0)
    

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for activation, color in zip(activations, colors):
    axes[0].plot(epochs[activation], test_loss[activation], label=activation, color=color, linewidth=2.5)
    axes[1].plot(epochs[activation], test_top1_acc[activation], label=activation, color=color, linewidth=2.5)

axes[0].set_title('ViT-B/32 ImageNet-1K Inference Loss', fontsize=20)
axes[0].set_xlabel('Epochs', fontsize=20)
axes[0].set_ylabel('Loss', fontsize=20)

axes[1].set_title('ViT-B/32 ImageNet-1K Inference Top-1 Accuracy', fontsize=20)
axes[1].set_xlabel('Epochs', fontsize=20)
axes[1].set_ylabel('Top-1 Accuracy (in %age)', fontsize=20)

for ax in axes:
    ax.grid(True, color='#afbab2', linewidth=1.5)

axes[0].legend(fontsize=20)
axes[1].legend(fontsize=20)

plt.tight_layout()
plt.savefig(f'{root_path}/vit_b_32_plots.png')




# BabyGPT on TinyStories -----------------------------------------------------------------------------------------------
activations = ['GoLU', 'GELU', 'Swish']
colors = sns.color_palette("Set1", len(activations))
babygpt_path = './results/tiny_stories/baby_gpt'

iterations = dict.fromkeys(activations, None)
test_loss = dict.fromkeys(activations, None)
test_acc = dict.fromkeys(activations, None)

for activation in activations:
    loss = []
    acc = []
    for seed in seeds:
        file_path = f'{babygpt_path}/{original_name.get(activation, activation)}/opt_trace_{seed}.json'
        opt_trace = load_json(file_path)
        loss.append(opt_trace['test_loss'][int(len(opt_trace['iteration'])*2/3):])
        acc.append(opt_trace['test_acc'][int(len(opt_trace['iteration'])*2/3):])
        if seed == 1:
            iterations[activation] = opt_trace['iteration'][int(len(opt_trace['iteration'])*2/3):]
    test_loss[activation] = np.mean(loss, axis=0)
    test_acc[activation] = np.mean(acc, axis=0)
    

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for activation, color in zip(activations, colors):
    axes[0].plot(iterations[activation], test_loss[activation], label=activation, color=color, linewidth=2.5)
    axes[1].plot(iterations[activation], test_acc[activation], label=activation, color=color, linewidth=2.5)

axes[0].set_title('BabyGPT TinyStories Inference Loss', fontsize=20)
axes[0].set_xlabel('Iterations', fontsize=20)
axes[0].set_ylabel('Loss', fontsize=20)

axes[1].set_title('BabyGPT TinyStories Inference Token Accuracy', fontsize=20)
axes[1].set_xlabel('Iterations', fontsize=20)
axes[1].set_ylabel('Token Accuracy (in %age)', fontsize=20)

for ax in axes:
    ax.grid(True, color='#afbab2', linewidth=1.5)

axes[0].legend(fontsize=20)
axes[1].legend(fontsize=20)

plt.tight_layout()
plt.savefig(f'{root_path}/baby_gpt_plots.png')



# GPT2-S on OpenWebText ------------------------------------------------------------------------------------------------
activations = ['GoLU', 'GELU', 'Swish']
colors = sns.color_palette("Set1", len(activations))
gpt2_path = './results/open_web_text/gpt2s'

iterations = dict.fromkeys(activations, None)
test_loss = dict.fromkeys(activations, None)
test_acc = dict.fromkeys(activations, None)
iters = 5

for activation in activations:
    loss = []
    acc = []
    for seed in seeds:
        file_path = f'{gpt2_path}/{original_name.get(activation, activation)}/opt_trace_{seed}.json'
        opt_trace = load_json(file_path)
        it = opt_trace['iteration'][int(len(opt_trace['iteration'])*2/3):]
        lo = opt_trace['test_loss'][int(len(opt_trace['iteration'])*2/3):]
        ac = opt_trace['test_acc'][int(len(opt_trace['iteration'])*2/3):]
        if len(iterations) % iters !=0:
             it = it[iters:]
             lo = lo[iters:]
             ac = ac[iters:]
        it = [j for i, j in enumerate(it) if (i+1)%iters == 0]
        lo = [j for i, j in enumerate(lo) if (i+1)%iters == 0]
        ac = [j for i, j in enumerate(ac) if (i+1)%iters == 0]
        loss.append(lo)
        acc.append(ac)
        if seed == 1:
            iterations[activation] = it
    test_loss[activation] = np.mean(loss, axis=0)
    test_acc[activation] = np.mean(acc, axis=0)
    

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for activation, color in zip(activations, colors):
    axes[0].plot(iterations[activation], test_loss[activation], label=activation, color=color, linewidth=2.5)
    axes[1].plot(iterations[activation], test_acc[activation], label=activation, color=color, linewidth=2.5)

axes[0].set_title('GPT2-S OpenWebText Inference Loss', fontsize=20)
axes[0].set_xlabel('Iterations', fontsize=20)
axes[0].set_ylabel('Loss', fontsize=20)

axes[1].set_title('GPT2-S OpenWebText Inference Token Accuracy', fontsize=20)
axes[1].set_xlabel('Iterations', fontsize=20)
axes[1].set_ylabel('Token Accuracy (in %age)', fontsize=20)

for ax in axes:
    ax.grid(True, color='#afbab2', linewidth=1.5)

axes[0].legend(fontsize=20)
axes[1].legend(fontsize=20)

plt.tight_layout()
plt.savefig(f'{root_path}/gpt2_plots.png')




# DeepLabV3-ResNet-50 on MS-COCO ---------------------------------------------------------------------------------------
activations = ['GoLU', 'GELU', 'ReLU']
colors = sns.color_palette("Set1", len(activations))

# LR = 0.01
dlv3_path = './results/coco/deeplabv3_resnet50_0.01'

epochs = dict.fromkeys(activations, None)
test_loss = dict.fromkeys(activations, None)
test_miou = dict.fromkeys(activations, None)

for activation in activations:
    loss = []
    miou = []
    for seed in seeds:
        file_path = f'{dlv3_path}/{original_name.get(activation, activation)}/opt_trace_{seed}.json'
        opt_trace = load_json(file_path)
        loss.append(opt_trace['test_loss'])
        miou.append(opt_trace['test_miou'])
        if seed == 1:
            epochs[activation] = opt_trace['epochs']
    test_loss[activation] = np.mean(loss, axis=0)
    test_miou[activation] = np.mean(miou, axis=0)
    

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for activation, color in zip(activations, colors):
    axes[0].plot(epochs[activation], test_loss[activation], label=activation, color=color, linewidth=2.5)
    axes[1].plot(epochs[activation], test_miou[activation], label=activation, color=color, linewidth=2.5)

axes[0].set_title('DeepLabV3-RN50 MS-COCO Inference Loss | LR=0.01', fontsize=18)
axes[0].set_xlabel('Epochs', fontsize=20)
axes[0].set_ylabel('Loss', fontsize=20)

axes[1].set_title('DeepLabV3-RN50 MS-COCO Inference mIoU | LR=0.01', fontsize=18)
axes[1].set_xlabel('Epochs', fontsize=20)
axes[1].set_ylabel('mIoU', fontsize=20)

for ax in axes:
    ax.grid(True, color='#afbab2', linewidth=1.5)

axes[0].legend(fontsize=20)
axes[1].legend(fontsize=20)

plt.tight_layout()
plt.savefig(f'{root_path}/dlv3_0.01_plots.png')


# LR=0.02
dlv3_path = './results/coco/deeplabv3_resnet50_0.02'

epochs = dict.fromkeys(activations, None)
test_loss = dict.fromkeys(activations, None)
test_miou = dict.fromkeys(activations, None)

for activation in activations:
    loss = []
    miou = []
    for seed in seeds:
        file_path = f'{dlv3_path}/{original_name.get(activation, activation)}/opt_trace_{seed}.json'
        opt_trace = load_json(file_path)
        loss.append(opt_trace['test_loss'])
        miou.append(opt_trace['test_miou'])
        if seed == 1:
            epochs[activation] = opt_trace['epochs']
    test_loss[activation] = np.mean(loss, axis=0)
    test_miou[activation] = np.mean(miou, axis=0)
    

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for activation, color in zip(activations, colors):
    axes[0].plot(epochs[activation], test_loss[activation], label=activation, color=color, linewidth=2.5)
    axes[1].plot(epochs[activation], test_miou[activation], label=activation, color=color, linewidth=2.5)

axes[0].set_title('DeepLabV3-RN50 MS-COCO Inference Loss | LR=0.02', fontsize=20)
axes[0].set_xlabel('Epochs', fontsize=20)
axes[0].set_ylabel('Loss', fontsize=20)

axes[1].set_title('DeepLabV3-RN50 MS-COCO Inference mIoU | LR=0.02', fontsize=20)
axes[1].set_xlabel('Epochs', fontsize=20)
axes[1].set_ylabel('mIoU', fontsize=20)

for ax in axes:
    ax.grid(True, color='#afbab2', linewidth=1.5)

axes[0].legend(fontsize=20)
axes[1].legend(fontsize=20)

plt.tight_layout()
plt.savefig(f'{root_path}/dlv3_0.02_plots.png')




# Faster-RCNN-FPN-ResNet-50 and RetinaNet-FPN-ResNet-50 on MS-COCO -----------------------------------------------------
activations = ['GoLU', 'GELU', 'ReLU']
colors = sns.color_palette("Set1", len(activations))
frcnn_path = './results/coco/fasterrcnn_resnet50_fpn'
retinanet_path = './results/coco/retinanet_resnet50_fpn_v2'

epochs = dict.fromkeys(activations, None)
frcnn_test_box_map = dict.fromkeys(activations, None)
retinanet_test_box_map = dict.fromkeys(activations, None)

for activation in activations:
    frcnn_box_map = []
    retinanet_box_map = []
    for seed in seeds:
        
        file_path = f'{frcnn_path}/{original_name.get(activation, activation)}/opt_trace_{seed}.json'
        opt_trace = load_json(file_path)
        frcnn_box_map.append([value * 100 for value in opt_trace['test_box_map']])
        
        file_path = f'{retinanet_path}/{original_name.get(activation, activation)}/opt_trace_{seed}.json'
        opt_trace = load_json(file_path)
        retinanet_box_map.append([value * 100 for value in opt_trace['test_box_map']])
        
        if seed == 1:
            epochs[activation] = opt_trace['epochs']
    frcnn_test_box_map[activation] = np.mean(frcnn_box_map, axis=0)
    retinanet_test_box_map[activation] = np.mean(retinanet_box_map, axis=0)
    

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for activation, color in zip(activations, colors):
    axes[0].plot(epochs[activation], frcnn_test_box_map[activation], label=activation, color=color, linewidth=2.5)
    axes[1].plot(epochs[activation], retinanet_test_box_map[activation], label=activation, color=color, linewidth=2.5)

axes[0].set_title('Faster R-CNN-FPN-RN50 MS-COCO Inference Box mAP', fontsize=20)
axes[0].set_xlabel('Epochs', fontsize=20)
axes[0].set_ylabel('Box mAP', fontsize=20)

axes[1].set_title('RetinaNet-FPN-RN50 MS-COCO Inference Box mAP', fontsize=20)
axes[1].set_xlabel('Epochs', fontsize=20)
axes[1].set_ylabel('Box mAP', fontsize=20)

for ax in axes:
    ax.grid(True, color='#afbab2', linewidth=1.5)

axes[0].legend(fontsize=20)
axes[1].legend(fontsize=20)

plt.tight_layout()
plt.savefig(f'{root_path}/obj_dec_plots.png')




# Mask-RCNN-FPN-ResNet-50 on MS-COCO -----------------------------------------------------------------------------------
activations = ['GoLU', 'GELU', 'ReLU']
colors = sns.color_palette("Set1", len(activations))
mrcnn_path = './results/coco/maskrcnn_resnet50_fpn'

epochs = dict.fromkeys(activations, None)
test_box_map = dict.fromkeys(activations, None)
test_mask_map = dict.fromkeys(activations, None)

for activation in activations:
    box_map = []
    mask_map = []
    for seed in seeds:
        file_path = f'{mrcnn_path}/{original_name.get(activation, activation)}/opt_trace_{seed}.json'
        opt_trace = load_json(file_path)
        box_map.append([value * 100 for value in opt_trace['test_box_map']])
        mask_map.append([value * 100 for value in opt_trace['test_mask_map']])
        if seed == 1:
            epochs[activation] = opt_trace['epochs']
    test_box_map[activation] = np.mean(box_map, axis=0)
    test_mask_map[activation] = np.mean(mask_map, axis=0)
    

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for activation, color in zip(activations, colors):
    axes[0].plot(epochs[activation], test_box_map[activation], label=activation, color=color, linewidth=2.5)
    axes[1].plot(epochs[activation], test_mask_map[activation], label=activation, color=color, linewidth=2.5)

axes[0].set_title('Mask R-CNN-FPN-RN50 MS-COCO Inference Box mAP', fontsize=18)
axes[0].set_xlabel('Epochs', fontsize=20)
axes[0].set_ylabel('Box mAP', fontsize=20)

axes[1].set_title('Mask R-CNN-FPN-RN50 MS-COCO Inference Mask mAP', fontsize=18)
axes[1].set_xlabel('Epochs', fontsize=20)
axes[1].set_ylabel('Mask mAP', fontsize=20)

for ax in axes:
    ax.grid(True, color='#afbab2', linewidth=1.5)

axes[0].legend(fontsize=20)
axes[1].legend(fontsize=20)

plt.tight_layout()
plt.savefig(f'{root_path}/ins_seg_plots.png')




# DDPM on MS-COCO ------------------------------------------------------------------------------------------------------
activations = ['GoLU', 'GELU', 'Swish']
colors = sns.color_palette("Set1", len(activations))
ddpm_default_lr_path = './results/celeba/ddpm_0.0003'
ddpm_best_lr_path = './results/celeba/ddpm_0.001'

epochs_default = dict.fromkeys(activations, None)
test_loss_default = dict.fromkeys(activations, None)

epochs_best = dict.fromkeys(activations, None)
test_loss_best = dict.fromkeys(activations, None)

for activation in activations:
    loss_default = []
    loss_best = []
    for seed in seeds:
        
        file_path = f'{ddpm_default_lr_path}/{original_name.get(activation, activation)}/opt_trace_{seed}.json'
        opt_trace = load_json(file_path)
        loss_default.append(opt_trace['test_loss'][20:])
        if seed == 1:
            epochs_default[activation] = opt_trace['epoch'][20:]
        
        file_path = f'{ddpm_best_lr_path}/{original_name.get(activation, activation)}/opt_trace_{seed}.json'
        opt_trace = load_json(file_path)
        loss_best.append(opt_trace['test_loss'][20:])
        if seed == 1:
            epochs_best[activation] = opt_trace['epoch'][20:]
            
    test_loss_default[activation] = np.mean(loss_default, axis=0)
    test_loss_best[activation] = np.mean(loss_best, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for activation, color in zip(activations, colors):
    axes[0].plot(epochs_default[activation], test_loss_default[activation], label=activation, color=color, linewidth=2.5)
    axes[1].plot(epochs_best[activation], test_loss_best[activation], label=activation, color=color, linewidth=2.5)

axes[0].set_title('DDPM CelebA Inference Loss | LR=0.0003', fontsize=18)
axes[0].set_xlabel('Epochs', fontsize=20)
axes[0].set_ylabel('Loss', fontsize=20)

axes[1].set_title('DDPM CelebA Inference Loss | LR=0.001', fontsize=18)
axes[1].set_xlabel('Epochs', fontsize=20)
axes[1].set_ylabel('Loss', fontsize=20)

for ax in axes:
    ax.grid(True, color='#afbab2', linewidth=1.5)

axes[0].legend(fontsize=20)
axes[1].legend(fontsize=20)
plt.tight_layout()
plt.savefig(f'{root_path}/ddpm_plots.png')
