import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.constants import precision
from torch.utils.data import DataLoader
import datasets
from sklearn.metrics import f1_score, roc_auc_score


import MultiTaskLoss
import module.unet_network as unet_network
import module.unet as unet
import module.unetplusplus as unetplusplus
import module.swin_unet as swin_unet
import module.trans_unet as trans_unet
import module.pre_encoder as pre_encoder
import module.trfeplus as trfeplus
import module.FCN as FCN
import module.DAC_Net as dacnet

import modeling.deeplab as deeplab


def get_current_seed():
    # Python随机数生成器种子
    python_seed = random.getstate()
    # NumPy随机数生成器种子
    numpy_seed = np.random.get_state()[1][0]
    # PyTorch随机数生成器种子
    torch_seed = torch.initial_seed()
    # 如果使用GPU
    cuda_seed = torch.cuda.seed()
    # 其它随机数生成器种子
    random_seed = torch.random.seed()

    return {
        "Python_seed": python_seed,
        "NumPy_seed": numpy_seed,
        "PyTorch_seed": torch_seed,
        "CUDA_seed": cuda_seed,
        "Random_seed": random_seed
    }

def train(root_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = pre_encoder.Unet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load("dendunet_76.pth", map_location=device),strict=False)
    model.to(device)
    model.train()
    print(model)
    print('device:', device)

    epoch_num = 151
    batch_size = 12
    train_folds = [1,2,3,4]
    test_folds = [5]

    train_dataset = datasets.datasets(root_dir=root_dir, fold_ids=train_folds)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

    test_dataset = datasets.datasets(root_dir=root_dir, fold_ids=test_folds)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

    criterion = MultiTaskLoss.MultiLoss(dice_weight=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.0001)

    print('epoch_num:', epoch_num)
    print('batch_size:', batch_size)
    print('train_folds:', train_folds)
    print('test_folds:', test_folds)
    print('train_dataloader:', len(train_loader))
    print('test_dataloaders:', len(test_loader))
    print('criteria:', criterion)
    print('optimizer:', optimizer)

    # 打印seed
    seeds = get_current_seed()
    print("当前随机种子状态：")
    for key, value in seeds.items():
        print(f"{key}: {value}")


    iou_list = []
    dice_list = []
    loss_list = []
    auc_list = []
    precision_list = []
    recall_list = []
    mpa_list = []

    print('start training......')
    from tqdm import tqdm
    for epoch in range(epoch_num):
        for i, (img, mask) in enumerate(tqdm(train_loader)):
            img = img.to(device)
            mask = mask.to(device)
            output = model(img)

            loss = criterion(output, mask)
            loss_list.append(loss.item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if epoch!=-1:
            print('epoch {}, loss {}'.format(epoch, loss.item()))

            # 使用测试集进行测试
            model.eval()
            with torch.no_grad():
                iou, dice, precision, recall, MPA = 0, 0, 0, 0, 0
                auc_total = []
                total_samples = 0  # 用于计算最终的平均准确度
                
                for i, (img, mask) in enumerate(test_loader):
                    img = img.to(device)
                    mask = mask.to(device)
                    output = model(img)
            
                    # 逐张图像进行评估
                    for j in range(img.size(0)):  # 对每张图像进行评估
                        single_img = img[j].unsqueeze(0)  # 选择一张图像
                        single_mask = mask[j].unsqueeze(0)  # 选择对应的掩膜
            
                        # 获取当前图像的预测结果
                        single_output = output[j].unsqueeze(0)  # 选择当前预测
                        
                        # 将张量转化为平展形式并在GPU上进行计算
                        single_mask_flat = single_mask.flatten()
                        single_output_flat = (single_output.flatten() >= 0.5).float()
            
                        # 计算IoU和Dice
                        intersection = torch.sum(single_output_flat * single_mask_flat)
                        union = torch.sum(single_output_flat) + torch.sum(single_mask_flat) - intersection
                        iou += intersection / (union + 1e-6)
                        dice += (2 * intersection) / (torch.sum(single_output_flat) + torch.sum(single_mask_flat) + 1e-6)
            
                        # 计算Precision和Recall
                        tp = intersection
                        fp = torch.sum(single_output_flat) - tp
                        fn = torch.sum(single_mask_flat) - tp
                        precision += tp / (tp + fp + 1e-6)
                        recall += tp / (tp + fn + 1e-6)
            
                        # Mean Pixel Accuracy (MPA)
                        MPA += torch.mean((single_output_flat == single_mask_flat).float())
            
                        # 计算AUC（仅二分类时）
                        if len(torch.unique(single_mask_flat)) == 2:
                            auc = roc_auc_score(single_mask_flat.cpu().numpy(), single_output_flat.cpu().numpy())
                            auc_total.append(auc)
            
                        total_samples += 1  # 更新总样本数
            
                # 计算平均值
                iou /= total_samples
                dice /= total_samples
                precision /= total_samples
                recall /= total_samples
                MPA /= total_samples
                avg_auc = np.mean(auc_total) if auc_total else 0
            
                # 输出结果
                print(f'IoU: {iou}')
                print(f'Dice: {dice}')
                print(f'AUC: {avg_auc}')
                print(f'Precision: {precision}')
                print(f'Recall: {recall}')
                print(f'Mean Pixel Accuracy: {MPA}')
            
                # 保存指标
                iou_list.append(iou)
                dice_list.append(dice)
                auc_list.append(avg_auc)
                precision_list.append(precision)
                recall_list.append(recall)
                mpa_list.append(MPA)

            
                # Optionally save the model every few epochs
                if epoch % 2 == 0:
                    torch.save(model.state_dict(), f'dendunet_{epoch}.pth')
                    print(f"Saved the model at epoch {epoch}")
            
    # 最终保存模型
    torch.save(model.state_dict(), 'unet.pth')
            
        # 保存日志
    with open('unet.txt', 'w') as f:
        f.write(f'IoU: {iou_list}\n')
        f.write(f'Dice: {dice_list}\n')
        f.write(f'AUC: {auc_list}\n')
        f.write(f'Precision: {precision_list}\n')
        f.write(f'Recall: {recall_list}\n')
        f.write(f'MPA: {mpa_list}\n')


    # 性能曲线
    plt.figure()
    plt.plot(iou_list, label='IoU')
    plt.plot(dice_list, label='Dice')
    plt.plot(auc_list, label='AUC')
    plt.plot(precision_list, label='Precision')
    plt.plot(recall_list, label='Recall')
    plt.plot(mpa_list, label='MPA')
    plt.legend()
    plt.savefig('unet_metrics.png')
    plt.show()


    # loss曲线
    plt.figure()
    plt.plot(loss_list, label='Loss')
    plt.legend()
    plt.savefig('unet_loss.png')
    plt.show()


if __name__ == '__main__':
    root_dir = r"dataset/DDTI dataset"
    train(root_dir)
