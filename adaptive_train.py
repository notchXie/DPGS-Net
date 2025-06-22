import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler
import datasets

import MultiTaskLoss
import module.MultiTaskUnet as MultiTaskUnet
import module.pre_encoder as pre_encoder


def train(root_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = GradScaler()

    model = pre_encoder.Unet(in_channels=3, out_channels=1)
    # model.load_state_dict(torch.load("endunet_32.pth"), strict=False)
    model.to(device)
    model.train()
    print(model)
    print('device:', device)

    epoch_num = 51
    batch_size = 12
    train_folds = [1, 2]
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
    print('test_dataloader:', len(test_loader))
    print('criteria:', criterion)
    print('optimizer:', optimizer)

    # 指标记录
    iou_list = []
    dice_list = []
    loss_list = []
    auc_list = []
    precision_list = []
    recall_list = []
    mpa_list = []
    test_loss_list = []

    min_test_loss = float('inf')
    best_model_path = ""

    print('start training......')
    from tqdm import tqdm
    for epoch in range(epoch_num):
        model.train()
        for i, (img, mask) in enumerate(tqdm(train_loader)):
            img = img.to(device)
            mask = mask.to(device)
            mask_output = model(img)

            loss = criterion(mask_output, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        print('epoch {}, loss {}'.format(epoch, loss.item()))

        # 测试阶段
        model.eval()
        iou, dice, precision, recall, MPA = 0, 0, 0, 0, 0
        auc_total = []
        class_correct = 0
        total_samples = 0
        total_test_loss = 0

        with torch.no_grad():
            for i, (img, mask) in enumerate(test_loader):
                img = img.to(device)
                mask = mask.to(device)

                output = model(img)

                test_loss = criterion(output, mask)
                total_test_loss += test_loss.item()

                for j in range(img.size(0)):
                    single_mask = mask[j].unsqueeze(0)
                    single_output = output[j].unsqueeze(0)

                    single_mask_flat = single_mask.flatten()
                    single_output_flat = (single_output.flatten() >= 0.5).float()

                    intersection = torch.sum(single_output_flat * single_mask_flat)
                    union = torch.sum(single_output_flat) + torch.sum(single_mask_flat) - intersection
                    iou += intersection / (union + 1e-6)
                    dice += (2 * intersection) / (torch.sum(single_output_flat) + torch.sum(single_mask_flat) + 1e-6)

                    tp = intersection
                    fp = torch.sum(single_output_flat) - tp
                    fn = torch.sum(single_mask_flat) - tp
                    precision += tp / (tp + fp + 1e-6)
                    recall += tp / (tp + fn + 1e-6)
                    MPA += torch.mean((single_output_flat == single_mask_flat).float())

                    if len(torch.unique(single_mask_flat)) == 2:
                        auc = roc_auc_score(single_mask_flat.cpu().numpy(), single_output_flat.cpu().numpy())
                        auc_total.append(auc)

                    total_samples += 1

        # 平均指标
        iou /= total_samples
        dice /= total_samples
        precision /= total_samples
        recall /= total_samples
        MPA /= total_samples
        avg_auc = np.mean(auc_total) if auc_total else 0

        print(f'IoU: {iou}')
        print(f'Dice: {dice}')
        print(f'AUC: {avg_auc}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'MPA: {MPA}')
        print(f'Test Loss (sum): {total_test_loss}')

        iou_list.append(iou)
        dice_list.append(dice)
        auc_list.append(avg_auc)
        precision_list.append(precision)
        recall_list.append(recall)
        mpa_list.append(MPA)
        test_loss_list.append(total_test_loss)

        # 域自适应最佳模型保存
        if total_test_loss < min_test_loss:
            min_test_loss = total_test_loss
            best_model_path = f'unet_best.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model at epoch {epoch} with test loss {min_test_loss}")


    # 最终模型
    torch.save(model.state_dict(), 'unet.pth')

    # 保存日志
    with open('unet.txt', 'w') as f:
        f.write(f'IoU: {iou_list}\n')
        f.write(f'Dice: {dice_list}\n')
        f.write(f'AUC: {auc_list}\n')
        f.write(f'Precision: {precision_list}\n')
        f.write(f'Recall: {recall_list}\n')
        f.write(f'MPA: {mpa_list}\n')
        f.write(f'Test Loss List: {test_loss_list}\n')
        f.write(f'Best Model Path: {best_model_path}, Min Test Loss: {min_test_loss}\n')

    print(f"\nBest model saved at {best_model_path} with minimum test loss {min_test_loss}")

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

    # loss 曲线
    plt.figure()
    plt.plot(loss_list, label='Train Loss')
    plt.plot(test_loss_list, label='Test Loss')
    plt.legend()
    plt.savefig('unet_loss.png')
    plt.show()


if __name__ == '__main__':
    root_dir = r"dataset/DDTI dataset"
    train(root_dir)
