from sklearn.metrics import roc_auc_score
import os
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import module.unet as unet
import numpy as np
import module.unet_network as unet_network
import module.unetplusplus as unetplusplus
import module.swin_unet as swin_unet
import module.trans_unet as trans_unet
import module.MultiTaskUnet as MultiTaskUnet
import module.pre_encoder as pre_encoder
import module.trfeplus as trfeplus
import module.FCN as FCN
import module.DAC_Net as dacnet


import modeling.deeplab as deeplab


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vgg_bn_xcrop_dir = r'dendunet_76.pth'

model = pre_encoder.Unet(in_channels=3, out_channels=1)
model.to(device)
model.load_state_dict(torch.load(vgg_bn_xcrop_dir, map_location=device))
model.eval()

def predict_all():
    img_dir = r'dataset/DDTI dataset/fold_5/p_image'
    mask_dir = r'dataset/DDTI dataset/fold_5/p_mask'
    imgs = os.listdir(img_dir)
    for img in imgs:
        img_name = img
        img_path = os.path.join(img_dir, img)
        mask_path = os.path.join(mask_dir, img)
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)

        img = cv2.resize(img, (256, 256))
        img = img.reshape(256, 256, 3)
        img = transforms.ToTensor()(img)
        img = img.reshape(1, 3, 256, 256)
        img = img.to(device, dtype=torch.float32)

        mask = cv2.resize(mask, (256, 256))
        mask = mask.reshape(256, 256, 3)
        mask = mask[:, :, 0]

        output = model(img)
        if isinstance(output, tuple):
            _, pred = output
        else:
            pred = output

        pred = pred.cpu().detach().numpy()
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        pred = pred.reshape(256, 256)

        # 显示图像、掩膜和预测结果
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
        plt.title('Original Image')

        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Ground Truth Mask')

        plt.subplot(1, 3, 3)
        plt.imshow(pred, cmap='gray')
        plt.title('Predicted Mask')

        plt.savefig('results/' + img_name)

        # 计算IoU，Dice，Precision，Recall，Mean Pixel Accuracy (MPA)，auc
        pred = np.where(pred > 0, 1, 0)
        mask = np.where(mask > 0, 1, 0)

        intersection = np.sum(pred * mask)
        union = np.sum(pred) + np.sum(mask) - intersection
        iou = intersection / (union + 1e-6)
        dice = (2 * intersection) / (np.sum(pred) + np.sum(mask) + 1e-6)
        precision = intersection / (np.sum(pred) + 1e-6)
        recall = intersection / (np.sum(mask) + 1e-6)
        MPA = np.sum(pred == mask) / (256 * 256)

        print('name: ' + img_name)
        print('iou: ' + str(iou))
        print('dice: ' + str(dice))
        print('precision: ' + str(precision))
        print('recall: ' + str(recall))
        print('MPA: ' + str(MPA))
        print('--' * 20)

        # 创建并写入csv文件
        with open('results/results.csv', 'a') as f:
            f.write(f'{img_name},{iou},{dice},{precision},{recall},{MPA}\n')


def predict_one():
    img_name = input('image name: ') + '.PNG'
    img_dir = 'dataset/DDTI dataset/fold_' + input('fold: ') + '/p_image/'
    mask_dir = 'dataset/DDTI dataset/fold_' + input('fold: ') + '/p_mask/'

    img = cv2.imread(img_dir + img_name)
    img = cv2.resize(img, (256, 256))
    mask = cv2.imread(mask_dir + img_name)
    mask = cv2.resize(mask, (256, 256))
    img = img.reshape(256, 256, 3)
    img = transforms.ToTensor()(img)
    img = img.reshape(1, 3, 256, 256)
    img = img.to(device, dtype=torch.float32)

    output = model(img)
    if isinstance(output, tuple):
        _, pred = output
        print('class: ' + str(_))
    else:
        pred = output

    pred = pred.cpu().detach().numpy()
    pred[pred >= 0.5] = 255
    pred[pred < 0.5] = 0

    pred = pred.reshape(256, 256)
    plt.imshow(pred, cmap='gray')
    plt.show()

    mask = mask.reshape(256, 256, 3)
    mask = mask[:, :, 0]
    plt.imshow(mask, cmap='gray')
    plt.show()

    image = cv2.imread(img_dir + img_name)
    plt.imshow(image)
    plt.show()

    pred = np.where(pred > 0, 1, 0)
    mask = np.where(mask > 0, 1, 0)

    # 计算IoU，Dice，Precision，Recall，Mean Pixel Accuracy (MPA)，auc
    intersection = np.sum(pred * mask)
    union = np.sum(pred) + np.sum(mask) - intersection
    iou = intersection / (union + 1e-6)
    dice = (2 * intersection) / (np.sum(pred) + np.sum(mask) + 1e-6)
    precision = intersection / (np.sum(pred) + 1e-6)
    recall = intersection / (np.sum(mask) + 1e-6)
    MPA = np.sum(pred == mask) / (256 * 256)

    print('iou: ' + str(iou))
    print('dice: ' + str(dice))
    print('precision: ' + str(precision))
    print('recall: ' + str(recall))
    print('MPA: ' + str(MPA))

if __name__ == '__main__':
    if(input("one or all：")=="one"):
        while True:
            predict_one()
    predict_all()
