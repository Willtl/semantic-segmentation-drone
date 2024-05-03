import os
import time

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

# Set device for model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths
IMAGE_PATH = './dataset/semantic_drone_dataset/original_images/'
MASK_PATH = './dataset/semantic_drone_dataset/label_images_semantic/'

# Number of classes
n_classes = 23


class DroneDataset(Dataset):

    def __init__(self, img_path, mask_path, X, mean, std, transform=None, patch=False):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.patches = patch
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']

        if self.transform is None:
            img = Image.fromarray(img)

        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)
        mask = torch.from_numpy(mask).long()

        if self.patches:
            img, mask = self.tiles(img, mask)

        return img, mask

    def tiles(self, img, mask):
        img_patches = img.unfold(1, 512, 512).unfold(2, 768, 768)
        img_patches = img_patches.contiguous().view(3, -1, 512, 768)
        img_patches = img_patches.permute(1, 0, 2, 3)

        mask_patches = mask.unfold(0, 512, 512).unfold(1, 768, 768)
        mask_patches = mask_patches.contiguous().view(-1, 512, 768)

        return img_patches, mask_patches


def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def mIoU(pred_mask, mask, smooth=1e-10, n_classes=23):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes):  # loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0:  # no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)


class DroneTestDataset(Dataset):

    def __init__(self, img_path, mask_path, X, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']

        if self.transform is None:
            img = Image.fromarray(img)

        mask = torch.from_numpy(mask).long()

        return img, mask


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def create_df():
    name = []
    for dirname, _, filenames in os.walk(IMAGE_PATH):
        for filename in filenames:
            name.append(filename.split('.')[0])
    return pd.DataFrame({'id': name}, index=np.arange(0, len(name)))


def load_model(path):
    return torch.load(path)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs):
    return fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler)


def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, patch=False):
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    val_iou = []
    val_acc = []
    train_iou = []
    train_acc = []
    lrs = []
    best_miou = -np.inf

    model.to(device)
    summary(model, (3, 32, 32))
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        # training loop
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            # training phase
            image_tiles, mask_tiles = data
            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()

                image_tiles = image_tiles.view(-1, c, h, w)
                mask_tiles = mask_tiles.view(-1, h, w)

            image = image_tiles.to(device)
            mask = mask_tiles.to(device)
            # forward
            output = model(image)
            loss = criterion(output, mask)
            # evaluation metrics
            iou_score += mIoU(output, mask)
            accuracy += pixel_accuracy(output, mask)
            # backward
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient

            # step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()

        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0
            # validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    # reshape to 9 patches from single image, delete batch size
                    image_tiles, mask_tiles = data

                    if patch:
                        bs, n_tiles, c, h, w = image_tiles.size()

                        image_tiles = image_tiles.view(-1, c, h, w)
                        mask_tiles = mask_tiles.view(-1, h, w)

                    image = image_tiles.to(device)
                    mask = mask_tiles.to(device)
                    output = model(image)
                    # evaluation metrics
                    val_iou_score += mIoU(output, mask)
                    test_accuracy += pixel_accuracy(output, mask)
                    # loss
                    loss = criterion(output, mask)
                    test_loss += loss.item()

            # calculatio mean for each batch
            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(val_loader))

            # Calculate mIoU for this epoch
            current_miou = val_iou_score / len(val_loader)
            if current_miou > best_miou:
                print('mIoU Improved: {:.3f} >> {:.3f}'.format(best_miou, current_miou))
                best_miou = current_miou
                torch.save(model, f'Unet-Mobilenet_v2_miou-{best_miou:.3f}.pt')

            # iou
            val_iou.append(val_iou_score / len(val_loader))
            train_iou.append(iou_score / len(train_loader))
            train_acc.append(accuracy / len(train_loader))
            val_acc.append(test_accuracy / len(val_loader))
            print("Epoch:{}/{}..".format(e + 1, epochs),
                  "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
                  "Val Loss: {:.3f}..".format(test_loss / len(val_loader)),
                  "Train mIoU:{:.3f}..".format(iou_score / len(train_loader)),
                  "Val mIoU: {:.3f}..".format(val_iou_score / len(val_loader)),
                  "Train Acc:{:.3f}..".format(accuracy / len(train_loader)),
                  "Val Acc:{:.3f}..".format(test_accuracy / len(val_loader)),
                  "Time: {:.2f}m".format((time.time() - since) / 60))

    history = {'train_loss': train_losses, 'val_loss': test_losses,
               'train_miou': train_iou, 'val_miou': val_iou,
               'train_acc': train_acc, 'val_acc': val_acc,
               'lrs': lrs}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    return history


def predict_image_mask_miou(model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        score = mIoU(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score


def predict_image_mask_pixel(model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        acc = pixel_accuracy(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, acc


def miou_score(model, test_set):
    score_iou = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, score = predict_image_mask_miou(model, img, mask)
        score_iou.append(score)
    return score_iou


def pixel_acc(model, test_set):
    accuracy = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, acc = predict_image_mask_pixel(model, img, mask)
        accuracy.append(acc)
    return accuracy


def test(model, X_test):
    print('Testing')
    t_test = A.Resize(768, 1152, interpolation=cv2.INTER_NEAREST)
    test_set = DroneTestDataset(IMAGE_PATH, MASK_PATH, X_test, transform=t_test)

    image, mask = test_set[3]
    pred_mask, score = predict_image_mask_miou(model, image, mask)

    mob_miou = miou_score(model, test_set)
    mob_acc = pixel_acc(model, test_set)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.imshow(image)
    ax1.set_title('Picture')

    ax2.imshow(mask)
    ax2.set_title('Ground truth')
    ax2.set_axis_off()

    ax3.imshow(pred_mask)
    ax3.set_title('UNet-MobileNet | mIoU {:.3f}'.format(score))
    ax3.set_axis_off()

    plt.tight_layout()
    plt.savefig('seg1.png', dpi=300)

    image2, mask2 = test_set[4]
    pred_mask2, score2 = predict_image_mask_miou(model, image2, mask2)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.imshow(image2)
    ax1.set_title('Picture')

    ax2.imshow(mask2)
    ax2.set_title('Ground truth')
    ax2.set_axis_off()

    ax3.imshow(pred_mask2)
    ax3.set_title('UNet-MobileNet | mIoU {:.3f}'.format(score2))
    ax3.set_axis_off()

    image3, mask3 = test_set[6]
    pred_mask3, score3 = predict_image_mask_miou(model, image3, mask3)
    plt.tight_layout()
    plt.savefig('seg2.png', dpi=300)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.imshow(image3)
    ax1.set_title('Picture')

    ax2.imshow(mask3)
    ax2.set_title('Ground truth')
    ax2.set_axis_off()

    ax3.imshow(pred_mask3)
    ax3.set_title('UNet-MobileNet | mIoU {:.3f}'.format(score3))
    ax3.set_axis_off()
    plt.tight_layout()
    plt.savefig('seg3.png', dpi=300)

    print('Test Set mIoU', np.mean(mob_miou))
    print('Test Set Pixel Accuracy', np.mean(mob_acc))


# Define color map dictionary
safe_keys = ['grass', 'gravel', 'paved-area', 'dirt', 'roof']

color_map = {
    'unlabeled': (0, 0, 0),
    'paved-area': (128, 64, 128),
    'dirt': (130, 76, 0),
    'grass': (0, 102, 0),
    'gravel': (112, 103, 87),
    'water': (28, 42, 168),
    'rocks': (48, 41, 30),
    'pool': (0, 50, 89),
    'vegetation': (107, 142, 35),
    'roof': (70, 70, 70),
    'wall': (102, 102, 156),
    'window': (254, 228, 12),
    'door': (254, 148, 12),
    'fence': (190, 153, 153),
    'fence-pole': (153, 153, 153),
    'person': (255, 22, 96),
    'dog': (102, 51, 0),
    'car': (9, 143, 150),
    'bicycle': (119, 11, 32),
    'tree': (51, 51, 0),
    'bald-tree': (190, 250, 190),
    'ar-marker': (112, 150, 146),
    'obstacle': (2, 135, 115),
    'conflicting': (255, 0, 0),
    'safe_landing': (0, 102, 0)
}


def plot_drone_segmentation(folder_path, model_path, safe_landing=True):
    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device)

    model.eval()
    model.to(device)

    # Set the image preprocessing steps, aligning with test set preprocessing
    transform = T.Compose([
        T.Resize((int(704 * 1.2), int(1056 * 1.2)), interpolation=InterpolationMode.NEAREST),
        T.CenterCrop((704, 1056)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Function to unnormalize an image
    def unnormalize(tensor):
        for t, m, s in zip(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
            t.mul_(s).add_(m)
        return tensor

    # List images in the specified directory
    images = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

    for image_name in tqdm(images):
        # Load image
        image_path = os.path.join(folder_path, image_name)
        image = Image.open(image_path).convert('RGB')

        # Preprocess image
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Generate segmentation mask
        with torch.no_grad():
            output = model(input_tensor)
            pred_mask = torch.argmax(output, dim=1).cpu().squeeze(0)

        # Replace classes with 'safe_landing'
        if safe_landing:
            for key in safe_keys:
                pred_mask[pred_mask == list(color_map.keys()).index(key)] = list(color_map.keys()).index('safe_landing')

        # Create colored mask using the color map
        colored_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
        for class_index, class_name in enumerate(color_map.keys()):
            color = color_map[class_name]
            colored_mask[pred_mask == class_index] = color

        # Unnormalize the tensor for visualization
        unnormalized_img = unnormalize(input_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()

        # Plotting the images and masks
        fig, axs = plt.subplots(1, 2, figsize=(30, 10))
        axs[0].imshow(unnormalized_img)
        axs[0].set_title('Transformed Image')
        axs[0].set_axis_off()

        axs[1].imshow(colored_mask)
        axs[1].set_title('Predicted Mask')
        axs[1].set_axis_off()

        # Create legend
        legend_elements = [plt.Rectangle((0, 0), 1, 1, color=np.array(color) / 255, label=label)
                           for label, color in color_map.items()]
        axs[1].legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(-0.1, -0.2), ncol=10, fontsize='small')

        # plt.show()

        plt.savefig(f'{folder_path}/seg_{image_name}', dpi=150)
        plt.close()


def main(perf_training=True, model_path='Unet-Mobilenet.pt', epochs=10, batch_size=3):
    # Data preparation
    df = create_df()
    # split data
    X_train, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=19)

    print('Train Size   : ', len(X_train))
    print('Test Size    : ', len(X_test))

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Augmentations
    # t_train = A.Compose([A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(), A.VerticalFlip(),
    #                      A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
    #                      A.GaussNoise()])

    t_train = A.Compose([
        A.RandomResizedCrop(704, 1056, scale=(0.25, 1.0), ratio=(1.0, 1.0), interpolation=cv2.INTER_NEAREST, always_apply=True, p=1.0, antialias=True),
        A.ColorJitter(brightness=(0.7, 1.3), contrast=(0.7, 1.3), saturation=(0.7, 1.3), hue=(-0.3, 0.3), p=0.75),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GridDistortion(p=0.2),
        A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
        A.GaussNoise()
    ])

    t_val = A.Compose([A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(), A.GridDistortion(p=0.2)])

    # Datasets and Dataloaders
    train_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_train, mean, std, t_train, patch=False)
    val_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_test, mean, std, t_val, patch=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # Model setup
    if perf_training:
        model = smp.Unet('mobilenet_v2', encoder_weights='imagenet', classes=23, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(train_loader))
        history = train_model(model, train_loader, val_loader, criterion, optimizer, sched, epochs=epochs)
        torch.save(model, 'Unet-Mobilenet.pt')
    else:
        model = load_model(model_path)

    # Evaluation
    # test(model, X_test)

    plot_drone_segmentation('./dataset/natsbag_terrains_202-263/', model_path=model_path)


if __name__ == "__main__":
    main(perf_training=False, model_path='Unet-Mobilenet_v2_miou-0.243.pt')
