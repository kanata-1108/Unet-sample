import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
# import albumentations as albu
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm

class MakeDataset(Dataset):
    def __init__(self, img_dir, msk_dir, transform):
        self.dir_length = len(os.listdir(img_dir))
        self.img_paths = [os.path.join(img_dir, str(index) + '.png') for index in range(self.dir_length)]
        self.msk_paths = [os.path.join(msk_dir, str(index) + '.png') for index in range(self.dir_length)]
        self.transform = transform

    def __getitem__(self, index):
        # https://note.nkmk.me/python-opencv-numpy-color-to-gray/
        # 訓練に用いるデータは厳密で合って欲しいので読み込んだ後にグレースケールに変換する
        img = cv2.imread(self.img_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        msk = cv2.imread(self.msk_paths[index], cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            transformd = self.transform(image = img, mask = msk)
            img, msk = transformd['image'], transformd['mask']

        # albuのToTensorはtorchvisionのToTensorのように規格化してくれないらしい
        img, msk = img / 255., msk / 255.

        # マスクの方にバッチ用の次元がないので追加
        msk = msk.unsqueeze(0)

        return img.to(torch.float32), msk.to(torch.float32)
    
    def __len__(self):
        return self.dir_length


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = "same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = "same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.convblock(x)

        return out

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upconvblock = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners = True),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size = 2, padding = "same"),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        out = self.upconvblock(x)

        return out

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # down
        self.CB1 = ConvBlock(in_channels, 64)
        self.CB2 = ConvBlock(64, 128)
        self.CB3 = ConvBlock(128, 256)
        self.CB4 = ConvBlock(256, 512)
        self.CB5 = ConvBlock(512, 1024)

        # up
        self.UCB1 = UpConvBlock(1024, 512)
        self.CB6 = ConvBlock(1024, 512)
        self.UCB2 = UpConvBlock(512, 256)
        self.CB7 = ConvBlock(512, 256)
        self.UCB3 = UpConvBlock(256, 128)
        self.CB8 = ConvBlock(256, 128)
        self.UCB4 = UpConvBlock(128, 64)
        self.CB9 = ConvBlock(128, 64)

        # other parts
        self.conv = nn.Conv2d(64, out_channels, kernel_size = 1, padding = "same")
        self.pool = nn.MaxPool2d(kernel_size = 2)

    def forward(self, x):
        x1 = self.CB1(x)
        x2 = self.pool(x1)

        x2 = self.CB2(x2)
        x3 = self.pool(x2)

        x3 = self.CB3(x3)
        x4 = self.pool(x3)

        x4 = self.CB4(x4)
        x5 = self.pool(x4)

        x5 = self.CB5(x5)
        z1 = self.UCB1(x5)

        z2 = torch.cat((x4, z1), dim = 1)
        z2 = self.CB6(z2)
        z2 = self.UCB2(z2)

        z3 = torch.cat((x3, z2), dim = 1)
        z3 = self.CB7(z3)
        z3 = self.UCB3(z3)

        z4 = torch.cat((x2, z3), dim = 1)
        z4 = self.CB8(z4)
        z4 = self.UCB4(z4)

        z5 = torch.cat((x1, z4), dim = 1)
        z5 = self.CB9(z5)
        z5 = self.conv(z5)

        return z5  

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dir_fullpath = os.path.dirname(__file__)

    train_path = dir_fullpath + '/Segmentation01/train/'
    test_path = dir_fullpath + '/Segmentation01/test/'

    train_dataset = MakeDataset(train_path + 'org', train_path + 'label', transform = ToTensorV2())
    test_dataset = MakeDataset(test_path + 'org', test_path + 'label', transform = ToTensorV2())
    
    BATCH_SIZE = 8
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers = 2, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, num_workers = 2, shuffle = False)

    model = Unet(1, 1)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.99)

    train_losses = []
    test_losses = []
    EPOCHS = 100

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for (inputs, labels) in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        test_loss = 0

        with torch.no_grad():
            for (inputs, labels) in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                train_loss += loss.item()
            test_loss /= len(test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f'Epoch {epoch + 1} :: train loss {train_loss:.4f}, val loss {test_loss:.4f}')
