import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

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
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = "same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
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
        self.dropout = nn.Dropout()
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
        x5 = self.dropout(x5)

        z1 = self.UCB1(x5)
        z1 = torch.cat((x4, z1), dim = 1)
        z1 = self.CB6(z1)

        z2 = self.UCB2(z1)
        z2 = torch.cat((x3, z2), dim = 1)
        z2 = self.CB7(z2)

        z3 = self.UCB3(z2)
        z3 = torch.cat((x2, z3), dim = 1)
        z3 = self.CB8(z3)

        z4 = self.UCB4(z3)
        z4 = torch.cat((x1, z4), dim = 1)
        z4 = self.CB9(z4)

        z = self.conv(z4)
        z = F.sigmoid(z)

        return z

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

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)

    train_losses = []
    test_losses = []
    EPOCHS = 200

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for (inputs, labels) in train_loader:
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
                inputs, labels = inputs.to(device).float(), labels.to(device).float()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
            test_loss /= len(test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f'Epoch {epoch + 1} :: train loss {train_loss:.6f}, val loss {test_loss:.6f}')

    # 結果を格納するディレクトリの作成
    result_savedir = dir_fullpath + '/result'
    if os.path.exists(result_savedir):
        pass
    else:
        os.mkdir(result_savedir)
    
    # 結果の描画
    plt.plot(range(EPOCHS), train_losses, c = 'orange', label = 'train loss')
    plt.plot(range(EPOCHS), test_losses, c = 'blue', label = 'test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend()
    plt.title('loss')
    plt.savefig(result_savedir + '/loss.png')

    # Dataloaderから１バッチ取り出す
    x_batch, y_batch = next(iter(test_loader))

    # 推論
    model.eval()
    preds = model(x_batch.to(device))
    preds = preds.to('cpu').detach().numpy()

    img = np.ones((256, 256 * 3))

    img_savedir = result_savedir + '/pred_img'
    if os.path.exists(img_savedir):
        pass
    else:
        os.mkdir(img_savedir)

    for index, i in enumerate(range(BATCH_SIZE)):
        img[:, :256] = x_batch[i, 0, :, :].numpy()
        img[:, 256 :256 * 2] = y_batch[i, 0, :, :].numpy()
        img[:, 256 * 2 :] = preds[i, 0, :, :]
        plt.figure(figsize = (9, 3))
        plt.imshow(img, cmap = 'gray')
        plt.savefig(img_savedir + '/' + str(index) + '.png')
        