#%%
import torch.nn as nn
import torch
from torch import optim
import matplotlib.pyplot as plt

class Generator(nn.Module):
    """
    生成器Gのクラス
    """
    def __init__(self, nz=100, nch_g=128, nch=1) -> None:
        """
        :param nz: 入力ベクトルzの次元
        :param nch_g: 最終層の入力チャンネル数
        :param nch: 出力画像のチャンネル数
        """
        super(Generator, self).__init__()

        # ニューラルネットワークの構造を定義する
        self.layers = nn.ModuleDict({
            "layer0": nn.Sequential(
                nn.ConvTranspose2d(nz, nch_g * 4, 3, 1, 0),# 転置畳み込み
                nn.BatchNorm2d(nch_g * 4), # バッチノーマライゼーション
                nn.ReLU()
            ), # (B, nz, 1, 1) -> (B, nch_g*4, 3, 3)
            "layer1": nn.Sequential(
                nn.ConvTranspose2d(nch_g * 4, nch_g * 2, 3, 2, 0),
                nn.BatchNorm2d(nch_g * 2),
                nn.ReLU()
            ),
            "layer2": nn.Sequential(
                nn.ConvTranspose2d(nch_g * 2, nch_g, 4, 2, 1),
                nn.BatchNorm2d(nch_g),
                nn.ReLU()
            ),
            "layer3": nn.Sequential(
                nn.ConvTranspose2d(nch_g, nch, 4, 2, 1),
                nn.Tanh()
            ) # (B, nch_g, 14, 14) -> (B, nch, 28, 28)
        })

    def forward(self, z):
        """
        順方向の演算
        :param z: 入力演算
        :return: 生成画像
        """
        for layer in self.layers.values():
            z = layer(z)
        return z
    
#%%
class Discriminator(nn.Module):
    """
    識別器Dのクラス
    """
    def __init__(self, nch=1, nch_d=128) -> None:
        super(Discriminator, self).__init__()

        # ニューラルネットワークの構造を定義する
        self.layers = nn.ModuleDict({
            "layer0": nn.Sequential(
                nn.Conv2d(nch, nch_d, 4, 2, 1),# 畳み込み
                nn.LeakyReLU(negative_slope=0.2)
            ),
            "layer1": nn.Sequential(
                nn.Conv2d(nch_d, nch_d * 2, 4, 2, 1),
                nn.BatchNorm2d(nch_d * 2),
                nn.LeakyReLU(negative_slope=0.2)
            ), # (B, nch_d, 14, 14) -> (B, nch_d*2, 7, 7)
            "layer2": nn.Sequential(
                nn.Conv2d(nch_d * 2, nch_d * 4, 3, 2, 0),
                nn.BatchNorm2d(nch_d * 4),
                nn.LeakyReLU(negative_slope=0.2)
            ), # (B, nch_d * 4, 3, 3) -> (B, 1, 1, 1)
            "layer3": nn.Sequential(
                nn.Conv2d(nch_d * 4, 1, 3, 1, 0),
                nn.Sigmoid() # Sigmoid関数
            )
            # (B, nch_d*4, 3, 3) -> (B, 1, 1, 1)
        })

    def forward(self, x):
        """
        順方向の演算
        :param x: 本物画像あるいは生成画像
        :return: 識別信号
        """
        for layer in self.layers.values():
            x = layer(x)
        return x.squeeze()

#%%
# 生成器G。ランダムベクトルから生成画像を生成する
device = torch.device("cpu")
netG = Generator().to(device)
print(netG)

#%%
netD = Discriminator().to(device)
# netD.apply(weight_init)

#%%
from torchvision import transforms as transforms
from torchvision import datasets as dset

dataset = dset.MNIST(
    root="./minist_root",
    download=True,
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))]))

# 訓練データをセットしたデータローダーを作成する
batch_size= 50
workers = 10

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=int(workers)
)

#%%
criterion = nn.BCELoss() # バイナリクロスエントロピー

# 生成器のエポックごとの画像生成に使用する確認用の固定ノイズ
fixed_noise = torch.randn()
