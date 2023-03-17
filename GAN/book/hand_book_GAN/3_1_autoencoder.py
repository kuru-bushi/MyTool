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

#%%p69
# 生成器G。ランダムベクトルから生成画像を生成する
device = torch.device("cpu")
netG = Generator().to(device)
print(netG)

#%%p70
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

# オプティマイザのセットアップ
lr = 0.001
beta1 = 0.9

display_interval = 10

optimizerD = optim.Adam(
    netD.parameters(),
    lr = lr,
    betas=(beta1, 0.999)
)
optimizerG = optim.Adam(
    netG.parameters(),
    lr=lr,
    betas=(beta1, 0.999),
    weight_decay = 1e-5,
)

n_epoch = 50
for epoch in range(n_epoch):
    for itr, data in enumerate(dataloader):
        real_image = data[0].to(device) # 本物画像
        sample_size = real_image.size(0) # 画像枚数

        # 標準正規分布からノイズ生成
        noise = torch.randn(sample_size, nz, 1, 1, device=device)
        # 本物画像に対する識別信号の目標値[1]
        real_target = torch.full(
            (sample_size, ),
            1.,
            device=device
        )
        fake_target = torch.full(
            (sample_size, ),
            0.,
            device=device
        )
        ##############################
        # #識別器Dの更新 p74
        ##############################
        netD.zero_grad() # 勾配の初期化

        output = netD(real_image) # 識別器Dで生成画像に対する識別信号を出力

        errD_real = criterion(output, real_target) # 生成画像に対する識別信号の損失関数

        D_x = output.mean().item()# 本物画像の識別信号の平均

        fake_image = netG(noise) # 生成器Gでノイズから生成画像を生成

        output = netD(fake_image.detach())

        errD_fake = criterion(output, fake_target)

        D_G_z1 = output.mean().item() 

        errD = errD_real + errD_fake # 識別器Dの全体の損失
        errD.backward() # 誤差逆伝播
        optimizerD.step() # Dのパラメタを更新

        ##############################
        # #識別器Dの更新 p74
        ##############################
        netG.zero_grad() # 勾配の初期化
        
        output = netD(fake_image) # 更新した識別器Dで改めて生成画像に対する対する

        errG = criterion(output, real_target) # 生成器Gの損失値。Dに生成画像を本物画像と誤認させるたいため目標値は1

        errG.backward() # 誤差逆伝播
        D_G_z2 = output.mean().item() # 更新した識別器Dによる生成画像の識別信号の平均

        optimizerG.step() # Gのパラメータを更新

        if itr % display_interval == 0:
            print(f"[{epoch + 1}/ {n_epoch}][{itr + 1}/{len(dataloader)} Loss_D: {errD.item()}, Loss_G: {errG.item()} D(x): {D_x}, D(G(z)): {D_G_z1} / {}]")
        
        if epoch == 0 and itr == 0:
            torch.utils.save_image(real_image, f"real_samples.png", normalize = True)
        
#%%
#p81
nz = 100
nch_g = 128
nch = 1

# https://nw.tsuda.ac.jp/lec/PyTorch/torch_book1/win/ch03/torch_book1_ch03_03.html
def weights_init(m):
    """
    ニューラルネットワークの重みを初期化する。作成したインスタンスに対しapplyメソッドで適用する
    :param m: ニューラルネットワークを構成する層
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: # 畳み込み層
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1: # 全結合層
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1: # バッチのーまりゼーション
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

netG = Generator(nz = nz+10, nch_g=nch_g).to(device) # 10はn_class=10を指す。だし訳に必要なラベル情報
netG.apply(weights_init)
print(netG)

#%%p81
nch_d = 128
netD = Discriminator(nch=1+10, nch_d = nch_d)
netD.apply(weights_init)
print(netD)

def onehot_encode(label, device, n_class=10):
    """
    カテゴリかる変数のラベルをワンほっと
    label: 対象用のラベル
    device: 学習にしようするデバイス
    n_class: ラベルのクラス数
    """
    eye = torch.eye(n_class, device = device)
    # ランダムベクトルあるいは画像と連結するために(B, c_class, 1, 1)のTensorにして戻す
    return eye[label].view(-1, n_class, 1, 1)

def concat_image_label(image, label, device, n_class=10):
    B, C, H, W = image.shape

    oh_label = onehot_encode(label, device) # ラベルをワンほっとベクトル化

    oh_label = oh_label.expand(B, n_class, H, W) # 画像のサイズに合わせるようラベルを拡張する

    return torch.cat((image, oh_label), dim=1) # 画像とラベルをチャンネル方向(dim=1)で連結する

# 生成器のエポ





