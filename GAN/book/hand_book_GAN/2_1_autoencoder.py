#%%
import torch.nn as nn
import torch
from torch import optim

class AutoEncoder(nn.Module):
    def __init__(self, device = "cpu") -> None:
        super().__init__()
        self.device = device
        self.l1 = nn.Linear(784, 200)
        self.l2 = nn.Linear(200, 784)

    def forward(self, x):
        # エンコーダ
        h = self.l1(x)
        # 活性化関数
        h = torch.relu(h)
        # デコーダ
        h = self.l2(h)
        # シグモイド関数で0-1の値域に変換
        y = torch.sigmoid(h)

        return y

#%%
import matplotlib.pyplot as plt
device = torch.device("cpu")
model = AutoEncoder().to(device)
# 損失関数の設定 1
criterion = nn.BCELoss()
# 最適化関数の設定
optimizer = optim.Adam(model.parameters())

# %%
from torchvision import datasets
from torchvision.transforms import ToTensor
train_dataloader = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_dataloader = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

#%%
epochs = 10
device 
for epoch in range(epochs):
    train_loss = 0
    # バッチサイズのループ
    for (x, _) in train_dataloader:
        x = x.to(device)
        # 訓練モードへの切り替え
        model.train()
        # 順伝ばん計算
        preds = model(x)
        # 入力画像xと復元画像predsの誤差電ばん
        loss = criterion(preds, x)
        # 勾配の初期化
        optimizer.zero_grad()
        # 誤差の勾配計算
        loss.backward()
        # パラメータの更新
        optimizer.step()
        # 訓練誤差の更新
        train_loss += loss.item()

        train_loss /= len(train_dataloader)

        print(f"Epoch: {epoch+1}, Loss: {train_loss}")

        # dataloader からのデータの取り出し
        x, _ = next(iter(test_dataloader))
        x = x.to(device)

        # 評価モードへの切り替え
        model.eval()
        # 復元画像
        x_rec = model(x)

        # 入力画像、復元画像の表示
        for i, image in enumerate([x, x_rec]):
            image = image.view(28, 28).detach().cpu().numpy()
            plt.subplot(1, 2, i+1)
            plt.imshow(image, cmap="binary_r")
            plt.axis("off")
        plt.show()

# %%
# 変分オートエンコーダ p40
class VAE(nn.Module):
    def __init__(self, device: str ="cpu") -> None:
        super().__init__()
        self.device = device
        self.encoder = Encoder(device = device)
        self.decoder = Decoder(device = device)

    def foward(self, x):
        # エンコーダ
        mean, var = self.encoder(x)
        # 潜在変数の作成
        z = self.reparameterize(mean, var)
        # デコーダ
        y = self.decoder(z)
        # 生成画像yyと潜在変数zが戻り値
        return y, z

    # 潜在変数 z
    def reparameterize(self, mean, var):
        eps = torch.randn(mean.size()).to(self.device)
        z = mean + torch.sqrt(var) * eps
        return z
    
    def lower_bound(self, x):
        mean, var = self.encoder(x)
        # 平均と分散から潜在変数zを計算
        z = self.reparameterize(mean, var)
        # 潜在変数zから生成画像を作成
        y = self.decoder(z)
        # 再構成誤差
        reconst = - torch.mean(torch.sum(x*torch.log(y) + (1 - x) * torch.log(1-y), dim=1))
        # 正則化
        k1 = - 1/2 * torch.mean(torch.sum(1 + torch.log(var) - mean**2 -var, dim=1))
        # 再構成誤差 + 正則化
        L = reconst + k1

        return L

# %%
class Encoder(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device
        self.l1 = nn.Linear(784, 200)
        self.l_mean = nn.Linear(200, 10)
        self.l_var = nn.Linear()

    def foward(self, x):
        # 784次元から200次元
        h = self.l1
        # 活性化関数
        h = torch.relu(h)
        # 200次元から10次元
        mean = self.l_mean(h)
        # 200次元から10次元の分散
        var = self.l_var(h)
        # 活性化関数sofplus
        var = F.sofplus(var)

        return mean, var

# %%
class Decoder(nn.Module):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.device = device
        self.l1 = nn.Linear(10, 200)
        self.out = nn.Linear(200, 784)

    def forward(self, x):
        # 10次元から200次元
        h = self.l1(x)
        # 活性化関数
        h = torch.relu(h)
        # 200次元から784次元
        h = self.out(h)
        # シグモイド関数
        y = torch.sigmoid(h)

        return y

#%%
torch.device("cpu")
# %%
# p46
model = VAE().to(device)
# 損失関数の設定
criterion = model.lower_bound
# 最適化関数の設定
optimizer = optim.Adam(model.parameters())


# %%
epochs = 10
for epoch in range(epochs):
    train_loss = 0.
    # バッチサイズのループ
    for (x, _) in train_dataloader:
        x = x.to(device)
        # 訓練モードへの切り替え
        model.train()
        # 本物画像と生成画像の誤差計算
        loss = criterion(x)
        # 勾配の初期化
        optimizer.zero_grad()
        # 誤差の勾配計算
        loss.backward()
        # パラメタの更新
        optimizer.step()
        # 訓練誤差の更新
        train_loss += loss.item()

        train_loss /= len(train_dataloader)

        print(f"Epoch: {epoch+1}, Loss: {train_loss}")

        # dataloader からのデータの取り出し
        x, _ = next(iter(test_dataloader))
        x = x.to(device)

        # 評価モードへの切り替え
        model.eval()
        # 復元画像
        x_rec = model(x)

        # 入力画像、復元画像の表示
        for i, image in enumerate([x, x_rec]):
            image = image.view(28, 28).detach().cpu().numpy()
            plt.subplot(1, 2, i+1)
            plt.imshow(image, cmap="binary_r")
            plt.axis("off")
        plt.show()

#%%
# ノイズの作成数
batch_size = 8
# デコーダ入力用に標準正規分布に従う10次元ノイズを作成
z = torch.randn(batch_size, 10, device = device)

# 評価モードへの切り替え
model.eval()
# デコーダにノイズzを入力
images = model.decoder(x)
images = model.view(-1, 28, 28)
images = images.squeeze().detach().cpu().numpy()

for i, image in enumerate(images):
    plt.subplot(2, 4, i+1)
    plt.imshow(image, cmap="binary_r")
    plt.axis("off")
plt.tight_layout()
plt.show()

#%%
fig = plt.figure(figsize=(10, 3))
model.eval()
for x, t in test_dataloader:
    # 本物画像
    for i, im in enumerate(x.view(-1, 28, 28).detach().numpy()[:10]):
        ax = fig.add_subplot(3, 10, i+1, xticks = [], yticks=[])
        ax.imshow(im, "gray")
    x = x.to(device)
    # 本物画像から生成画像
    y, z = model(x)
    y = y.view(-1, 28, 28)
    for i, im in enumerate(y.cpu().detach().numpy()[:10]):
        ax = fig.add_subplot(3, 10, i+11, xticks=[], yticks=[])
        ax.imshow(im, "gray")
    # 1つ目の画像と2つ目の画像の潜在変数を連続的にへんか
    z1to0 = torch.cat([z[1] * (i * 0.1) + z[0] * ((9 - i) * 0.1) for i in range(10)]).reshape(10, 10)
    y2 = model.decoder(z1to0).view(-1, 28, 28)
    for i, im in enumerate(y2.cpu().detach().numpy()):
        ax = fig.add_subplot(3, 10, i+21, xticks=[], yticks=[])
        ax.imshow(im, "gray")
    break

#%%



