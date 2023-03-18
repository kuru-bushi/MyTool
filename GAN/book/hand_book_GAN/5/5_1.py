#%%
import torch

class Dscriminator(torch.nn.Module):
    def __init__(self) -> None:
        super(Dscriminator, self).__init__()
        # 70*70 PatchGAN識別器モデルの定義
        # 2つの画像を結合したものが入力となるため、チャンネル数は3*2=6となる
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2, True),
            self.__layer(64, 128),
            self.__layer(128, 256),
            self.__layer(256, 512, stride=1),
            torch.nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def __layer(self, input, output, stride=2):
        return torch.nn.Sequential(
            torch.nn.Conv2d(input, output, kernel_size=4, stride=stride, pading=1),
            torch.nn.BatchNorm2d(output),
            torch.nn.LeakyReLU(0.2, True),
        )
    
    def forward(self, x):
        return self.model(x)
    
#%% p159
class Generator(torch.nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        # U-Net のエンコーダ部分
        self.down0 = torch.nn.Conv2d(3, 64, kernel_size=4, stride=2, pading=1)
        self.down1 = self__encoder_block(64, 128)
        self.down2 = self__encoder_block(128, 256)
        self.down3 = self__encoder_block(256, 512)
        self.down4 = self__encoder_block(512, 512)
        self.down5 = self__encoder_block(512, 512)
        self.down6 = self__encoder_block(512, 512)
        self.down7 = self__encoder_block(512, 512, use_norm=False)

        # U-NetのDecoder部分
        self.up7 = self.__decoder_block(512, 512)
        self.up6 = self.__decoder_block(1024, 512, user_dropout=True)

        self.up5 = self.__decoder_block(1024, 512, user_dropout=True)
        self.up4 = self.__decoder_block(1024, 512, user_dropout=True)
        self.up3 = self.__decoder_block(1024, 256)
        self.up3 = self.__decoder_block(512, 128)
        self.up2 = self.__decoder_block(256, 64)
        
        self.up0 = torch.nn.Sequential(
            self.__decoder_block(128, 3, use_norm=False),
            torch.nn.Tanh(),
        )

    def __encoder_block(self, input, output, use_norm=True):
        # LeakyReLU+Downsampling
        layer = [
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv2d(input, output, kernel_size=4, stride=2, padding=1)
        ]
        # BatchNormalization
        if use_norm:
            layer.append(torch.nn.BatchNorm2d())
        return torch.nn.Sequential(*layer)
    
    def __decoder_block(self, input, output, use_norm=True, use_dropout=False)
        # ReLU + Upsampling
        layer = [
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(input, output, kernel_size=4, stride=2, padding=1)
        ]

        if user_norm:
            layer.append(torch.nn.LazyBatchNorm2d(output))
        # Dropout
        # pa161

