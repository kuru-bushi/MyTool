 #%%
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F 
from torch.autograd import Variable

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
        # Dropout pa161
        if use_dropout:
            layer.append(torch.nn.Dropout(0.5))
        return torch.nn.Sequential(*layer)
    
    def foward(self, x):
        # 偽画像の生成
        x0 = self.down0(x)
        x1 = self.down0(x0)
        x2 = self.down0(x1)
        x3 = self.down0(x2)
        x4 = self.down0(x3)
        x5 = self.down0(x4)
        x6 = self.down0(x5)
        x7 = self.down0(x6)
        y7 = self.up7(x7)
        # Encoderの出力をDecodeの入力にSkipConnectionで接続
        y6 = self.up6(self.concat(x6, y7))
        y5 = self.up5(self.concat(x5, y6))
        y4 = self.up4(self.concat(x4, y5))
        y3 = self.up3(self.concat(x3, y4))
        y2 = self.up2(self.concat(x2, y3))
        y1 = self.up1(self.concat(x1, y2))
        y0 = self.up0(self.concat(x0, y1))

        return y0
    
    def concat(self, x, y):
        # 特徴量マップの結合
        return torch.cat([x, y], dim=1)

#p162
class GANLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(GANLoss, self).__init__()

        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))

        # Real/Fake伊σ器べつの損失を、シグモイド＋バイナリークロスエントロピーで計算
        # TODO
        self.loss = torch.nn.BCEWithLogitsLoss()
    
    def __call__(self, prediction, is_real):
        if is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label

        return self.loss(prediction, target_tensor.expand_as(prediction))
#%%
#p163
# Pix2Pixモデルの定義クラス
class Pix2Pix():
    def __init__(self, config) -> None:
        self.config = config

        # 生成器Gのオブジェクト取得とデバイス設定
        self.netG = Generator().to(self.config.device)
        # ネットワークの初期化
        self.netG.apply(self.__weights_init)

        # 識別器Dのオブジェクト取得とデバイス設定
        self.netD = Discriminator().to(self.config.device)
        # Dのネットワークの初期化
        self.netD.apply(self.__weights_init)

        # オプティマイザの初期化
        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
        # 目的（損失関数）の設定
        # GAN 損失(Adversarial損失)
        self.criterionGAN = GANLoss().to(self.config.device)
        # L1損失
        self.criterionL1 = torch.nn.L1Loss()

        # 学習率のスケジューラー設定
        self.schedulerG = torch.optim.lr_scheduler.LambdaLR(self.optimizerG, self.__modify_learning_rate)
        self.schedulerD = torch.optim.lr_scheduler.LambdaLR(self.optimizerD, self.__modify_learning_rate)

        self.training_start_time = time.time()
        
        self.writer = SummaryWriter(log_dir=config.log_dir)

    def __modify_leaning_rate(self):
        # 学習率の更新、毎エポック後に呼ばれる
        self.schedulerG.step()
        self.schedulerD.step()

    def __modify_learning_rate(self, epoch):
        # 学習率の計算
        if self.config.epoch_lr_decay_start < 0:
            return 1.0
        
        delta = max(0, epoch - self.config.epochs_lr_decay_start) / float(self.config.epochs_lr_decay)
        return max(0.0, 1.0 - delta)

    def __weights_init(self, m):
        # パラメータ初期値の設定
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def train(self, databatches_done):
        # ドメインAどめいんあんぺあのラベル画像とドメインBの正解画像を設定
        self.realA = data['A'].to(self.config.device)
        self.realB = data['B'].to(self.config.device)

        # 生成器Gで画像生成
        fakeB =self.netG(self.realA)

        # 識別器Dの学習開始
        # 条件画像(A)と生成画像(B)を結合
        fakeAB = torch.cat((self.realA, fakeB), dim=1)
        pred_fake = self.netD(fakeAB.detach())
        # 偽物画像を入力した時の識別器DのGAN損失を算出
        lossD_fake = self.criterionGAN(pred_fake, False)

        # 条件画像(A)と正解画像(B)を結合
        realAB = torch.cat((self.realA, self.realB), dim=1)
        # 識別器Dに正解画像を入力
        pred_real = self.criterionGAN(pred_real, True)

        # 偽物画像と正解画像のGAN損失の合計に0.5をかける　
        lossD = (lossD_fake + lossD_real) * 0.5

        # Dの勾配をゼロに設定
        self.optimizerD.zero_grad()
        # Dの逆電版を計算
        lossD.backward()
        # Dの重みを更新
        self.optimizerD.step()

        # 生成器Gの学習開始
        # 識別器Dに生成画像を入力
        with torch.no_grad():
            pred_fake = self.netD(fakeAB)

        # 生成器GのGAN損失を算出
        lossG_GAN = self.criterionGAN(pred_fake, True)
        # 生成器GのL1損失を算出
        lossG_L1 = self.criterionL1(fakeB, self.realB) * self.config.lambda_L1

        # 生成器Gの損失を合計
        lossG = lossG_GAN + lossG_L1

        # Gの勾配をゼロに設定
        self.optimizerG.zero_grad()
        # Gの逆電版を計算
        loss.backward()
        # Gの重みを更新
        self.optimizerG.step()
#%%
#p167
for epoch in range(1, opt.epochs + 1):
    for batch_num, data in enumerate(dataloader):
        batches_done = (epoch - 1) * len(dataloader) ; batch_num
        model.train(data, batches_done)

    model.update_learning_rate()

#%%
#p176
class Discriminator(torch.nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(64, 128, 4, stride=2, padding=1),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(128, 256, 4, stride=2, padding=1),
            torch.nn.InstanceNorm2d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(256, 512, 4, stride=2, padding=1),
            torch.nn.InstanceNorm2d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),


            torch.nn.Conv2d(512, 1, 4, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).vie(x.size()[0], -1)
    
class Generator(torch.nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_block=9) -> None:
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(input_nc, 64, 7),
            torch.nn.InstanceNorm2d(64),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
            torch.nn.InstanceNorm2d(128),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(128, 256, 3, stride=2, padding=1),
            torch.nn.InstanceNorm2d(256),
            torch.nn.ReLU(inplace=True),

            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),

            torch.nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            torch.nn.InstanceNorm2d(128),
            torch.nn.ReLU(inplace=True),

            torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            torch.nn.InstanceNorm2d(64),
            torch.nn.ReLU(inplace=True),

            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(64, 3, 7),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

#%%p178
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.conv_block = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_features, in_features, 3),
            torch.nn.InstanceNorm2d(in_features),
            torch.nn.ReLU(inplace=True),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_features, in_features, 3),
            torch.nn.InstanceNorm2d(in_features)
        )
    def forward(self, x):
        return x + self.conv_block(x)
    
#%%p179
for epoch in range(opt.start_epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # モデルの入力
        real_A = Variable(input_A.copy_(batch["A"]))
        real_B = Variable(input_B.copy_(batch["B"]))

        #### 生成器A2B, B2A の処理
        optimizer_G.zero_grad()

        # 同一損失の計算
        # G_A2B(B)はBと一致
        same_B = netG_B2A(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0
        # G_B2A(A)はAと一致
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * 5.0

        # 敵対性損失(GAN Loss)
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # サイクル一貫性損失 (Cycle-consistency loss)
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

        recoverd_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recoverd_B, real_B) * 10.0

        #生成器の合計損失関数(Total loss)
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()

        optimizer_G.step()

        # ドメインAの識別器
        optimizer_D_A.zero_grad()
        
        # ドメインAの本物画像の識別結果(Real loss)
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # ドメインAの生成画像の識別結果(Fake loss)
        fake_A = fake_A_buffer.push_and_pop(fakeA)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # 識別器(ドメインA)の合計損失(Total loss)
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        
        ##### ドメインBの識別器
        optimizer_D_B.zero_grad()

        # ドメインBの本物画像の識別結果(Real loss)
        pred_real = netD_G(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # ドメインBの生成画像の識別結果(Fake loss)
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # 識別器(ドメインB)の合計損失（Total loss)
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        #######################################

    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()


#%%p183
from icarawler.builtin imort BaiduImageCrawler, BingImageCrawler, GoogleImageCrawler

dir_name = 'portrait'
search_key = 'portrait painting'

google_crawler = GoogleImageCrawler(
    feeder_threads=1,
    parser_threads = 1,
    downloader_threads=4,
    storage={'root_dir': dir_name})
filters = dict(
    size='large',
    color='orange',
    license='commercial,modify',
    date=((2017, 1, 1), (2017, 11, 30))
)

google_crawler.crawl(keyward=search_key, filters=filters, offset=0, max_num=1000, min_size=(200, 200), max_size=None, file_idx_offset=0)

bing_crawler = BingImageCrawler(downloader_threads=4, storage={'root_dir': dir_name})
bing_crawler.crawl(keword=search_key, filters=None, offset=0, max_num=1000)

baidu_crawler = BailduImageCrawler(storage={'root_dir': dir_name})
baidu_crawler.crawl(keyward=search_key, offset=0, max_num=1000, min_size=(200, 200), max_size=None)





