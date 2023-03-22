#%%
#p192
# ドメインAとドメインBの画像データセット生成クラス
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F 
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision
import glob
import os
from PIL import Image
import numpy as np
import options
import itertools

opt = options.test_options()
opt2 = options.test_options()

from torch.utils.data import DataLoader

class FaceDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train', files_A='fadg0/video/', files_B='faks0/video/') -> None:
        # TODO 4,5 のtransforms をtorchvisionに
        self.transform = torchvision.transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, files_A + '/*/*')))
        self.files_B = sorted(glob.glob(os.path.join(root, files_B + '/*/*')))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))
        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[np.random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
        else:
            item_B = self.transform(Image.open(self.files_B[np.random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
        return {'A': item_A, 'B': item_B}
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    
# p193
dataloader = DataLoader(
    FaceDataset(opt.dataroot,
                transforms_ = transforms_,
                unaligned=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu
)

#p194
class FaceDatasetVideo(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train', files='fadg0/video/head') -> None:
        self.transform = torchvision.transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files = sorted(glob.glob(os.path.join(root, files) + ("/*")))

    def __getitem__(self, index):
        item = self.transform(Image.open(self.files[index % len(self.files)]).convert('RGB'))
        return item
    
    def __len__(self):
        return len(self.files)
    
#%%
if not os.path.exists(os.path.join('output/cycle', opt2.log_base_name, 'output_video')):
    os.makedirs(os.path.join('output/cycle', opt2.log_base_name, 'output_video'))

if not os.path.exists(os.path.join('output/cycle', opt2.log_base_name, 'output_video/A2B')):
    os.makedirs(os.path.join('output/cycle', opt2.log_base_name, 'output_video/A2B'))

if not os.path.exists(os.path.join('output/cycle', opt2.log_base_name, 'output_video/B2A')):
    os.makedirs(os.path.join('output/cycle', opt2.log_base_name, 'output_video/B2A'))
    
#%%p195
import cv2
in_video_w = 256*2
in_video_h = 256
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

def make_video(output_video_path, input_dir_path, domain):
    video = cv2.VideoWriter(output_video_path, fourcc, 10.0, (in_video_2, in_video_h))

    # ネットワーク呼び出し
    # 生成器G

    netG_A2B = Generator(opt2.input_nc, opt2.output_nc)
    netG_B2A = Generator(opt2.output_nc, opt2.input_nc)

    # CUDA
    if opt2.cuda:
        negG_A2B.cuda()
        negG_B2A.cuda()

    # Load state dicts
    netG_A2B.load_state_dict(torch.load(os.path.join(opt2.model_load_path, 'netG_A2B.pth')))
    netG_B2A.load_state_dict(torch.load(os.path.join(opt2.model_load_path, 'netG_A2B.pth')))

    # Set model's test mode
    netG_A2B.eval()
    netG_B2A.eval()

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt2.cuda else torch.Tensor
    input_A = Tensor(opt2.batch_size, opt2.input_nc, opt2.size, opt2.size)
    input_B = Tensor(opt2.batch_size, opt2.output_nc, opt2.size, opt2.size)

    # Data loader
    tansform_ = [torchvision.transforms.Resize(int(opt2.size*1.0), Image.BICUBIC),
                 torchvision.transforms.CenterCrop(opt2.size),
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    dataloader = DataLoader(FaceDatasetVideo(opt2.dataroot, transforms_=tansform_, model='train', files=input_dir_path),
                            batch_size=opt2.batch_size,
                            shuffle=False,
                            num_workers=opt,
                            num_workers=opt2.n_cpu
                            )

    #p196
    for i, batch in enumerate(dataloader):
        if domain == 'A':
            # Set model input
            real_A = Variable(input_A.copy_(batch))
            # Generate output
            fake_B = netG_A2B(real_A).data
            out_img1 = torch.cat([real_A, fake_B], dim=3)
        else:
            # Set model input
            real_B = Variable(input_B.copy_(batch))
            # Generate output
            fake_A = netG_B2A(real_B).data
            out_img1 = torch.cat([real_B, fake_A], dim=3)

        image = 127.5 * (out_img[0].cpu().float().detach().numpy() + 1.0)
        image = image.transpose(1,2,0).astype(np.unit8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video.write(image)
#%% p197
output_vide_path = os.path.join('output/cycle', opt2.log_base_name, 'output_video/A2B/head.mp4')
input_dir_path = 'fadg0/video/head'
domain = 'A'
make_video(output_vide_path, input_dir_path, domain=domain)

#%%
output_video_path = os.path.join('output/cycle/', opt2.log_base_name, 'output_video/B2A/head.mp4')
input_dir_path = 'faks0/video/head'
domain = 'B'
make_video(output_video_path, input_dir_path, domain=domain)

#%%p203
class FaceDatasetSequence(Dataset):
    def __init__(self, root, transform_=None,
                 unaligned=False,
                 mode='train',
                 files_B='faks0/video/',
                 skip=0):
        self.skip = skip
        self.remove_num = (skip + 1) * 2
        self.transform = torchvision.transforms.Compose(transform_)
        self.unaligned = unaligned
        dir_A = os.listdir(path=os.path.join(root, file_A))
        dir_B = os.listdir(path=os.path.join(root, file_B))
        self.files_A = []
        self.files_B = []
        for dir1 in dir_A:
            all_files = sorted(glob.glob(os.path.join(root, files_A)))
            self.files_A += all_files[:- self.remove_num] # 最後から(skip + 1) * 2つは削除
            for dir1 in dir_B:
                all_files = sorted(glob.glob(os.path.join(root, files_B, dir1 + '/*')))
                self.files_B += all_files[: - self.remove_num] # 最後から (skip + 1) * 2つは削除
                print("len(self.files_A):{}, len(self.files_B):{}".format(len(self.files_A), len(self.files_B)))
                self.count = 1234

    def __getitem__(self, index):
        seed = self.count
        file_A1 = self.files_A[index % len(self.files_A)]
        item_A1, item_A2, item_A3 = self.get_sequential_data(file_A1, seed)
        if self.unaligned:
            num_tmp = np.randint(0, len(self.files_B), - 1)
            file_B1 = self.files_B[index % len(self.files_B)]
        else:
            file_B1 = self.files_B[index % len(self.files_B)]
            item_B1, item_B2, item_B3 = self.get_sequential_data(file_B1, seed+1)
        self.count += 1
        return {'A1': item_A1, 'A2': item_A2, 'A3': item_A3,
                'B1': item_B1, 'B2': item_B2, 'B3': item_B3}
    
    def get_sequential_data(self, file1, seed):
        dir_name, file_num = file1.rsplit('/', 1)
        file2 = os.path.join(dir_name, '{:0=3}'.format(int(file_num), + self.skip))
        file3 = os.path.join(dir_name, '{:0=3}'.format(int(file_num), + self.skip * 2))

        np.random.seed(seed)
        item1 = self.transform(Image.open(file1).convert('RGB'))
        np.random.seed(seed)

        item2 = self.transform(Image.open(file2).convert('RGB'))
        np.random.seed(seed)
        item3 = self.transform(Image.open(file3).convert('RGB'))
        return item1, item2, item3
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    
#%%
class ConvUnit(torch.nn.Module):
    def __init__(self, in_channels, out_channels, mid_channel=None):
        super().__init__()
        if not mid_channel:
            mid_channels = out_channels
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(input=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)
    
class Down(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            ConvUnit(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvUnit(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # torch.nn.functional.pad(input, pad, mode='constant', value=0.0)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2,
                        diffX - diffX // 2,
                        diffY //2,
                        diffY - diffX // 2 ])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)
    
class Predictor(torch.nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Predictor, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.inc = ConvUnit(input_nc, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(128, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, output_nc)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        out = self.tanh(logits)
        return out

#%%p208
class Opts():
    def __init__(self):
        self.start_epoch = 0
        self.n_epoch = 40
        self.batch_size = 1
        self.dataroot = 'datasets/'
        self.lr = 0.0002
        self.decay_epoch = 200
        self.size = 256
        self.input_nc = 3
        self.output_nc = 3
        self.cpu = False
        self.n_cpu = 8
        self.device_name = 'cuda:0'
        self.device = torch.device(self.device_name)
        self.load_weight = False
        self.log_base_name = 'train11'
        self.model_load_path = 'output/recycle/train10/output_model/'

        self.file_a_dir = 'fadg0/video/'
        self.file_b_dir = 'faks0/video/'
        self.id_loss_rate = 5.0
        self.gan_loss_rate = 5.0
        self.recy_loss_rate = 10.0
        self.recu_loss_rate = 10.0
        self.skip = 2

opt = Opts()

#%%p209
#生成器
netG_A2B =Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc)

# 識別器
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)


# 予測器 ------------ recycle -------------
netP_A = Predictor(opt.input_nc*2, opt.input_nc)
netP_B = Predictor(opt.output_nc*2, opt.output_nc)


# GPU
if not opt.cpu:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()
    netP_A.cuda()
    netP_B.cuda()

# TODO ネットより。正しい？
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight, 1.0, 0.02)
        torch.nn.init.constant(m.bias, 0.0)

# 重みパラメター初期化
netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)


# p210保存したモデルのロード
if opt.load_weight is True:
    netG_A2B.load_state_dict(torch.load(os.path.join(opt.model_load_path, "netG_A2B.pth"),
                                map_location='cuda:0'),
                                strict=False)
    netG_B2A.load_state_dict(torch.load(os.path.join(opt.model_load_path, "netG_B2A.pth"),
                                map_location='cuda:0'),
                                strict=False)
    
    netD_A.load_state_dict(torch.load(os.path.join(opt.model_load_path, "netD_A.pth"),
                                map_location='cuda:0'),
                                strict=False)
    netD_B.load_state_dict(torch.load(os.path.join(opt.model_load_path, "netD_A.pth"),
                                map_location='cuda:0'),
                                strict=False)
    netP_A.load_state_dict(torch.load(os.path.join(opt.model_load_path, "netP_A.pth"),
                                map_location='cuda:0'),
                                strict=False)
    netP_B.load_state_dict(torch.load(os.path.join(opt.model_load_path, "netP_B.pth"),
                                map_location='cuda:0'),
                                strict=False) # Predictor
    

# 損失関数
criterion_GAN = torch.nn.MSELoss()
criterion_recycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
criterion_recurrent = torch.nn.L1Loss() # predictor

# Optimizer & LR schedulers
optimizer_PG = torch.optim.Adam(itertools.chain(netG_A2B.parameters(),
                                                netG_B2A.parameters(),
                                                netP_A.parameters(),
                                                netP_B.parameters(),
                                                lr=opt.lr,
                                                betas=(0.5, 0.999)))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(),
                                 lr=opt.lr,
                                 betas=(0.5, 0.999))

optimizer_D_B = torch.optim.Adam(netD_B.parameters(),
                                 lr=opt.lr,
                                 betas=(0.5, 0.999))

lr_scheduler_PG = torch.optim.lr_scheduler.LambdaLR(optimizer_PG,
                                                    lr_lambda=LambdaLR(opt.n_epoch,
                                                                       opt.start_epoch,
                                                                       opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                     lr_lambda=LambdaLR(opt.n_epoch,
                                                                        opt.start_epoch,
                                                                        opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                     lr_lambda=LambdaLR(opt.n_epoch,
                                                                        opt.start_epoch,
                                                                        opt.decay_epoch).step)

# 入出力メモリ確保
Tensor = torch.cuda.FloatTensor if not opt.cpu else torch.Tensor
input_A1 = Tensor(opt.batch_size,
                  opt.input_nc,
                  opt.size,
                  opt.size)
input_A2 = Tensor(opt.batch_size,
                  opt.input_nc,
                  opt.size,
                  opt.size)
input_A3 = Tensor(opt.batch_size,
                  opt.input_nc,
                  opt.size,
                  opt.size)
input_B1 = Tensor(opt.batch_size,
                  opt.input_nc,
                  opt.size,
                  opt.size)
input_B2 = Tensor(opt.batch_size,
                  opt.input_nc,
                  opt.size,
                  opt.size)
input_B3 = Tensor(opt.batch_size,
                  opt.input_nc,
                  opt.size,
                  opt.size)
target_real = Variable(Tensor(opt.batch_size).fill_(1.0),
                       requires_grad=False)

target_fake = Variable(Tensor(opt.batch_size).fill_(0.0),
                       requires_grad=False)


# 過去データ分のメモリ確保
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# データローダー
transforms_ = [torchvision.transforms.Resize(int(opt.size*1.12), Image.BICUBIC),
               torchvision.transforms.RandomCrop(opt.size),
            #    torchvision.transforms.RandomHorizontalFlip(),
               torchvision.transforms.ToTensor(),
               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
               ]

# dataloader = DataLoader(FaceDatasetSequence(opt.dataroot,
                # transforms_=transforms_,
                # unaligned=True,
                # files_A=opt.file_a_dir,
                # files_B=opt.file_b_dir,
                # skip=opt.skip),
                # batch_size=opt.batch_size,
                # shuffle=True,
                # num_workers=opt.n_cpu)

#%% p212
# for epoch in range(opt.start_epoch, opt.n_epochs):
    


    





