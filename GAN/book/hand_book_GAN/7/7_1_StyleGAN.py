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


# TODO ネットより
# class BaseLayer:
#     def update(self, eta):
#         self.w -= eta * self.grad_w
#         self.b -= eta * self.grad_b

# p268
class Generator(BaseLayer):
    def __init__(self, opt) -> None:
        super(Generator, self).__init__()
        self.opt = opt
        self.mapping_nework = MappingNetwork(dlaten_size=opt.latent_dim, opt=opt)
        self.synthesis_network = SynthesisNetwork(opt=opt)

    def forward(self, z):
        s = self.mapping_nework(z)
        x = self.synthesis_nework(s)
        return x, s

class MappingNetwork(BaseLayer):
    def __init__(self, dlaten_size, out) -> None:
        super(MappingNetwork, self).__init__()
        self.mapping_layers = 8
        self.out_features = 512

        resolution_log2 = int(np.log2(opt.resolution))
        self.num_layers = resolution_log2 * 2 - 2
        self.dense_layers = torch.nn.ModuleDict()
        self.fused_bias_acts = torch.nn.ModuleDict()
        for layer_idx in range(self.mapping_layers):
            self.dense_layers[str(layer_idx)] = DenseLayer(dlaten_size, self.out_features, lmul=0.01)
            self.fused_bias_acts[str(layer_idx)] = FusedBIasActivation(dlaten_size, lrmul=0.01, act='LeakyRelu')
        
    def forward(self, z):
        x = self.normalize(z)
        for layer_idx in range(self.mapping_layers):
            x = self.dense_layers[str(layer_idx)](x)
            x = self.fused_bias_acts[str(layer_idx)](x)
        x = x.unsqueeze(1)
        x = x.repeat([1, self.num_layers, 1])
        return x

    def normalize(self, x):
        x_var = torch.mean(x**2, dim=1, keepdim=True)
        x_rstd = torch.rsqrt(x_var + 1e-8)
        normalize = x * x_rstd
        return normalize

#%%p274
class SynthesisNetwork(BaseLayer):
    def __init__(self, opt) -> None:
        super(SynthesisNetwork, self).__init__()
        self.opt = opt
        c = self.Tensor(np.random.normal(loc=0, scale=1, size=(1, 512, 4, 4)))
        self.const = torch.nn.Parameter(c)
        # TODO Layer ってなに？torch?
        self.layer = Layer(
            x_channel=self.in_channel,
            style_layer_index=0,
            style_in_dim=self.dlatent_size,
            style_out_dim=self.dlatent_size,
            feature_map=self.feature_maps,
            res=1
        )
        # TODO Layer ってなに？torch?
        self.to_rgb = ToRGB(
            x_channel=self.in_channel,
            out_chanel=self.out_rgb_channel,
            kernel=1,
            style_dim=self.dlatent_size,
            res=2
        )
        self.block_dict = torch.nn.ModuleDict()
        self.to_rgb_dict = torch.nn.ModuleDict()
        self.upsample_2d_dict = torch.nn.ModuleDict()
        for res in range(3, self.resolution_log2 + 1):
            self.block_dict[str(res)] = Block(res, style_dim=self.dlatent_size)
            self.upsample_2d_dict[str(res)] = torch.nn.Upsample(res, resample_kernel=[1, 3, 3, 1])
            self.to_rgb_dict[str(res)] = ToRGB(
                self.cliped_features(res - 1),
                self.out_rgb_channel,
                kernel=1,
                style_dim=self.dlatent_size,
                res=res
            )

    def forward(self, style):
        image = None
        const_input = self.const.repeat([self.opt.batch_size, 1, 1,1])
        x = self.layer(const_input, style)
        image = self.to_rgb(x, style, image)
        
        for res in range(3, self.resolution_log2 + 1):
            x = self.block_dict[str(res)](x, style)
            upsample_image = self.upsample_2d_dict[str(res)](image)
            image = self.to_rgb[str(res)](x, style, upsample_image)

        return image

#%%p279
class Layer(BaseLayer):
    def __init__(self, x_channel, style_layer_index, style_in_dim, style_out_dim, feature_map, res, kernel=3, is_up=False) -> None:
        super(Layer, self).__init__()
        self.modlate_conv2d = ModulateConv(
            x_channel=x_channel,
            feature_map=feature_map,
            style_in_dim=style_in_dim,
            kernel=kernel,
            padding=1,
            is_demodulate=True,
            is_up=is_up
        )
        self.fused_bias_act = FusedBiasActivation(feature_map, cat='LeakyReLU')

    def forward(self, x, style):
        s = style[:, self.style_layer_index]
        x = self.modlate_conv2d(x, s)
        batch_size, channel, height, width = x.shape
        noise = self.Tensor(np.random.normal(loc=0, scale=1, size=(batch_size, 1, height, width)))

        noise = noise * self.noise_strength
        x = x + noise
        x = self.fused_bias_act(x)
        return x

#%%p283
class ModulateConv(BaseLayer):
    def __init__(self,
                 x_channel,
                 feature_map,
                 style_in_dim,
                 style_out_dim,
                 kernel=3,
                 padding=0,
                 is_demodulate=True,
                 is_up=False):
        super(ModulateConv, self).__init__()
        self.is_demodulate = is_demodulate
        self.is_up = is_up
        # TODO get_weight ... なんの関数？
        x, runtime_coef = self.get_weight_and_runtime_coef(
            shape=[self.feature_map, x_channel, kernel, kernel],
            gain=1,
            use_wscale=True,
            lrmul=1
        )
        self.weight = torch.nn.Parameter(x)
        self.runtime_coef = runtime_coef

        self.dense_layer = DenseLayer(in_channel=styhle_in_dim,feature_map=x_channel)
        self.fused_bias_act = FusedBiasActivation(channel=x_channel, act='Linear')

        if self.is_up:
            self.upsample_conv_2d = UpsampleConv2d()

    def foward(self, x, style):
        w_m = self.modulate(self.weight * self.runtime_coef, style)
        if self.is_demodulate:
            w_ = self.demodulate(w_m)
        else:
            w_ = w_m
        
        batch, out_c, in_c, kh, kw = w_.shape
        w = w_.view(batch * out_c, in_c, kh, kw)

        x_batch, x_c, x_h, x_w = x.shape
        x = x.view(1, x_batch * x_c, x_h, x_w)

        if self.is_up:
            x = self.upsample_conv_2d(x, w)
        else:
            x = F.conv2d(x, w, padding=self.padding, stride=self.stride, groups=batch)


        x = x.view(x_batch, self.feature_map, x.shape[2], x.shape[3])
        return x
    
    def modulate(self, weight, style_data):
        out_c, in_c, kh, kw = weight.shape

        w = weight.view(1, out_c, in_c, kh, kw)

        batch, f_map = style_data.shape
        s_d = self.dense_layer(style_data)
        s_b = self.fused_bias_act(s_d) + 1
        s = s_b.view(batch, 1, in_c, 1, 1)

        weight_scaled = w * s
        return weight_scaled
    
    def demodulate(self, weight):
        batch, out_c, in_c, kh, kw = weight.shape
        ww_sum = weight.pow(2).sum(dim=[2, 3, 4]) + 1e-8
        r_dev = torch.rsqrt(ww_sum).view(batch, out_c, 1, 1,1)
        ret = weight * r_dev
        return ret
    
#%%p290
class Block(BaseLayer):
    def __init__(self, res, style_dim):
        super(BaseLayer, self).__init__()
        self.x_channel = self.cliped_features(res -2)
        self.feature_map = self.cliped_featurees(res -1)

        layer_index_upsample = res * 2 - 5
        self.layer_upsample = Layer(
            x_channel=self.x_channel,
            style_layer_index=layer_index_upsample,
            style_in_dim=style_dim,
            style_out_dim=style_dim,
            feature_map=self.feature_map,
            res=res,
            is_up=True
        )

        layer_index = res * 2 - 4
        self.layer = Layer(
            x_channel=self.feature_map,
            style_layer_index=layer_index,
            style_in_dim=style_dim,
            style_out_dim=self.cliped_features(res-1),
            feature_map=self.feature_map,
            res=res
        )
    
    def forward(self, x, style):
        x = self.layer_upsample(x, style)
        x = self.layer(x, style)
        return x
    
#%%p292
class ToRGB(BaseLayer):
    def __init__(self, x_channel, out_channel, kernel, style_dim, res):
        super(ToRGB, self).__init__()
        self.style_layer_index = res * 2 - 3

        self.modulate_conv2d = ModulateConv(
            x_channel=x_channel,
            feature_map=out_channel,
            style_in_dim=style_dim,
            style_out_dim=self.cliped_features(res - 1),
            kernel=kernel,
            padding=0,
            is_demodulate=False,)
        self.fused_bias_act = FusedBiasActivation(out_channel, act='Linear')

    def forward(self, x, style, before_image):
        s = style[:, self.style_layer_index]

        image = self.modulate_conv2d(x,s)
        image = self.fused_bias_act(image)
        if before_image is not None:
            image = image + before_image
        return image

#%%p296
class Discriminator(BaseLayer):
    def __init__(self, opt) -> None:
        super(Discriminator, self).__init__()
        self.block_dict = torch.nn.ModuleDict()
        for res in range(self.resolution_log2, 2, -1):
            if res == self.resolution_log2:
                self.fromrgb = FromRGB(res)
            self.block_dict[str(res)] = Block(res)
        self.minibatch_stddev_layer = MiniBatchStddevLayer(self.mbstd_group_size, self.mbstd_num_features)

        in_feature_map = self.cliped_features(1) + 1
        self.conv2d_layer = Conv2dLayer(
            in_feature_map=in_feature_map,
            out_feature_map=self.cliped_features(1),
            kernel=3,
            padding=1
        )
        self.fused_bias_act = FusedBiasActivation(channel=self.cliped_features(1), act='LeakyRelu')

        in_channel = self.cliped_faetures(0) * 4 ** 2
        self.dense_layer1 = DenseLayer(
            in_channel=in_channel,
            feature_map=self.cliped_features(0))
        self.fused_bias_act1 = FusedBiasActivation(channel=self.cliped_features(0), act='LeakyRelu')
        self.dense_layer2 = DenseLayer(in_channel=self.cliped_features(0), feature_map=1)
        self.fused_bias_act2 = FusedBiasActivation(channel=1)

    def forward(self, image):
        x = None
        for res in range(self.resolution_log2, 2, -1):
            if res == self.resolution_log2:
                x = self.fromrgb(x, image)
            x = self.block_dict[str(res)](x)

            if self.mbstd_group_size > 1:

                x = self.minibatch_stddev_layer(x)

            x = self.conv2d_layer(x)
            x = self.fused_bias_act(x)

            x = self.dense_layer1(x)
            x = self.fused_bias_act1(x)

            x = self.dense_layer2(x)
            x = self.fused_bias_act2(x)

            return x
        
#%%p300
class FromRGB(BaseLayer):
    def __init__(self, res) -> None:
        super(FromRGB, self).__init__()
        slef.conv2d_layer = Conv2dLayer(in_feature_map=3, out_feature_map=self.cliped_features(res - 1), kernel=1)
        self.fused_bias_act = FusedBiasActivation(channel=self.cliped_features(res - 1), act='LeakyRelu')

    def forward(self, image):
        t = self.conv2d_layer(image)
        t = self.fused_bias_act(t)
        return t
    
#%%p302
class Block(BaseLayer):
    def __init__(self, res) -> None:
        super(Block, self).__init__()

        self.conv2d_layer = Conv2dLayer(
            in_feature_map=self.cliped_features(res - 1),
            out_feature_map=self.cliped_features(res - 1),
            kernel=3,
            padding=1
        )
        self.fused_bias_act = FusedBiasActivation(channel=self.cliped_features(res - 1), act='LeakyRelu')

        self.conv2d_layer_down1 = Conv2dLayer(
            in_feature_map = self.cliped_featuers(res - 1),
            out_feature_map = self.cliped_features(res - 2),
            kernel=3,
            padding=1,
            down=True,
            resample_kernel=[1, 3, 3, 1])
        self.fused_bias_act1 = FusedBiasActivation(
            channel=self.cliped_features(res - 2), act='LeakyRelu'
        )

        self.conv2d_layer_down2 = Conv2dLayer(
            in_feature_map = self.cliped_featuers(res - 1),
            out_feature_map = self.cliped_features(res - 2),
            kernel=1,
            down=True,
            resample_kernel=[1, 3, 3, 1])
        
    def forward(self, x):
        t = x
        x = self.conv2d_layer(x)
        x = self.fused_bias_act(x)

        x = self.conv2d_layer_down1(x)
        x = self.fused_bias_act1(x)

        t = self.conv2d_layer_down2(t)
        x = (x + t) * (1 / np.sqrt(2))
        return x

#%%p304
class Conv2dLayer(BaseLayer):
    def __init__(self, in_feature_map, out_feature_map, kernel, padding=0, down=False, resmple_kernel=None, gain=1, use_wscale=True, lrmul=1):
        super(Conv2dLayer, self).__init__()

        self.down = down
        self.padding = padding
        w, runtime_coef = self.get_weight_and_runtime_coef(shape=[out_feature_map, in_feature_map, kernel, kernel], gain=gain, use_wscale=use_wscale, lrmul=lrmul)

        self.weight = torch.nn.Parameter(w)
        self.runtime_coef = runtime_coef
        if self.down:
            self.downsample_conv2_2d = DownSampleConv2d(resample_kernel=resample_kernel)

    def forward(self, x):
        if self.down:
            x = self.downsample_conv2_2d(x, self.weight * self.runtime_coef)
        else:
            x = F.conv2d(x, self.weight * self.runtime_coef, padding=self.padding, stride=1)
        return x

#%%p306
class GeneratorLoss(BaseLayer):
    def __init__(self) -> None:
        super(GeneratorLoss, self).__init__()
        self.sofplus = torch.nn.Softplus()

    def forward(self, fake_scores_out):
        loss = torch.mean(self.softplus(-fake_scores_out))
        return losos
    

class DiscriminatorLoss(BaseLayer):
    def __init__(self) -> None:
        super(DiscriminatorLoss, self).__init__()
        self.softplus_fake = torch.nn.Softplus()
        self.softplus_real = torch.nn.Softplus()

    def forward(self, fake_scores_out, real_scores_out):
         loss = torch.mean(self.softplus_fake(fake_scores_out)) + torch.mean(self.softplus_real(-real_scores_out))
         return loss

#%%p307
class GeneratorLossPathReg(BaseLayer):
    def __init__(self, pl_decay=0.01, pl_weight=2.0, opt=None):
        super(GeneratorLossPathReg, self).__init__()
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean_var = self.Tensor(np.zeros(1,))
        self.reg_interval = 4 if opt is None else opt.g_reg_interval
        self.opt = opt

    def forward(self, fake_images_out, fake_dlatents_out):
        pl_noise = self.Tensor(np.random.normal(0, 1, fake_images_out.shape)) / np.sqrt(np.prod(fake_images_out.shape[2:]))
        f_img_out_pl_n = torch.sum(fake_images_out * pl_noise)
        pl_grads = torch.autograd.grad(outputs=f_img_out_pl_n, iputs=fake_dlatents_out, create_graph=True)[0]
        pl_grads_sum_mean = pl_grads.pow(2).sum(dim=2).mean(dim=1)
        pl_length = torch.sqrt(pl_grads_sum_mean)

        pl_mean = self.pl_mean_var + self.pl_decay * (pl_length.mean() - self.pl_mean_var)
        self.pl_mean_var = pl_mean.detach()
        pl_penalty = (pl_length - pl_mean).pow(2).mean()
        reg = pl_penalty * self.pl_weight * self.reg_interval
        return reg, pl_length.mean()
    
#%%p309
class Trainer:
    def __init__(self, opt) -> None:
        self.opt = opt
        self.generator = Generator(opt)
        self.generator_predict = Generator(opt)
        self.discriminator = Discriminator(opt)

        self.decay = 0.5 ** (opt.batch_size / (10 * 1000)) * opt.adjust_decay_param
        first_decay = 0
        if opt.is_restore_model:
            models = BaseLayer.restore_model(opt.model_path)
            if models is not None:
                self.generator, self.generator_predict, self.discriminator = models
                first_decay = self.decay
        self.generator.train()
        self.generator_predict.eval()

        Generator.apply_decay_parametes(
            self.generator, self.generator_predict, decay=first_decay
        )
        self.discriminator.train()

        self.generator_loss = GeneratorLoss()
        self.generator_loss_path_reg = GeneratorLossPathReg(opt=opt)

        self.dataloader = get_dataloader(
            opt.data_path, opt.resolution, opt.batch_size)
        self.fid = FrechetInceptionDistance(
            self.generator_predict, self.dataloader, opt
        )

        leaning_rate, beta1, beta2 = self.get_adam_parametes_adjust_interval(opt.g_reg_interval, opt)
        self.optimizer_g = torch.optim.Adam(self.generatoor.parameters(), lr=leaning_rate, betas=(beta1, beta2))

        leaning_rate, beta1, beta2 = self.get_adam_params_adjust_interval(opt.d_reg_interval, opt)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=leaning_rate, betas=(beta1, beta2))

    def train_generator(self, current_loop_num):
        BaseLayer.set_model_parameter_grad_all(self.generator, True)
        BaseLayer.set_model_parameter_requires_grad_all(self.discriminator, False)

        for index in range(0, self.opt.generator_train_num):
            train_z = self.Tensor(np.random.normal(loc=0, scale=1, size=(self.opt.batch_size, self.opt.latent_dim)))
            fake_imgs, fake_dlatents_out = self.generator(train_z)
            fake_validity = self.discriminator(fake_imgs)

            g_loss = self.generator_loss(fake_validity)
            self.optimizer_g.zero_grad()
            g_loss.backward()
            self.optimizer_g.step()

        run_g_reg = current_loop_num % self.opt.lgreg_interval == 0
        if run_g_reg:
            g_reg_maxcount = 4 if 4 < self.opt.generator_train_num else self.opt.generator_train_num
            for _ in range(0, g_reg_maxcount):
                z = self.Tensor(np.random.normal(loc=0, scale=1, size=(self.opt.batch_size, self.opt.latent_dim)))
                pl_fake_imgs, pl_fake_dlatents_out = self.generator(z)
                g_reg, pl_length = self.generator_loss_path_reg(pl_fake_imgs, pl_fake_dlatents_out)
                self.optimizer_g.zero_grad()
                g_reg.backward()
                self.optimizer_g.step()
        Generator.apply_decay_parameters(self.generator, self.generator_predict, decay=self.decay)

        return g_loss
    
    def train_discriminator(self. current_loop_num):
        BaseLayer.set_model_parameter_requires_grad_all(self.generator, False)
        BaseLayer.set_model_parameter_requries_grad_all(self.discriminator, True)

        for index in range(0, self.opt.discriminator_train_num):
            data_iterator = self.dataloader.__iter__()
            imgs = data_iterator.next()
            real_imgs = Variable(imgs.type(self.Tensor), requires_grad=False)
            
            z = self.Tensor(np.random.normal(loc=0, scale=1, size=(self.opt.batch_size, self.opt.latent_dim)))
            fake_imgs, fake_dlatents_out = self.generator(z)

            real_validity = self.discriminator(real_imgs)
            fake_validity = self.discriminator(fake_imgs)

            d_loss = self.discriminator_loss(fake_validity, real_validity)
            self.optimizer_d.zero_grad()
            d_loss.backward()
            self.optimizer_d.step()

        run_d_reg = current_loop_num %  self.opt.d_reg_interval == 0
        if run_d_reg:
            d_reg_maxcount = 4 if 4 < self.opt.discriminator_train_num else self.opt.discriminator_train_num

            for index in range(0, d_reg_maxcount):
                real_imgs.requires_grad = True
                real_validity = self.discriminator(real_imgs)

                d_reg = self.discriminator_loss_r1(real_validity, real_imgs)

                self.optimizer_d.zero_grad()
                d_reg.backward()
                self.optimizer_d.step()

        return d_loss
    
#%p319
def main(opt):
    trainer = Trainer(opt)

    for current_loop_num in range(opt.max_loop_num):
        # generatorの学習
        g_loss = trainer.train_generator(current_loop_num)

        # discriminatorの学習
        d_loss = trainer.train_discriminator(current_loop_num)

        # ログの出力
        print('current_loop_num: {}, d_loss: {}, g_loss: {}'.format(current_loop_num, d_loss, g_loss))

        if current_loop_num % opt.save_model_interval == 0:
            # save model
            trainer.save_model()

        if current_loop_num % opt.fid_score_interval == 0 and 0 < current_loop_num:
            # FID score を計算する
            trainer.calculate_fid_score()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # コマンドライン引数の定義
    ###### 
    # TODO 省略されている コードを探す
    #############

    option = parser.parse_args()
    print(option)

    main(option)




# %%
