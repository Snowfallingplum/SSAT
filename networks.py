import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler

####################################################################
# ------------------------- Discriminators --------------------------
####################################################################
class Dis(nn.Module):
    def __init__(self, input_dim, norm='None', sn=False):
        super(Dis, self).__init__()
        ch = 64
        n_layer = 5
        self.model = self._make_net(ch, input_dim, n_layer, norm, sn)

    def _make_net(self, ch, input_dim, n_layer, norm, sn):
        model = []
        model += [LeakyReLUConv2d(input_dim, ch, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)]
        tch = ch
        for i in range(1, n_layer):
            model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)]
            tch *= 2
        model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm='None', sn=sn)]
        tch *= 2
        if sn:
            model += [nn.utils.spectral_norm(nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0))]
        else:
            model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]
        return nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        out = out.view(-1)
        outs = []
        outs.append(out)
        return outs


class MultiScaleDis(nn.Module):
    def __init__(self, input_dim, n_scale=3, n_layer=4, norm='None', sn=False):
        super(MultiScaleDis, self).__init__()
        ch = 64
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.Diss = nn.ModuleList()
        for _ in range(n_scale):
            self.Diss.append(self._make_net(ch, input_dim, n_layer, norm, sn))

    def _make_net(self, ch, input_dim, n_layer, norm, sn):
        model = []
        model += [LeakyReLUConv2d(input_dim, ch, 4, 2, 1, norm, sn)]
        tch = ch
        for _ in range(1, n_layer):
            model += [LeakyReLUConv2d(tch, tch * 2, 4, 2, 1, norm, sn)]
            tch *= 2
        if sn:
            model += [nn.utils.spectral_norm(nn.Conv2d(tch, 1, 1, 1, 0))]
        else:
            model += [nn.Conv2d(tch, 1, 1, 1, 0)]
        return nn.Sequential(*model)

    def forward(self, x):
        outs = []
        for Dis in self.Diss:
            outs.append(Dis(x))
            x = self.downsample(x)
        return outs


####################################################################
# ---------------------------- Encoders -----------------------------
####################################################################

class E_content(nn.Module):
    def __init__(self, input_dim, ngf=64):
        super(E_content, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=ngf, kernel_size=7, stride=1, padding=3)
        self.Instance1 = nn.InstanceNorm2d(ngf)
        self.relu1 = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(in_channels=ngf, out_channels=ngf * 2, kernel_size=3, stride=2, padding=1)
        self.Instance2 = nn.InstanceNorm2d(ngf * 2)
        self.relu2 = nn.LeakyReLU(0.2, True)
        self.conv3 = nn.Conv2d(in_channels=ngf * 2, out_channels=ngf * 4, kernel_size=3, stride=2, padding=1)
        self.Instance3 = nn.InstanceNorm2d(ngf * 4)
        self.relu3 = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        # x (3, 256, 256)
        out = self.conv1(x)
        out = self.Instance1(out)
        out = self.relu1(out)
        feature_map1 = out
        # x (64, 256, 256)
        out = self.conv2(out)
        out = self.Instance2(out)
        out = self.relu2(out)
        feature_map2 = out
        # x (64*2, 128, 128)
        out = self.conv3(out)
        out = self.Instance3(out)
        out = self.relu3(out)
        feature_map3 = out
        # x (64*4, 64, 64)
        return feature_map1, feature_map2, feature_map3

class E_makeup(nn.Module):
    def __init__(self, input_dim, ngf=64):
        super(E_makeup, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=ngf, kernel_size=7, stride=1, padding=3)
        self.Instance1 = nn.InstanceNorm2d(ngf)
        self.relu1 = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(in_channels=ngf, out_channels=ngf * 2, kernel_size=3, stride=2, padding=1)
        self.Instance2 = nn.InstanceNorm2d(ngf * 2)
        self.relu2 = nn.LeakyReLU(0.2, True)
        self.conv3 = nn.Conv2d(in_channels=ngf * 2, out_channels=ngf * 4, kernel_size=3, stride=2, padding=1)
        self.Instance3 = nn.InstanceNorm2d(ngf * 4)
        self.relu3 = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        # x (3, 256, 256)
        out = self.conv1(x)
        out = self.Instance1(out)
        out = self.relu1(out)
        # x (64, 256, 256)
        out = self.conv2(out)
        out = self.Instance2(out)
        out = self.relu2(out)
        # x (64*2, 128, 128)
        out = self.conv3(out)
        out = self.Instance3(out)
        out = self.relu3(out)
        # x (64*4, 64, 64)
        return out

class E_semantic(nn.Module):
    def __init__(self, input_dim, ngf=32):
        super(E_semantic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=ngf, kernel_size=7, stride=1, padding=3)
        self.Instance1 = nn.InstanceNorm2d(ngf)
        self.relu1 = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(in_channels=ngf, out_channels=ngf * 2, kernel_size=3, stride=2, padding=1)
        self.Instance2 = nn.InstanceNorm2d(ngf * 2)
        self.relu2 = nn.LeakyReLU(0.2, True)
        self.conv3 = nn.Conv2d(in_channels=ngf * 2, out_channels=ngf * 4, kernel_size=3, stride=2, padding=1)
        self.Instance3 = nn.InstanceNorm2d(ngf * 4)
        self.relu3 = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        # x (3, 256, 256)
        out = self.conv1(x)
        out = self.Instance1(out)
        out = self.relu1(out)
        # x (16, 256, 256)
        out = self.conv2(out)
        out = self.Instance2(out)
        out = self.relu2(out)
        # x (32, 128, 128)
        out = self.conv3(out)
        out = self.Instance3(out)
        out = self.relu3(out)
        # x (64, 64, 64)
        return out


class FeatureFusion(nn.Module):
    def __init__(self, ngf=64):
        super(FeatureFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=(32 * 4 + 64 * 4), out_channels=ngf * 8, kernel_size=3, stride=2, padding=1)
        self.Instance1 = nn.InstanceNorm2d(ngf * 8)
        self.relu1 = nn.LeakyReLU(0.2, True)

    def forward(self, x, y):
        # x[0] (64*1, 256, 256)
        # x[1] (64*2, 128, 128)
        # x[2] (64*4, 64, 64)
        # y (32*4, 64,64)
        content_feature_map1 = x[0]
        content_feature_map2 = x[1]
        content_feature_map3 = x[2]
        semantic_feature_map = y
        out = torch.cat([content_feature_map3, semantic_feature_map], dim=1)
        out = self.conv1(out)
        out = self.Instance1(out)
        out = self.relu1(out)

        feature_map1 = F.interpolate(content_feature_map1, scale_factor=0.25, mode='bilinear')  # (64*1, 64, 64)
        feature_map2 = F.interpolate(content_feature_map2, scale_factor=0.5, mode='bilinear')  # (64*2, 64, 64)
        feature_map3 = F.interpolate(content_feature_map3, scale_factor=1, mode='bilinear')  # (64*4, 64, 64)
        feature_map4 = F.interpolate(out, scale_factor=2, mode='bilinear')  # (64*8, 64, 64)
        feature_map5 = F.interpolate(semantic_feature_map, scale_factor=1, mode='bilinear')  # (64*2, 64, 64)

        output = torch.cat([feature_map1,
                            feature_map2,
                            feature_map3,
                            feature_map4,
                            feature_map5
                            ], dim=1)
        return output

####################################################################
# ----------------------- Symmetry Transformer----------------------
####################################################################
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.fusion=FeatureFusion()
        self.atte=SymmetryAttention()

    def forward(self, x_c, y_c,x_s,y_s,x_m,y_m):
        x_f = self.fusion(x_c,x_s)
        y_f = self.fusion(y_c, y_s)
        attention_x,attention_y,x_m_warp,y_m_warp=self.atte(x_f,y_f,x_m,y_m)
        return attention_x,attention_y,x_m_warp,y_m_warp


####################################################################
# ----------------------- Attention Moudle -----------------------
####################################################################
class SymmetryAttention(nn.Module):
    def __init__(self, norm='Instance', sn=False):
        super(SymmetryAttention, self).__init__()
        in_dim = 64 * 17
        self.chanel_in = in_dim
        self.softmax_alpha = 100
        self.fa_conv = LeakyReLUConv2d(in_dim, in_dim // 8, kernel_size=1, stride=1, padding=0, norm=norm, sn=sn)
        self.fb_conv = LeakyReLUConv2d(in_dim, in_dim // 8, kernel_size=1, stride=1, padding=0, norm=norm, sn=sn)

    def warp(self, fa, fb, a_raw, b_raw, alpha):
        '''
            calculate correspondence matrix and warp the exemplar features
        '''
        assert fa.shape == fb.shape, \
            'Feature shape must match. Got %s in a and %s in b)' % (fa.shape, fb.shape)
        n, c, h, w = fa.shape
        _, raw_c, _, _ = a_raw.shape
        # subtract mean
        fa = fa - torch.mean(fa, dim=(2, 3), keepdim=True)
        fb = fb - torch.mean(fb, dim=(2, 3), keepdim=True)

        # vectorize (merge dim H, W) and normalize channelwise vectors
        fa = fa.view(n, c, -1)
        fb = fb.view(n, c, -1)
        fa = fa / torch.norm(fa, dim=1, keepdim=True)
        fb = fb / torch.norm(fb, dim=1, keepdim=True)

        # warp the exemplar features b, taking softmax along the b dimension
        energy_ab_T = torch.bmm(fb.transpose(-2, -1), fa) * alpha
        # fix bug dim=1 in softmax, see https://github.com/Snowfallingplum/SSAT/issues/5
        corr_ab_T = F.softmax(energy_ab_T, dim=1)  # n*HW*C @ n*C*HW -> n*HW*HW 
        # print(softmax_weights.shape, b_raw.shape)
        b_warp = torch.bmm(b_raw.view(n, raw_c, h * w), corr_ab_T)  # n*HW*1
        b_warp = b_warp.view(n, raw_c, h, w)

        energy_ba_T = torch.bmm(fa.transpose(-2, -1), fb) * alpha
        # fix bug dim=1 in softmax, see https://github.com/Snowfallingplum/SSAT/issues/5
        corr_ba_T = F.softmax(energy_ba_T, dim=1)  # n*HW*C @ n*C*HW -> n*HW*HW

        a_warp = torch.bmm(a_raw.view(n, raw_c, h * w), corr_ba_T)  # n*HW*1
        a_warp = a_warp.view(n, raw_c, h, w)
        return corr_ab_T, corr_ba_T, a_warp, b_warp


    def forward(self, fa, fb, a_raw, b_raw):
        fa = self.fa_conv(fa)
        fb = self.fb_conv(fb)
        X, Y, a_warp, b_warp = self.warp(fa, fb, a_raw, b_raw, self.softmax_alpha)
        return X, Y, a_warp, b_warp


####################################################################
# -------------------------- Decoder --------------------------
####################################################################

class Decoder(nn.Module):
    def __init__(self, output_dim=3, ngf=64):
        super(Decoder, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.SPADE1 = SPADEResnetBlock(ngf * 4, ngf * 4, ngf * 4)
        self.SPADE2 = SPADEResnetBlock(ngf * 4, ngf * 2, ngf * 4)
        self.SPADE3 = SPADEResnetBlock(ngf * 2, ngf * 1, ngf * 4)
        self.img_conv = nn.Conv2d(ngf * 1, output_dim, kernel_size=7, stride=1, padding=3)
        self.tanh = nn.Tanh()

    def forward(self, x, y):
        content=x[-1]
        makeup=y
        out = self.SPADE1(content,makeup)
        out = self.up(out)
        out = self.SPADE2(out, makeup)
        out = self.up(out)
        out = self.SPADE3(out, makeup)
        out=self.img_conv(out)
        out = self.tanh(out)
        return out

####################################################################
# -------------------------- Basic Blocks --------------------------
####################################################################
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, semantic_nc):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # define normalization layers
        self.norm_0 = SPADE(fin, semantic_nc)
        self.norm_1 = SPADE(fmiddle, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        ks = 3
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


# conv + (spectral) + (instance) + leakyrelu
class LeakyReLUConv2d(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding=0, norm='None', sn=False):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        if sn:
            model += [nn.utils.spectral_norm(
                nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
        else:
            model += [nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        if norm == 'Instance':
            model += [nn.InstanceNorm2d(outplanes, affine=False)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Residule_Block(nn.Module):
    def __init__(self, nc):
        super(Residule_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.instance1 = nn.InstanceNorm2d(nc)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.instance2 = nn.InstanceNorm2d(nc)

    def forward(self, x):
        out = self.conv1(x)
        out = self.instance1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.instance2(out)
        return out + x


####################################################################
# ------------------------- Basic Functions -------------------------
####################################################################
def get_scheduler(optimizer, opts, cur_ep=-1):
    if opts.lr_policy == 'lambda':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / float(opts.n_ep - opts.n_ep_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
    elif opts.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
    elif opts.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opts.lr_policy)
    return scheduler


def init_weights(net, init_type, gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fainplanes')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, gpu, init_type='normal', gain=0.02):
    assert (torch.cuda.is_available())
    net.to(gpu)
    init_weights(net, init_type, gain)
    return net
