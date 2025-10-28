import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import torch.nn.functional as F
from vmamba import VSSBlock

################
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = self.sigmoid(y)
        return x * y 
    
################
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * residual
    

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi
  
def init_model_weight(model):
    print("Start init weight")
    i = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            # nn.init.xavier_uniform_(m.weight.data)
            # nn.init.normal_(m.weight.data.bias, std=1e-6)
            i+=1
        
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            i+=1
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                i+=1

        elif isinstance(m, nn.LayerNorm):
            i+=1
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    print(f"End init weight per total of layers {i}")
    
#########################
def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        # print(w.shape)
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)



##### Double Convs
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
                
        self.conv = nn.Sequential(
            conv3x3(in_channels, out_channels, 1, 1, bias=False),
            nn.GroupNorm(32, out_channels, eps=1e-6),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels, 1, 1, bias=False),
            nn.GroupNorm(32, out_channels, eps=1e-6),
            nn.ReLU(inplace=True),
        ) 
        self.skip = nn.Sequential(
            conv1x1(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels, eps=1e-6),
            nn.ReLU(inplace=True))           

    def forward(self, x):
        return self.conv(x) + self.skip(x) 
    
########################################
class DoubleConv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv2, self).__init__()
                
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ) 
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))           

    def forward(self, x):
        return self.conv(x) + self.skip(x) 
    
#############################

###################
class MambaConv(nn.Module):
    def __init__(self,        
            hidden_dim: int = 0,
            out_dim: int = 0
        ):
        super().__init__()

        self.vssb = VSSBlock(hidden_dim,drop_path=0.6)

        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        ) 
        self.skip = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True))  

    def forward(self, x:torch.tensor):
        B,C,H, W = x.shape
        x = rearrange(x,'b c h w -> b h w c',b=B,h=H,w=W)
        z = self.vssb(x)
        # print("z=",z.shape)
        z_ = rearrange(z,'b h w c-> b c h w',b=B,h=H,w=W)
        z1 = self.conv(z_)
        z2 = self.skip(z_)
        # z1 = self.sa(z1)
        return z1 + z2
###### Attention Block (Sum)
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi        
###################
###### CoAttention
class CoAttention(nn.Module):
    def __init__(self, F_x, F_t):
        super(CoAttention,self).__init__()
        self.att1 = Attention_block(F_x, F_t, F_x//2)
        self.att2 = Attention_block(F_t, F_x, F_t//2)
        self.ca = ChannelAttention(F_x+F_t)        
    def forward(self, x, t):
        x1 = self.att1(x, t)
        x2 = self.att2(t, x)
        x = torch.cat((x1,x2), dim=1) 
        x = self.ca(x)
        return x    
###################
################
class CoASMamba(nn.Module):
    def __init__(self, F_x, F_t, F_r, F_out):
        super(CoASMamba,self).__init__()
        self.coatt = CoAttention(F_x, F_t)
        self.att = Attention_block(F_r, F_x+F_t, F_r//2) 
        self.sa =  SpatialAttention() 
        self.m = MambaConv(F_x+F_t, F_out)    
    def forward(self, x, t, r):
        x1 = self.coatt(x, t)
        x2 = self.sa(r)
        x = self.att(x2, x1) 
        x = self.m(x)        
        return x    
###################
class CoAMamba(nn.Module):
    def __init__(self, F_x1, F_x2, F_out):
        super(CoAMamba,self).__init__()
        self.coatt = CoAttention(F_x1, F_x2)
        self.m = MambaConv(F_x1+F_x2, F_out)    
    def forward(self, x, t):
        x = self.coatt(x, t)
        x = self.m(x)        
        return x    
############################# 
# CoAttentionp
class CoAttentionp(nn.Module):
    def __init__(self, F_x, F_t):
        super(CoAttentionp,self).__init__()
        self.att1 = Attention_block(F_x, F_t, F_x//2)
        self.att2 = Attention_block(F_t, F_x, F_t//2)       
    def forward(self, x, t):
        x1 = self.att1(x, t)
        x2 = self.att2(t, x)
        x = torch.cat((x1,x2), dim=1) 
        return x
###################
class DoubleLCoA(nn.Module):
    def __init__(self, F_x1, F_x2, F_d, F_out):
        super(DoubleLCoA, self).__init__()
        self.coatt1 = CoAttentionp(F_x1, F_x2)
        self.coatt2 = CoAttentionp(F_x1+F_x2, F_d)
        self.conv = DoubleConv2(F_x1+F_x2+F_d, F_out)    
    def forward(self, x1, x2, d):
        x = self.coatt1(x1, x2)
        x = self.coatt2(x, d) 
        x = self.conv(x)        
        return x    
###################


