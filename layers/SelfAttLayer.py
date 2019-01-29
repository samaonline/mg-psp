import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from pdb import set_trace as st

# TODO: implement dot_product and other non-local formats
class SelfAttLayer(nn.Module):

    def __init__(self, in_channels, reduction = 2, mode='embedded_gaussian'):
        super(SelfAttLayer, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian']

        self.g = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size = 1)
        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size = 1)
        self.phi = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size = 1)
        self.conv_mask = nn.Conv2d(self.inter_channels, self.in_channels, kernel_size = 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.scalar = Parameter(torch.Tensor(1), requires_grad=True)

        self.init_weights()

    def init_weights(self):
        msra_list = [self.g, self.theta, self.phi]
        for m in msra_list:
            nn.init.kaiming_normal_(m.weight.data)
            m.bias.data.zero_()
        self.conv_mask.weight.data.zero_()
        self.scalar.data.zero_()

    def embedded_gaussian(self, x):
        # embedded_gaussian cal self-attention, which may not strong enough
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        map_t_p = torch.matmul(theta_x, phi_x)
        mask_t_p = F.softmax(map_t_p, dim=-1)

        map_ = torch.matmul(mask_t_p, g_x)
        map_ = map_.permute(0, 2, 1).contiguous()
        map_ = map_.view(batch_size, self.inter_channels, x.size(2), x.size(3))
        mask = self.conv_mask(map_)
        # final = mask + x
        final = self.scalar * mask + x
        return final

    def forward(self, x):
        if self.mode == 'embedded_gaussian':
            output = self.embedded_gaussian(x)
        else:
            raise NotImplemented("The code has not been implemented.")
        return output
