import torch
from torch import nn
from torch.nn import functional as F
from pdb import set_trace as st
from layers.SelfAttLayer import SelfAttLayer

import extractors


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, n_classes=3, n_meiboclass=4, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
                 pretrained=True, inchannel = 3):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained, inchannel)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.LogSoftmax()
        )
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=3, padding=1)           
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=3, padding=1)           
        )
 
        self.encoder3 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=7),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=3, padding=1),
            nn.Conv2d(20, 40, kernel_size=5),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=3, padding=1),
            nn.Conv2d(40, 80, kernel_size=3),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=3, padding=1),
            nn.Conv2d(80, 160, kernel_size=3),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=3, padding=1)    
        )
               
        self.classifier2 = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),            
            nn.Linear(128, n_meiboclass),#n_classes)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),            
            nn.Linear(128, n_meiboclass)
        )
        
        self.classifier3 = nn.Sequential(
            nn.Linear(2, n_meiboclass),
        )
        
        self.classifierF = nn.Sequential(
            nn.Linear(12, n_meiboclass)
        )
        
        self.segcoef = nn.Sequential(
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        
        self.clascoef = nn.Sequential(
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        
        self.fincoef = nn.Sequential(
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        self.selfatt = SelfAttLayer(in_channels=2048)
        self.selfatt2 = SelfAttLayer(in_channels=1024)
        
    def forward(self, x):
        f, class_f = self.feats(x)
        
        # att_ori
        #f = self.selfatt(f)
        # att_end
        
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        # att_cls
        f = self.selfatt(f)
        class_f = self.selfatt2(class_f)
        # end
        
        auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))
        seg_map = self.final(p)
        
        temp1 = F.adaptive_max_pool2d(input=f, output_size=(1, 1)).view(-1, f.size(1))
        #temp2 = F.adaptive_max_pool2d(input=seg_map, output_size=(1, 1)).view(-1, seg_map.size(1))        
        #all_feat = torch.cat((auxiliary, temp1, temp2), 1)
        all_feat = torch.cat((auxiliary, temp1), 1)
        classifier = self.classifier(auxiliary)
        
        count_temp = torch.max(seg_map, 1)[1]
        c1 = torch.sum( torch.sum(count_temp==1, dim=-1), dim=-1, keepdim = True)
        c2 = torch.sum( torch.sum(count_temp==2, dim=-1), dim=-1, keepdim = True)
        
        const = (seg_map.shape[-1])**2
        c2p = c2.float()/ const
        c1p = c1.float()/ const
        c_all = self.classifier3(torch.cat((c2p, c1p), dim=1) )
        feature = self.classifier2(all_feat)
        
        classifier_fin = self.classifierF(torch.cat((feature, classifier, c_all), 1) )
        return seg_map, classifier, classifier_fin, self.segcoef(c_all), self.clascoef(classifier), self.fincoef(classifier_fin)  #, self.classifier2(all_feat), all_feat
