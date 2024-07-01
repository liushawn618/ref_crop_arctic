import torch

from src.nets.hand_heads.hand_hmr import HandHMR
from src.nets.obj_heads.obj_hmr import ObjectHMR

class WrappedHandHMR(HandHMR):
    def forward(self, features, posi_vec, use_pool=True):
        if use_pool:
            feat = self.avgpool(features)
            feat = feat.view(feat.size(0), -1)
        else:
            feat = features

        feat = torch.concat([feat, posi_vec], dim=1)
        return super().forward(feat, False)
    
class WrappedObjectHMR(ObjectHMR):
    def forward(self, features, posi_vec, use_pool=True):
        if use_pool:
            feat = self.avgpool(features)
            feat = feat.view(feat.size(0), -1)
        else:
            feat = features

        feat = torch.concat([feat, posi_vec], dim=1)
        return super().forward(feat, False)