import torch
from torch import nn
from models.encoder import FPN
from models.afp import AFP
from models.ids import ISD
from models.transformer import Transformer
import torch.nn.functional as F
from models.regressor import Regressor

class Model(nn.Module):
    def __init__(self, mode='train', iters=2):
        super(Model, self).__init__()
        self.mode = mode
        self.backbone = FPN()

        self.afp = AFP()
        self.transformer = Transformer(mlp_ratio=4., injection=True)
        self.transformer2 = Transformer(mlp_ratio=4., injection=False)

        self.iters = iters
        self.ids = ISD()

        self.regressor = Regressor()

    def forward(self, img, targets=None):
        feature_img, fpn = self.backbone(img)

        idirs = None
        for i in range(self.iters):
            feature_img = self.transformer(feature_img, feature_img)
            idirs = self.afp(feature_img, idirs)

        for i in range(self.iters):
            feature_img = self.transformer2(feature_img, feature_img)

        out = self.ids(fpn, idirs)

        if self.mode == 'train':
            gt_mano_params = torch.cat([targets['mano_pose'], targets['mano_shape']], dim=1)
        else:
            gt_mano_params = None

        pred_mano_results, gt_mano_results, preds_joints_img = self.regressor(feature_img, gt_mano_params)

        return (pred_mano_results, gt_mano_results, preds_joints_img), out


    def init_weights(self, m):
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight,std=0.001)
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight,std=0.001)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight,std=0.01)
            nn.init.constant_(m.bias,0)

    def save_model(self, state, epoch, best=False, last=False):
        import os
        if not best:
            file_path = os.path.join('/home/ana/Study/CVPR/cvpr2/models', "snapshot_{}.pth.tar".format(str(epoch)))
        else:
            file_path = os.path.join('/home/ana/Study/CVPR/cvpr2/models', "best.pth.tar")

        if last:
            file_path = os.path.join('/home/ana/Study/CVPR/cvpr2/models', "last.pth.tar")

        torch.save(state, file_path)

if __name__ == '__main__':
    from losses.losses import idisc_loss
    model = Model('test').to('cuda')
    model.train()
    a = torch.rand((1, 3, 256, 256)).to('cuda')
    out, out2 = model(a)
    gt = torch.rand((2, 1, 256, 256)).to('cuda')
    mask = torch.rand((2, 1, 256, 256))
    x = idisc_loss(out ,gt)
    model.save_model({"network": model.state_dict()}, 1, True)


