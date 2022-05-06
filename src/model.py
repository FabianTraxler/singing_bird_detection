import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from torch.nn.parameter import Parameter
import torchaudio as ta
from torch.cuda.amp import autocast
import timm


class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()

        self.cfg = cfg

        self.n_classes = cfg.n_classes

        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=0,
            global_pool="",
            in_chans=cfg.in_chans,
        )

        if "efficientnet" in cfg.backbone:
            backbone_out = self.backbone.num_features
        else:
            backbone_out = self.backbone.feature_info[-1]["num_chs"]

        self.global_pool = GeM()

        self.head = nn.Linear(backbone_out, self.n_classes)

        if cfg.pretrained_weights is not None:
            sd = torch.load(cfg.pretrained_weights, map_location="cpu")["model"]
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            self.load_state_dict(sd, strict=True)
            print("weights loaded from", cfg.pretrained_weights)

        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        self.mixup = Mixup(mix_beta=cfg.mix_beta)


    def forward(self, data):
        x = data

        if self.cfg.mel_norm:
            x = (x + 80) / 80

        x = x.permute(0, 2, 1)
        x = x[:, None, :, :]

        x = self.backbone(x)

        x = self.global_pool(x)
        x = x[:, :, 0, 0]
        logits = self.head(x)

        return {"logits": logits.sigmoid(), "logits_raw": logits}


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    # Generalized mean: https://arxiv.org/abs/1711.02512
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (self.__class__.__name__+ "(p="+ "{:.4f}".format(self.p.data.tolist()[0])+ ", eps="+ str(self.eps)+ ")")


class Mixup(nn.Module):
    def __init__(self, mix_beta):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, X, Y, weight=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]

        Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]

        if weight is None:
            return X, Y
        else:
            weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
            return X, Y, weight
