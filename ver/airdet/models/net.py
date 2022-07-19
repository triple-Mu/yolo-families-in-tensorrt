import torch
import torch.nn as nn


class Detector(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.backbone = build_backbone(config.model.backbone)
        self.neck = build_neck(config.model.neck)
        self.head = build_head(config.model.head)

        self.config = config
        self.conf_thre = self.config.testing.conf_threshold
        self.nms_thre = self.config.testing.nms_iou_threshold

    def init_bn(self, M):

        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def init_model(self):

        self.apply(self.init_bn)

        if self.config.training.pretrain_model is None:
            self.backbone.init_weights()
            self.neck.init_weights()
            self.head.init_weights()
        else:
            self.load_pretrain_detector(self.config.training.pretrain_model)

    def load_pretrain_detector(self, pretrain_model):

        state_dict = torch.load(pretrain_model, map_location='cpu')['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('module.', '')
            if 'head' in k:
                new_state_dict[k] = self.state_dict()[k]
                continue
            new_state_dict[k] = v

        self.load_state_dict(new_state_dict)
        print(f'load params from {pretrain_model}')


    def forward(self, x, targets=None):
        images = to_image_list(x)
        feature_outs = self.backbone(images.tensors)  # list of tensor
        fpn_outs = self.neck(feature_outs)
        if self.training:
            loss_dict = self.head(
                            xin=fpn_outs,
                            labels=targets,
                            imgs=images,
                            )
            return loss_dict
        else:
            outputs = self.head(
            fpn_outs,
            imgs=images,
            conf_thre=self.conf_thre,
            nms_thre=self.nms_thre
            )
            return outputs