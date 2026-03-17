import os
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from models.net import MobileNetV1
from models.net import FPN
from models.net import SSH


class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.conv1x1 = nn.Conv2d(
            inchannels,
            num_anchors * 2,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(
            inchannels,
            num_anchors * 4,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(
            inchannels,
            num_anchors * 10,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):
    def __init__(self, cfg=None, phase='train'):
        super(RetinaFace, self).__init__()
        self.phase = phase

        if cfg is None:
            raise ValueError("cfg must not be None")

        # ===============================
        # Backbone
        # ===============================
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()

            if cfg['pretrain']:
                #  FIXED ABSOLUTE PATH
                current_dir = os.path.dirname(os.path.abspath(__file__))
                weight_path = os.path.join(
                    current_dir,
                    "..",
                    "weights",
                    "mobilenetV1X0.25_pretrain.tar"
                )

                weight_path = os.path.abspath(weight_path)

                if not os.path.exists(weight_path):
                    raise FileNotFoundError(
                        f"Pretrain weight not found at {weight_path}"
                    )

                checkpoint = torch.load(
                    weight_path,
                    map_location=torch.device('cpu')
                )

                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove "module."
                    new_state_dict[name] = v

                backbone.load_state_dict(new_state_dict)

        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])

        else:
            raise ValueError("Unsupported backbone type")

        # ===============================
        # FPN
        # ===============================
        self.body = _utils.IntermediateLayerGetter(
            backbone,
            cfg['return_layers']
        )

        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]

        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list, out_channels)

        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        # ===============================
        # Heads
        # ===============================
        self.ClassHead = self._make_class_head(3, out_channels)
        self.BboxHead = self._make_bbox_head(3, out_channels)
        self.LandmarkHead = self._make_landmark_head(3, out_channels)

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        return nn.ModuleList([
            ClassHead(inchannels, anchor_num)
            for _ in range(fpn_num)
        ])

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        return nn.ModuleList([
            BboxHead(inchannels, anchor_num)
            for _ in range(fpn_num)
        ])

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        return nn.ModuleList([
            LandmarkHead(inchannels, anchor_num)
            for _ in range(fpn_num)
        ])

    # ===============================
    # Forward
    # ===============================
    def forward(self, inputs):

        out = self.body(inputs)
        fpn = self.fpn(out)

        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])

        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat(
            [self.BboxHead[i](feature) for i, feature in enumerate(features)],
            dim=1
        )

        classifications = torch.cat(
            [self.ClassHead[i](feature) for i, feature in enumerate(features)],
            dim=1
        )

        ldm_regressions = torch.cat(
            [self.LandmarkHead[i](feature) for i, feature in enumerate(features)],
            dim=1
        )

        if self.phase == 'train':
            return bbox_regressions, classifications, ldm_regressions
        else:
            return (
                bbox_regressions,
                F.softmax(classifications, dim=-1),
                ldm_regressions
            )