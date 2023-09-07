import logging
import os
from functools import partial
from pathlib import Path
from typing import Type, Union, List, Dict, Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False,
    )


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
    ).zero_()
    # TODO
    # if not no_cuda:
    #     if isinstance(out.data, torch.cuda.FloatTensor):
    #         zero_pads = zero_pads.cuda()
    # zero_pads = zero_pads.to(torch_directml.device(torch_directml.default_device()))

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    # TODO CHECK
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int,
        shortcut_type: str = "B",
        no_cuda=False,
    ) -> None:
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4
        )

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classification_layer = nn.Linear(512 * block.expansion, num_classes)
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classification_layer(x)
        x = self.softmax(x)

        return x


def resnet10(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


# TODO add typing
def resnet34(**kwargs):
    """Constructs a ResNet-34 model."""
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


RESNET_MODELS_MAP: Dict[int, Callable[..., ResNet]] = {
    10: resnet10,
    18: resnet18,
    34: resnet34,
    50: resnet50,
    101: resnet101,
    152: resnet152,
    200: resnet200,
}


def create_resnet_model(
    model_depth: int,
    resnet_shortcut_type: str,
    number_of_classes: int,
    pretrained_model_path: Path,
    device: torch.device,
    use_23_dataset_pretrained_model: bool = False,
) -> Tuple[ResNet, List[Parameter], List[Parameter]]:
    assert model_depth in RESNET_MODELS_MAP
    assert resnet_shortcut_type in ["A", "B"]

    no_cuda = False if device == torch.device("cuda") else True

    model = RESNET_MODELS_MAP[model_depth](
        shortcut_type=resnet_shortcut_type,
        no_cuda=no_cuda,
        num_classes=number_of_classes,
    )

    # TODO figure out if no_cuda is needed
    if no_cuda:
        model = model.to(device)
        # model = nn.DataParallel(model)
        initialized_model_state_dict = model.state_dict()
    else:
        gpus_ids = [d for d in range(torch.cuda.device_count())]
        assert len(gpus_ids) > 0
        if len(gpus_ids) > 1:
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=gpus_ids)
            initialized_model_state_dict = model.state_dict()
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus_ids[0])
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=None)
            initialized_model_state_dict = model.state_dict()

    # load pretrained model
    logging.info(f"Loading pretrained model {pretrained_model_path}")
    pretrain: ResNet = torch.load(
        pretrained_model_path.joinpath(
            f"resnet_{model_depth}"
            f"{'' if not use_23_dataset_pretrained_model else '23dataset'}.pth"
        ),
        map_location=device,
    )

    if device == torch.device("cpu"):
        pretrained_model_state_dict = {
            k.replace("module.", ""): v for k, v in pretrain["state_dict"].items()
        }
    else:
        pretrained_model_state_dict = pretrain["state_dict"]

    initialized_model_state_dict.update(pretrained_model_state_dict)
    logging.info(
        f"Loaded the following layers from the pretrained model: {pretrained_model_state_dict.keys()}"
    )
    model.load_state_dict(initialized_model_state_dict)

    pretrained_model_parameters = []
    parameters_to_fine_tune = []

    for pname, p in model.named_parameters():
        if pname in pretrained_model_state_dict:
            pretrained_model_parameters.append(p)
        else:
            parameters_to_fine_tune.append(p)

    return model, pretrained_model_parameters, parameters_to_fine_tune
