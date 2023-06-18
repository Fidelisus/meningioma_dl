import logging
from pathlib import Path
from typing import List, Tuple, Optional
import torch
import torch_directml
import os
import torch
from torch import nn
from torch.nn import Parameter

from meningioma_dl.config import TaskType
from meningioma_dl.models.resnet import ResNet, RESNET_MODELS_MAP


def create_resnet_model(
    model_depth: int,
    resnet_shortcut_type: str,
    no_cuda: bool,
    number_of_classes: int,
    gpus_ids: tuple[int],
    pretrained_model_path: Optional[Path],
    device: torch.device,
) -> tuple[ResNet, list[Parameter], list[Parameter]]:
    assert model_depth in RESNET_MODELS_MAP
    assert resnet_shortcut_type in ["A", "B"]

    model = RESNET_MODELS_MAP[model_depth](
        shortcut_type=resnet_shortcut_type,
        no_cuda=no_cuda,
        num_classes=number_of_classes,
    )

    if no_cuda:
        model = model.to(device)
        # model = nn.DataParallel(model)
        initialized_model_state_dict = model.state_dict()
    else:
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
        pretrained_model_path,
        # TODO change to gpu
        # map_location=torch.device(device),
        map_location="cpu",
    )
    pretrain_dict = {
        k: v
        for k, v in pretrain["state_dict"].items()
        if k in initialized_model_state_dict
    }

    initialized_model_state_dict.update(pretrain_dict)
    logging.info(
        f"Loaded the following layers from the pretrained model: {pretrain_dict.keys()}"
    )
    model.load_state_dict(initialized_model_state_dict)

    pretrained_model_parameters = []
    parameters_to_fine_tune = []

    for pname, p in model.named_parameters():
        if pname in pretrain_dict:
            pretrained_model_parameters.append(p)
        else:
            parameters_to_fine_tune.append(p)

    return model, pretrained_model_parameters, parameters_to_fine_tune
