from typing import Type
import torch.nn as nn
from .algos import Approach
import importlib


class Replacer():
    approach : Type[Approach] = None
    
    def __init__(self, approach: Type[Approach] = Approach.BNN) -> None:
        self.approach = approach
        
        module_path = "quantization.layers." + approach.value
        module = importlib.import_module(module_path)
        
        self.q_conv2d = module.QConv2d
        self.q_conv1d = module.QConv1d
        self.q_linear = module.QLinear
        
    def layer_replacer(
        self, 
        model, 
        if_conv2d: bool = True, 
        if_conv1d: bool = True, 
        if_linear: bool = True
    ):
        for name, layer in model.named_children():
            if isinstance(layer, nn.Conv2d) and if_conv2d:
                q_layer = self.q_conv2d(
                    layer.in_channels, 
                    layer.out_channels, 
                    kernel_size=layer.kernel_size[0],
                    stride=layer.stride[0], 
                    padding=layer.padding[0],
                    bias=layer.bias is not None
                )
                setattr(model, name, q_layer)
            elif isinstance(layer, nn.Conv1d) and if_conv1d:
                q_layer = self.q_conv1d(
                    layer.in_channels, 
                    layer.out_channels, 
                    kernel_size=layer.kernel_size[0],
                    stride=layer.stride[0],
                    padding=layer.padding[0], 
                    bias=layer.bias is not None
                )
                setattr(model, name, q_layer)
            elif isinstance(layer, nn.Linear) and if_linear:
                q_layer = self.q_linear(
                    layer.in_features, 
                    layer.out_features, 
                    bias=layer.bias is not None
                )
                setattr(model, name, q_layer)
            elif isinstance(layer, nn.Module):
               self.layer_replacer(layer)
        return model

