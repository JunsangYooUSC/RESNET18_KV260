import numpy as np

##
from torchvision.models import resnet18, ResNet18_Weights
import random
import copy
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
import torchvision
from torchvision.transforms import Resize
import torchvision.transforms as transforms
import torchvision.models as models
import torch.quantization
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import pandas as pd
import time

device = torch.device("cpu")
print(f"Using device: {device}")

SCALE = 32

##
import torch
import torch.nn as nn
from torchvision.models import resnet18

class ModifiedResNet18(nn.Module):
    # def __init__(self, input_channels=3, input_size=224):
    def __init__(self, input_channels=1, input_size=224):
        """
        Initialize the modified ResNet-18 model with reduced channel sizes.

        Args:
            input_channels (int): Number of input channels (default: 3).
            input_size (int): Height and width of the input image (default: 224).
        """
        super(ModifiedResNet18, self).__init__()

        # Load the base ResNet-18 model
        self.base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Adjust only the internal channels, keeping input parameters the same
        self.base_model.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=max(1, self.base_model.conv1.out_channels // SCALE),
            kernel_size=self.base_model.conv1.kernel_size,
            stride=self.base_model.conv1.stride,
            padding=self.base_model.conv1.padding,
            bias=False
        )

        self.base_model.bn1 = nn.BatchNorm2d(max(1, self.base_model.bn1.num_features // SCALE))

        # Scale down the layers
        def scale_down_layer(layer):
            for block in layer:
                block.conv1 = nn.Conv2d(
                    in_channels=max(1, block.conv1.in_channels // SCALE),
                    out_channels=max(1, block.conv1.out_channels // SCALE),
                    kernel_size=block.conv1.kernel_size,
                    stride=block.conv1.stride,
                    padding=block.conv1.padding,
                    bias=False
                )
                block.bn1 = nn.BatchNorm2d(max(1, block.bn1.num_features // SCALE))
                block.conv2 = nn.Conv2d(
                    in_channels=max(1, block.conv2.in_channels // SCALE),
                    out_channels=max(1, block.conv2.out_channels // SCALE),
                    kernel_size=block.conv2.kernel_size,
                    stride=block.conv2.stride,
                    padding=block.conv2.padding,
                    bias=False
                )
                block.bn2 = nn.BatchNorm2d(max(1, block.bn2.num_features // SCALE))
                if block.downsample is not None:
                    block.downsample[0] = nn.Conv2d(
                        in_channels=max(1, block.downsample[0].in_channels // SCALE),
                        out_channels=max(1, block.downsample[0].out_channels // SCALE),
                        kernel_size=block.downsample[0].kernel_size,
                        stride=block.downsample[0].stride,
                        padding=block.downsample[0].padding,
                        bias=False
                    )
                    block.downsample[1] = nn.BatchNorm2d(max(1, block.downsample[1].num_features // SCALE))

        scale_down_layer(self.base_model.layer1)
        scale_down_layer(self.base_model.layer2)
        scale_down_layer(self.base_model.layer3)
        scale_down_layer(self.base_model.layer4)

        # Replace the fully connected layer
        self.base_model.fc = nn.Linear(512//SCALE, 10)

    def forward(self, x):
        """Forward pass for the modified ResNet-18 model."""
        return self.base_model(x)

# Define the input channels and size
input_channels = 1  # for simulation
input_size = 224  # Standard input size for ResNet

# Initialize the model
model = ModifiedResNet18(input_channels=input_channels, input_size=input_size)

# Example input tensor
x = torch.randn(1, input_channels, input_size, input_size)
output = model(x)
print("Output shape:", output.shape)

##
def fixed_point_quantize(weights, total_bits, int_bits):
    frac_bits = total_bits - int_bits
    delta = 2 ** (-frac_bits)
    max_val = (2 ** (total_bits - 1) - 1) * delta
    min_val = -2 ** (total_bits - 1) * delta
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights)
    q_weights = torch.clamp(torch.round(weights / delta), min_val / delta, max_val / delta) * delta
    return q_weights

class QuantizedConv2d(nn.Conv2d):
    def __init__(self, *args, total_bits=8, weight_int_bits=2, input_int_bits=2, **kwargs):
        super(QuantizedConv2d, self).__init__(*args, **kwargs)
        self.total_bits = total_bits
        self.weight_int_bits = weight_int_bits
        self.input_int_bits = input_int_bits
    def forward(self, input):
        # quantize input
        quantized_input = fixed_point_quantize(input, self.total_bits, self.input_int_bits)
        # quantize weights
        original_weights = self.weight.data
        quantized_weights = fixed_point_quantize(original_weights, self.total_bits, self.weight_int_bits)
        output = F.conv2d(quantized_input, quantized_weights, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
        return output

## quantize conv
def quantize_conv2d(model, total_bits, weight_int_bits, input_int_bits):
    for name, m in model.named_children():
        if isinstance(m, nn.Conv2d):
            new_layer = QuantizedConv2d(
                in_channels=m.in_channels,
                out_channels=m.out_channels,
                kernel_size=m.kernel_size,
                stride=m.stride,
                padding=m.padding,
                dilation=m.dilation,
                groups=m.groups,
                bias=(m.bias is not None),
                total_bits=total_bits,
                weight_int_bits=weight_int_bits,
                input_int_bits=input_int_bits
            )
            new_layer.weight.data = fixed_point_quantize(m.weight.data.clone(), total_bits, weight_int_bits)
            if m.bias is not None:
                new_layer.bias.data = fixed_point_quantize(m.bias.data.clone(), total_bits, weight_int_bits)

            setattr(model, name, new_layer)
        elif len(list(m.children())) > 0:
            quantize_conv2d(m, total_bits, weight_int_bits, input_int_bits)

##
import copy
quantized_model = copy.deepcopy(model.base_model)

# case 1:
total_bits = 16
weight_int_bits = 8
input_int_bits = 8
dtype_weight = np.int16
dtype_bn_weight = np.float32
dtype_input = np.int16
dtype_output = np.int16
# case 2:
# total_bits = 8
# weight_int_bits = 2
# input_int_bits = 3
# dtype_weight = np.int8
# dtype_bn_weight = np.float32
# dtype_input = np.int8
# dtype_output = np.int8


quantize_conv2d(quantized_model, total_bits, weight_int_bits, input_int_bits)
quantized_model = quantized_model.to(device)
quantized_model.eval()

##
# function to save the values in given format
def save_to_bin(data, filename, dtype, total_bits, int_bits):
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    if dtype == np.int8:
        data = data * (2 ** (total_bits - int_bits))
    if dtype == np.int16:
        data = data * (2 ** (total_bits - int_bits))
    data = dtype(data)
    data.tofile(filename)

def load_from_bin(filename, dtype, total_bits, int_bits):
    data = np.fromfile(filename, dtype)
    if dtype == np.int8:
        data = data / (2 ** (total_bits - int_bits))
    if dtype == np.int16:
        data = data / (2 ** (total_bits - int_bits))
    return data

##
conv_weights_raw = np.array([])
bn_params_raw = np.array([])
eps = 1e-5

for name, module in quantized_model.named_modules():
    if isinstance(module, nn.Conv2d):
        data = module.weight.detach().numpy().flatten()
        conv_weights_raw = np.append(conv_weights_raw, data)
        print(name, data.flatten().shape[0])
    if isinstance(module, nn.BatchNorm2d):
        mult_factor = module.weight / torch.sqrt(module.running_var + eps)
        params = torch.stack([
            module.running_mean,
            mult_factor,
            module.bias
        ])
        data = params.detach().numpy().flatten()
        bn_params_raw = np.append(bn_params_raw, data)
        print(name, data.flatten().shape[0])
    if isinstance(module, nn.Linear):
        data = module.weight.detach().numpy().flatten();
        data = np.append(data, module.bias.detach().numpy().flatten())
        bn_params_raw = np.append(bn_params_raw, data)
        print(name, data.flatten().shape[0])


conv_weights = fixed_point_quantize(conv_weights_raw, total_bits, weight_int_bits)
bn_params = dtype_bn_weight(bn_params_raw)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# input gen
x = torch.randn(1, input_channels, input_size, input_size)
x = fixed_point_quantize(x, total_bits, input_int_bits)
y = quantized_model(x)
y = fixed_point_quantize(y, total_bits, input_int_bits)

print()
print(f"dtype weight: {dtype_weight}")
print(f"dtype bn weight: {dtype_bn_weight}")
print(f"dtype input: {dtype_input}")
print(f"dtype output: {dtype_output}")
print(f"Total parameters: {count_parameters(quantized_model)}")
print(f"Total size of conv weights: {conv_weights.size}")
print(f"Total size of bn params: {bn_params.size}")
print(f"Total size of input: {x.size()}")
print(f"Total size of output: {y.size()}")
print()

# save params, input, output
save_to_bin(conv_weights, "data/conv_all_params.bin", dtype_weight, total_bits, weight_int_bits)
save_to_bin(bn_params, "data/bn_all_params.bin", dtype_bn_weight, total_bits, weight_int_bits)
save_to_bin(x, "data/input.bin", dtype_input, total_bits, input_int_bits)
save_to_bin(y, "data/output.bin", dtype_output, total_bits, input_int_bits)

##
conv_weights_get = load_from_bin("data/conv_all_params.bin", dtype_weight, total_bits, weight_int_bits)
bn_params_get = load_from_bin("data/bn_all_params.bin", dtype_bn_weight, total_bits, weight_int_bits)
input_get = load_from_bin("data/input.bin", dtype_input, total_bits, input_int_bits)
output_get = load_from_bin("data/output.bin", dtype_output, total_bits, input_int_bits)

print(f"conv_weights_get: {conv_weights_get.size}")
print(f"bn_params_get: {bn_params_get.size}")
print(f"input_get: {input_get.size}")
print(f"output_get: {output_get.size}")

if (conv_weights.flatten() == conv_weights_get).all():
    print("Conv weights match")
else:
    print("Conv weights do not match")
if (bn_params.flatten() == bn_params_get).all():
    print("BN params match")
else:
    print("BN params do not match")
if (x.flatten() == input_get).all():
    print("Input match")
else:
    print("Input do not match")
if (y.flatten() == output_get).all():
    print("Output match")
else:
    print("Output do not match")

## params match
# cnt = 0
# for name, module in quantized_model.named_modules():
#     if isinstance(module, nn.Conv2d):
#         print(name, sum(p.numel() for p in module.parameters()))
#         cnt += 1
#     if isinstance(module, nn.Linear):
#         print(name, sum(p.numel() for p in module.parameters()))
#         cnt += 1
#     if isinstance(module, nn.BatchNorm2d):
#         print(name, sum(p.numel() for p in module.parameters()))
#
# print("layers with params:", cnt)


## intermediate vals
with torch.no_grad():
    after_conv1 = quantized_model.conv1(x)
    after_bn1 = quantized_model.bn1(after_conv1)
    after_relu = quantized_model.relu(after_bn1)
    after_maxpool = quantized_model.maxpool(after_relu)
    after_layer1_0 = quantized_model.layer1[0](after_maxpool)
    after_layer1_1 = quantized_model.layer1[1](after_layer1_0)
    after_layer1 = quantized_model.layer1(after_maxpool)
    after_layer2 = quantized_model.layer2(after_layer1)
    after_layer3 = quantized_model.layer3(after_layer2)
    after_layer4 = quantized_model.layer4(after_layer3)
    after_avgpool = quantized_model.avgpool(after_layer4)
    after_fc = quantized_model.fc(after_avgpool.flatten())
    if (y == after_fc).all():
        print("Intermediate values match")

    # save intermediate results
    save_to_bin(after_relu, "data/after_relu.bin", dtype_output, total_bits, input_int_bits)
    save_to_bin(after_maxpool, "data/after_maxpool.bin", dtype_output, total_bits, input_int_bits)
    save_to_bin(after_layer1_0, "data/after_layer1_0.bin", dtype_output, total_bits, input_int_bits)
    save_to_bin(after_layer1_1, "data/after_layer1_1.bin", dtype_output, total_bits, input_int_bits)
    save_to_bin(after_layer4, "data/after_layer4.bin", dtype_output, total_bits, input_int_bits)
    save_to_bin(after_avgpool, "data/after_avgpool.bin", dtype_output, total_bits, input_int_bits)
    save_to_bin(after_fc, "data/after_fc.bin", dtype_output, total_bits, input_int_bits)

##
for idx in range(50):
    print(idx, np.round(after_conv1.flatten()[idx].item(), 5))
    # print(idx, np.round(after_bn1.flatten()[idx].item(), 5))


