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

# use GPU if available
if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available and being used.")
else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU instead.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")
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
        self.base_model = resnet18(pretrained=False)

        # Adjust only the internal channels, keeping input parameters the same
        self.base_model.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=max(1, self.base_model.conv1.out_channels // 32),
            kernel_size=self.base_model.conv1.kernel_size,
            stride=self.base_model.conv1.stride,
            padding=self.base_model.conv1.padding,
            bias=False
        )

        self.base_model.bn1 = nn.BatchNorm2d(max(1, self.base_model.bn1.num_features // 32))

        # Scale down the layers
        def scale_down_layer(layer):
            for block in layer:
                block.conv1 = nn.Conv2d(
                    in_channels=max(1, block.conv1.in_channels // 32),
                    out_channels=max(1, block.conv1.out_channels // 32),
                    kernel_size=block.conv1.kernel_size,
                    stride=block.conv1.stride,
                    padding=block.conv1.padding,
                    bias=False
                )
                block.bn1 = nn.BatchNorm2d(max(1, block.bn1.num_features // 32))
                block.conv2 = nn.Conv2d(
                    in_channels=max(1, block.conv2.in_channels // 32),
                    out_channels=max(1, block.conv2.out_channels // 32),
                    kernel_size=block.conv2.kernel_size,
                    stride=block.conv2.stride,
                    padding=block.conv2.padding,
                    bias=False
                )
                block.bn2 = nn.BatchNorm2d(max(1, block.bn2.num_features // 32))
                if block.downsample is not None:
                    block.downsample[0] = nn.Conv2d(
                        in_channels=max(1, block.downsample[0].in_channels // 32),
                        out_channels=max(1, block.downsample[0].out_channels // 32),
                        kernel_size=block.downsample[0].kernel_size,
                        stride=block.downsample[0].stride,
                        padding=block.downsample[0].padding,
                        bias=False
                    )
                    block.downsample[1] = nn.BatchNorm2d(max(1, block.downsample[1].num_features // 32))

        scale_down_layer(self.base_model.layer1)
        scale_down_layer(self.base_model.layer2)
        scale_down_layer(self.base_model.layer3)
        scale_down_layer(self.base_model.layer4)

        # Replace the fully connected layer
        self.base_model.fc = nn.Linear(16, 10)

    def forward(self, x):
        """Forward pass for the modified ResNet-18 model."""
        return self.base_model(x)

# Example usage:
# Define the input channels and size
input_channels = 3  # Standard input channels for images
input_channels = 1  # Standard input channels for images
input_size = 224  # Standard input size for ResNet

# Initialize the model
model = ModifiedResNet18(input_channels=input_channels, input_size=input_size)

# Check the model structure
print(model)

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

weight_int_bits = 2
input_int_bits = 3

quantize_conv2d(quantized_model, 8, weight_int_bits, input_int_bits)
quantized_model = quantized_model.to(device)
##
def save_bn_params_to_bin(bn_layer, filename, eps=1e-5):
    """
    Combines running_var and gamma into mult_factor and saves BN params.

    Args:
        bn_layer: The Batch Normalization layer.
        filename: The path to save the binary file.
        eps: A small value added to running_var for numerical stability.
    """
    # Calculate mult_factor
    mult_factor = bn_layer.weight / torch.sqrt(bn_layer.running_var + eps)

    # Stack parameters in the desired order
    params = torch.stack([
        bn_layer.running_mean,
        mult_factor,
        bn_layer.bias
    ])

    # Save to binary file
    np_array = params.cpu().detach().numpy()
    np_array.tofile(filename)

def save_all_bn_params(model, save_dir):
    """
    Iterates through the model and saves parameters for all BN layers.

    Args:
        model: The PyTorch model.
        save_dir: The directory to save the binary files.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):  # Check if the module is a BN layer
            filename = f"{save_dir}{name.replace('.', '_')}_combined.bin"
            save_bn_params_to_bin(module, filename)

def save_conv_params_to_bin(conv_layer, filename):
    np_array = conv_layer.weight.cpu().detach().numpy()
    np_array.tofile(filename)

def save_all_conv_params(model, save_dir):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):  # Check if the module is a BN layer
            filename = f"{save_dir}{name.replace('.', '_')}_combined.bin"
            save_conv_params_to_bin(module, filename)

def save_fc_params_to_bin(fc_layer, filename):
    np_array = fc_layer.weight.cpu().detach().numpy()
    np_array.tofile(filename)

def save_all_fc_params(model, save_dir):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):  # Check if the module is a BN layer
            filename = f"{save_dir}{name.replace('.', '_')}_combined.bin"
            save_fc_params_to_bin(module, filename)

save_dir = 'small_model/'
save_all_bn_params(model, save_dir)
save_all_conv_params(model, save_dir)
save_all_fc_params(model, save_dir)


##
conv_weights_raw = np.array([])
bn_params_raw = np.array([])
eps = 1e-5

for name, module in quantized_model.named_modules():
    if isinstance(module, nn.Conv2d):
        conv_weights_raw = np.append(conv_weights_raw, module.weight.detach().numpy().flatten())
    if isinstance(module, nn.BatchNorm2d):
        mult_factor = module.weight / torch.sqrt(module.running_var + eps)
        params = torch.stack([
            module.running_mean,
            mult_factor,
            module.bias
        ])
        bn_params_raw = np.append(bn_params_raw, params.detach().numpy().flatten())
    if isinstance(module, nn.Linear):
        bn_params_raw = np.append(bn_params_raw, module.weight.detach().numpy().flatten())
        bn_params_raw = np.append(bn_params_raw, module.bias.detach().numpy().flatten())

conv_weights = np.int8(conv_weights_raw*(2**(8-weight_int_bits)))
bn_params = np.float32(bn_params_raw)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

print(f"Total parameters: {count_parameters(quantized_model)}")

conv_weights.tofile("conv_all_params.bin")
bn_params.tofile("bn_all_params.bin")
input = np.int8(x*(2**(8-input_int_bits))).flatten()
input.tofile("input.bin")

##
print(conv_weights.shape)
print(bn_params.shape)
print(conv_weights.shape[0]+bn_params_raw.shape[0])

##
conv_weights_get = np.fromfile("conv_all_params.bin",np.int8)/(2**(8-weight_int_bits))
bn_params_get = np.fromfile("bn_all_params.bin", np.float32)
input_get = np.fromfile("input.bin", np.int8)/(2**(8-input_int_bits))

print(f"conv_weights: {conv_weights.shape}")
print(f"conv_weights_get: {conv_weights_get.shape}")
print(f"bn_params: {bn_params.shape}")
print(f"bn_params_get: {bn_params_get.shape}")
print(f"input: {input.shape}")
print(f"input_get: {input_get.shape}")



##
import torch
import torch.nn as nn
from torchvision.models import resnet18

def count_layers_excluding_bn_relu(model):
    """
    Counts the total number of layers in a PyTorch model, excluding BatchNorm and ReLU layers.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        int: Total number of layers excluding BatchNorm and ReLU layers.
    """
    exclude_types = (nn.BatchNorm2d, nn.BatchNorm1d, nn.ReLU, nn.ReLU6)
    total_layers = sum(1 for layer in model.modules() if not isinstance(layer, exclude_types))
    return total_layers

# Example usage with ResNet18
total_layers = count_layers_excluding_bn_relu(quantized_model)
print(f"Total number of layers in the model (excluding BatchNorm and ReLU): {total_layers}")

##
quantized_model.eval()
a = quantized_model.conv1(x)
b = quantized_model.bn1(a)
c = quantized_model.relu(b)
d = quantized_model.maxpool(c)
e = quantized_model.layer1[0].conv1(d)
f = quantized_model.layer1[0].bn1(e)
g = quantized_model.layer1[0].relu(f)
for idx in range(20):
    for jdx in range(5):
        print(idx*5+jdx, np.round(d.flatten()[idx*5+jdx].item(),5))




