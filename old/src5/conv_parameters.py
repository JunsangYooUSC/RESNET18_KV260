## Import Library
import torch
import torchvision
import torchvision.models as models
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
# use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

## Torch Pretrained Model and Copied model
torch_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
torch_model.fc = nn.Linear(torch_model.fc.in_features, 100)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 100)

## quantized conv layer
def fixed_point_quantize_weights(weights, total_bits, int_bits):
    frac_bits = total_bits - int_bits
    delta = 2 ** (-frac_bits)
    max_val = (2 ** (total_bits - 1) - 1) * delta
    min_val = -2 ** (total_bits - 1) * delta
    q_weights = torch.clamp(torch.round(weights / delta), min_val / delta, max_val / delta) * delta
    return q_weights

class QuantizedConv2d(nn.Conv2d):
    def __init__(self, *args, input_bits=8, input_int=2, weight_bits=8, weight_int=2, **kwargs):
        super(QuantizedConv2d, self).__init__(*args, **kwargs)
        self.input_bits = input_bits
        self.input_int = input_int
        self.weight_bits = weight_bits
        self.weight_int = weight_int
    def forward(self, input):
        # quantize input
        quantized_input = fixed_point_quantize_weights(input, self.input_bits, self.input_int)
        # quantize weights
        original_weights = self.weight.data
        quantized_weights = fixed_point_quantize_weights(original_weights, self.weight_bits, self.weight_int)
        self.weight.data = quantized_weights
        output = super().forward(quantized_input)
        return output

# ## Test quantized output
# X = (torch.rand((1, 64,28,28))-1).to(device)
# total_bits = 8
# int_bits = 3
# Xq = fixed_point_quantize_weights(X, total_bits, int_bits)
# conv = conv1
# Y = conv(X)
# convq = conv1
# convq.weight.data = fixed_point_quantize_weights(convq.weight.data, total_bits, int_bits)
# Yq = convq(Xq)
# Yq = fixed_point_quantize_weights(Yq, total_bits, int_bits)
#
# diff_X = X-Xq
# diff_Y = Y-Yq
# # print(f"diff_X.std: {diff_X.std()}")
# # print(f"diff_X.rms_mean: {torch.mean(torch.sqrt(diff_X**2))}")
# # print(f"diff_X.max: {diff_X.max()}")
# print(f"diff_Y.std: {diff_Y.std()}")
# print(f"diff_Y.rms_mean: {torch.mean(torch.sqrt(diff_Y**2))}")
# print(f"diff_Y.max: {diff_Y.max()}")

## quantize conv
input_bits=8
input_int=3
weight_bits=8
weight_int=1
def quantize_conv2d(model, input_bits, input_int, weight_bits, weight_int):
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
                input_bits = input_bits,
                input_int = input_int,
                weight_bits = weight_bits,
                weight_int = weight_int
            )
            new_layer.weight.data = fixed_point_quantize_weights(m.weight.data.clone(), weight_bits, weight_int)
            if m.bias is not None:
                new_layer.bias.data = fixed_point_quantize_weights(m.bias.data.clone(), weight_bits, weight_int)

            setattr(model, name, new_layer)
        elif len(list(m.children())) > 0:
            quantize_conv2d(m, input_bits, input_int, weight_bits, weight_int)

quantize_conv2d(model, input_bits, input_int, weight_bits, weight_int)

X = torch.randn((1,256,14,14))
Y = torch_model.layer4[0](X)
Yq = model.layer4[0](X)
diff = (Y-Yq)
print(f"diff max: {diff.max()}")
print(f"diff min: {diff.min()}")
print(f"diff mean: {diff.mean()}")
print(f"diff std: {diff.std()}")

##
import torch
import numpy as np

def save_conv_parameters(layer, weight_bits, weight_int, prefix):
    mult = 2**(weight_bits-weight_int)
    if hasattr(layer, 'weight') and layer.weight is not None:
        data = np.int8(layer.weight.data*mult)
        data.tofile(f"{prefix}_weights.bin")


def save_bn_parameters(layer, prefix):
    if hasattr(layer, 'weight') and layer.weight is not None:
        data = np.float32(layer.weight.data)
        data.tofile(f"{prefix}_gamma.bin")
    if hasattr(layer, 'bias') and layer.bias is not None:
        data = np.float32(layer.bias.data)
        data.tofile(f"{prefix}_beta.bin")
    if hasattr(layer, 'running_mean'):
        data = np.float32(layer.running_mean)
        data.tofile(f"{prefix}_running_mean.bin")
    if hasattr(layer, 'running_var'):
        data = 1/np.sqrt(np.float32(layer.running_var)+layer.eps)
        data.tofile(f"{prefix}_running_var.bin")


# Save model.layer4[0] parameters
layer4_0 = model.layer4[0]

# conv1 weights
save_conv_parameters(layer4_0.conv1, weight_bits, weight_int, "data/layer4_0_conv1")

# bn1 parameters
save_bn_parameters(layer4_0.bn1, "data/layer4_0_bn1")

# conv2 weights
save_conv_parameters(layer4_0.conv2, weight_bits, weight_int, "data/layer4_0_conv2")

# bn2 parameters
save_bn_parameters(layer4_0.bn2, "data/layer4_0_bn2")

# conv3 weights
save_conv_parameters(layer4_0.downsample[0], weight_bits, weight_int, "data/layer4_0_conv3")

# bn3 parameters
save_bn_parameters(layer4_0.downsample[1], "data/layer4_0_bn3")









