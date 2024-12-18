## Import Library
import torch
import torchvision
import torchvision.models as models
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import numpy as np
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



## small size
small_data = model.layer4[0].conv1.weight.data[:32,:16,:,:]
small_layer = torch.nn.Conv2d(16,32,3,1,1)
small_layer.weight.data = small_data
x = torch.randn((1,16,14,14))
y = small_layer(x)

## gen random small layer
def quantize_np(weights, total_bits, int_bits):
    frac_bits = total_bits - int_bits
    delta = 2 ** (-frac_bits)
    max_val = (2 ** (total_bits - 1) - 1) * delta
    min_val = -2 ** (total_bits - 1) * delta
    q_weights = np.clip(np.round(weights / delta), min_val / delta, max_val / delta) * delta
    return q_weights
input = np.random.rand(16,14,14)-0.5
weight1 = np.random.rand(32,16,3,3)-0.5
weight2 = np.random.rand(32,32,3,3)-0.5
weight3 = np.random.rand(32,16,1,1)-0.5
bn_weight1 = np.random.rand(4,32)-0.5
bn_weight2 = np.random.rand(4,32)-0.5
bn_weight3 = np.random.rand(4,32)-0.5
bn_weight1[1] = np.abs(bn_weight1[1])
bn_weight2[1] = np.abs(bn_weight2[1])
bn_weight3[1] = np.abs(bn_weight3[1])

bn_hw_weight1 = np.zeros((3,32))
bn_hw_weight2 = np.zeros((3,32))
bn_hw_weight3 = np.zeros((3,32))
bn_hw_weight1[0] = bn_weight1[0]
bn_hw_weight2[0] = bn_weight2[0]
bn_hw_weight3[0] = bn_weight3[0]
bn_hw_weight1[2] = bn_weight1[3]
bn_hw_weight2[2] = bn_weight2[3]
bn_hw_weight3[2] = bn_weight3[3]
bn_hw_weight1[1] = bn_weight1[2]/np.sqrt(bn_weight1[1]+1e-5)
bn_hw_weight2[1] = bn_weight2[2]/np.sqrt(bn_weight2[1]+1e-5)
bn_hw_weight3[1] = bn_weight3[2]/np.sqrt(bn_weight3[1]+1e-5)

sim_mode = 1
if (sim_mode):
    input_bits = 16
    input_int = 8
    weight_bits = 16
    weight_int = 8
    dtype = np.int16
else:
    input_bits = 8
    input_int = 3
    weight_bits = 8
    weight_int = 2
    dtype = np.int8

input = quantize_np(input, input_bits, input_int)
weight1 = quantize_np(weight1, weight_bits, weight_int)
weight2 = quantize_np(weight2, weight_bits, weight_int)
weight3 = quantize_np(weight3, weight_bits, weight_int)

dtype(input.flatten()*2**(input_bits-input_int)).tofile("C:/Users/junsa/OneDrive - University of Southern California/2024_Fall/EE511/Assignments/hw4/RESNET18_KV260/src/data/input.bin")
dtype(weight1.flatten()*2**(weight_bits-weight_int)).tofile("C:/Users/junsa/OneDrive - University of Southern California/2024_Fall/EE511/Assignments/hw4/RESNET18_KV260/src/data/weight1.bin")
dtype(weight2.flatten()*2**(weight_bits-weight_int)).tofile("C:/Users/junsa/OneDrive - University of Southern California/2024_Fall/EE511/Assignments/hw4/RESNET18_KV260/src/data/weight2.bin")
dtype(weight3.flatten()*2**(weight_bits-weight_int)).tofile("C:/Users/junsa/OneDrive - University of Southern California/2024_Fall/EE511/Assignments/hw4/RESNET18_KV260/src/data/weight3.bin")
np.float32(bn_weight1.flatten()).tofile("C:/Users/junsa/OneDrive - University of Southern California/2024_Fall/EE511/Assignments/hw4/RESNET18_KV260/src/data/bn_weight1.bin")
np.float32(bn_weight2.flatten()).tofile("C:/Users/junsa/OneDrive - University of Southern California/2024_Fall/EE511/Assignments/hw4/RESNET18_KV260/src/data/bn_weight2.bin")
np.float32(bn_weight3.flatten()).tofile("C:/Users/junsa/OneDrive - University of Southern California/2024_Fall/EE511/Assignments/hw4/RESNET18_KV260/src/data/bn_weight3.bin")
np.float32(bn_hw_weight1.flatten()).tofile("C:/Users/junsa/OneDrive - University of Southern California/2024_Fall/EE511/Assignments/hw4/RESNET18_KV260/src/data/bn_hw_weight1.bin")
np.float32(bn_hw_weight2.flatten()).tofile("C:/Users/junsa/OneDrive - University of Southern California/2024_Fall/EE511/Assignments/hw4/RESNET18_KV260/src/data/bn_hw_weight2.bin")
np.float32(bn_hw_weight3.flatten()).tofile("C:/Users/junsa/OneDrive - University of Southern California/2024_Fall/EE511/Assignments/hw4/RESNET18_KV260/src/data/bn_hw_weight3.bin")

##
print(f"first 3 input val: {input.flatten()[0]}, {input.flatten()[1]}, {input.flatten()[2]}")
print(f"first 3 weight1 val: {weight1.flatten()[0]}, {weight1.flatten()[1]}, {weight1.flatten()[2]}")
print(f"first 3 bn_weight1 val: {bn_weight1.flatten()[0]}, {bn_weight1.flatten()[1]}, {bn_weight1.flatten()[2]}")
conv1 = torch.nn.Conv2d(32,16,3,2,1, bias=False)
conv1.weight.data = torch.tensor(weight1)
conv2 = torch.nn.Conv2d(32,32,3,1,1, bias=False)
conv2.weight.data = torch.tensor(weight2)
conv3 = torch.nn.Conv2d(32,16,1,2,0, bias=False)
conv3.weight.data = torch.tensor(weight3)
bn1 = torch.nn.BatchNorm2d(32, 1e-5)
bn1.running_mean.data = torch.tensor(bn_weight1[0])
bn1.running_var.data = torch.tensor(bn_weight1[1])
bn1.weight.data = torch.tensor(bn_weight1[2])
bn1.bias.data = torch.tensor(bn_weight1[3])
bn2 = torch.nn.BatchNorm2d(32, 1e-5)
bn2.running_mean.data = torch.tensor(bn_weight2[0])
bn2.running_var.data = torch.tensor(bn_weight2[1])
bn2.weight.data = torch.tensor(bn_weight2[2])
bn2.bias.data = torch.tensor(bn_weight2[3])
bn3 = torch.nn.BatchNorm2d(32, 1e-5)
bn3.running_mean.data = torch.tensor(bn_weight3[0])
bn3.running_var.data = torch.tensor(bn_weight3[1])
bn3.weight.data = torch.tensor(bn_weight3[2])
bn3.bias.data = torch.tensor(bn_weight3[3])
relu1 = torch.nn.ReLU()
relu2 = torch.nn.ReLU()
bn1.eval()
bn2.eval()
bn3.eval()


with torch.no_grad():
    input = torch.tensor(input.reshape((1,16,14,14)))
    out1 = conv1(input)
    out2 = bn1(out1)
    out3 = relu1(out2)
    out4 = conv2(out3)
    out5 = bn2(out4)
    add1 = conv3(input)
    add2 = bn3(add1)
    out6 = out5 + add2
    out7 = relu2(out6)

    out11 = conv1(input)
    out22 = out11.clone()
    epsilon = 1e-5
    for idx in range(32):
        mean = torch.tensor(bn_weight1[0][idx])
        var = torch.tensor(bn_weight1[1][idx])
        gamma = torch.tensor(bn_weight1[2][idx])
        beta = torch.tensor(bn_weight1[3][idx])
        out22[:, idx, :, :] = ((out11[:, idx, :, :] - mean) / torch.sqrt(var + epsilon)) * gamma + beta
    out33 = relu1(out22)
    out44 = conv2(out33)
    out55 = bn2(out44)
    add11 = conv3(input)
    add22 = bn3(add11)
    out66 = out55 + add22
    out77 = relu2(out66)

    out111 = conv1(input)
    out222 = out111.clone()
    for idx in range(32):
        out222[:,idx,:,:] = (out111[:,idx,:,:]-torch.tensor(bn_hw_weight1[0][idx]))*torch.tensor(bn_hw_weight1[1][idx])+torch.tensor(bn_hw_weight1[2][idx])
    out333 = relu1(out222)
    out444 = conv2(out333)
    out555 = bn2(out444)
    add111 = conv3(input)
    add222 = bn3(add111)
    out666 = out555 + add222
    out777 = relu2(out666)


    print((out1-out11).max())
    print((out2-out22).max())
    print((out3-out33).max())
    print((out4-out44).max())
    print((out5-out55).max())
    print((out5-out55).max())
    print((out6-out66).max())
    print((out7-out77).max())

    print((out1-out111).max())
    print((out2-out222).max())
    print((out3-out333).max())
    print((out4-out444).max())
    print((out5-out555).max())
    print((out6-out666).max())
    print((out7-out777).max())

    print((out11-out111).max())
    print((out22-out222).max())
    print((out33-out333).max())
    print((out44-out444).max())
    print((out55-out555).max())
    print((out66-out666).max())
    print((out77-out777).max())


    print(out3.flatten()[:10])
    print(out5.flatten()[:10])
    print(out7.flatten()[:10])

##
np_in1 = np.fromfile("input.bin", dtype=np.int8)
np_in2 = np.array([], dtype=np.int8)
np_in2 = np.append(np_in2,(np.fromfile("weight1.bin", dtype=np.int8)))
np_in2 = np.append(np_in2,(np.fromfile("weight2.bin", dtype=np.int8)))
np_in2 = np.append(np_in2,(np.fromfile("weight3.bin", dtype=np.int8)))
np_in3 = np.array([], dtype=np.float32)
np_in3 = np.append(np_in3,(np.fromfile("bn_hw_weight1.bin", dtype=np.float32)))
np_in3 = np.append(np_in3,(np.fromfile("bn_hw_weight2.bin", dtype=np.float32)))
np_in3 = np.append(np_in3,(np.fromfile("bn_hw_weight3.bin", dtype=np.float32)))

print("input size:", input.flatten().shape[0])
print("load input size:", np_in1.flatten().shape[0])
print("weight size:", weight1.flatten().shape[0]+weight2.flatten().shape[0]+weight3.flatten().shape[0])
print("weight input size:", np_in2.flatten().shape[0])
print("bn_hw_weight size:", bn_hw_weight1.flatten().shape[0]+bn_hw_weight2.flatten().shape[0]+bn_hw_weight3.flatten().shape[0])
print("bn_hw_weight input size:", np_in3.flatten().shape[0])



