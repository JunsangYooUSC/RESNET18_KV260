# RESNET18_KV260

src3: BUF2PE implementation
src: main src code

src host2.cpp uses small parameters to test in vitis testbench
full size won't be able to simulate
in vitis testbench, with the current small layer configuration, 
The conv_parameters_test.py generates the test weights
- conv_all_params.bin
- bn_all_params.bin
- input.bin

The resnet module can select the start and the end layer
in current sample layers 0 through 5 is passed

after the simulation is done, output file is generated as kernel_out.bin


I think I messed up very bad and now I basically don't have much results
kernel.cpp is the main
kernel_syn.cpp is the one tested for synthesis
kernel_with_buf2pe.cpp was the original plan, but did not make it
