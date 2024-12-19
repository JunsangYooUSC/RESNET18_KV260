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
for hw4, we can choose 20 to 22
this is done by
	unsigned start_layer = 20;
	unsigned end_layer = 22;

