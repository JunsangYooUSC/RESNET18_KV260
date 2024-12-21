src:
	version 0
	- uses only off-chip memory
	- no optimizations
	- no parallel
	- no packing
	- use float

src1: **most recent and verified version**
	version 1
	- uses only off-chip memory
	- minimum pragmas applied
	- no parallel
	- no packing
	- use float

src2:
	version 2
	- uses on-chip memory for activations
	- pragmas applied
	- no parallel
	- packing
	- use 8bit fixed

to run the C simulation:
	0. note that small sizes were used to test the model due to large size of arrays having problem with simulation.
	1. run the python in each src file (make sure to move the directory to that file)
	2. run simulation using the dataset (change the base_fname in tb host code)
	3. compare with the python file outputs. rmse is calculated after the testbench is complete
	4. RMSE of 0.0247 for random input, random weight configuration. tested in C simulation on src1.

conv_kernel:
	- start_layer and end_layer variables are used to control the input and output stage.
		for example, start_layer, end_layer of 0, 0 will run the first convolution layer (conv, bn, relu because layer fusion)

host_tb_functions.cpp is used to test each functions
host.cpp is used for whole inference


resnet18_src1 is used for hardware inference
	- start_layer of 0 and end_layer of 0 is tested first
	- plenty of room for act_mem memory was given
*** currently output not generating ***
