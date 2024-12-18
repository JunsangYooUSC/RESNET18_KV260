/******************************************************************************
 * Filename: host.cpp
 * Author: Junsang Yoo
 *
 * Description:
 * host testbench code for convolutional layer
 *
 * Features:
 * - main function that includes
 *     - print and check current convolutional layer configuration
 *     - golden convolution function
 ******************************************************************************/

#include <iostream>
#include <unistd.h>
#include <limits.h>
#include <string>
// Include Vitis HLS headers
#include "ap_fixed.h"
// Include C++ headers
// Include project headers
#include "conv_config.h"
#include "host_utils.h"
#include "kernel.h"

// Print the configuration information
#define CHECK_CONFIG		1

#define INPUT_SIZE 			(BB6_SKIP_C*BB7_CONV1_H*BB7_CONV1_W*BB7_CONV1_S*BB7_CONV1_S)
#define OUTPUT_SIZE			(BB7_CONV1_C*BB7_CONV1_H*BB7_CONV1_W)
#define FILTER_SIZE			(BB6_SKIP_C*BB7_CONV1_C*BB7_CONV1_H*BB7_CONV1_W)


// Function: golden convolution
template<typename D_ACT, typename D_FILTER, typename D_MULT, typename D_MAC>
void convolution_golden(D_ACT *in_act, D_FILTER *in_fil, D_ACT *out_act,
	unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox,
	unsigned int stride,
	unsigned int pad
) {
	// create output MAC and initialize to 0
	D_MAC out_act_mac[nof*noy*nox];
	for (int idx = 0; idx < nof*noy*nox; idx++) {
		out_act_mac[idx] = 0;
	}
	unsigned int nix = nox*stride;
	unsigned int niy = noy*stride;
	for (int kdx = 0; kdx < nif; kdx++) {
		for (int ndx = 0; ndx < niy; ndx += stride) {
			for (int mdx = 0; mdx < nix; mdx += stride) {
				for (int cdx = 0; cdx < nof; cdx++) {
					for (int hdx = 0; hdx < nky; hdx++) {
						for (int wdx = 0; wdx < nkx; wdx++) {

							D_ACT in_act_element;
							// when accessing zero padded index
							if ( (ndx + hdx < pad) || (ndx + hdx >= niy + pad) || (mdx + wdx < pad) || (mdx + wdx >= nix + pad) ) {
								in_act_element = 0;
							}
							else {
								// input address calc
								unsigned int in_addr = kdx * niy * nix;
								in_addr += (ndx + hdx - pad) * nix + (mdx + wdx - pad);
								// load input
								in_act_element = in_act[in_addr];
								if (in_addr >= nif*niy*nix) {
									std::cout << "input index out-of-bounds: " << in_addr << std::endl;
								}
							}

							// filter address calc
							unsigned int filter_addr = cdx * nif * nky * nkx;
							filter_addr += kdx * nky * nkx;
							filter_addr += hdx * nkx;
							filter_addr += wdx;
							// load filter
							D_FILTER in_fil_element = in_fil[filter_addr];
							if (filter_addr >= nof*nif*nky*nkx) {
								std::cout << "filter index out-of-bounds: " << filter_addr << std::endl;
							}

							// mult
							D_MULT mult_element = in_act_element * in_fil_element;

							// output address calc
							unsigned int out_addr = cdx * noy * nox;
							out_addr += ndx / stride * nox + mdx / stride;
							if (out_addr >= nof*noy*nox) {
								std::cout << "output index out-of-bounds: " << out_addr << std::endl;
							}
							// load output mac
							out_act_mac[out_addr] += mult_element;
						}
					}
				}
			}
		}
	}

	for (int idx = 0; idx < nof*noy*nox; idx++) {
		out_act[idx] = out_act_mac[idx];
	}
}

template<typename D_ACT, typename D_FILTER, typename D_MULT, typename D_MAC>
void convolution_bn_golden(D_ACT *in_act, D_FILTER *in_fil, D_ACT *out_act, float *bn_weight_mem,
	unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox,
	unsigned int stride,
	unsigned int pad
) {
	// create output MAC and initialize to 0
	D_MAC out_act_mac[nof*noy*nox];
	for (int idx = 0; idx < nof*noy*nox; idx++) {
		out_act_mac[idx] = 0;
	}
	unsigned int nix = nox*stride;
	unsigned int niy = noy*stride;
	for (int kdx = 0; kdx < nif; kdx++) {
		for (int ndx = 0; ndx < niy; ndx += stride) {
			for (int mdx = 0; mdx < nix; mdx += stride) {
				for (int cdx = 0; cdx < nof; cdx++) {
					for (int hdx = 0; hdx < nky; hdx++) {
						for (int wdx = 0; wdx < nkx; wdx++) {

							D_ACT in_act_element;
							// when accessing zero padded index
							if ( (ndx + hdx < pad) || (ndx + hdx >= niy + pad) || (mdx + wdx < pad) || (mdx + wdx >= nix + pad) ) {
								in_act_element = 0;
							}
							else {
								// input address calc
								unsigned int in_addr = kdx * niy * nix;
								in_addr += (ndx + hdx - pad) * nix + (mdx + wdx - pad);
								// load input
								in_act_element = in_act[in_addr];
								if (in_addr >= nif*niy*nix) {
									std::cout << "input index out-of-bounds: " << in_addr << std::endl;
								}
							}

							// filter address calc
							unsigned int filter_addr = cdx * nif * nky * nkx;
							filter_addr += kdx * nky * nkx;
							filter_addr += hdx * nkx;
							filter_addr += wdx;
							// load filter
							D_FILTER in_fil_element = in_fil[filter_addr];
							if (filter_addr >= nof*nif*nky*nkx) {
								std::cout << "filter index out-of-bounds: " << filter_addr << std::endl;
							}

							// mult
							D_MULT mult_element = in_act_element * in_fil_element;

							// output address calc
							unsigned int out_addr = cdx * noy * nox;
							out_addr += ndx / stride * nox + mdx / stride;
							if (out_addr >= nof*noy*nox) {
								std::cout << "output index out-of-bounds: " << out_addr << std::endl;
							}
							// load output mac
							out_act_mac[out_addr] += mult_element;
						}
					}
				}
			}
		}
	}

	for (int f = 0; f < nof; f++) {
		float mean = bn_weight_mem[f];
		float var_mult = bn_weight_mem[f+nof];
		float gamma = bn_weight_mem[f+2*nof];
		float beta = bn_weight_mem[f+3*nof];
		for (int y = 0; y < noy; y++) {
			for (int x = 0; x < nox; x++) {
				int idx = f*noy*nox + y*nox + x;
				float val = (float) out_act_mac[idx];
				val = (val-mean)*var_mult*gamma+beta;
				out_act[idx] = val;
			}
		}
	}
}

template<typename D_ACT, typename D_FILTER, typename D_MULT, typename D_MAC>
void convolution_bn_skip_relu_golden(D_ACT *in_act, D_FILTER *in_fil, D_ACT *out_act, float *bn_weight_mem, D_ACT *add_act,
	unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox,
	unsigned int stride,
	unsigned int pad
) {
	// create output MAC and initialize to 0
	D_MAC out_act_mac[nof*noy*nox];
	float out_vals[nof*noy*nox];
	for (int idx = 0; idx < nof*noy*nox; idx++) {
		out_act_mac[idx] = 0;
	}
	unsigned int nix = nox*stride;
	unsigned int niy = noy*stride;
	for (int kdx = 0; kdx < nif; kdx++) {
		for (int ndx = 0; ndx < niy; ndx += stride) {
			for (int mdx = 0; mdx < nix; mdx += stride) {
				for (int cdx = 0; cdx < nof; cdx++) {
					for (int hdx = 0; hdx < nky; hdx++) {
						for (int wdx = 0; wdx < nkx; wdx++) {

							D_ACT in_act_element;
							// when accessing zero padded index
							if ( (ndx + hdx < pad) || (ndx + hdx >= niy + pad) || (mdx + wdx < pad) || (mdx + wdx >= nix + pad) ) {
								in_act_element = 0;
							}
							else {
								// input address calc
								unsigned int in_addr = kdx * niy * nix;
								in_addr += (ndx + hdx - pad) * nix + (mdx + wdx - pad);
								// load input
								in_act_element = in_act[in_addr];
								if (in_addr >= nif*niy*nix) {
									std::cout << "input index out-of-bounds: " << in_addr << std::endl;
								}
							}

							// filter address calc
							unsigned int filter_addr = cdx * nif * nky * nkx;
							filter_addr += kdx * nky * nkx;
							filter_addr += hdx * nkx;
							filter_addr += wdx;
							// load filter
							D_FILTER in_fil_element = in_fil[filter_addr];
							if (filter_addr >= nof*nif*nky*nkx) {
								std::cout << "filter index out-of-bounds: " << filter_addr << std::endl;
							}

							// mult
							D_MULT mult_element = in_act_element * in_fil_element;

							// output address calc
							unsigned int out_addr = cdx * noy * nox;
							out_addr += ndx / stride * nox + mdx / stride;
							if (out_addr >= nof*noy*nox) {
								std::cout << "output index out-of-bounds: " << out_addr << std::endl;
							}
							// load output mac
							out_act_mac[out_addr] += mult_element;
						}
					}
				}
			}
		}
	}

	for (int f = 0; f < nof; f++) {
		float mean = bn_weight_mem[f];
		float var_mult = bn_weight_mem[f+nof];
		float gamma = bn_weight_mem[f+2*nof];
		float beta = bn_weight_mem[f+3*nof];
		for (int y = 0; y < noy; y++) {
			for (int x = 0; x < nox; x++) {
				int idx = f*noy*nox + y*nox + x;
				float val = (float) out_act_mac[idx];
				val = (val-mean)*var_mult*gamma+beta;
				out_vals[idx] = val;
			}
		}
	}
	for (int f = 0; f < nof; f++) {
		for (int y = 0; y < noy; y++) {
			for (int x = 0; x < nox; x++) {
				int idx = f*noy*nox + y*nox + x;
				out_vals[idx] += add_act[idx];
				out_act[idx] = (out_vals[idx] > 0) ? out_vals[idx] : (D_ACT) 0;
			}
		}
	}
}

int main(){
	// Print configuration information
#if CHECK_CONFIG
	print_conv_config();
	print_data_types();
	// Assertion to check the configuration
#endif
	// DTYPE_FIL *weight_mem;		// todo: weight packing
	// float *bn_weight_mem;

	DTYPE_ACT act_mem[MEM0_SIZE+MEM1_SIZE+MEM2_SIZE];
	DTYPE_FIL weight_mem[WEIGHT_MEM_SIZE];
	float bn_weight_mem[BN_WEIGHT_MEM_SIZE];
	for (int idx = 0; idx < MEM0_SIZE+MEM1_SIZE+MEM2_SIZE; idx++) act_mem[idx] = 0;
	for (int idx = 0; idx < WEIGHT_MEM_SIZE; idx++) weight_mem[idx] = 0;
	for (int idx = 0; idx < BN_WEIGHT_MEM_SIZE; idx++) bn_weight_mem[idx] = 0;
	

	int result1;
	int result2;

	// fill data
	std::string fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/data/input.bin";
	read_bin_fixed<DTYPE_ACT>(fname, act_mem, 0, INPUT_SIZE);
	fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/data/weight1.bin";
	read_bin_fixed<DTYPE_FIL>(fname, weight_mem, BB7_CONV1_WEIGHT_BASE, BB7_CONV1_CONV_WEIGHT_SIZE);
	fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/data/weight2.bin";
	read_bin_fixed<DTYPE_FIL>(fname, weight_mem, BB7_CONV2_WEIGHT_BASE, BB7_CONV2_CONV_WEIGHT_SIZE);
	fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/data/weight3.bin";
	read_bin_fixed<DTYPE_FIL>(fname, weight_mem, BB7_SKIP_WEIGHT_BASE, BB7_SKIP_CONV_WEIGHT_SIZE);
	fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/data/bn_weight1.bin";
	read_bin_float(fname, bn_weight_mem, BB7_CONV1_BN_WEIGHT_BASE, BB7_CONV1_BN_WEIGHT_SIZE);
	fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/data/bn_weight2.bin";
	read_bin_float(fname, bn_weight_mem, BB7_CONV2_BN_WEIGHT_BASE, BB7_CONV2_BN_WEIGHT_SIZE);
	fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/data/bn_weight3.bin";
	read_bin_float(fname, bn_weight_mem, BB7_SKIP_BN_WEIGHT_BASE, BB7_SKIP_BN_WEIGHT_SIZE);

	std::cout << "first 3 input val: " << act_mem[0] << " " << act_mem[1] << " " << act_mem[2] << std::endl;
	std::cout << "first 3 weight val: " << weight_mem[0] << " " << weight_mem[1] << " " << weight_mem[2] << std::endl;
	std::cout << "first 3 bn_weight_mem val: " << bn_weight_mem[0] << " " << bn_weight_mem[1] << " " << bn_weight_mem[2] << std::endl;

	conv_kernel(act_mem, weight_mem, bn_weight_mem, &result1, &result2);
	
	// golden conv gen
	float in_act_host_float[INPUT_SIZE];
	float in_fil_host_float[FILTER_SIZE];
	float out_act_host_float[OUTPUT_SIZE];
	for (int idx = 0; idx < INPUT_SIZE; idx++) {
		in_act_host_float[idx] = (float) act_mem[idx];
	}
	for (int idx = 0; idx < FILTER_SIZE; idx++) {
		in_fil_host_float[idx] = (float) weight_mem[idx];
	}
 	convolution_golden<float, float, float, float>(in_act_host_float, in_fil_host_float, out_act_host_float,
 			BB7_CONV1_K, BB7_CONV1_K, BB7_CONV1_C, BB6_SKIP_C, BB7_CONV1_H, BB7_CONV1_W, BB7_CONV1_S, BB7_CONV1_PAD);

    for (int idx = 0; idx < OUTPUT_SIZE; idx++) {
        if (out_act_host_float[idx] != act_mem[MEM0_SIZE+idx]){
			std::cout << "idx: " << idx << " host: " << out_act_host_float[idx] << " kernel: " << act_mem[idx] << std::endl;
            result2 = 0;
        }
    }

	std::cout << "result1: " << result1 << std::endl;
	std::cout << "result2: " << result2 << std::endl;

}
