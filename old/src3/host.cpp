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
#include "kernel_test.hpp"
#include "kernel_test2.hpp"
#include "kernel_test3.hpp"

// Print the configuration information
#define CHECK_CONFIG		1

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

	DTYPE_FIL *weight_mem = new DTYPE_FIL[WEIGHT_MEM_SIZE];
	float *bn_weight_mem = new float[BN_WEIGHT_MEM_SIZE];

	// // load weights
	// const std::string fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/data/layer4_0_conv1_weights.bin";
	// read_bin_fixed<DTYPE_FIL>(fname, weight_mem, BB7_CONV1_CONV_WEIGHT_SIZE);
	// fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/data/layer4_0_conv2_weights.bin";
	// read_bin_fixed<DTYPE_FIL>(fname, BB7_CONV2_WEIGHT_BASE+weight_mem, BB7_CONV2_CONV_WEIGHT_SIZE);
	// 
	// std::cout << "weight_mem[BB7_CONV2_WEIGHT_BASE]: " << weight_mem[BB7_CONV2_WEIGHT_BASE] << std::endl;
	// std::cout << "weight_mem[BB7_CONV2_WEIGHT_BASE+1]: " << weight_mem[BB7_CONV2_WEIGHT_BASE+1] << std::endl;
	
	// read_bin_fixed<DTYPE_FIL>(fname, weight_mem, BB7_CONV1_CONV_WEIGHT_SIZE);
	gen_rand<DTYPE_FIL, BB7_CONV1_CONV_WEIGHT_SIZE>(weight_mem+BB7_CONV1_WEIGHT_BASE, -1, 1);
	gen_rand<DTYPE_FIL, BB7_CONV2_CONV_WEIGHT_SIZE>(weight_mem+BB7_CONV2_WEIGHT_BASE, -1, 1);
	gen_rand<DTYPE_FIL, BB7_SKIP_CONV_WEIGHT_SIZE>(weight_mem+BB7_SKIP_WEIGHT_BASE, -1, 1);
	gen_rand<float, BB7_CONV1_BN_WEIGHT_SIZE>(bn_weight_mem+BB7_CONV1_BN_WEIGHT_BASE, -0.5,0.5);
	gen_rand<float, BB7_CONV2_BN_WEIGHT_SIZE>(bn_weight_mem+BB7_CONV2_BN_WEIGHT_BASE, -0.5,0.5);
	gen_rand<float, BB7_SKIP_BN_WEIGHT_SIZE>(bn_weight_mem+BB7_SKIP_BN_WEIGHT_BASE, -0.5,0.5);

	// host-side data
	DTYPE_ACT in_act_host[BB6_SKIP_C * BB6_SKIP_H * BB6_SKIP_W];
	// DTYPE_FIL in_fil_host[TOTAL_FIL_LEN];
	DTYPE_ACT out_act_host[BB7_CONV1_C * BB7_CONV1_H * BB7_CONV1_W];
	DTYPE_ACT in_add_host[BB7_CONV1_C * BB7_CONV1_H * BB7_CONV1_W];
	float in_act_host_float[BB6_SKIP_C * BB6_SKIP_H * BB6_SKIP_W];
	float in_fil_host_float[BB6_SKIP_C * BB7_CONV1_C * BB7_CONV1_H * BB7_CONV1_W];
	float out_act_host_float[BB7_CONV1_C * BB7_CONV1_H * BB7_CONV1_W];
	float in_add_host_float[BB7_CONV1_C * BB7_CONV1_H * BB7_CONV1_W];
	// generate random input activation and filter value with float
	gen_rand<DTYPE_ACT, BB6_SKIP_C * BB6_SKIP_H * BB6_SKIP_W>(in_act_host, -1, 1);
	gen_rand<DTYPE_ACT, BB7_CONV1_C * BB7_CONV1_H * BB7_CONV1_W>(in_add_host, -1, 1);
	// gen_rand<DTYPE_FIL, TOTAL_FIL_LEN>(in_fil_host, -1, 1);
	for (int idx = 0; idx < BB6_SKIP_C * BB6_SKIP_H * BB6_SKIP_W; idx++) {
		in_act_host_float[idx] = in_act_host[idx];
	}
	for (int idx = 0; idx < BB6_SKIP_C * BB7_CONV1_C * BB7_CONV1_K * BB7_CONV1_K; idx++) {
		in_fil_host_float[idx] = weight_mem[idx];
	}
	for (int idx = 0; idx < BB7_CONV1_C * BB7_CONV1_H * BB7_CONV1_W; idx++) {
		in_add_host_float[idx] = in_add_host[idx];
	}

	// golden convolution result with fixed point and float
 	convolution_golden<DTYPE_ACT, DTYPE_FIL, DTYPE_MUL, DTYPE_MAC>(in_act_host, weight_mem, out_act_host,
 			BB7_CONV1_K, BB7_CONV1_K, BB7_CONV1_C, BB6_SKIP_C, BB7_CONV1_H, BB7_CONV1_W, BB7_CONV1_S, BB7_CONV1_PAD);
 	convolution_golden<float, float, float, float>(in_act_host_float, in_fil_host_float, out_act_host_float,
 			BB7_CONV1_K, BB7_CONV1_K, BB7_CONV1_C, BB6_SKIP_C, BB7_CONV1_H, BB7_CONV1_W, BB7_CONV1_S, BB7_CONV1_PAD);

	//std::cout << "out_act_host[0] = " << out_act_host[0] << std::endl;

	// compare with golden result
	compare_result<DTYPE_ACT, float, BB7_CONV1_C * BB7_CONV1_H * BB7_CONV1_W>(out_act_host, out_act_host_float, 2.0/(1<<(W_ACT-I_ACT)));

	// kernel_func(in_act_host, weight_mem, bn_weight_mem, out_act_host);

	kernel_test_func(in_act_host, weight_mem, bn_weight_mem, out_act_host);
	compare_result<DTYPE_ACT, float, BB7_CONV1_C * BB7_CONV1_H * BB7_CONV1_W>(out_act_host, out_act_host_float, 2.0/(1<<(W_ACT-I_ACT)));

	// conv with bn
	convolution_bn_golden<float, float, float, float>(in_act_host_float, in_fil_host_float, out_act_host_float, bn_weight_mem,
			BB7_CONV1_K, BB7_CONV1_K, BB7_CONV1_C, BB6_SKIP_C, BB7_CONV1_H, BB7_CONV1_W, BB7_CONV1_S, BB7_CONV1_PAD);
	kernel_test2_func(in_act_host, weight_mem, bn_weight_mem, out_act_host);
	compare_result<DTYPE_ACT, float, BB7_CONV1_C * BB7_CONV1_H * BB7_CONV1_W>(out_act_host, out_act_host_float, 2.0/(1<<(W_ACT-I_ACT)));

	// conv, bn, skip conn, relu
	convolution_bn_skip_relu_golden<float, float, float, float>(in_act_host_float, in_fil_host_float, out_act_host_float, bn_weight_mem, in_add_host_float,
			BB7_CONV1_K, BB7_CONV1_K, BB7_CONV1_C, BB6_SKIP_C, BB7_CONV1_H, BB7_CONV1_W, BB7_CONV1_S, BB7_CONV1_PAD);
	kernel_test3_func(in_act_host, in_add_host, weight_mem, bn_weight_mem, out_act_host);

	// DTYPE_ACT in_act_host[BB6_SKIP_C * BB6_SKIP_H * BB6_SKIP_W];
	float weight_mem0[BB7_CONV1_CONV_WEIGHT_SIZE];
	float weight_mem1[BB7_CONV2_CONV_WEIGHT_SIZE];
	float weight_mem2[BB7_SKIP_CONV_WEIGHT_SIZE];
	float bn_weight_mem0[BB7_CONV1_BN_WEIGHT_SIZE];
	float bn_weight_mem1[BB7_CONV2_BN_WEIGHT_SIZE];
	float bn_weight_mem2[BB7_SKIP_BN_WEIGHT_SIZE];
	for (int idx = 0; idx < BB7_CONV1_CONV_WEIGHT_SIZE; idx++) weight_mem0[idx] = weight_mem[idx+BB7_CONV1_WEIGHT_BASE];
	for (int idx = 0; idx < BB7_CONV2_CONV_WEIGHT_SIZE; idx++) weight_mem1[idx] = weight_mem[idx+BB7_CONV2_WEIGHT_BASE];
	for (int idx = 0; idx < BB7_SKIP_CONV_WEIGHT_SIZE; idx++) weight_mem2[idx] = weight_mem[idx+BB7_SKIP_WEIGHT_BASE];
	for (int idx = 0; idx < BB7_CONV1_BN_WEIGHT_SIZE; idx++) bn_weight_mem0[idx] = weight_mem[idx+BB7_CONV1_BN_WEIGHT_BASE];
	for (int idx = 0; idx < BB7_CONV2_BN_WEIGHT_SIZE; idx++) bn_weight_mem1[idx] = weight_mem[idx+BB7_CONV2_BN_WEIGHT_BASE];
	for (int idx = 0; idx < BB7_SKIP_BN_WEIGHT_SIZE; idx++) bn_weight_mem2[idx] = weight_mem[idx+BB7_SKIP_BN_WEIGHT_BASE];

	float host_mem0[BB6_SKIP_C * BB6_SKIP_H * BB6_SKIP_W];
	float host_mem1[BB6_SKIP_C * BB6_SKIP_H * BB6_SKIP_W];
	float host_mem2[BB6_SKIP_C * BB6_SKIP_H * BB6_SKIP_W];
	for (int idx = 0; idx < BB6_SKIP_C * BB6_SKIP_H * BB6_SKIP_W; idx++) {
		host_mem0[idx] = in_act_host[idx];
	}
	convolution_bn_golden<float, float, float, float>(host_mem0, weight_mem0, host_mem1, bn_weight_mem0,
			BB7_CONV1_K, BB7_CONV1_K, BB7_CONV1_C, BB6_SKIP_C, BB7_CONV1_H, BB7_CONV1_W, BB7_CONV1_S, BB7_CONV1_PAD);
	for (int idx = 0; idx < BB7_CONV1_C*BB7_CONV1_H*BB7_CONV1_W; idx++) {
		host_mem1[idx] = (host_mem1[idx] > 0) ? host_mem1[idx] : 0;
	}
	convolution_bn_golden<float, float, float, float>(host_mem1, weight_mem1, host_mem2, bn_weight_mem1,
			BB7_CONV2_K, BB7_CONV2_K, BB7_CONV2_C, BB7_CONV1_C, BB7_CONV2_H, BB7_CONV2_W, BB7_CONV2_S, BB7_CONV2_PAD);
	convolution_bn_skip_relu_golden<float, float, float, float>(host_mem0, weight_mem2, host_mem1, bn_weight_mem2, host_mem2,
			BB7_SKIP_K, BB7_SKIP_K, BB7_SKIP_C, BB7_CONV2_C, BB7_SKIP_H, BB7_SKIP_W, BB7_SKIP_S, BB7_SKIP_PAD);
	
	kernel_func(in_act_host, weight_mem, bn_weight_mem, out_act_host);
	compare_result<DTYPE_ACT, float, BB7_SKIP_C*BB7_SKIP_H*BB7_SKIP_W>(out_act_host, host_mem1);

}
