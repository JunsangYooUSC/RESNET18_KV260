/******************************************************************************
 * Filename: host_utils.h
 * Author: Junsang Yoo
 *
 * Description:
 * util functions for host
 *
 * Functions:
 * - print_conv_config: print convolution configuration
 * - print_data_types: print information about defined data types
 * - print_PE_config: print information about Load / Store / PE configuration
 * - gen_rand: generate random array
 * - compare_result: compare fixed-point and floating-point outputs
 ******************************************************************************/

#ifndef HOST_UTILS_H
#define HOST_UTILS_H

// Include C++ headers
#include <fstream>
#include <iostream>
#include <iostream>
#include <string>
#include <cmath>
#include <random>
// Include project headers
#include "conv_config.h"

// print_conv_config: function to print current convolutional layer configuration
void print_conv_config() {
    // std::cout << std::string(20, '-') << "input layer information" << std::string(20, '-') << std::endl;
    std::cout << "PKX:        " << PKX << std::endl;
    std::cout << "PKY:        " << PKY << std::endl;
    std::cout << "PIF:        " << PIF << std::endl;
    std::cout << "POX:        " << POX << std::endl;
    std::cout << "POY:        " << POY << std::endl;
    std::cout << "POF:        " << POF << std::endl;
}

// print_data_types: print information about defined data types
void print_data_types() {
    std::cout << std::string(20, '-') << "activation datatype information" << std::string(20, '-') << std::endl;
    std::cout << "activation total bits:            " << W_ACT << std::endl;
    std::cout << "activation integer bits:          " << I_ACT << std::endl;
    std::cout << std::string(20, '-') << "FILTER datatype information" << std::string(20, '-') << std::endl;
    std::cout << "filter total bits:                " << W_FIL << std::endl;
    std::cout << "filter integer bits:              " << I_FIL << std::endl;
    std::cout << std::string(20, '-') << "mult datatype information" << std::string(20, '-') << std::endl;
    std::cout << "mult total bits:                  " << W_MUL << std::endl;
    std::cout << "mult integer bits:                " << I_MUL << std::endl;
    std::cout << std::string(20, '-') << "MAC datatype information" << std::string(20, '-') << std::endl;
    std::cout << "MAC total bits:                   " << W_MAC << std::endl;
    std::cout << "MAC integer bits:                 " << I_MAC << std::endl;
}

// gen_rand: generate random array
template<typename DTYPE, unsigned int LEN>
void gen_rand(DTYPE arr[LEN], float min_val, float max_val, unsigned int seed=1) {
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> dist(min_val, max_val);
	for (int i = 0; i < LEN; i++) {
		// Generate random float within range
		float rand_val = dist(rng);
		// Convert to fixed point
		arr[i] = static_cast<DTYPE>(rand_val);
	}
}

// compare_result: compare values of two datatypes
template<typename DTYPE1, typename DTYPE2, unsigned int LEN>
void compare_result(DTYPE1 *mat1, DTYPE2 *mat2, float tolerance = 0.2) {
    bool mismatch_flag = false;
    bool diff_flag;
    int cnt = 0;
    for (int idx = 0; idx < LEN; idx++) {
        float val1 = static_cast<float>(mat1[idx]);
        float val2 = static_cast<float>(mat2[idx]);
        float diff = std::abs(val1 - val2);
        float max_val = (std::abs(val1) > std::abs(val2)) ? std::abs(val1) : std::abs(val2);
		if (max_val < 1) {
			if (diff < tolerance) diff_flag = false;
			else diff_flag = true;
		} else {
			if (diff < tolerance * max_val) diff_flag = false;
			else diff_flag = true;
		}
		if (diff_flag) {
			mismatch_flag = true;
			if (cnt < 500) {
				std::cout << "idx: " << idx << ", dtype1: " << mat1[idx] << ", dtype2: " << mat2[idx] << std::endl;
				cnt++;
			}
		}
	}
	if (mismatch_flag == 1) {
		std::cout << "float and fixed result mismatch" << std::endl;
	} else {
		std::cout << "float and fixed result match" << std::endl;
	}
}

void read_bin_float(const std::string &filename, float *data, unsigned int base_addr, unsigned int size) {
	FILE *file = fopen(filename.c_str(), "rb");
	unsigned total_reads = fread(data+base_addr, sizeof(float), size, file);
	assert(size == total_reads && "read mismatch");
	fclose(file);
}

template<typename T>
void read_bin_fixed(const std::string &filename, T *data, unsigned int base_addr, unsigned int size) {
	FILE *file = fopen(filename.c_str(), "rb");
	unsigned total_reads = fread(data+base_addr, sizeof(T), size, file);
	// assert(size == total_reads && "read mismatch");
	fclose(file);
}

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
	for (int cdx = 0; cdx < nof; cdx++) {
		for (int ndx = 0; ndx < niy; ndx += stride) {
			for (int mdx = 0; mdx < nix; mdx += stride) {
				for (int kdx = 0; kdx < nif; kdx++) {
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
		float mult_factor = bn_weight_mem[f+nof];
		float beta = bn_weight_mem[f+2*nof];
		for (int y = 0; y < noy; y++) {
			for (int x = 0; x < nox; x++) {
				int idx = f*noy*nox + y*nox + x;
				float val = (float) out_act_mac[idx];
				val = (val-mean)*mult_factor+beta;
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
		float mult_factor = bn_weight_mem[f+nof];
		float beta = bn_weight_mem[f+2*nof];
		for (int y = 0; y < noy; y++) {
			for (int x = 0; x < nox; x++) {
				int idx = f*noy*nox + y*nox + x;
				float val = (float) out_act_mac[idx];
				val = (val-mean)*mult_factor+beta;
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


#endif