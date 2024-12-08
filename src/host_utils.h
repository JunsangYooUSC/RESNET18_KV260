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
#include <iostream>
#include <string>
#include <cmath>
#include <random>
// Include project headers
#include "conv_config.h"

// print_conv_config: function to print current convolutional layer configuration
void print_conv_config() {
    // std::cout << std::string(20, '-') << "input layer information" << std::string(20, '-') << std::endl;
    std::cout << "NKX:        " << NKX << std::endl;
    std::cout << "NKY:        " << NKY << std::endl;
    std::cout << "NIF:        " << NIF << std::endl;
    std::cout << "NOX:        " << NOX << std::endl;
    std::cout << "NOY:        " << NOY << std::endl;
    std::cout << "NOF:        " << NOF << std::endl;
    
    std::cout << "PKX:        " << PKX << std::endl;
    std::cout << "PKY:        " << PKY << std::endl;
    std::cout << "PIF:        " << PIF << std::endl;
    std::cout << "POX:        " << POX << std::endl;
    std::cout << "POY:        " << POY << std::endl;
    std::cout << "POF:        " << POF << std::endl;
    
    std::cout << "PAD:        " << POF << std::endl;
    std::cout << "STRIDE:     " << POF << std::endl;
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
    std::cout << "extra bits needed for MAC:        " << MAC_EXTRA_BITS << std::endl;
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

/*
// compare_result: compare fixed-point and floating-point outputs
template<typename DTYPE, unsigned int LEN>
void compare_result(DTYPE *mat_fixed, float *mat_float, float tolerance = 0.2) {
    bool mismatch_flag = false;
    bool diff_flag;
    int cnt = 0;
    for (int idx = 0; idx < LEN; idx++) {
        float val_fixed = static_cast<float>(mat_fixed[idx]);
        float val_float = mat_float[idx];
        float diff = std::abs(val_fixed - val_float);
        float max_val = (std::abs(val_fixed) > std::abs(val_float)) ? std::abs(val_fixed) : std::abs(val_float);
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
				std::cout << "idx: " << idx << ", float: " << mat_float[idx] << ", fixed: " << mat_fixed[idx] << std::endl;
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
*/

#endif