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
#define CHECK_CONFIG		0
#define SHOW_ALL_OUTPUT		1

int main(){
	// Print configuration information
#if CHECK_CONFIG
	print_conv_config();
	print_data_types();
	// Assertion to check the configuration
#endif

	std::cout << "****************************************" << std::endl;
	std::cout << "host_tb_functions.cpp" << std::endl;
	std::cout << "****************************************" << std::endl;

	// mimic controller 
	unsigned start_layer;
	unsigned end_layer;
	unsigned layer_cnt;
	unsigned nif;
	unsigned nof;
	unsigned noy;
	unsigned nox;
	unsigned nkx;
	unsigned nky;
	unsigned stride;
	unsigned pad;
	bool bb_en;
	bool conv_en;
	bool bn_en;
	bool skip_en;
	bool relu_en;
	bool max_pool_en;
	bool avg_pool_en;
	bool fc_en;
	unsigned base_addr_in;
	unsigned base_addr_out;
	unsigned base_addr_add;
	unsigned weight_base;
	unsigned weight_size;
	unsigned bn_weight_base;
	unsigned bn_weight_size;
	unsigned in_size;
	unsigned out_size;

	// mem check
	std::cout << "WEIGHT_MEM_SIZE: " << WEIGHT_MEM_SIZE << std::endl;
	std::cout << "BN_WEIGHT_MEM_SIZE: " << BN_WEIGHT_MEM_SIZE << std::endl;
	std::cout << "ACT_MEM_SIZE: " << ACT_MEM_SIZE << std::endl;
	std::cout << "MAX_ACT_MEM_SIZE: " << MAX_ACT_MEM_SIZE << std::endl << std::endl;

	// layer size check
	for (layer_cnt = 0; layer_cnt <= 27; layer_cnt++) {
		controller (
			&layer_cnt, &nif, &nof, &noy, &nox, &nkx, &nky, &stride, &pad,
			&bb_en, &conv_en, &bn_en, &skip_en, &relu_en, &max_pool_en, &avg_pool_en, &fc_en,
			&base_addr_in, &base_addr_out, &base_addr_add, 
			&weight_base, &weight_size, &bn_weight_base, &bn_weight_size, &in_size, &out_size
		);
		if (conv_en) {
			std::cout << "layer_cnt: " << layer_cnt << std::endl;
			std::cout << "  weight_size: " << weight_size << std::endl;
			std::cout << "  bn_weight_size: " << bn_weight_size << std::endl;
		}
		if (fc_en) {
			std::cout << "layer_cnt: " << layer_cnt << std::endl;
			std::cout << "bn_weight_size: " << bn_weight_size << std::endl;
		}
		// // for debugging
		// std::cout << "****************************************" << std::endl;
		// std::cout << "layer_cnt: " << layer_cnt << std::endl;
		// std::cout << "nif: " << nif << std::endl;
		// std::cout << "nof: " << nof << std::endl;
		// std::cout << "noy: " << noy << std::endl;
		// std::cout << "nox: " << nox << std::endl;
		// std::cout << "nkx: " << nkx << std::endl;
		// std::cout << "nky: " << nky << std::endl;
		// std::cout << "stride: " << stride << std::endl;
		// std::cout << "pad: " << pad << std::endl;
		// std::cout << "bb_en: " << bb_en << std::endl;
		// std::cout << "conv_en: " << conv_en << std::endl;
		// std::cout << "bn_en: " << bn_en << std::endl;
		// std::cout << "skip_en: " << skip_en << std::endl;
		// std::cout << "relu_en: " << relu_en << std::endl;
		// std::cout << "max_pool_en: " << max_pool_en << std::endl;
		// std::cout << "avg_pool_en: " << avg_pool_en << std::endl;
		// std::cout << "fc_en: " << fc_en << std::endl;
		// std::cout << "base_addr_in: " << base_addr_in << std::endl;
		// std::cout << "base_addr_out: " << base_addr_out << std::endl;
		// std::cout << "base_addr_add: " << base_addr_add << std::endl;
		// std::cout << "weight_base: " << weight_base << std::endl;
		// std::cout << "weight_size: " << weight_size << std::endl;
		// std::cout << "bn_weight_base: " << bn_weight_base << std::endl;
		// std::cout << "bn_weight_size: " << bn_weight_size << std::endl;
		// std::cout << "in_size: " << in_size << std::endl;
		// std::cout << "out_size: " << out_size << std::endl;
		// std::cout << "****************************************" << std::endl;
		// std::cout << std::endl;
	}

	// kernel IO
	DTYPE_ACT act_in[MAX_ACT_MEM_SIZE];
	DTYPE_ACT act_out[MAX_ACT_MEM_SIZE];
	for (int idx = 0; idx < MAX_ACT_MEM_SIZE; idx++) act_in[idx] = 0;
	for (int idx = 0; idx < MAX_ACT_MEM_SIZE; idx++) act_out[idx] = 0;

	// kernel offchip memory
	DTYPE_FIL weight_mem[WEIGHT_MEM_SIZE];
	float bn_weight_mem[BN_WEIGHT_MEM_SIZE];
	for (int idx = 0; idx < WEIGHT_MEM_SIZE; idx++) weight_mem[idx] = 0;
	for (int idx = 0; idx < BN_WEIGHT_MEM_SIZE; idx++) bn_weight_mem[idx] = 0;
	// load input, filter, bn_weight
	std::string base_fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/data/";
	read_bin<DTYPE_ACT>(base_fname+"input.bin", act_in, 0, CONV1_IN_SIZE);
	read_bin<DTYPE_FIL>(base_fname+"conv_all_params.bin", weight_mem, 0, WEIGHT_MEM_SIZE);
	read_bin<float>(base_fname+"bn_all_params.bin", bn_weight_mem, 0, BN_WEIGHT_MEM_SIZE);
	std::cout << "input, filter, bn_weight loaded" << std::endl << std::endl;

	// host memory
	float act_mem_host[ACT_MEM_HOST_SIZE];
	float weight_mem_host[WEIGHT_MEM_SIZE];
	float bn_weight_mem_host[BN_WEIGHT_MEM_SIZE];
	for (int idx = 0; idx < WEIGHT_MEM_SIZE; idx++) weight_mem_host[idx] = 0;
	for (int idx = 0; idx < BN_WEIGHT_MEM_SIZE; idx++) bn_weight_mem_host[idx] = 0;
	// copy input, filter, bn_weight to host
	for (int idx = 0; idx < WEIGHT_MEM_SIZE; idx++) weight_mem_host[idx] = (float) weight_mem[idx];
	for (int idx = 0; idx < BN_WEIGHT_MEM_SIZE; idx++) bn_weight_mem_host[idx] = bn_weight_mem[idx];

	// conv1 test
	start_layer = 0;
	end_layer = 0;
	layer_cnt = 0;
	controller (
		&layer_cnt, &nif, &nof, &noy, &nox, &nkx, &nky, &stride, &pad,
		&bb_en, &conv_en, &bn_en, &skip_en, &relu_en, &max_pool_en, &avg_pool_en, &fc_en,
		&base_addr_in, &base_addr_out, &base_addr_add, 
		&weight_base, &weight_size, &bn_weight_base, &bn_weight_size, &in_size, &out_size
	);
	// load input for test
	read_bin<float>(base_fname+"input.bin", act_mem_host, base_addr_in, in_size);
	for (int idx = 0; idx < in_size; idx++) act_in[base_addr_in+idx] = act_mem_host[base_addr_in+idx];
	// conv, bn
	convolution_bn_golden<float, float, float, float>(
			act_mem_host+base_addr_in, 
			weight_mem_host+weight_base, 
			act_mem_host+base_addr_out, 
			bn_weight_mem_host+bn_weight_base,
			nky, nkx, nof, nif, noy, nox, stride, pad);
	// relu
	for (int idx = 0; idx < out_size; idx++) {
		act_mem_host[base_addr_out+idx] = (act_mem_host[base_addr_out+idx] > 0) ? act_mem_host[base_addr_out+idx] : 0;
	}
	conv_kernel(act_in, act_out, weight_mem, bn_weight_mem, &start_layer, &end_layer);
	std::cout << "act_out size: " << out_size << std::endl;
	// for (int idx = 0; idx < out_size; idx++) {
	for (int idx = 0; idx < out_size; idx++) {
		std::cout << "act_out[" << idx << "]: " << act_out[idx] << std::endl;
	}
	for (int idx = 0; idx < out_size; idx++) {
		std::cout << "act_mem_host[" << idx << "]: " << act_mem_host[base_addr_out+idx] << std::endl;
	}

	// max pool test
	start_layer = 1;
	end_layer = 1;
	layer_cnt = 1;
	controller (
		&layer_cnt, &nif, &nof, &noy, &nox, &nkx, &nky, &stride, &pad,
		&bb_en, &conv_en, &bn_en, &skip_en, &relu_en, &max_pool_en, &avg_pool_en, &fc_en,
		&base_addr_in, &base_addr_out, &base_addr_add, 
		&weight_base, &weight_size, &bn_weight_base, &bn_weight_size, &in_size, &out_size
	);
	// load input for test
	read_bin<float>(base_fname+"after_relu.bin", act_mem_host, base_addr_in, in_size);
	for (int idx = 0; idx < in_size; idx++) act_in[idx] = act_mem_host[base_addr_in+idx];
	// max pool
	max_pool_golden<float>(
			act_mem_host, 
			base_addr_in, 
			base_addr_out, 
			nky, nkx, nof, nif, noy, nox, stride, pad, max_pool_en);
	conv_kernel(act_in, act_out, weight_mem, bn_weight_mem, &start_layer, &end_layer);

	// avg pool test
	start_layer = 26;
	end_layer = 26;
	layer_cnt = 26;
	controller (
		&layer_cnt, &nif, &nof, &noy, &nox, &nkx, &nky, &stride, &pad,
		&bb_en, &conv_en, &bn_en, &skip_en, &relu_en, &max_pool_en, &avg_pool_en, &fc_en,
		&base_addr_in, &base_addr_out, &base_addr_add, 
		&weight_base, &weight_size, &bn_weight_base, &bn_weight_size, &in_size, &out_size
	);
	// load input for test
	read_bin<float>(base_fname+"after_layer4.bin", act_mem_host, base_addr_in, in_size);
	for (int idx = 0; idx < in_size; idx++) act_in[idx] = act_mem_host[base_addr_in+idx];
	for (int idx = 0; idx < in_size; idx++) {
		std::cout << "act_mem[" << idx << "]: " << act_mem[base_addr_in+idx] << std::endl;
	}
	for (int idx = 0; idx < in_size; idx++) {
		std::cout << "act_mem_host[" << idx << "]: " << act_mem_host[base_addr_in+idx] << std::endl;
	}
	std::cout << "****************************************" << std::endl;
	std::cout << "base_addr_in: " << base_addr_in << std::endl;
	std::cout << "base_addr_out: " << base_addr_out << std::endl;
	std::cout << "****************************************" << std::endl;
	// avg pool
	avg_pool_golden<float>(
			act_mem_host, 
			base_addr_in, 
			base_addr_out, 
			nky, nkx, nof, nif, noy, nox, stride, pad, avg_pool_en);
	conv_kernel(act_in, act_out, weight_mem, bn_weight_mem, &start_layer, &end_layer);
	// show all outputs for debugging
#if SHOW_ALL_OUTPUT
	std::cout << "act_out size: " << out_size << std::endl;
	// for (int idx = 0; idx < out_size; idx++) {
	for (int idx = 0; idx < out_size; idx++) {
		std::cout << "act_out[" << idx << "]: " << act_out[idx] << std::endl;
	}
	for (int idx = 0; idx < out_size; idx++) {
		std::cout << "act_mem_host[" << idx << "]: " << act_mem_host[base_addr_out+idx] << std::endl;
	}
#endif

	// fc test
	start_layer = 27;
	end_layer = 27;
	layer_cnt = 27;
	controller (
		&layer_cnt, &nif, &nof, &noy, &nox, &nkx, &nky, &stride, &pad,
		&bb_en, &conv_en, &bn_en, &skip_en, &relu_en, &max_pool_en, &avg_pool_en, &fc_en,
		&base_addr_in, &base_addr_out, &base_addr_add, 
		&weight_base, &weight_size, &bn_weight_base, &bn_weight_size, &in_size, &out_size
	);
	// load input for test
	read_bin<float>(base_fname+"after_avgpool.bin", act_mem_host, base_addr_in, in_size);
	for (int idx = 0; idx < in_size; idx++) act_in[idx] = act_mem_host[base_addr_in+idx];
	// fc
	fc_golden<float>(
			act_mem_host,
			bn_weight_mem_host,
			base_addr_in, 
			base_addr_out, 
			bn_weight_base,
			nof, nif, fc_en);
	conv_kernel(act_in, act_out, weight_mem, bn_weight_mem, &start_layer, &end_layer);

	// show all outputs for debugging
#if SHOW_ALL_OUTPUT
	std::cout << "act_out size: " << out_size << std::endl;
	// for (int idx = 0; idx < out_size; idx++) {
	for (int idx = 0; idx < out_size; idx++) {
		std::cout << "act_out[" << idx << "]: " << act_out[idx] << std::endl;
	}
	for (int idx = 0; idx < out_size; idx++) {
		std::cout << "act_mem_host[" << idx << "]: " << act_mem_host[base_addr_out+idx] << std::endl;
	}
#endif

}
