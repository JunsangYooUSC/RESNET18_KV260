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
#define BN_WEIGHT_SIZE		BB7_CONV1_BN_WEIGHT_SIZE

int main(){
	// Print configuration information
#if CHECK_CONFIG
	print_conv_config();
	print_data_types();
	// Assertion to check the configuration
#endif
	// memory block check
	static_assert(MEM1_BASE_ADDR >= MEM0_BASE_ADDR + MEM0_SIZE, 
			"Memory overlap between MEM0 and MEM1");
	static_assert(MEM2_BASE_ADDR >= MEM1_BASE_ADDR + MEM1_SIZE, 
			"Memory overlap between MEM1 and MEM2");

	// weight memory size check
	static_assert(WEIGHT_MEM_SIZE >= (BB8_CONV2_WEIGHT_BASE + BB8_CONV2_WEIGHT_SIZE), 
			"WEIGHT_MEM_SIZE not enough");

	// BN weight memory size check
	static_assert(BN_WEIGHT_MEM_SIZE >= (FC_BN_WEIGHT_BASE + FC_BN_WEIGHT_SIZE), 
			"BN_WEIGHT_MEM_SIZE not enough");

	// activation memory size check
	static_assert(MAX_ACT_MEM_SIZE >= MEM0_SIZE && 
			MAX_ACT_MEM_SIZE >= MEM1_SIZE && 
			MAX_ACT_MEM_SIZE >= MEM2_SIZE, 
			"MAX_ACT_MEM_SIZE not enough");

	// packing check
	static_assert((MEM0_SIZE % ACT_PACK == 0) && 
			(MEM1_SIZE % ACT_PACK == 0) && 
			(MEM2_SIZE % ACT_PACK == 0), 
			"Memory sizes must align with ACT_PACK configuration.");


	// kernel IO
	DTYPE_ACT act_in_host[MAX_ACT_MEM_SIZE];
	DTYPE_ACT act_out_host[MAX_ACT_MEM_SIZE];
	for (int idx = 0; idx < MAX_ACT_MEM_SIZE; idx++) act_in_host[idx] = 0;
	for (int idx = 0; idx < MAX_ACT_MEM_SIZE; idx++) act_out_host[idx] = 0;

	unsigned start_layer = 0;
	unsigned end_layer = 1;
	
	// kernel offchip memory
	DTYPE_FIL weight_mem[WEIGHT_MEM_SIZE];
	float bn_weight_mem[BN_WEIGHT_MEM_SIZE];
	for (int idx = 0; idx < WEIGHT_MEM_SIZE; idx++) weight_mem[idx] = 0;
	for (int idx = 0; idx < BN_WEIGHT_MEM_SIZE; idx++) bn_weight_mem[idx] = 0;

	// host memory
	float act_host_float[ACT_MEM_SIZE];
	float weight_host_float[WEIGHT_MEM_SIZE];
	float bn_weight_host_float[BN_WEIGHT_MEM_SIZE];
	for (int idx = 0; idx < ACT_MEM_SIZE; idx++) act_host_float[idx] = 0;
	for (int idx = 0; idx < WEIGHT_MEM_SIZE; idx++) weight_host_float[idx] = 0;
	for (int idx = 0; idx < BN_WEIGHT_MEM_SIZE; idx++) bn_weight_host_float[idx] = 0;

	// layer configuration for validation
	std::string fname;

	// load weight and bn_weight as a whole
	fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/conv_all_params.bin";
	read_bin_fixed<DTYPE_FIL>(fname, weight_mem, 0, WEIGHT_MEM_SIZE);
	fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/bn_all_params.bin";
	read_bin_float(fname, bn_weight_mem, 0, BN_WEIGHT_MEM_SIZE);
	for (int idx = 0; idx < WEIGHT_MEM_SIZE; idx++) weight_host_float[idx] = weight_mem[idx];
	for (int idx = 0; idx < BN_WEIGHT_MEM_SIZE; idx++) bn_weight_host_float[idx] = bn_weight_mem[idx];

	// mimic controller 
	unsigned layer_cnt = 0;
	unsigned nif = 0;
	unsigned nof = 0;
	unsigned noy = 0;
	unsigned nox = 0;
	unsigned nkx = 0;
	unsigned nky = 0;
	unsigned stride = 0;
	unsigned pad = 0;
	bool bb_en = 0;
	bool conv_en = 0;
	bool bn_en = 0;
	bool skip_en = 0;
	bool relu_en = 0;
	bool max_pool_en = 0;
	bool avg_pool_en = 0;
	bool fc_en = 0;
	unsigned base_addr_in = 0;
	unsigned base_addr_out = 0;
	unsigned base_addr_add = 0;
	unsigned weight_base = 0;
	unsigned weight_size = 0;
	unsigned bn_weight_base = 0;
	unsigned bn_weight_size = 0;
	unsigned in_size = 0;
	unsigned out_size = 0;

	// for debug
    for (layer_cnt = 0; layer_cnt <= 27; layer_cnt++) {
        controller (
            &layer_cnt,
            &nif,
            &nof,
            &noy,
            &nox,
            &nkx,
            &nky,
            &stride,
            &pad,
            &bb_en,
            &conv_en,
            &bn_en,
            &skip_en,
            &relu_en,
            &max_pool_en,
            &avg_pool_en,
            &fc_en,
            &base_addr_in,
            &base_addr_out,
            &base_addr_add,
            &weight_base,
            &weight_size,
            &bn_weight_base,
            &bn_weight_size,
            &in_size,
            &out_size
        );
		// for debugging
		std::cout << "****************************************" << std::endl;
		std::cout << "layer_cnt: " << layer_cnt << std::endl;
		std::cout << "nif: " << nif << std::endl;
		std::cout << "nof: " << nof << std::endl;
		std::cout << "noy: " << noy << std::endl;
		std::cout << "nox: " << nox << std::endl;
		std::cout << "nkx: " << nkx << std::endl;
		std::cout << "nky: " << nky << std::endl;
		std::cout << "stride: " << stride << std::endl;
		std::cout << "pad: " << pad << std::endl;
		std::cout << "bb_en: " << bb_en << std::endl;
		std::cout << "conv_en: " << conv_en << std::endl;
		std::cout << "bn_en: " << bn_en << std::endl;
		std::cout << "skip_en: " << skip_en << std::endl;
		std::cout << "relu_en: " << relu_en << std::endl;
		std::cout << "max_pool_en: " << max_pool_en << std::endl;
		std::cout << "avg_pool_en: " << avg_pool_en << std::endl;
		std::cout << "fc_en: " << fc_en << std::endl;
		std::cout << "base_addr_in: " << base_addr_in << std::endl;
		std::cout << "base_addr_out: " << base_addr_out << std::endl;
		std::cout << "base_addr_add: " << base_addr_add << std::endl;
		std::cout << "weight_base: " << weight_base << std::endl;
		std::cout << "weight_size: " << weight_size << std::endl;
		std::cout << "bn_weight_base: " << bn_weight_base << std::endl;
		std::cout << "bn_weight_size: " << bn_weight_size << std::endl;
		std::cout << "in_size: " << in_size << std::endl;
		std::cout << "out_size: " << out_size << std::endl;
		std::cout << "****************************************" << std::endl;
		std::cout << std::endl;
		if (layer_cnt == start_layer) {
			// load input
			fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/input.bin";
			read_bin_fixed<DTYPE_ACT>(fname, act_in_host, base_addr_in, in_size);
		}
		// CONV1 layer cnt 0
		if (layer_cnt == 0) {
			// conv, bn
			convolution_bn_golden<float, float, float, float>(
					act_host_float+base_addr_in, 
					weight_host_float+weight_base, 
					act_host_float+base_addr_out, 
					bn_weight_host_float+bn_weight_base,
					nky, nkx, nof, nif, noy, nox, stride, pad);
			// relu
			for (int idx = 0; idx < out_size; idx++) {
				act_host_float[base_addr_out+idx] = (act_host_float[base_addr_out+idx] > 0) ? act_host_float[base_addr_out+idx] : 0;
			}
		}
		// MAX_POOL layer cnt 1
		if (layer_cnt == 1) {
			// max_pool
			max_pool_golden<float>(act_host_float, base_addr_in, base_addr_out, nky, nkx, nof, nif, noy, nox, stride, pad, max_pool_en);
		}



		if (layer_cnt == end_layer){
			// kernel calculation
			conv_kernel(act_in_host, act_out_host, weight_mem, bn_weight_mem, &start_layer, &end_layer);
			for (int idx = 0; idx < out_size; idx++) {
				std::cout << "idx: " << idx << "  kernel out: " << act_out_host[idx] << std::endl;
			}
		}
	}
	std::cout << "WEIGHT_MEM_SIZE: " << WEIGHT_MEM_SIZE << std::endl;
	std::cout << "BN_WEIGHT_MEM_SIZE: " << BN_WEIGHT_MEM_SIZE << std::endl;
	std::cout << "ACT_MEM_SIZE: " << ACT_MEM_SIZE << std::endl;
	std::cout << "MEM0_SIZE: " << MEM0_SIZE << std::endl;
	std::cout << "MEM1_SIZE: " << MEM1_SIZE << std::endl;
	std::cout << "MEM2_SIZE: " << MEM2_SIZE << std::endl;

	// mimic controller
//    for (layer_cnt = start_layer; layer_cnt <= end_layer; layer_cnt++) {
//        controller (
//            &layer_cnt,
//            &nif,
//            &nof,
//            &noy,
//            &nox,
//            &nkx,
//            &nky,
//            &stride,
//            &pad,
//            &bb_en,
//            &conv_en,
//            &bn_en,
//            &skip_en,
//            &relu_en,
//            &max_pool_en,
//            &avg_pool_en,
//            &fc_en,
//            &base_addr_in,
//            &base_addr_out,
//            &base_addr_add,
//            &weight_base,
//            &weight_size,
//            &bn_weight_base,
//            &bn_weight_size,
//            &in_size,
//            &out_size
//        );
//		if (layer_cnt == start_layer){
//			std::cout << "WEIGHT_MEM_SIZE: " << WEIGHT_MEM_SIZE << std::endl;
//			std::cout << "BN_WEIGHT_MEM_SIZE: " << BN_WEIGHT_MEM_SIZE << std::endl;
//			std::cout << "ACT_MEM_SIZE: " << ACT_MEM_SIZE << std::endl;
//			std::cout << "MEM0_SIZE: " << MEM0_SIZE << std::endl;
//			std::cout << "MEM1_SIZE: " << MEM1_SIZE << std::endl;
//			std::cout << "MEM2_SIZE: " << MEM2_SIZE << std::endl;
//			// load input
//			fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/input.bin";
//			read_bin_fixed<DTYPE_ACT>(fname, act_in_host, base_addr_in, in_size);
//			// copy for host validation
//			for (int idx = 0; idx < in_size; idx++) {
//				act_host_float[base_addr_in+idx] = act_in_host[base_addr_in+idx];
//			}
//			// load weight and bn_weight as a whole
//			fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/conv_all_params.bin";
//			read_bin_fixed<DTYPE_FIL>(fname, weight_mem, weight_base, WEIGHT_MEM_SIZE);
//			fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/bn_all_params.bin";
//			read_bin_float(fname, bn_weight_mem, bn_weight_base, BN_WEIGHT_MEM_SIZE);
//		}
//		// // CONV1 layer cnt 0
//		// if (layer_cnt == 0) {
//		// 	// conv, bn
//		// 	convolution_bn_golden<float, float, float, float>(
//		// 			act_host_float+base_addr_in, 
//		// 			weight_host_float, 
//		// 			act_host_float+base_addr_out, 
//		// 			bn_weight_host_float,
//		// 			nky, nkx, nof, nif, noy, nox, stride, pad);
//		// 	// relu
//		// 	for (int idx = 0; idx < out_size; idx++) {
//		// 		act_host_float[base_addr_out+idx] = (act_host_float[base_addr_out+idx] > 0) ? act_host_float[base_addr_out+idx] : 0;
//		// 	}
//		// }
//		// 
//// 
//		// // BB7_CONV1 layer cnt 21
//		// if (layer_cnt == 21) {
//		// 	// load weight
//		// 	fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/data/weight1.bin";
//		// 	read_bin_fixed<DTYPE_FIL>(fname, weight_mem, weight_base, weight_size);
//		// 	// load bn_weight
//		// 	fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/data/bn_hw_weight1.bin";
//		// 	read_bin_float(fname, bn_weight_mem, bn_weight_base, bn_weight_size);
//		// 	// copy weight for host validation
//		// 	for (int idx = 0; idx < weight_size; idx++) {
//		// 		weight_host_float[idx] = weight_mem[weight_base+idx];
//		// 	}
//		// 	// copy bn_weight for host validation
//		// 	for (int idx = 0; idx < bn_weight_size; idx++) {
//		// 		bn_weight_host_float[idx] = bn_weight_mem[bn_weight_base+idx];
//		// 	}
//		// 	// host calculation
//		// 	// conv, bn
//		// 	convolution_bn_golden<float, float, float, float>(
//		// 			act_host_float+base_addr_in, 
//		// 			weight_host_float, 
//		// 			act_host_float+base_addr_out, 
//		// 			bn_weight_host_float,
//		// 			nky, nkx, nof, nif, noy, nox, stride, pad);
//		// 	// relu
//		// 	for (int idx = 0; idx < out_size; idx++) {
//		// 		act_host_float[base_addr_out+idx] = (act_host_float[base_addr_out+idx] > 0) ? act_host_float[base_addr_out+idx] : 0;
//		// 	}
//		// }
//		// // BB7_CONV2 layer cnt 22
//		// if (layer_cnt == 22) {
//		// 	// load weight
//		// 	fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/data/weight2.bin";
//		// 	read_bin_fixed<DTYPE_FIL>(fname, weight_mem, weight_base, weight_size);
//		// 	// load bn_weight
//		// 	fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/data/bn_hw_weight2.bin";
//		// 	read_bin_float(fname, bn_weight_mem, bn_weight_base, bn_weight_size);
//		// 	// copy weight for host validation
//		// 	for (int idx = 0; idx < weight_size; idx++) {
//		// 		weight_host_float[idx] = weight_mem[weight_base+idx];
//		// 	}
//		// 	// copy bn_weight for host validation
//		// 	for (int idx = 0; idx < bn_weight_size; idx++) {
//		// 		bn_weight_host_float[idx] = bn_weight_mem[bn_weight_base+idx];
//		// 	}
//		// 	// host calculation
//		// 	// conv, bn
//		// 	convolution_bn_golden<float, float, float, float>(
//		// 			act_host_float+base_addr_in, 
//		// 			weight_host_float, 
//		// 			act_host_float+base_addr_out, 
//		// 			bn_weight_host_float,
//		// 			nky, nkx, nof, nif, noy, nox, stride, pad);
//		// }
//		// // BB7_SKIP layer cnt 23
//		// if (layer_cnt == 23) {
//		// 	// load weight
//		// 	fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/data/weight3.bin";
//		// 	read_bin_fixed<DTYPE_FIL>(fname, weight_mem, weight_base, weight_size);
//		// 	// load bn_weight
//		// 	fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/data/bn_hw_weight3.bin";
//		// 	read_bin_float(fname, bn_weight_mem, bn_weight_base, bn_weight_size);
//		// 	// copy weight for host validation
//		// 	for (int idx = 0; idx < weight_size; idx++) {
//		// 		weight_host_float[idx] = weight_mem[weight_base+idx];
//		// 	}
//		// 	// copy bn_weight for host validation
//		// 	for (int idx = 0; idx < bn_weight_size; idx++) {
//		// 		bn_weight_host_float[idx] = bn_weight_mem[bn_weight_base+idx];
//		// 	}
//		// 	// host calculation
//		// 	// conv, bn
//		// 	convolution_bn_skip_relu_golden<float, float, float, float>(
//		// 			act_host_float+base_addr_in, 
//		// 			weight_host_float, 
//		// 			act_host_float+base_addr_out, 
//		// 			bn_weight_host_float,
//		// 			act_host_float+base_addr_add,
//		// 			nky, nkx, nof, nif, noy, nox, stride, pad);
//		// }
//		// 
//		// // for debugging
//		// std::cout << "****************************************" << std::endl;
//		// std::cout << "layer_cnt: " << layer_cnt << std::endl;
//		// std::cout << "nif: " << nif << std::endl;
//		// std::cout << "nof: " << nof << std::endl;
//		// std::cout << "noy: " << noy << std::endl;
//		// std::cout << "nox: " << nox << std::endl;
//		// std::cout << "nkx: " << nkx << std::endl;
//		// std::cout << "nky: " << nky << std::endl;
//		// std::cout << "stride: " << stride << std::endl;
//		// std::cout << "pad: " << pad << std::endl;
//		// std::cout << "bb_en: " << bb_en << std::endl;
//		// std::cout << "conv_en: " << conv_en << std::endl;
//		// std::cout << "bn_en: " << bn_en << std::endl;
//		// std::cout << "skip_en: " << skip_en << std::endl;
//		// std::cout << "relu_en: " << relu_en << std::endl;
//		// std::cout << "max_pool_en: " << max_pool_en << std::endl;
//		// std::cout << "avg_pool_en: " << avg_pool_en << std::endl;
//		// std::cout << "fc_en: " << fc_en << std::endl;
//		// std::cout << "base_addr_in: " << base_addr_in << std::endl;
//		// std::cout << "base_addr_out: " << base_addr_out << std::endl;
//		// std::cout << "base_addr_add: " << base_addr_add << std::endl;
//		// std::cout << "weight_base: " << weight_base << std::endl;
//		// std::cout << "weight_size: " << weight_size << std::endl;
//		// std::cout << "bn_weight_base: " << bn_weight_base << std::endl;
//		// std::cout << "bn_weight_size: " << bn_weight_size << std::endl;
//		// std::cout << "in_size: " << in_size << std::endl;
//		// std::cout << "out_size: " << out_size << std::endl;
//		// std::cout << "****************************************" << std::endl;
//		// std::cout << std::endl;
//	}

	// kernel calculation
	// conv_kernel(act_in_host, act_out_host, weight_mem, bn_weight_mem, &start_layer, &end_layer);
	// for (int idx = 0; idx < out_size; idx++) {
	// 	std::cout << "idx: " << idx << "kernel out: " << act_out_host[idx] << std::endl;
	// }
	// compare host and kernel
	// compare_result<DTYPE_ACT, float, MAX_ACT_MEM_SIZE>(act_out_host, act_host_float+base_addr_out);

/*
	// fill data
	fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/data/weight2.bin";
	read_bin_fixed<DTYPE_FIL>(fname, weight_mem, BB7_CONV2_WEIGHT_BASE, BB7_CONV2_CONV_WEIGHT_SIZE);
	fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/data/weight3.bin";
	read_bin_fixed<DTYPE_FIL>(fname, weight_mem, BB7_SKIP_WEIGHT_BASE, BB7_SKIP_CONV_WEIGHT_SIZE);
	fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/data/bn_hw_weight2.bin";
	read_bin_float(fname, bn_weight_mem, BB7_CONV2_BN_WEIGHT_BASE, BB7_CONV2_BN_WEIGHT_SIZE);
	fname = "/home/junsang/projects/EE511/hw4/RESNET18_KV260/src/data/bn_hw_weight3.bin";
	read_bin_float(fname, bn_weight_mem, BB7_SKIP_BN_WEIGHT_BASE, BB7_SKIP_BN_WEIGHT_SIZE);

	std::cout << "first 3 input val: " << act_mem[0] << " " << act_mem[1] << " " << act_mem[2] << std::endl;
	std::cout << "first 3 weight val: " << weight_mem[0] << " " << weight_mem[1] << " " << weight_mem[2] << std::endl;
	std::cout << "first 3 bn_weight_mem val: " << bn_weight_mem[0] << " " << bn_weight_mem[1] << " " << bn_weight_mem[2] << std::endl;
	DTYPE_ACT act_out_host[MEM0_SIZE];
	conv_kernel(act_mem, weight_mem, bn_weight_mem, act_out_host);
	
	// golden conv gen
	float act_host_float[MEM0_SIZE+MEM1_SIZE+MEM2_SIZE];
	float fil_host_float[WEIGHT_MEM_SIZE];
	float bn_weight_host_float[BN_WEIGHT_MEM_SIZE];
	float out_act_host_float[OUTPUT_SIZE];
	for (int idx = 0; idx < INPUT_SIZE; idx++) {
		act_host_float[idx] = (float) act_mem[idx];
	}
	for (int idx = 0; idx < FILTER_SIZE; idx++) {
		fil_host_float[idx] = (float) weight_mem[idx];
	}
	for (int idx = 0; idx < BN_WEIGHT_SIZE; idx++) {
		bn_weight_host_float[idx] = bn_weight_mem[idx];
	}
 	// convolution_golden<float, float, float, float>(in_act_host_float, in_fil_host_float, out_act_host_float,
 	// 		BB7_CONV1_K, BB7_CONV1_K, BB7_CONV1_C, BB6_SKIP_C, BB7_CONV1_H, BB7_CONV1_W, BB7_CONV1_S, BB7_CONV1_PAD);
 	// convolution_bn_golden<float, float, float, float>(in_act_host_float, in_fil_host_float, out_act_host_float, bn_weight_host_float,
 	// 		BB7_CONV1_K, BB7_CONV1_K, BB7_CONV1_C, BB6_SKIP_C, BB7_CONV1_H, BB7_CONV1_W, BB7_CONV1_S, BB7_CONV1_PAD);
	// convolution_bn_skip_relu_golden<float, float, float, float>(in_act_host_float, in_fil_host_float, out_act_host_float, bn_weight_host_float, in_act_host_float,
	// 		BB7_CONV1_K, BB7_CONV1_K, BB7_CONV1_C, BB6_SKIP_C, BB7_CONV1_H, BB7_CONV1_W, BB7_CONV1_S, BB7_CONV1_PAD);
    // for (int idx = 0; idx < OUTPUT_SIZE; idx++) {
    //     if (out_act_host_float[idx] != act_mem[MEM0_SIZE+idx]){
	// 		std::cout << "idx: " << idx << " host: " << out_act_host_float[idx] << " kernel: " << act_mem[MEM0_SIZE+idx] << std::endl;
    //         result2 = 0;
    //     }
    // }
	convolution_bn_golden<float, float, float, float>(
			act_host_float, 
			fil_host_float+BB7_CONV1_WEIGHT_BASE, 
			act_host_float+MEM0_SIZE, 
			bn_weight_host_float+BB7_CONV1_BN_WEIGHT_BASE,
			BB7_CONV1_K, BB7_CONV1_K, BB7_CONV1_C, BB6_SKIP_C, BB7_CONV1_H, BB7_CONV1_W, BB7_CONV1_S, BB7_CONV1_PAD);
	for (int idx = 0; idx < BB7_CONV1_C*BB7_CONV1_H*BB7_CONV1_W; idx++) {
		act_host_float[MEM0_SIZE+idx] = (act_host_float[MEM0_SIZE+idx] > 0) ? act_host_float[MEM0_SIZE+idx] : 0;
	}
	convolution_bn_golden<float, float, float, float>(
			act_host_float+MEM0_SIZE,
			fil_host_float+BB7_CONV2_WEIGHT_BASE,
			act_host_float+MEM0_SIZE+MEM1_SIZE,
			bn_weight_host_float+BB7_CONV2_BN_WEIGHT_BASE,
			BB7_CONV2_K, BB7_CONV2_K, BB7_CONV2_C, BB7_CONV1_C, BB7_CONV2_H, BB7_CONV2_W, BB7_CONV2_S, BB7_CONV2_PAD);
	//convolution_bn_skip_relu_golden<float, float, float, float>(
	//		act_host_float,
	//		fil_host_float+BB7_SKIP_WEIGHT_BASE,
	//		act_host_float+MEM0_SIZE,
	//		bn_weight_host_float+BB7_SKIP_BN_WEIGHT_BASE,
	//		act_host_float+MEM0_SIZE+MEM1_SIZE,
	//		BB7_SKIP_K, BB7_SKIP_K, BB7_SKIP_C, BB7_CONV2_C, BB7_SKIP_H, BB7_SKIP_W, BB7_SKIP_S, BB7_SKIP_PAD);
	
	// compare_result<DTYPE_ACT, float, OUTPUT_SIZE>(act_out_host, act_host_float+MEM0_SIZE, 0.1);
	compare_result<DTYPE_ACT, float, OUTPUT_SIZE>(act_out_host, act_host_float+MEM0_SIZE+MEM1_SIZE, 0.1);
	for (int idx = 0; idx < 10; idx++) {
		std::cout << "output[" << idx << "]: " << act_out_host[idx] << std::endl;
	}
	*/
}
