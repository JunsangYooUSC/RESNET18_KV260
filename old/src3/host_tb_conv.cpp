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

// Include Vitis HLS headers
#include "ap_fixed.h"
// Include C++ headers
// Include project headers
#include "conv_config.h"
#include "host_utils.h"
#include "kernel.h"

// Print the configuration information
#define CHECK_CONFIG		1

// Function: golden convolution
template<typename D_ACT, typename D_FILTER, typename D_MULT, typename D_MAC>
void convolution_golden(D_ACT *in_act, D_FILTER *in_fil, D_ACT *out_act) {
	// create output MAC and initialize to 0
	D_MAC out_act_mac[TOTAL_OUT_LEN];
	for (int idx = 0; idx < TOTAL_OUT_LEN; idx++) {
		out_act_mac[idx] = 0;
	}
	
	for (int kdx = 0; kdx < NIF; kdx++) {
		for (int ndx = 0; ndx < NIY; ndx += STRIDE) {
			for (int mdx = 0; mdx < NIX; mdx += STRIDE) {
				for (int cdx = 0; cdx < NOF; cdx++) {
					for (int hdx = 0; hdx < NKY; hdx++) {
						for (int wdx = 0; wdx < NKX; wdx++) {

							D_ACT in_act_element;
							// when accessing zero padded index
							if ( (ndx + hdx < PAD) || (ndx + hdx >= NIY + PAD) || (mdx + wdx < PAD) || (mdx + wdx >= NIX + PAD) ) {
								in_act_element = 0;
							}
							else {
								// input address calc
								unsigned int in_addr = kdx * NIY * NIX;
								in_addr += (ndx + hdx - PAD) * NIX + (mdx + wdx - PAD);
								// load input
								in_act_element = in_act[in_addr];
								if (in_addr >= TOTAL_IN_LEN) {
									std::cout << "input index out-of-bounds: " << in_addr << std::endl;
								}
							}

							// filter address calc
							unsigned int filter_addr = cdx * NIF * NKY * NKX;
							filter_addr += kdx * NKY * NKX;
							filter_addr += hdx * NKX;
							filter_addr += wdx;
							// load filter
							D_FILTER in_fil_element = in_fil[filter_addr];
							if (filter_addr >= TOTAL_FIL_LEN) {
								std::cout << "filter index out-of-bounds: " << filter_addr << std::endl;
							}

							// mult
							D_MULT mult_element = in_act_element * in_fil_element;

							// output address calc
							unsigned int out_addr = cdx * NOY * NOX;
							out_addr += ndx / STRIDE * NOX + mdx / STRIDE;
							if (out_addr >= TOTAL_OUT_LEN) {
								std::cout << "fioutputlter index out-of-bounds: " << out_addr << std::endl;
							}
							// load output mac
							out_act_mac[out_addr] += mult_element;
						}
					}
				}
			}
		}
	}

	for (int idx = 0; idx < TOTAL_OUT_LEN; idx++) {
		out_act[idx] = out_act_mac[idx];
	}
}


int main(){
	// Print configuration information
#if CHECK_CONFIG
	print_conv_config();
	print_data_types();
	// Assertion to check the configuration
	assert((NIX % STRIDE == 0) && "Input width should be divisible by stride");
	assert((NIY / STRIDE == NOY) && "IN_HEIGHT/STRIDE should be same as OUT_HEIGHT");
	assert((NIX / STRIDE == NOX) && "IN_WIDTH/STRIDE should be same as OUT_WIDTH");
#endif

	// host-side data
	DTYPE_ACT in_act_host[TOTAL_IN_LEN];
	DTYPE_FIL in_fil_host[TOTAL_FIL_LEN];
	DTYPE_ACT out_act_host[TOTAL_OUT_LEN];
	float in_act_host_float[TOTAL_IN_LEN];
	float in_fil_host_float[TOTAL_FIL_LEN];
	float out_act_host_float[TOTAL_OUT_LEN];
	// generate random input activation and filter value with float
	gen_rand<DTYPE_ACT, TOTAL_IN_LEN>(in_act_host, -1, 1);
	gen_rand<DTYPE_FIL, TOTAL_FIL_LEN>(in_fil_host, -1, 1);

	/////////////////////////////////////////////////////////////////////////////
	// for (int idx = 0; idx < TOTAL_IN_LEN; idx++) {
	// 	in_act_host[idx] = 0.125;
	// }
	// std::cout << "in_act_host[0]: " << in_act_host[0] << std::endl;
	// for (int idx = 0; idx < TOTAL_FILTER_LEN; idx++) {
	// 	in_fil_host[idx] = idx%(FILTER_HEIGHT * FILTER_WIDTH);
	// }
	// std::cout << "in_fil_host[0]: " << in_fil_host[0] << std::endl;
	// std::cout << "in_fil_host[1]: " << in_fil_host[1] << std::endl;
	/////////////////////////////////////////////////////////////////////////////

	// copy input and filter from fixed point arrays to float arrays
	for (int idx = 0; idx < TOTAL_IN_LEN; idx++) {
		in_act_host_float[idx] = in_act_host[idx];
	}
	for (int idx = 0; idx < TOTAL_FIL_LEN; idx++) {
		in_fil_host_float[idx] = in_fil_host[idx];
	}

	// input debug
	std::cout << "input debug host\n";
	for (int idx = 0; idx < POY*STRIDE; idx+=STRIDE) {
		for (int jdx = 0; jdx < POX*STRIDE; jdx+=STRIDE) {
			int f = NIF - 1;
			int y = NIY - PIY + idx + NKY-1;
			int x = NIX - PIX + jdx + NKX-1;
			DTYPE_ACT val;
			if ( (y < PAD) || (y >= NIY + PAD) || (x < PAD) || (x >= NIX + PAD) ) {
				val = 0;
			}
			else {
				unsigned int act_idx = f*NIY*NIX+(y-PAD)*NIX+(x-PAD);
				val = in_act_host[act_idx];
			}
			std::cout << std::setw(5) << (val << 8) << " ";
		}
		std::cout << std::endl;
	}
	// weight debug
	std::cout << "filter debug host\n";
	for (int fo = 0; fo < POF; fo++) {
		unsigned int fi = NIF-1;
		unsigned int idx = NKY-1;
		unsigned int jdx = NKX-1;
		unsigned int fil_idx = (fo+NOF-POF)*NIF*NKY*NKX + fi*NKY*NKX + idx*NKX + jdx;
		std::cout << std::setw(5) << (in_fil_host[fil_idx] << 8) << " ";
	}
	std::cout << std::endl;
	// std::cout << "filter debug host all\n";
	// for (int fo = 0; fo < POF; fo++) {
	// 	for (int idx = 0; idx < NKY; idx++) {
	// 		for (int jdx = 0; jdx < NKX; jdx++) {
	// 			unsigned int fi = NIF-1;
	// 			unsigned int fil_idx = (fo+NOF-POF)*NIF*NKY*NKX + fi*NKY*NKX + idx*NKX + jdx;
	// 			std::cout << std::setw(5) << (in_fil_host[fil_idx] << 8) << " ";
	// 		}
	// 		std::cout << std::endl;
	// 	}
	// 	std::cout << std::endl;
	// }
	// std::cout << std::endl;

	// golden convolution result with fixed point and float
	convolution_golden<DTYPE_ACT, DTYPE_FIL, DTYPE_MUL, DTYPE_MAC>(in_act_host, in_fil_host, out_act_host);
	convolution_golden<float, float, float, float>(in_act_host_float, in_fil_host_float, out_act_host_float);

	std::cout << "out_act_host[0] = " << out_act_host[0] << std::endl;

	// compare with golden result
	compare_result<DTYPE_ACT, float, TOTAL_OUT_LEN>(out_act_host, out_act_host_float, 2.0/(1<<(W_ACT-I_ACT)));

	kernel_func(in_act_host, in_fil_host, out_act_host);
	compare_result<DTYPE_ACT, float, TOTAL_OUT_LEN>(out_act_host, out_act_host_float, 2.0/(1<<(W_ACT-I_ACT)));

	// print some results
	// std::cout << "in_act_host[0]:" << in_act_host[0] << std::endl;
	// std::cout << "in_act_host_float[0]:" << in_act_host_float[0] << std::endl;
	// std::cout << "in_act_host[1]:" << in_act_host[1] << std::endl;
	// std::cout << "in_act_host_float[1]:" << in_act_host_float[1] << std::endl;
	// std::cout << "out_act_host[0]:" << out_act_host[0] << std::endl;
	// std::cout << "out_act_host_float[0]:" << out_act_host_float[0] << std::endl;
	// std::cout << "out_act_host[1]:" << out_act_host[1] << std::endl;
	// std::cout << "out_act_host_float[1]:" << out_act_host_float[1] << std::endl;
}
