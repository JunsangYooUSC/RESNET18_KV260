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
#define CHECK_CONFIG		0

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
							}

							// filter address calc
							unsigned int filter_addr = cdx * NIF * NKY * NKX;
							filter_addr += kdx * NKY * NKX;
							filter_addr += hdx * NKX * wdx;
							// load filter
							D_FILTER in_fil_element = in_fil[filter_addr];

							// mult
							D_MULT mult_element = in_act_element * in_fil_element;

							// output address calc
							unsigned int out_addr = cdx * NOY * NOX;
							out_addr += ndx / STRIDE * NOX + mdx / STRIDE;
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
	print_PE_config();
	// Assertion to check the configuration
	assert(TOTAL_PE < 1000 && "Limit total PE under 1000");
	assert((IN_WIDTH % STRIDE == 0) && "Input width should be divisible by stride");
	assert((IN_HEIGHT / STRIDE == OUT_HEIGHT) && "IN_HEIGHT/STRIDE should be same as OUT_HEIGHT");
	assert((IN_WIDTH / STRIDE == OUT_WIDTH) && "IN_WIDTH/STRIDE should be same as OUT_WIDTH");
#endif

	DTYPE_ACT in_act_host[TOTAL_IN_LEN];
	DTYPE_ACT in_fil_host[TOTAL_FIL_LEN];
	DTYPE_ACT out_act_host[TOTAL_OUT_LEN];
	float in_act_host_float[TOTAL_IN_LEN];
	float in_fil_host_float[TOTAL_FIL_LEN];
	float out_act_host_float[TOTAL_OUT_LEN];
	// generate random input activation and filter value with float
	gen_rand<DTYPE_ACT, TOTAL_IN_LEN>(in_act_host, -1, 1);
	gen_rand<DTYPE_ACT, TOTAL_FIL_LEN>(in_fil_host, -1, 1);
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
	// copy input and filter to fixed point arrays
	for (int idx = 0; idx < TOTAL_IN_LEN; idx++) {
		in_act_host_float[idx] = in_act_host[idx];
	}
	for (int idx = 0; idx < TOTAL_IN_LEN; idx++) {
		in_fil_host_float[idx] = in_fil_host[idx];
	}

	// golden convolution result with fixed point and float
	convolution_golden<DTYPE_ACT, DTYPE_FILTER, DTYPE_MULT, DTYPE_MAC>(in_act_host, in_fil_host, out_act_host);
	convolution_golden<float, float, float, float>(in_act_host_float, in_fil_host_float, out_act_host_float);

	// compare with golden result

	compare_result<DTYPE_ACT, TOTAL_OUT_LEN>(out_act_host, out_act_host_float);

	DTYPE_ACT out_act_kernel[TOTAL_OUT_LEN];
	kernel_func(in_act_host, in_fil_host, out_act_kernel);

	compare_result<DTYPE_ACT, TOTAL_OUT_LEN>(out_act_kernel, out_act_host_float);
}
