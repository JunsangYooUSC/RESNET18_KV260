#ifndef KERNEL_H
#define KERNEL_H

// Include Vitis HLS headers
#include "ap_fixed.h"
// Include project headers
#include "conv_config.h"

// Declare functions
void load_input_burst(DTYPE_ACT *in_offchip, DTYPE_ACT in_onchip[FILTER_HEIGHT][PADDING + BURST * 2], 
                      unsigned int m0, unsigned int m0_onchip, unsigned int k, unsigned int n);

void load_filter(DTYPE_FILTER *filter_offchip, DTYPE_FILTER filter_onchip[PARALLEL_FILTERS][FILTER_HEIGHT][FILTER_WIDTH], 
                 unsigned int c0);

void store_output_burst(DTYPE_MAC out_onchip[PARALLEL_FILTERS][OUT_WIDTH], DTYPE_ACT *out_offchip, 
                        unsigned int c0, unsigned int n);

void calc_burst(DTYPE_ACT in_onchip[FILTER_HEIGHT][PADDING + BURST * 2],
                DTYPE_FILTER filter_onchip[PARALLEL_FILTERS][FILTER_HEIGHT][FILTER_WIDTH],
                DTYPE_MAC out_onchip[PARALLEL_FILTERS][OUT_WIDTH],
                unsigned int m0);

void kernel_func(DTYPE_ACT *in_offchip, DTYPE_ACT *filter_offchip, DTYPE_ACT *out_offchip);

#endif
