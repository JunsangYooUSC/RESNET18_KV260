/******************************************************************************
 * Filename: kernel.cpp
 * Author: Junsang Yoo
 *
 * Description:
 * util functions for host
 *
 * Functions:
 * - load_input: load input from off-chip memory to on-chip memeory
 * - load_filter: load filter from off-chip memory to on-chip memory
 * - store_output: store output from on-chip memory to off-chip memory
 ******************************************************************************/

#ifndef KERNEL_CPP
#define KERNEL_CPP

#include <iomanip> // For std::setw

// Include Vitis HLS headers
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_vector.h>
#include <hls_stream.h>
// Include project headers
#include "conv_config.h"
#include "kernel.h"
#include "dummy_func.h"
#include "kernel_func.hpp"

// kernel function
void kernel_func(
    DTYPE_ACT *in_host,
    DTYPE_MEM_WEIGHT *weight_mem,
    DTYPE_ACT *out_host
) {
    // on-chip buffers
    // DTYPE_ACT input_buffer[2*INPUT_BUFFER_SIZE];
    // DTYPE_ACT input_buffer[2][POY+PAD*2][POX+PAD*2];
    // #pragma HLS BIND_STORAGE variable=input_buffer type=register
    // DTYPE_ACT input_buffer_stride[2][POY*MAX_STRIDE+PAD*2][POX*MAX_STRIDE+PAD*2];
    // #pragma HLS BIND_STORAGE variable=input_buffer_stride type=register
    // DTYPE_FIL filter_buffer[2*FILTER_BUFFER_SIZE];
    // #pragma HLS BIND_STORAGE variable=filter_buffer type=register
    // DTYPE_ACT output_buffer[2*OUTPUT_BUFFER_SIZE];
    // #pragma HLS BIND_STORAGE variable=output_buffer type=register

    // on-chip memory
    DTYPE_MEM_ACT mem0[MEM0_SIZE];
    #pragma HLS bind_storage variable=mem0 impl=uram
    #pragma HLS ARRAY_PARTITION variable=mem0 dim=1 complete
    DTYPE_MEM_ACT mem1[MEM1_SIZE];
    #pragma HLS bind_storage variable=mem1 impl=uram
    #pragma HLS ARRAY_PARTITION variable=mem1 dim=1 complete
    DTYPE_MEM_ACT mem2[MEM2_SIZE];
    #pragma HLS bind_storage variable=mem2 impl=uram
    #pragma HLS ARRAY_PARTITION variable=mem2 dim=1 complete

    // off-chip memory
    // DTYPE_MEM_WEIGHT weight_mem[WEIGHT_MEM_SIZE];
    #pragma HLS INTERFACE mode=m_axi port=weight_mem offset=slave bundle=gmem0
    #pragma HLS INTERFACE mode=s_axilite port=weight_mem bundle=control

    // interface
    #pragma HLS INTERFACE mode=m_axi port=in_host offset=slave bundle=gmem0
    #pragma HLS INTERFACE mode=m_axi port=out_host offset=slave bundle=gmem0
    #pragma HLS INTERFACE mode=s_axilite port=in_host bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=out_host bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

}

#endif
