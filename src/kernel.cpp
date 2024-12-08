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

// Include Vitis HLS headers
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_vector.h>
#include <hls_stream.h>
// Include project headers
#include "conv_config.h"
#include "kernel.h"

// BUF2PE
template<typename DTYPE, unsigned int LEN>
void BUF2PE(DTYPE_ACT *input_buffer, hls::stream<BUF2PEVEC> fifo_arr[POF][POY-1], hls::stream<BUF2PEVEC> mac_in_fifo_arr[POF][POY]) {
    std::cout << "hello" << std::endl;
}

// kernel function
void kernel_func(DTYPE_ACT *in_host,
                DTYPE_ACT *filter_offchip,
                DTYPE_ACT *out_host
) {
    // on-chip buffers
    DTYPE_ACT input_buffer[2*INPUT_BUFFER_SIZE];
    #pragma HLS BIND_STORAGE variable=input_buffer type=register
    DTYPE_ACT buf2pe_reg[2*BUF2PE_REG_SIZE];
    #pragma HLS BIND_STORAGE variable=buf2pe_reg type=register
    DTYPE_FIL filter_buffer[2*FILTER_BUFFER_SIZE];
    #pragma HLS BIND_STORAGE variable=filter_buffer type=register
    DTYPE_ACT output_buffer[2*OUTPUT_BUFFER_SIZE];
    #pragma HLS BIND_STORAGE variable=output_buffer type=register

    // fifo
    hls::stream<BUF2PEVEC> fifo_arr[POF][POY-1];
    #pragma HLS STREAM variable=fifo_arr depth=FIFO_ARR_DEPTH
    hls::stream<BUF2PEVEC> mac_in_fifo_arr[POF][POY];
    #pragma HLS STREAM variable=mac_in_fifo_arr depth=FIFO_ARR_DEPTH

    // on-chip memory
    DTYPE_MEM act_mem[ACT_MEM_SIZE];
    #pragma HLS bind_storage variable=act_mem impl=uram
    DTYPE_MEM fil_mem[FIL_MEM_SIZE];
    #pragma HLS bind_storage variable=fil_mem impl=bram

}

#endif
