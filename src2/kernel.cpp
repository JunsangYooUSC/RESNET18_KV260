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
void kernel_func(DTYPE_ACT *in_host,
                DTYPE_FIL *filter_host,
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
    DTYPE_MEM act_mem[2][ACT_MEM_SIZE];
    #pragma HLS bind_storage variable=act_mem impl=uram
    DTYPE_FIL fil_mem[FIL_MEM_SIZE];
    // DTYPE_MEM fil_mem[2][FIL_MEM_SIZE];
    // #pragma HLS bind_storage variable=fil_mem impl=bram

    // fifo
    hls::stream<DTYPE_ACT> mac_in_fifo_arr[POY][POX];
    #pragma HLS STREAM variable=mac_in_fifo_arr depth=FIFO_ARR_DEPTH
    hls::stream<DTYPE_FIL> weight_in_fifo_arr[POF];
    #pragma HLS STREAM variable=weight_in_fifo_arr depth=FIFO_ARR_DEPTH
    hls::stream<DTYPE_MAC> out_fifo_arr[POF][POY][POX];
    #pragma HLS STREAM variable=out_fifo_arr depth=FIFO_ARR_DEPTH

    // dummy function to fill input buffer
    // dummy_fill_input_buffer(input_buffer);

    std::cout << "in_host[0]: " << in_host[0] << std::endl;

    // load in_host to act_mem
    DTYPE_MEM block;
    for (int idx = 0; idx < TOTAL_IN_LEN; idx++) {
        unsigned int idx2 = idx % MEM_PACK;
        block.range(W_ACT*(idx2+1)-1, W_ACT*(idx2)) = in_host[idx].range();
        if (idx2 == MEM_PACK-1) {
            act_mem[0][idx/MEM_PACK] = block;
        }
    }

    // load filter_host to fil_mem
    for (int idx = 0; idx < TOTAL_FIL_LEN; idx++) {
        // fil_mem[idx] = filter_host[idx];
        fil_mem[idx] = idx % I_FIL;
    }

    unsigned int nky = NKY;
    unsigned int nkx = NKX;
    unsigned int nof = NOF;
    unsigned int nif = NIF;
    unsigned int noy = NOY;
    unsigned int nox = NOX;    
    unsigned int s = 1;
    unsigned int pad = PAD;
    

    BUF2PE_stride(act_mem, mac_in_fifo_arr,
            nky, nkx, nof, nif, noy, nox, s, pad, 0);
    load_weight_fifo(fil_mem, weight_in_fifo_arr,
            nky, nkx, nof, nif, noy, nox);
    PE(mac_in_fifo_arr, weight_in_fifo_arr, out_fifo_arr,
            nky, nkx, nof, nif, noy, nox);
    store_output_fifo(act_mem, out_fifo_arr,
            nky, nkx, nof, nif, noy, nox, 1);

    // DTYPE_ACT val;
    // val.range() = act_mem[1][0].range(W_ACT-1,0)
	// std::cout << "kernel output[0] = " << val[0] << std::endl;

    // write out_host from act_mem
    for (int idx = 0; idx < TOTAL_OUT_LEN; idx++) {
        unsigned int idx2 = idx % MEM_PACK;
        if (idx%MEM_PACK == 0) {
            block = act_mem[1][idx/MEM_PACK];
        }
        out_host[idx].range() = block.range(W_ACT*(idx2+1)-1, W_ACT*(idx2));
    }
}

#endif
