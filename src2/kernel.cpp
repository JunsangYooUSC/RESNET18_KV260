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

    std::cout << "in_host[0]: " << (in_host[0] << (W_ACT-I_ACT)) << std::endl;
    std::cout << "in_host[" << TOTAL_IN_LEN-4 << "]: " << (in_host[TOTAL_IN_LEN-4] << (W_ACT-I_ACT)) << std::endl;
    std::cout << "in_host[" << TOTAL_IN_LEN-3 << "]: " << (in_host[TOTAL_IN_LEN-3] << (W_ACT-I_ACT)) << std::endl;
    std::cout << "in_host[" << TOTAL_IN_LEN-2 << "]: " << (in_host[TOTAL_IN_LEN-2] << (W_ACT-I_ACT)) << std::endl;
    std::cout << "in_host[" << TOTAL_IN_LEN-1 << "]: " << (in_host[TOTAL_IN_LEN-1] << (W_ACT-I_ACT)) << std::endl;
    
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
        fil_mem[idx] = filter_host[idx];
    }

    unsigned int nky = NKY;
    unsigned int nkx = NKX;
    unsigned int nof = NOF;
    unsigned int nif = NIF;
    unsigned int noy = NOY;
    unsigned int nox = NOX;    
    unsigned int s = STRIDE;
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

    // buf2pe test
    hls::stream<DTYPE_ACT> in_fifo_arr[POY][POX];
    BUF2PE_stride(act_mem, in_fifo_arr,
            nky, nkx, nof, nif, noy, nox, s, pad, 0);
    for (int f_out = 0; f_out < nof; f_out+=POF) {
        for (int y0 = 0; y0 < noy; y0 += PIY) {
            for (int x0 = 0; x0 < nox; x0 += PIX) {
                for (int f_in = 0; f_in < nif; f_in ++) {
                    // parallel
                    for (int f = 0; f < POF; f++) {
                        for (int y = 0; y < PIY; y+=STRIDE) {
                            for (int x = 0; x < PIX; x+=STRIDE) {
                                for (int i = 0; i < NKY; i++) {
                                    for (int j = 0; j < NKX; j++){
                                        unsigned int yidx = y0 + y + i;
                                        unsigned int xidx = x0 + x + j;
                                        DTYPE_ACT val1;
                                        if ( (yidx < PAD) || (yidx >= NIY+PAD) || (xidx < PAD) || (xidx >= NIX+PAD) ) {
                                            val1 = 0;
                                        }
                                        else {
                                            int in_mem_idx;
                                            in_mem_idx = f_in*NIY*NIX + (yidx-PAD)*NIX + (xidx-PAD);
                                            int idx1 = in_mem_idx / MEM_PACK;
                                            int idx2 = in_mem_idx % MEM_PACK;
                                            DTYPE_MEM block = act_mem[0][idx1];
                                            val1.range() = block.range(W_ACT*(idx2+1)-1, W_ACT*(idx2));
                                        }
                                        DTYPE_ACT val2 = in_fifo_arr[y+i][x+i].read();
                                        if (val1 != val2) {
                                            std::cout << "f_in: " << std::setw(5) << f_in << " ";
                                            std::cout << "y: " << std::setw(5) << y0+y+i << " ";
                                            std::cout << "x: " << std::setw(5) << x0+x+j << " ";
                                            std::cout << "val1: " << std::setw(5) << val1 << " ";
                                            std::cout << "val2: " << std::setw(5) << val2 << std::endl;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#endif
