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

#ifndef KERNEL_TEST_CPP
#define KERNEL_TEST_CPP

#include <iomanip> // For std::setw

// Include Vitis HLS headers
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_vector.h>
#include <hls_stream.h>
// Include project headers
#include "conv_config.h"
#include "kernel.h"
#include "kernel_func.hpp"
#include "utils.h"

// kernel function
void kernel_test_func(
    DTYPE_ACT *in_host,
    DTYPE_FIL *weight_mem,
    float *bn_weight_mem,
    // DTYPE_MEM_WEIGHT *weight_mem,
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

    // parameters for kernel functions
    DTYPE_MEM_ACT *mem_in;
    DTYPE_MEM_ACT *mem_out;
    DTYPE_MEM_ACT *mem_add;
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
    bool lin_en;
    nif             = BB6_SKIP_C;
    nof             = BB7_CONV1_C;
    noy             = BB7_CONV1_H;
    nox             = BB7_CONV1_W;
    nkx             = BB7_CONV1_K;
    nky             = BB7_CONV1_K;
    stride          = BB7_CONV1_S;
    pad             = BB7_CONV1_PAD;
    bb_en           = BB7_CONV1_BB_EN;
    conv_en         = BB7_CONV1_CONV_EN;
    bn_en           = BB7_CONV1_BN_EN;
    skip_en         = BB7_CONV1_SKIP_EN;
    relu_en         = BB7_CONV1_RELU_EN;
    max_pool_en     = BB7_CONV1_MAX_POOL;
    avg_pool_en     = BB7_CONV1_AVG_POOL;
    lin_en          = BB7_CONV1_LIN_EN;
    mem_in          = mem0;
    mem_out         = mem1;
    unsigned niy = noy*stride;
    unsigned nix = nox*stride;
    

    // load mem0, mem1, mem2 for testing
    for (int idx = 0; idx < nif*niy*nix/POX; idx++) {
        DTYPE_MEM_ACT block;
        for (int x = 0; x < POX; x++) {
            DTYPE_ACT val;
            block.range(W_ACT*(x+1)-1, W_ACT*(x)) = in_host[idx*POX+x].range();
        }
        mem0[idx] = block;
    }

    hls::stream<DTYPE_MAC> out_fifo_arr[POF][POY][POX];
    #pragma HLS STREAM variable=out_fifo_arr depth=FIFO_ARR_DEPTH
    // conv pass test
    bb_en = 1;
    conv_en = 0;
    conv(mem0, weight_mem, out_fifo_arr,
            0, nky, nkx, nof, nif, noy, nox, stride, pad, bb_en, conv_en);
    
    // consume
    for (int out_f = 0; out_f < nof; out_f+=POF) {
        for (int y0 = 0; y0 < noy; y0+=POY) {
            for (int x0 = 0; x0 < nox; x0+=POX) {
                for (int f = 0; f < POF; f++) {
                    for (int y = 0; y < POY; y++) {
                        for (int x = 0; x < POX; x++) {
                            out_fifo_arr[f][y][x].read();
                        }
                    }
                }
            }
        }
    }

    // conv test
    bb_en = 1;
    conv_en = 1;
    conv(mem0, weight_mem, out_fifo_arr,
            0, nky, nkx, nof, nif, noy, nox, stride, pad, bb_en, conv_en);
    // consume
    for (int out_f = 0; out_f < nof; out_f+=POF) {
        for (int y0 = 0; y0 < noy; y0+=POY) {
            for (int x0 = 0; x0 < nox; x0+=POX) {
                for (int f = 0; f < POF; f++) {
                    for (int y = 0; y < POY; y++) {
                        for (int x = 0; x < POX; x++) {
                            out_fifo_arr[f][y][x].read();
                        }
                    }
                }
            }
        }
    }
}
#endif