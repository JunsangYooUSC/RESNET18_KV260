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

void load_input(
    DTYPE_ACT *act_mem,
    hls::stream<DTYPE_ACT> load_input_fifo,
    unsigned int base_addr,
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox,
    unsigned int stride,
    unsigned int pad
) {

    for (int f_out = 0; f_out < nof; f_out += POF) {
        for (int y0 = 0; y0 < noy*stride; y0 += POY*stride) {
            for (int x0 = 0; x0 < nox*stride; x0 += POX*stride) {
                for (int f_in = 0; f_in < nif; f_in ++) {
                    for (int y = 0; y < POY*stride; y+=stride) {
                        for (int x = 0; x < POX*stride; x+=stride) {
                            for (int i = 0; i < nky; i++) {
                                for (int j = 0; j < nkx; j++) {
                                    DTYPE_ACT in_val;
                                    if ( (y0 + y + i < pad) || (y0 + y + i >= noy*stride + pad) || (x0 + x + j < pad) || (x0 + x + j >= pad) ){
                                        in_val = 0;
                                    }
                                    else {
                                        unsigned addr = f_in*noy*nox + (y0+y+i-pad)*nox + (x0+x+j-pad);
                                        in_val = act_mem[base_addr+addr];
                                    }
                                    load_input_fifo.write(in_val);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void store_input_test(
    DTYPE_ACT *act_mem,
    hls::stream<DTYPE_ACT> load_input_fifo,
    unsigned int base_addr,
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox,
    unsigned int stride,
    unsigned int pad
) {

    for (int f_out = 0; f_out < nof; f_out += POF) {
        for (int y0 = 0; y0 < noy*stride; y0 += POY*stride) {
            for (int x0 = 0; x0 < nox*stride; x0 += POX*stride) {
                for (int f_in = 0; f_in < nif; f_in ++) {
                    for (int y = 0; y < POY*stride; y+=stride) {
                        for (int x = 0; x < POX*stride; x+=stride) {
                            for (int i = 0; i < nky; i++) {
                                for (int j = 0; j < nkx; j++) {
                                    DTYPE_ACT in_val;
                                    in_val = load_input_fifo.read();
                                    if ( (y0 + y + i < pad) || (y0 + y + i >= noy*stride + pad) || (x0 + x + j < pad) || (x0 + x + j >= pad) ){
                                        in_val = 0;
                                    }
                                    else {
                                        unsigned addr = f_in*noy*nox + (y0+y+i-pad)*nox + (x0+x+j-pad);
                                        act_mem[base_addr+addr] = in_val;
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

void kernel(
    DTYPE_ACT *act_mem,
    DTYPE_FIL *weight_mem,
    float *bn_weight_mem
) {

    // interface
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS INTERFACE m_axi depth=16 port=act_mem bundle=gmem
    #pragma HLS INTERFACE m_axi depth=16 port=weight_mem bundle=gmem
    #pragma HLS INTERFACE m_axi depth=16 port=bn_weight_mem bundle=gmem

    // fill act_mem
    load_input(act_mem, load_input_fifo, 0,
            nky, nkx, nof, nif, noy, nox, stride, pad);
    store_input_test(act_mem, load_input_fifo, MEM0_SIZE,
            nky, nkx, nof, nif, noy, nox, stride, pad);

}
#endif
