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
    hls::stream<DTYPE_ACT> &load_input_fifo,
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
    int cnt = 0;
    for (int f_out = 0; f_out < nof; f_out += POF) {
        for (int y0 = 0; y0 < noy*stride; y0 += POY*stride) {
            for (int x0 = 0; x0 < nox*stride; x0 += POX*stride) {
                for (int f_in = 0; f_in < nif; f_in ++) {
                    for (int f = 0; f < POF; f++) {
                        for (int y = 0; y < POY*stride; y+=stride) {
                            for (int x = 0; x < POX*stride; x+=stride) {
                                for (int i = 0; i < nky; i++) {
                                    for (int j = 0; j < nkx; j++) {
                                        DTYPE_ACT in_val;
                                        if ( (y0 + y + i < pad) || (y0 + y + i >= noy*stride + pad) || (x0 + x + j < pad) || (x0 + x + j >= nox*stride + pad) ){
                                            in_val = 0;
                                        }
                                        else {
                                            unsigned addr = f_in*noy*nox + (y0+y+i-pad)*nox + (x0+x+j-pad);
                                            in_val = act_mem[base_addr+addr];
                                        }
                                        load_input_fifo.write(in_val);
                                        cnt ++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    std::cout << "load_input cnt: " << cnt << std::endl;
}

void store_input_test(
    DTYPE_ACT *act_mem,
    hls::stream<DTYPE_ACT> &load_input_fifo,
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
    int cnt = 0;
    for (int f_out = 0; f_out < nof; f_out += POF) {
        for (int y0 = 0; y0 < noy*stride; y0 += POY*stride) {
            for (int x0 = 0; x0 < nox*stride; x0 += POX*stride) {
                for (int f_in = 0; f_in < nif; f_in ++) {
                    for (int f = 0; f < POF; f++) {
                        for (int y = 0; y < POY*stride; y+=stride) {
                            for (int x = 0; x < POX*stride; x+=stride) {
                                for (int i = 0; i < nky; i++) {
                                    for (int j = 0; j < nkx; j++) {
                                        DTYPE_ACT in_val;
                                        in_val = load_input_fifo.read();
                                        if ( (y0 + y + i < pad) || (y0 + y + i >= noy*stride + pad) || (x0 + x + j < pad) || (x0 + x + j >= nox*stride + pad) ){
                                            in_val = 0;
                                        }
                                        else {
                                            unsigned addr = f_in*noy*nox + (y0+y+i-pad)*nox + (x0+x+j-pad);
                                            act_mem[base_addr+addr] = in_val;
                                        }
                                        cnt++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    std::cout << "store_input cnt: " << cnt << std::endl;    
}

void load_weight(
    DTYPE_FIL *weight_mem,
    hls::stream<DTYPE_FIL> &load_weight_fifo,
    unsigned int base_addr,
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox
) {
    int cnt = 0; 
    for (int f_out = 0; f_out < nof; f_out += POF) {
        for (int y0 = 0; y0 < noy; y0 += POY) {
            for (int x0 = 0; x0 < nox; x0 += POX) {
                for (int f_in = 0; f_in < nif; f_in ++) {
                    for (int f = 0; f < POF; f++) {
                        for (int y = 0; y < POY; y++) {
                            for (int x = 0; x < POX; x++) {
                                for (int i = 0; i < nky; i++) {
                                    for (int j = 0; j < nkx; j++) {
                                        unsigned int addr = (f_out+f)*nif*nky*nkx + f_in*nky*nkx + i*nky + j;
                                        load_weight_fifo.write(weight_mem[addr]);
                                        cnt++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    std::cout << "load_weight cnt: " << cnt << std::endl;
}

void PE(
    hls::stream<DTYPE_ACT> &load_input_fifo,
    hls::stream<DTYPE_FIL> &load_weight_fifo,
    hls::stream<float> &pe_out_fifo,
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox
) {
    int cnt = 0;
    int cnt1 = 0;
    DTYPE_MAC mac_vals[POF][POY][POX];
    for (int f_out = 0; f_out < nof; f_out += POF) {
        for (int y0 = 0; y0 < noy; y0 += POY) {
            for (int x0 = 0; x0 < nox; x0 += POX) {
                // init mac
                for (int f = 0; f < POF; f++) {
                    for (int y = 0; y < POY; y++) {
                        for (int x = 0; x < POX; x++) {
                            mac_vals[f][y][x] = 0;
                        }
                    }
                }
                // calc 
                for (int f_in = 0; f_in < nif; f_in ++) {
                    for (int f = 0; f < POF; f++) {
                        for (int y = 0; y < POY; y++) {
                            for (int x = 0; x < POX; x++) {
                                for (int i = 0; i < nky; i++) {
                                    for (int j = 0; j < nkx; j++) {
                                        DTYPE_ACT act_in = load_input_fifo.read();
                                        DTYPE_FIL fil_in = load_weight_fifo.read();
                                        DTYPE_MUL mul_val = act_in * fil_in;
                                        mac_vals[f][y][x] += mul_val;
                                        cnt++;
                                    }
                                }
                            }
                        }
                    }
                }
                // pass to fifo
                for (int f = 0; f < POF; f++) {
                    for (int y = 0; y < POY; y++) {
                        for (int x = 0; x < POX; x++) {
                            unsigned int addr = (f_out+f)*noy*nox + y*nox + x;
                            pe_out_fifo.write(mac_vals[f][y][x]);
                            cnt1++;
                        }
                    }
                }
            }
        }
    }
    std::cout << "PE cnt: " << cnt << std::endl;
    std::cout << "PE output cnt: " << cnt1 << std::endl;
}

void store_output_fifo(
    DTYPE_ACT *act_mem,
    hls::stream<float> &out_fifo_arr,
    unsigned int base_addr,
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox
) {
    int cnt = 0;
    int cnt1 = 0;
    for (int f_out = 0; f_out < nof; f_out += POF) {
        for (int y0 = 0; y0 < noy; y0 += POY) {
            for (int x0 = 0; x0 < nox; x0 += POX) {
                for (int f_in = 0; f_in < nif; f_in ++) {
                    for (int f = 0; f < POF; f++) {
                        for (int y = 0; y < POY; y++) {
                            for (int x = 0; x < POX; x++) {
                                for (int i = 0; i < nky; i++) {
                                    for (int j = 0; j < nkx; j++) {
                                        cnt++;
                                    }
                                }
                            }
                        }
                    }
                }
                for (int f = 0; f < POF; f++) {
                    for (int y = 0; y < POY; y++) {
                        for (int x = 0; x < POX; x++) {
                            unsigned int addr = (f_out+f)*noy*nox + (y0+y)*nox + (x0+x);
                            act_mem[base_addr+addr] = out_fifo_arr.read();
                            cnt1++;
                        }
                    }
                }
            }
        }
    }
    std::cout << "store cnt: " << cnt << std::endl;
    std::cout << "store cnt1: " << cnt1 << std::endl;
}

void conv_kernel(
    DTYPE_ACT *act_mem,
    DTYPE_FIL *weight_mem,
    float *bn_weight_mem,
    int *result1,
    int *result2
) {

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
    unsigned weight_base;
    unsigned weight_size;
    unsigned bn_weight_base;
    unsigned bn_weight_size;
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
    weight_base     = BB7_CONV1_WEIGHT_BASE;
    weight_size     = BB7_CONV1_CONV_WEIGHT_SIZE;
    bn_weight_base  = BB7_CONV1_BN_WEIGHT_BASE;
    bn_weight_size  = BB7_CONV1_BN_WEIGHT_SIZE;

    // interface
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS INTERFACE mode=m_axi port=act_mem offset=slave bundle=gmem0 depth = 10000000
    #pragma HLS INTERFACE mode=m_axi port=weight_mem offset=slave bundle=gmem0 depth = 10000000
    #pragma HLS INTERFACE mode=m_axi port=bn_weight_mem offset=slave bundle=gmem0 depth = 10000000
    #pragma HLS INTERFACE mode=m_axi port=result1 offset=slave bundle=gmem0 depth = 1
    #pragma HLS INTERFACE mode=m_axi port=result2 offset=slave bundle=gmem0 depth = 1
    #pragma HLS INTERFACE mode=s_axilite port=act_mem bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=weight_mem bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=bn_weight_mem bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=result1 bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=result2 bundle=control

    // fifo
    hls::stream<DTYPE_ACT> load_input_fifo;
    #pragma HLS STREAM variable=load_input_fifo depth=FIFO_ARR_DEPTH
    hls::stream<DTYPE_FIL> load_weight_fifo;
    #pragma HLS STREAM variable=load_weight_fifo depth=FIFO_ARR_DEPTH
    hls::stream<float> pe_out_fifo;
    #pragma HLS STREAM variable=pe_out_fifo depth=FIFO_ARR_DEPTH

    // load input check
    load_input(act_mem, load_input_fifo, 0,
            nky, nkx, nof, nif, noy, nox, stride, pad);
    store_input_test(act_mem, load_input_fifo, MEM0_SIZE,
            nky, nkx, nof, nif, noy, nox, stride, pad);
    (*result1) = 1;
    for (int idx = 0; idx < nif*noy*nox; idx++) {
        if (act_mem[idx] != act_mem[MEM0_SIZE+idx]){
            (*result1) = 0;
        }
    }

    load_input(act_mem, load_input_fifo, 0,
            nky, nkx, nof, nif, noy, nox, stride, pad);
    load_weight(weight_mem, load_weight_fifo, 0,
            nky, nkx, nof, nif, noy, nox);
    PE(load_input_fifo, load_weight_fifo, pe_out_fifo,
            nky, nkx, nof, nif, noy, nox);
    store_output_fifo(act_mem, pe_out_fifo, MEM0_SIZE, 
            nky, nkx, nof, nif, noy, nox);
    (*result2) = 1;
}

#endif
