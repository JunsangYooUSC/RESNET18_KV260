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
                                            // unsigned addr = f_in*noy*nox + (y0+y+i-pad)*nox + (x0+x+j-pad);
                                            unsigned addr = f_in*noy*stride*nox*stride + (y0+y+i-pad)*nox*stride + (x0+x+j-pad);
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
                                            // unsigned addr = f_in*noy*nox + (y0+y+i-pad)*nox + (x0+x+j-pad);
                                            unsigned addr = f_in*noy*stride*nox*stride + (y0+y+i-pad)*nox*stride + (x0+x+j-pad);
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
                        }
                    }
                }
            }
        }
    }
}

void store_output(
    DTYPE_ACT *act_mem,
    hls::stream<float> &out_fifo,
    unsigned int base_addr,
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox
) {
    for (int f_out = 0; f_out < nof; f_out += POF) {
        for (int y0 = 0; y0 < noy; y0 += POY) {
            for (int x0 = 0; x0 < nox; x0 += POX) {
                for (int f = 0; f < POF; f++) {
                    for (int y = 0; y < POY; y++) {
                        for (int x = 0; x < POX; x++) {
                            unsigned int addr = (f_out+f)*noy*nox + (y0+y)*nox + (x0+x);
                            act_mem[base_addr+addr] = out_fifo.read();
                        }
                    }
                }
            }
        }
    }
}

void batch_norm(
    float *bn_weight_mem,
    hls::stream<float> &in_fifo,
    hls::stream<float> &out_fifo,
    unsigned int bn_weight_base_addr,
    unsigned int nof,
    unsigned int noy,
    unsigned int nox,
    unsigned int bb_en,
    unsigned int bn_en
) {
    if (!bb_en) return;
    
    float mean;
    float mult_factor;
    float beta;
    for (int f_out = 0; f_out < nof; f_out+=POF) {
        for (int y0 = 0; y0 < noy; y0+=POY) {
            for (int x0 = 0; x0 < nox; x0+=POX) {
                // parallel
                for (int f = 0; f < POF; f++) {
                    mean = bn_weight_mem[(f_out+f)];
                    mult_factor = bn_weight_mem[nof+(f_out+f)];
                    beta = bn_weight_mem[nof*2+(f_out+f)];
                    for (int y = 0; y < POY; y++) {
                        for (int x = 0; x < POX; x++) {
                            float val;
                            val = in_fifo.read();
                            // batch norm when enabled
                            if (bn_en) {
                                val = (val-mean)*mult_factor+beta;
                            }
                            out_fifo.write(val);
                        }
                    }
                }

            }
        }
    }
}

void skip_conn(
    DTYPE_ACT *act_mem,
    hls::stream<float> &in_fifo,
    hls::stream<float> &out_fifo,
    unsigned int base_addr,
    unsigned int nof,
    unsigned int noy,
    unsigned int nox,
    unsigned int bb_en,
    unsigned int skip_en,
    unsigned int relu_en
) {
    if (!bb_en) return;

    for (int f_out = 0; f_out < nof; f_out+=POF) {
        for (int y0 = 0; y0 < noy; y0+=POY) {
            for (int x0 = 0; x0 < nox; x0+=POX) {
                for (int f = 0; f < POF; f++) {
                    for (int y = 0; y < POY; y++) {
                        for (int x = 0; x < POX; x++) {
                            float val = in_fifo.read();
                            if (skip_en) {
                                unsigned add_addr = (f_out+f)*noy*nox + (y0+y)*nox + x0+x;
                                DTYPE_ACT add_val = act_mem[base_addr + add_addr];
                                val = val + (float)add_val;
                            }
                            if (relu_en) {
                                val = (val > 0) ? val : 0;
                            }
                            out_fifo.write(val);
                        }
                    }
                }
            }
        }
    }
}

void conv_kernel(
    DTYPE_ACT *act_mem_host,
    DTYPE_FIL *weight_mem,
    float *bn_weight_mem,
    DTYPE_ACT *act_out_host
) {
    DTYPE_ACT act_mem[MEM0_SIZE+MEM1_SIZE+MEM2_SIZE];
    // interface
    // #pragma HLS INTERFACE s_axilite port=return bundle=control
    // #pragma HLS INTERFACE mode=m_axi port=act_mem offset=slave bundle=gmem0 depth = 10000000
    // #pragma HLS INTERFACE mode=m_axi port=weight_mem offset=slave bundle=gmem0 depth = 10000000
    // #pragma HLS INTERFACE mode=m_axi port=bn_weight_mem offset=slave bundle=gmem0 depth = 10000000
    // #pragma HLS INTERFACE mode=m_axi port=result1 offset=slave bundle=gmem0 depth = 1
    // #pragma HLS INTERFACE mode=m_axi port=result2 offset=slave bundle=gmem0 depth = 1
    // #pragma HLS INTERFACE mode=s_axilite port=act_mem bundle=control
    // #pragma HLS INTERFACE mode=s_axilite port=weight_mem bundle=control
    // #pragma HLS INTERFACE mode=s_axilite port=bn_weight_mem bundle=control
    // #pragma HLS INTERFACE mode=s_axilite port=result1 bundle=control
    // #pragma HLS INTERFACE mode=s_axilite port=result2 bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS INTERFACE mode=m_axi port=act_mem_host offset=slave bundle=gmem0 depth = 10000000
    #pragma HLS INTERFACE mode=m_axi port=weight_mem offset=slave bundle=gmem0 depth = 10000000
    #pragma HLS INTERFACE mode=m_axi port=bn_weight_mem offset=slave bundle=gmem0 depth = 10000000
    #pragma HLS INTERFACE mode=s_axilite port=act_mem_host bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=weight_mem bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=bn_weight_mem bundle=control
    
    #pragma HLS BIND_STORAGE variable=act_mem type=ram_2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=act_mem block factor=8

    // fifo
    hls::stream<DTYPE_ACT> load_input_fifo;
    #pragma HLS STREAM variable=load_input_fifo depth=FIFO_ARR_DEPTH
    hls::stream<DTYPE_FIL> load_weight_fifo;
    #pragma HLS STREAM variable=load_weight_fifo depth=FIFO_ARR_DEPTH
    hls::stream<float> pe_out_fifo;
    #pragma HLS STREAM variable=pe_out_fifo depth=FIFO_ARR_DEPTH
    hls::stream<float> bn_out_fifo;
    #pragma HLS STREAM variable=bn_out_fifo depth=FIFO_ARR_DEPTH
    hls::stream<float> skip_out_fifo;
    #pragma HLS STREAM variable=skip_out_fifo depth=FIFO_ARR_DEPTH
    
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
    unsigned base_addr_in;
    unsigned base_addr_out;
    unsigned base_addr_add;
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
    base_addr_in    = 0;
    base_addr_out   = MEM0_SIZE;
    base_addr_add   = 0;
    weight_base     = BB7_CONV1_WEIGHT_BASE;
    weight_size     = BB7_CONV1_CONV_WEIGHT_SIZE;
    bn_weight_base  = BB7_CONV1_BN_WEIGHT_BASE;
    bn_weight_size  = BB7_CONV1_BN_WEIGHT_SIZE;

    int loops = 2;
    for (int opcnt = 0; opcnt < loops; opcnt++) {
        if (opcnt == 0) {
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
            base_addr_in    = 0;
            base_addr_out   = MEM0_SIZE;
            base_addr_add   = 0;
            weight_base     = BB7_CONV1_WEIGHT_BASE;
            weight_size     = BB7_CONV1_CONV_WEIGHT_SIZE;
            bn_weight_base  = BB7_CONV1_BN_WEIGHT_BASE;
            bn_weight_size  = BB7_CONV1_BN_WEIGHT_SIZE;
        } 
        else if (opcnt == 1) {
            nif             = BB7_CONV1_C;
            nof             = BB7_CONV2_C;
            noy             = BB7_CONV2_H;
            nox             = BB7_CONV2_W;
            nkx             = BB7_CONV2_K;
            nky             = BB7_CONV2_K;
            stride          = BB7_CONV2_S;
            pad             = BB7_CONV2_PAD;
            bb_en           = BB7_CONV2_BB_EN;
            conv_en         = BB7_CONV2_CONV_EN;
            bn_en           = BB7_CONV2_BN_EN;
            skip_en         = BB7_CONV2_SKIP_EN;
            relu_en         = BB7_CONV2_RELU_EN;
            max_pool_en     = BB7_CONV2_MAX_POOL;
            avg_pool_en     = BB7_CONV2_AVG_POOL;
            lin_en          = BB7_CONV2_LIN;
            base_addr_in    = MEM0_SIZE;
            base_addr_out   = MEM0_SIZE+MEM1_SIZE;
            base_addr_add   = 0;
            weight_base     = BB7_CONV2_WEIGHT_BASE;
            weight_size     = BB7_CONV2_CONV_WEIGHT_SIZE;
            bn_weight_base  = BB7_CONV2_BN_WEIGHT_BASE;
            bn_weight_size  = BB7_CONV2_BN_WEIGHT_SIZE;
        }
        else if (opcnt == 2) {
            nif             = BB7_CONV2_C;
            nof             = BB7_SKIP_C;
            noy             = BB7_SKIP_H;
            nox             = BB7_SKIP_W;
            nkx             = BB7_SKIP_K;
            nky             = BB7_SKIP_K;
            stride          = BB7_SKIP_S;
            pad             = BB7_SKIP_PAD;
            bb_en           = BB7_SKIP_BB_EN;
            conv_en         = BB7_SKIP_CONV_EN;
            bn_en           = BB7_SKIP_BN_EN;
            skip_en         = BB7_SKIP_SKIP_EN;
            relu_en         = BB7_SKIP_RELU_EN;
            max_pool_en     = BB7_SKIP_MAX_POOL;
            avg_pool_en     = BB7_SKIP_AVG_POOL;
            lin_en          = BB7_SKIP_LIN;
            base_addr_in    = 0;
            base_addr_out   = MEM0_SIZE;
            base_addr_add   = MEM0_SIZE+MEM1_SIZE;
            weight_base     = BB7_SKIP_WEIGHT_BASE;
            weight_size     = BB7_SKIP_CONV_WEIGHT_SIZE;
            bn_weight_base  = BB7_SKIP_BN_WEIGHT_BASE;
            bn_weight_size  = BB7_SKIP_BN_WEIGHT_SIZE;
        }

        // initial input
        if (opcnt == 0) {
            // load input
            for (int idx = 0; idx < nif*noy*nox*stride*stride; idx++){
                act_mem[idx] = act_mem_host[idx];
            }
        }

        // conv
        load_input(act_mem, load_input_fifo, base_addr_in,
                nky, nkx, nof, nif, noy, nox, stride, pad);
        load_weight(weight_mem, load_weight_fifo, weight_base,
                nky, nkx, nof, nif, noy, nox);
        PE(load_input_fifo, load_weight_fifo, pe_out_fifo,
                nky, nkx, nof, nif, noy, nox);
        batch_norm(bn_weight_mem, pe_out_fifo, bn_out_fifo, bn_weight_base,
                nof, noy, nox, bb_en, bn_en);
        skip_conn(act_mem, bn_out_fifo, skip_out_fifo, base_addr_add,
                nof, noy, nox, bb_en, skip_en, relu_en);
        store_output(act_mem, skip_out_fifo, base_addr_out, 
                nky, nkx, nof, nif, noy, nox);
        // output back to host
        if (opcnt == loops-1) {
            for (int idx = 0; idx < nof*noy*nox; idx++){
                // act_mem_host[MEM0_SIZE+idx] = act_mem[MEM0_SIZE+idx];
                act_out_host[idx] = act_mem[base_addr_out+idx];
                // act_mem_host[MEM0_SIZE+MEM1_SIZE+idx] = act_mem[MEM0_SIZE+MEM1_SIZE+idx];
            }
        }
    }

}

#endif
