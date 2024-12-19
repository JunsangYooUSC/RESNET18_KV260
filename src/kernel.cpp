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
    DTYPE_MEM_ACT *act_mem,
    // hls::stream<DTYPE_ACT> &load_input_fifo,
    hls::stream<DTYPE_ACT> &load_input_fifo,
    unsigned int base_addr,
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox,
    unsigned int stride,
    unsigned int pad,
    unsigned int bb_en,
    unsigned int conv_en
) {
    if (!bb_en) return;
    if (!conv_en) {
        for (int f_out = 0; f_out < nof; f_out += POF) {
            for (int y0 = 0; y0 < noy; y0 += POY) {
                for (int x0 = 0; x0 < nox; x0 += POX) {
                    for (int f = 0; f < POF; f++) {
                        for (int y = 0; y < POY; y++) {
                            for (int x = 0; x < POX; x++) {
                                DTYPE_ACT in_val;
                                unsigned addr = (f_out+f)*noy*nox + (y0+y)*nox + x0+x;
                                in_val = act_mem[base_addr+addr/ACT_PACK].range((addr%ACT_PACK+1)*W_ACT-1, (addr%ACT_PACK)*W_ACT);
                                load_input_fifo.write(in_val);
                            }
                        }
                    }
                }
            }
        }
    }
    else {
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
                                                in_val = act_mem[base_addr+addr/ACT_PACK].range((addr%ACT_PACK+1)*W_ACT-1, (addr%ACT_PACK)*W_ACT);
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
    unsigned int nox,
    unsigned int bb_en,
    unsigned int conv_en
) {
    if (!bb_en) return;
    if (!conv_en) return;

    for (int f_out = 0; f_out < nof; f_out += POF) {
        for (int y0 = 0; y0 < noy; y0 += POY) {
            for (int x0 = 0; x0 < nox; x0 += POX) {
                for (int f_in = 0; f_in < nif; f_in ++) {
                    for (int f = 0; f < POF; f++) {
                        for (int y = 0; y < POY; y++) {
                            for (int x = 0; x < POX; x++) {
                                for (int i = 0; i < nky; i++) {
                                    for (int j = 0; j < nkx; j++) {
                                        unsigned int addr = (f_out+f)*nif*nky*nkx + f_in*nky*nkx + i*nkx + j;
                                        load_weight_fifo.write(weight_mem[base_addr+addr]);
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
    unsigned int nox,
    unsigned int bb_en,
    unsigned int conv_en
) {
    if (!bb_en) return;
    if (!conv_en) {
        for (int f_out = 0; f_out < nof; f_out += POF) {
            for (int y0 = 0; y0 < noy; y0 += POY) {
                for (int x0 = 0; x0 < nox; x0 += POX) {
                    for (int f = 0; f < POF; f++) {
                        for (int y = 0; y < POY; y++) {
                            for (int x = 0; x < POX; x++) {
                                pe_out_fifo.write(load_input_fifo.read());
                            }
                        }
                    }
                }
            }
        }
    }
    else {
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
}

void store_output(
    DTYPE_MEM_ACT *act_mem,
    hls::stream<float> &out_fifo,
    unsigned int base_addr,
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox,
    unsigned int bb_en
) {
    if (!bb_en) return;
    for (int f_out = 0; f_out < nof; f_out += POF) {
        for (int y0 = 0; y0 < noy; y0 += POY) {
            for (int x0 = 0; x0 < nox; x0 += POX) {
                for (int f = 0; f < POF; f++) {
                    for (int y = 0; y < POY; y++) {
                        for (int x = 0; x < POX; x++) {
                            DTYPE_MEM_ACT pack;
                            unsigned int addr = (f_out+f)*noy*nox + (y0+y)*nox + (x0+x);
                            act_mem[base_addr+addr/ACT_PACK].range((addr%ACT_PACK+1)*W_ACT-1, (addr%ACT_PACK)*W_ACT) = out_fifo.read();
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
                    mean = bn_weight_mem[bn_weight_base_addr+(f_out+f)];
                    mult_factor = bn_weight_mem[bn_weight_base_addr+nof+(f_out+f)];
                    beta = bn_weight_mem[bn_weight_base_addr+nof*2+(f_out+f)];
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
    DTYPE_MEM_ACT *act_mem,
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
                                DTYPE_ACT add_val = act_mem[base_addr+add_addr/ACT_PACK].range((add_addr%ACT_PACK+1)*W_ACT-1, (add_addr%ACT_PACK)*W_ACT);
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

void max_pool(
    DTYPE_MEM_ACT *act_mem,
    unsigned int in_base_addr,
    unsigned int out_base_addr,
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox,
    unsigned int stride,
    unsigned int pad,
    unsigned int max_pool_en
) {
    if (!max_pool_en) return;
    // todo: burst read and write
    DTYPE_ACT max_pool_kernel[MAX_POOL_K * MAX_POOL_K];

    for (int f = 0; f < nif; f++) {
        for (int y = 0; y < noy; y++) {
            for (int x = 0; x < nox; x++) {
                for (int i = 0; i < MAX_POOL_K; i++) {
                    for (int j = 0; j < MAX_POOL_K; j++) {
                        DTYPE_ACT in_val;
                        int y_in = y*stride + i - pad;
                        int x_in = x*stride + j - pad;
                        if ( (y*stride + i < pad) || (y*stride + i >= noy*stride + pad) || (x*stride + j < pad) || (x*stride + j >= nox*stride + pad) ){
                            in_val = 0;
                        }
                        else {
                            unsigned in_addr = f*noy*stride*nox*stride + (y*stride+i-pad)*nox*stride + (x*stride+j-pad);
                            in_val = act_mem[in_base_addr + in_addr/ACT_PACK].range((in_addr%ACT_PACK+1)*W_ACT-1, (in_addr%ACT_PACK)*W_ACT);
                        }
                        max_pool_kernel[i * MAX_POOL_K + j] = in_val;
                    }
                }
                DTYPE_ACT max = max_pool_kernel[0];
                for (int idx = 1; idx < MAX_POOL_K * MAX_POOL_K; idx++) {
                    if (max_pool_kernel[idx] > max) {
                        max = max_pool_kernel[idx];
                    }
                }
                unsigned out_addr = f*noy*nox + y*nox + x;
                act_mem[out_base_addr + out_addr/ACT_PACK].range((out_addr%ACT_PACK+1)*W_ACT-1, (out_addr%ACT_PACK)*W_ACT) = max;
            }
        }
    }
}

void avg_pool(
    DTYPE_MEM_ACT *act_mem,
    unsigned int in_base_addr,
    unsigned int out_base_addr,
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox,
    unsigned int stride,
    unsigned int pad,
    unsigned int avg_pool_en
) {
    if (!avg_pool_en) return;
    
    for (int f = 0; f < nif; f++) {
        float sum = 0;
        for (int y = 0; y < noy; y++) {
            for (int x = 0; x < nox; x++) {
                unsigned in_addr = f*noy*nox + y*nox + x;
                sum = sum + (float) act_mem[in_base_addr + in_addr/ACT_PACK].range((in_addr%ACT_PACK+1)*W_ACT-1, (in_addr%ACT_PACK)*W_ACT);
            }
        }
        act_mem[out_base_addr+f/ACT_PACK].range((f%ACT_PACK+1)*W_ACT-1, (f%ACT_PACK)*W_ACT) = (DTYPE_ACT) (sum / (noy * nox));
    }
}

void fc(
    DTYPE_MEM_ACT *act_mem,
    float *bn_weight_mem,
    unsigned int in_base_addr,
    unsigned int out_base_addr,
    unsigned int bn_weight_base_addr,
    unsigned int nof,
    unsigned int nif,
    unsigned int fc_en
) {
    if (!fc_en) return;

    for (int f_out = 0; f_out < nof; f_out++) {
        float sum = 0;
        for (int f_in = 0; f_in < nif; f_in++) {
            unsigned weight_addr = f_out*nif + f_in;
            sum = ((float) act_mem[in_base_addr + f_in%ACT_PACK].range((f_in%ACT_PACK+1)*W_ACT-1, (f_in%ACT_PACK)*W_ACT)) * bn_weight_mem[bn_weight_base_addr + weight_addr];
        }
        act_mem[out_base_addr+f_out%ACT_PACK].range() = sum + (float) bn_weight_mem[bn_weight_base_addr + AVG_POOL_C*FC_C + f_out];
    }
}

void controller (
    unsigned *layer_cnt,
    unsigned *nif,
    unsigned *nof,
    unsigned *noy,
    unsigned *nox,
    unsigned *nkx,
    unsigned *nky,
    unsigned *stride,
    unsigned *pad,
    bool *bb_en,
    bool *conv_en,
    bool *bn_en,
    bool *skip_en,
    bool *relu_en,
    bool *max_pool_en,
    bool *avg_pool_en,
    bool *fc_en,
    unsigned *base_addr_in,
    unsigned *base_addr_out,
    unsigned *base_addr_add,
    unsigned *weight_base,
    unsigned *weight_size,
    unsigned *bn_weight_base,
    unsigned *bn_weight_size,
    unsigned *in_size,
    unsigned *out_size
) {
    if (*layer_cnt == 0) {
        *nif                = IN_C;
        *nof                = CONV1_C;
        *noy                = CONV1_H;
        *nox                = CONV1_W;
        *nkx                = CONV1_K;
        *nky                = CONV1_K;
        *stride             = CONV1_S;
        *pad                = CONV1_PAD;
        *bb_en              = CONV1_BB_EN;
        *conv_en            = CONV1_CONV_EN;
        *bn_en              = CONV1_BN_EN;
        *skip_en            = CONV1_SKIP_EN;
        *relu_en            = CONV1_RELU_EN;
        *max_pool_en        = CONV1_MAX_POOL_EN;
        *avg_pool_en        = CONV1_AVG_POOL_EN;
        *fc_en             = CONV1_FC_EN;
        *base_addr_in       = CONV1_BASE_ADDR_IN;
        *base_addr_out      = CONV1_BASE_ADDR_OUT;
        *base_addr_add      = CONV1_BASE_ADDR_ADD;
        *weight_base        = CONV1_WEIGHT_BASE;
        *weight_size        = CONV1_WEIGHT_SIZE;
        *bn_weight_base     = CONV1_BN_WEIGHT_BASE;
        *bn_weight_size     = CONV1_BN_WEIGHT_SIZE;
        *in_size            = CONV1_IN_SIZE;
        *out_size           = CONV1_OUT_SIZE;
    }
    else if (*layer_cnt == 1) {
        *nif                = CONV1_C;
        *nof                = MAX_POOL_C;
        *noy                = MAX_POOL_H;
        *nox                = MAX_POOL_W;
        *nkx                = MAX_POOL_K;
        *nky                = MAX_POOL_K;
        *stride             = MAX_POOL_S;
        *pad                = MAX_POOL_PAD;
        *bb_en              = MAX_POOL_BB_EN;
        *conv_en            = MAX_POOL_CONV_EN;
        *bn_en              = MAX_POOL_BN_EN;
        *skip_en            = MAX_POOL_SKIP_EN;
        *relu_en            = MAX_POOL_RELU_EN;
        *max_pool_en        = MAX_POOL_MAX_POOL_EN;
        *avg_pool_en        = MAX_POOL_AVG_POOL_EN;
        *fc_en             = MAX_POOL_FC_EN;
        *base_addr_in       = MAX_POOL_BASE_ADDR_IN;
        *base_addr_out      = MAX_POOL_BASE_ADDR_OUT;
        *base_addr_add      = MAX_POOL_BASE_ADDR_ADD;
        *weight_base        = MAX_POOL_WEIGHT_BASE;
        *weight_size        = MAX_POOL_WEIGHT_SIZE;
        *bn_weight_base     = MAX_POOL_BN_WEIGHT_BASE;
        *bn_weight_size     = MAX_POOL_BN_WEIGHT_SIZE;
        *in_size            = MAX_POOL_IN_SIZE;
        *out_size           = MAX_POOL_OUT_SIZE;
    }
    else if(*layer_cnt == 2) {
        *nif                = MAX_POOL_C;
        *nof                = BB1_CONV1_C;
        *noy                = BB1_CONV1_H;
        *nox                = BB1_CONV1_W;
        *nkx                = BB1_CONV1_K;
        *nky                = BB1_CONV1_K;
        *stride             = BB1_CONV1_S;
        *pad                = BB1_CONV1_PAD;
        *bb_en              = BB1_CONV1_BB_EN;
        *conv_en            = BB1_CONV1_CONV_EN;
        *bn_en              = BB1_CONV1_BN_EN;
        *skip_en            = BB1_CONV1_SKIP_EN;
        *relu_en            = BB1_CONV1_RELU_EN;
        *max_pool_en        = BB1_CONV1_MAX_POOL_EN;
        *avg_pool_en        = BB1_CONV1_AVG_POOL_EN;
        *fc_en             = BB1_CONV1_FC_EN;
        *base_addr_in       = BB1_CONV1_BASE_ADDR_IN;
        *base_addr_out      = BB1_CONV1_BASE_ADDR_OUT;
        *base_addr_add      = BB1_CONV1_BASE_ADDR_ADD;
        *weight_base        = BB1_CONV1_WEIGHT_BASE;
        *weight_size        = BB1_CONV1_WEIGHT_SIZE;
        *bn_weight_base     = BB1_CONV1_BN_WEIGHT_BASE;
        *bn_weight_size     = BB1_CONV1_BN_WEIGHT_SIZE;
        *in_size            = BB1_CONV1_IN_SIZE;
        *out_size           = BB1_CONV1_OUT_SIZE;
    }
    else if(*layer_cnt == 3) {
        *nif                = BB1_CONV1_C;
        *nof                = BB1_CONV2_C;
        *noy                = BB1_CONV2_H;
        *nox                = BB1_CONV2_W;
        *nkx                = BB1_CONV2_K;
        *nky                = BB1_CONV2_K;
        *stride             = BB1_CONV2_S;
        *pad                = BB1_CONV2_PAD;
        *bb_en              = BB1_CONV2_BB_EN;
        *conv_en            = BB1_CONV2_CONV_EN;
        *bn_en              = BB1_CONV2_BN_EN;
        *skip_en            = BB1_CONV2_SKIP_EN;
        *relu_en            = BB1_CONV2_RELU_EN;
        *max_pool_en        = BB1_CONV2_MAX_POOL_EN;
        *avg_pool_en        = BB1_CONV2_AVG_POOL_EN;
        *fc_en             = BB1_CONV2_FC_EN;
        *base_addr_in       = BB1_CONV2_BASE_ADDR_IN;
        *base_addr_out      = BB1_CONV2_BASE_ADDR_OUT;
        *base_addr_add      = BB1_CONV2_BASE_ADDR_ADD;
        *weight_base        = BB1_CONV2_WEIGHT_BASE;
        *weight_size        = BB1_CONV2_WEIGHT_SIZE;
        *bn_weight_base     = BB1_CONV2_BN_WEIGHT_BASE;
        *bn_weight_size     = BB1_CONV2_BN_WEIGHT_SIZE;
        *in_size            = BB1_CONV2_IN_SIZE;
        *out_size           = BB1_CONV2_OUT_SIZE;
    }
    else if(*layer_cnt == 4) {
        *nif                = BB1_CONV2_C;
        *nof                = BB1_SKIP_C;
        *noy                = BB1_SKIP_H;
        *nox                = BB1_SKIP_W;
        *nkx                = BB1_SKIP_K;
        *nky                = BB1_SKIP_K;
        *stride             = BB1_SKIP_S;
        *pad                = BB1_SKIP_PAD;
        *bb_en              = BB1_SKIP_BB_EN;
        *conv_en            = BB1_SKIP_CONV_EN;
        *bn_en              = BB1_SKIP_BN_EN;
        *skip_en            = BB1_SKIP_SKIP_EN;
        *relu_en            = BB1_SKIP_RELU_EN;
        *max_pool_en        = BB1_SKIP_MAX_POOL_EN;
        *avg_pool_en        = BB1_SKIP_AVG_POOL_EN;
        *fc_en             = BB1_SKIP_FC_EN;
        *base_addr_in       = BB1_SKIP_BASE_ADDR_IN;
        *base_addr_out      = BB1_SKIP_BASE_ADDR_OUT;
        *base_addr_add      = BB1_SKIP_BASE_ADDR_ADD;
        *weight_base        = BB1_SKIP_WEIGHT_BASE;
        *weight_size        = BB1_SKIP_WEIGHT_SIZE;
        *bn_weight_base     = BB1_SKIP_BN_WEIGHT_BASE;
        *bn_weight_size     = BB1_SKIP_BN_WEIGHT_SIZE;
        *in_size            = BB1_SKIP_IN_SIZE;
        *out_size           = BB1_SKIP_OUT_SIZE;
    }
    else if(*layer_cnt == 5) {
        *nif                = BB1_SKIP_C;
        *nof                = BB2_CONV1_C;
        *noy                = BB2_CONV1_H;
        *nox                = BB2_CONV1_W;
        *nkx                = BB2_CONV1_K;
        *nky                = BB2_CONV1_K;
        *stride             = BB2_CONV1_S;
        *pad                = BB2_CONV1_PAD;
        *bb_en              = BB2_CONV1_BB_EN;
        *conv_en            = BB2_CONV1_CONV_EN;
        *bn_en              = BB2_CONV1_BN_EN;
        *skip_en            = BB2_CONV1_SKIP_EN;
        *relu_en            = BB2_CONV1_RELU_EN;
        *max_pool_en        = BB2_CONV1_MAX_POOL_EN;
        *avg_pool_en        = BB2_CONV1_AVG_POOL_EN;
        *fc_en             = BB2_CONV1_FC_EN;
        *base_addr_in       = BB2_CONV1_BASE_ADDR_IN;
        *base_addr_out      = BB2_CONV1_BASE_ADDR_OUT;
        *base_addr_add      = BB2_CONV1_BASE_ADDR_ADD;
        *weight_base        = BB2_CONV1_WEIGHT_BASE;
        *weight_size        = BB2_CONV1_WEIGHT_SIZE;
        *bn_weight_base     = BB2_CONV1_BN_WEIGHT_BASE;
        *bn_weight_size     = BB2_CONV1_BN_WEIGHT_SIZE;
        *in_size            = BB2_CONV1_IN_SIZE;
        *out_size           = BB2_CONV1_OUT_SIZE;
    }
    else if(*layer_cnt == 6) {
        *nif                = BB2_CONV1_C;
        *nof                = BB2_CONV2_C;
        *noy                = BB2_CONV2_H;
        *nox                = BB2_CONV2_W;
        *nkx                = BB2_CONV2_K;
        *nky                = BB2_CONV2_K;
        *stride             = BB2_CONV2_S;
        *pad                = BB2_CONV2_PAD;
        *bb_en              = BB2_CONV2_BB_EN;
        *conv_en            = BB2_CONV2_CONV_EN;
        *bn_en              = BB2_CONV2_BN_EN;
        *skip_en            = BB2_CONV2_SKIP_EN;
        *relu_en            = BB2_CONV2_RELU_EN;
        *max_pool_en        = BB2_CONV2_MAX_POOL_EN;
        *avg_pool_en        = BB2_CONV2_AVG_POOL_EN;
        *fc_en             = BB2_CONV2_FC_EN;
        *base_addr_in       = BB2_CONV2_BASE_ADDR_IN;
        *base_addr_out      = BB2_CONV2_BASE_ADDR_OUT;
        *base_addr_add      = BB2_CONV2_BASE_ADDR_ADD;
        *weight_base        = BB2_CONV2_WEIGHT_BASE;
        *weight_size        = BB2_CONV2_WEIGHT_SIZE;
        *bn_weight_base     = BB2_CONV2_BN_WEIGHT_BASE;
        *bn_weight_size     = BB2_CONV2_BN_WEIGHT_SIZE;
        *in_size            = BB2_CONV2_IN_SIZE;
        *out_size           = BB2_CONV2_OUT_SIZE;
    }
    else if(*layer_cnt == 7) {
        *nif                = BB2_CONV2_C;
        *nof                = BB2_SKIP_C;
        *noy                = BB2_SKIP_H;
        *nox                = BB2_SKIP_W;
        *nkx                = BB2_SKIP_K;
        *nky                = BB2_SKIP_K;
        *stride             = BB2_SKIP_S;
        *pad                = BB2_SKIP_PAD;
        *bb_en              = BB2_SKIP_BB_EN;
        *conv_en            = BB2_SKIP_CONV_EN;
        *bn_en              = BB2_SKIP_BN_EN;
        *skip_en            = BB2_SKIP_SKIP_EN;
        *relu_en            = BB2_SKIP_RELU_EN;
        *max_pool_en        = BB2_SKIP_MAX_POOL_EN;
        *avg_pool_en        = BB2_SKIP_AVG_POOL_EN;
        *fc_en             = BB2_SKIP_FC_EN;
        *base_addr_in       = BB2_SKIP_BASE_ADDR_IN;
        *base_addr_out      = BB2_SKIP_BASE_ADDR_OUT;
        *base_addr_add      = BB2_SKIP_BASE_ADDR_ADD;
        *weight_base        = BB2_SKIP_WEIGHT_BASE;
        *weight_size        = BB2_SKIP_WEIGHT_SIZE;
        *bn_weight_base     = BB2_SKIP_BN_WEIGHT_BASE;
        *bn_weight_size     = BB2_SKIP_BN_WEIGHT_SIZE;
        *in_size            = BB2_SKIP_IN_SIZE;
        *out_size           = BB2_SKIP_OUT_SIZE;
    }
    else if(*layer_cnt == 8) {
        *nif                = BB2_SKIP_C;
        *nof                = BB3_CONV1_C;
        *noy                = BB3_CONV1_H;
        *nox                = BB3_CONV1_W;
        *nkx                = BB3_CONV1_K;
        *nky                = BB3_CONV1_K;
        *stride             = BB3_CONV1_S;
        *pad                = BB3_CONV1_PAD;
        *bb_en              = BB3_CONV1_BB_EN;
        *conv_en            = BB3_CONV1_CONV_EN;
        *bn_en              = BB3_CONV1_BN_EN;
        *skip_en            = BB3_CONV1_SKIP_EN;
        *relu_en            = BB3_CONV1_RELU_EN;
        *max_pool_en        = BB3_CONV1_MAX_POOL_EN;
        *avg_pool_en        = BB3_CONV1_AVG_POOL_EN;
        *fc_en             = BB3_CONV1_FC_EN;
        *base_addr_in       = BB3_CONV1_BASE_ADDR_IN;
        *base_addr_out      = BB3_CONV1_BASE_ADDR_OUT;
        *base_addr_add      = BB3_CONV1_BASE_ADDR_ADD;
        *weight_base        = BB3_CONV1_WEIGHT_BASE;
        *weight_size        = BB3_CONV1_WEIGHT_SIZE;
        *bn_weight_base     = BB3_CONV1_BN_WEIGHT_BASE;
        *bn_weight_size     = BB3_CONV1_BN_WEIGHT_SIZE;
        *in_size            = BB3_CONV1_IN_SIZE;
        *out_size           = BB3_CONV1_OUT_SIZE;
    }
    else if(*layer_cnt == 9) {
        *nif                = BB3_CONV1_C;
        *nof                = BB3_CONV2_C;
        *noy                = BB3_CONV2_H;
        *nox                = BB3_CONV2_W;
        *nkx                = BB3_CONV2_K;
        *nky                = BB3_CONV2_K;
        *stride             = BB3_CONV2_S;
        *pad                = BB3_CONV2_PAD;
        *bb_en              = BB3_CONV2_BB_EN;
        *conv_en            = BB3_CONV2_CONV_EN;
        *bn_en              = BB3_CONV2_BN_EN;
        *skip_en            = BB3_CONV2_SKIP_EN;
        *relu_en            = BB3_CONV2_RELU_EN;
        *max_pool_en        = BB3_CONV2_MAX_POOL_EN;
        *avg_pool_en        = BB3_CONV2_AVG_POOL_EN;
        *fc_en             = BB3_CONV2_FC_EN;
        *base_addr_in       = BB3_CONV2_BASE_ADDR_IN;
        *base_addr_out      = BB3_CONV2_BASE_ADDR_OUT;
        *base_addr_add      = BB3_CONV2_BASE_ADDR_ADD;
        *weight_base        = BB3_CONV2_WEIGHT_BASE;
        *weight_size        = BB3_CONV2_WEIGHT_SIZE;
        *bn_weight_base     = BB3_CONV2_BN_WEIGHT_BASE;
        *bn_weight_size     = BB3_CONV2_BN_WEIGHT_SIZE;
        *in_size            = BB3_CONV2_IN_SIZE;
        *out_size           = BB3_CONV2_OUT_SIZE;
    }
    else if(*layer_cnt == 10) {
        *nif                = BB3_CONV2_C;
        *nof                = BB3_SKIP_C;
        *noy                = BB3_SKIP_H;
        *nox                = BB3_SKIP_W;
        *nkx                = BB3_SKIP_K;
        *nky                = BB3_SKIP_K;
        *stride             = BB3_SKIP_S;
        *pad                = BB3_SKIP_PAD;
        *bb_en              = BB3_SKIP_BB_EN;
        *conv_en            = BB3_SKIP_CONV_EN;
        *bn_en              = BB3_SKIP_BN_EN;
        *skip_en            = BB3_SKIP_SKIP_EN;
        *relu_en            = BB3_SKIP_RELU_EN;
        *max_pool_en        = BB3_SKIP_MAX_POOL_EN;
        *avg_pool_en        = BB3_SKIP_AVG_POOL_EN;
        *fc_en             = BB3_SKIP_FC_EN;
        *base_addr_in       = BB3_SKIP_BASE_ADDR_IN;
        *base_addr_out      = BB3_SKIP_BASE_ADDR_OUT;
        *base_addr_add      = BB3_SKIP_BASE_ADDR_ADD;
        *weight_base        = BB3_SKIP_WEIGHT_BASE;
        *weight_size        = BB3_SKIP_WEIGHT_SIZE;
        *bn_weight_base     = BB3_SKIP_BN_WEIGHT_BASE;
        *bn_weight_size     = BB3_SKIP_BN_WEIGHT_SIZE;
        *in_size            = BB3_SKIP_IN_SIZE;
        *out_size           = BB3_SKIP_OUT_SIZE;
    }
    else if(*layer_cnt == 11) {
        *nif                = BB3_SKIP_C;
        *nof                = BB4_CONV1_C;
        *noy                = BB4_CONV1_H;
        *nox                = BB4_CONV1_W;
        *nkx                = BB4_CONV1_K;
        *nky                = BB4_CONV1_K;
        *stride             = BB4_CONV1_S;
        *pad                = BB4_CONV1_PAD;
        *bb_en              = BB4_CONV1_BB_EN;
        *conv_en            = BB4_CONV1_CONV_EN;
        *bn_en              = BB4_CONV1_BN_EN;
        *skip_en            = BB4_CONV1_SKIP_EN;
        *relu_en            = BB4_CONV1_RELU_EN;
        *max_pool_en        = BB4_CONV1_MAX_POOL_EN;
        *avg_pool_en        = BB4_CONV1_AVG_POOL_EN;
        *fc_en             = BB4_CONV1_FC_EN;
        *base_addr_in       = BB4_CONV1_BASE_ADDR_IN;
        *base_addr_out      = BB4_CONV1_BASE_ADDR_OUT;
        *base_addr_add      = BB4_CONV1_BASE_ADDR_ADD;
        *weight_base        = BB4_CONV1_WEIGHT_BASE;
        *weight_size        = BB4_CONV1_WEIGHT_SIZE;
        *bn_weight_base     = BB4_CONV1_BN_WEIGHT_BASE;
        *bn_weight_size     = BB4_CONV1_BN_WEIGHT_SIZE;
        *in_size            = BB4_CONV1_IN_SIZE;
        *out_size           = BB4_CONV1_OUT_SIZE;
    }
    else if(*layer_cnt == 12) {
        *nif                = BB4_CONV1_C;
        *nof                = BB4_CONV2_C;
        *noy                = BB4_CONV2_H;
        *nox                = BB4_CONV2_W;
        *nkx                = BB4_CONV2_K;
        *nky                = BB4_CONV2_K;
        *stride             = BB4_CONV2_S;
        *pad                = BB4_CONV2_PAD;
        *bb_en              = BB4_CONV2_BB_EN;
        *conv_en            = BB4_CONV2_CONV_EN;
        *bn_en              = BB4_CONV2_BN_EN;
        *skip_en            = BB4_CONV2_SKIP_EN;
        *relu_en            = BB4_CONV2_RELU_EN;
        *max_pool_en        = BB4_CONV2_MAX_POOL_EN;
        *avg_pool_en        = BB4_CONV2_AVG_POOL_EN;
        *fc_en             = BB4_CONV2_FC_EN;
        *base_addr_in       = BB4_CONV2_BASE_ADDR_IN;
        *base_addr_out      = BB4_CONV2_BASE_ADDR_OUT;
        *base_addr_add      = BB4_CONV2_BASE_ADDR_ADD;
        *weight_base        = BB4_CONV2_WEIGHT_BASE;
        *weight_size        = BB4_CONV2_WEIGHT_SIZE;
        *bn_weight_base     = BB4_CONV2_BN_WEIGHT_BASE;
        *bn_weight_size     = BB4_CONV2_BN_WEIGHT_SIZE;
        *in_size            = BB4_CONV2_IN_SIZE;
        *out_size           = BB4_CONV2_OUT_SIZE;
    }
    else if(*layer_cnt == 13) {
        *nif                = BB4_CONV2_C;
        *nof                = BB4_SKIP_C;
        *noy                = BB4_SKIP_H;
        *nox                = BB4_SKIP_W;
        *nkx                = BB4_SKIP_K;
        *nky                = BB4_SKIP_K;
        *stride             = BB4_SKIP_S;
        *pad                = BB4_SKIP_PAD;
        *bb_en              = BB4_SKIP_BB_EN;
        *conv_en            = BB4_SKIP_CONV_EN;
        *bn_en              = BB4_SKIP_BN_EN;
        *skip_en            = BB4_SKIP_SKIP_EN;
        *relu_en            = BB4_SKIP_RELU_EN;
        *max_pool_en        = BB4_SKIP_MAX_POOL_EN;
        *avg_pool_en        = BB4_SKIP_AVG_POOL_EN;
        *fc_en             = BB4_SKIP_FC_EN;
        *base_addr_in       = BB4_SKIP_BASE_ADDR_IN;
        *base_addr_out      = BB4_SKIP_BASE_ADDR_OUT;
        *base_addr_add      = BB4_SKIP_BASE_ADDR_ADD;
        *weight_base        = BB4_SKIP_WEIGHT_BASE;
        *weight_size        = BB4_SKIP_WEIGHT_SIZE;
        *bn_weight_base     = BB4_SKIP_BN_WEIGHT_BASE;
        *bn_weight_size     = BB4_SKIP_BN_WEIGHT_SIZE;
        *in_size            = BB4_SKIP_IN_SIZE;
        *out_size           = BB4_SKIP_OUT_SIZE;
    }
    else if(*layer_cnt == 14) {
        *nif                = BB4_SKIP_C;
        *nof                = BB5_CONV1_C;
        *noy                = BB5_CONV1_H;
        *nox                = BB5_CONV1_W;
        *nkx                = BB5_CONV1_K;
        *nky                = BB5_CONV1_K;
        *stride             = BB5_CONV1_S;
        *pad                = BB5_CONV1_PAD;
        *bb_en              = BB5_CONV1_BB_EN;
        *conv_en            = BB5_CONV1_CONV_EN;
        *bn_en              = BB5_CONV1_BN_EN;
        *skip_en            = BB5_CONV1_SKIP_EN;
        *relu_en            = BB5_CONV1_RELU_EN;
        *max_pool_en        = BB5_CONV1_MAX_POOL_EN;
        *avg_pool_en        = BB5_CONV1_AVG_POOL_EN;
        *fc_en             = BB5_CONV1_FC_EN;
        *base_addr_in       = BB5_CONV1_BASE_ADDR_IN;
        *base_addr_out      = BB5_CONV1_BASE_ADDR_OUT;
        *base_addr_add      = BB5_CONV1_BASE_ADDR_ADD;
        *weight_base        = BB5_CONV1_WEIGHT_BASE;
        *weight_size        = BB5_CONV1_WEIGHT_SIZE;
        *bn_weight_base     = BB5_CONV1_BN_WEIGHT_BASE;
        *bn_weight_size     = BB5_CONV1_BN_WEIGHT_SIZE;
        *in_size            = BB5_CONV1_IN_SIZE;
        *out_size           = BB5_CONV1_OUT_SIZE;
    }
    else if(*layer_cnt == 15) {
        *nif                = BB5_CONV1_C;
        *nof                = BB5_CONV2_C;
        *noy                = BB5_CONV2_H;
        *nox                = BB5_CONV2_W;
        *nkx                = BB5_CONV2_K;
        *nky                = BB5_CONV2_K;
        *stride             = BB5_CONV2_S;
        *pad                = BB5_CONV2_PAD;
        *bb_en              = BB5_CONV2_BB_EN;
        *conv_en            = BB5_CONV2_CONV_EN;
        *bn_en              = BB5_CONV2_BN_EN;
        *skip_en            = BB5_CONV2_SKIP_EN;
        *relu_en            = BB5_CONV2_RELU_EN;
        *max_pool_en        = BB5_CONV2_MAX_POOL_EN;
        *avg_pool_en        = BB5_CONV2_AVG_POOL_EN;
        *fc_en             = BB5_CONV2_FC_EN;
        *base_addr_in       = BB5_CONV2_BASE_ADDR_IN;
        *base_addr_out      = BB5_CONV2_BASE_ADDR_OUT;
        *base_addr_add      = BB5_CONV2_BASE_ADDR_ADD;
        *weight_base        = BB5_CONV2_WEIGHT_BASE;
        *weight_size        = BB5_CONV2_WEIGHT_SIZE;
        *bn_weight_base     = BB5_CONV2_BN_WEIGHT_BASE;
        *bn_weight_size     = BB5_CONV2_BN_WEIGHT_SIZE;
        *in_size            = BB5_CONV2_IN_SIZE;
        *out_size           = BB5_CONV2_OUT_SIZE;
    }
    else if(*layer_cnt == 16) {
        *nif                = BB5_CONV2_C;
        *nof                = BB5_SKIP_C;
        *noy                = BB5_SKIP_H;
        *nox                = BB5_SKIP_W;
        *nkx                = BB5_SKIP_K;
        *nky                = BB5_SKIP_K;
        *stride             = BB5_SKIP_S;
        *pad                = BB5_SKIP_PAD;
        *bb_en              = BB5_SKIP_BB_EN;
        *conv_en            = BB5_SKIP_CONV_EN;
        *bn_en              = BB5_SKIP_BN_EN;
        *skip_en            = BB5_SKIP_SKIP_EN;
        *relu_en            = BB5_SKIP_RELU_EN;
        *max_pool_en        = BB5_SKIP_MAX_POOL_EN;
        *avg_pool_en        = BB5_SKIP_AVG_POOL_EN;
        *fc_en             = BB5_SKIP_FC_EN;
        *base_addr_in       = BB5_SKIP_BASE_ADDR_IN;
        *base_addr_out      = BB5_SKIP_BASE_ADDR_OUT;
        *base_addr_add      = BB5_SKIP_BASE_ADDR_ADD;
        *weight_base        = BB5_SKIP_WEIGHT_BASE;
        *weight_size        = BB5_SKIP_WEIGHT_SIZE;
        *bn_weight_base     = BB5_SKIP_BN_WEIGHT_BASE;
        *bn_weight_size     = BB5_SKIP_BN_WEIGHT_SIZE;
        *in_size            = BB5_SKIP_IN_SIZE;
        *out_size           = BB5_SKIP_OUT_SIZE;
    }
    else if(*layer_cnt == 17) {
        *nif                = BB5_SKIP_C;
        *nof                = BB6_CONV1_C;
        *noy                = BB6_CONV1_H;
        *nox                = BB6_CONV1_W;
        *nkx                = BB6_CONV1_K;
        *nky                = BB6_CONV1_K;
        *stride             = BB6_CONV1_S;
        *pad                = BB6_CONV1_PAD;
        *bb_en              = BB6_CONV1_BB_EN;
        *conv_en            = BB6_CONV1_CONV_EN;
        *bn_en              = BB6_CONV1_BN_EN;
        *skip_en            = BB6_CONV1_SKIP_EN;
        *relu_en            = BB6_CONV1_RELU_EN;
        *max_pool_en        = BB6_CONV1_MAX_POOL_EN;
        *avg_pool_en        = BB6_CONV1_AVG_POOL_EN;
        *fc_en             = BB6_CONV1_FC_EN;
        *base_addr_in       = BB6_CONV1_BASE_ADDR_IN;
        *base_addr_out      = BB6_CONV1_BASE_ADDR_OUT;
        *base_addr_add      = BB6_CONV1_BASE_ADDR_ADD;
        *weight_base        = BB6_CONV1_WEIGHT_BASE;
        *weight_size        = BB6_CONV1_WEIGHT_SIZE;
        *bn_weight_base     = BB6_CONV1_BN_WEIGHT_BASE;
        *bn_weight_size     = BB6_CONV1_BN_WEIGHT_SIZE;
        *in_size            = BB6_CONV1_IN_SIZE;
        *out_size           = BB6_CONV1_OUT_SIZE;
    }
    else if(*layer_cnt == 18) {
        *nif                = BB6_CONV1_C;
        *nof                = BB6_CONV2_C;
        *noy                = BB6_CONV2_H;
        *nox                = BB6_CONV2_W;
        *nkx                = BB6_CONV2_K;
        *nky                = BB6_CONV2_K;
        *stride             = BB6_CONV2_S;
        *pad                = BB6_CONV2_PAD;
        *bb_en              = BB6_CONV2_BB_EN;
        *conv_en            = BB6_CONV2_CONV_EN;
        *bn_en              = BB6_CONV2_BN_EN;
        *skip_en            = BB6_CONV2_SKIP_EN;
        *relu_en            = BB6_CONV2_RELU_EN;
        *max_pool_en        = BB6_CONV2_MAX_POOL_EN;
        *avg_pool_en        = BB6_CONV2_AVG_POOL_EN;
        *fc_en             = BB6_CONV2_FC_EN;
        *base_addr_in       = BB6_CONV2_BASE_ADDR_IN;
        *base_addr_out      = BB6_CONV2_BASE_ADDR_OUT;
        *base_addr_add      = BB6_CONV2_BASE_ADDR_ADD;
        *weight_base        = BB6_CONV2_WEIGHT_BASE;
        *weight_size        = BB6_CONV2_WEIGHT_SIZE;
        *bn_weight_base     = BB6_CONV2_BN_WEIGHT_BASE;
        *bn_weight_size     = BB6_CONV2_BN_WEIGHT_SIZE;
        *in_size            = BB6_CONV2_IN_SIZE;
        *out_size           = BB6_CONV2_OUT_SIZE;
    }
    else if (*layer_cnt == 19) {
        *nif                = BB6_CONV2_C;
        *nof                = BB6_SKIP_C;
        *noy                = BB6_SKIP_H;
        *nox                = BB6_SKIP_W;
        *nkx                = BB6_SKIP_K;
        *nky                = BB6_SKIP_K;
        *stride             = BB6_SKIP_S;
        *pad                = BB6_SKIP_PAD;
        *bb_en              = BB6_SKIP_BB_EN;
        *conv_en            = BB6_SKIP_CONV_EN;
        *bn_en              = BB6_SKIP_BN_EN;
        *skip_en            = BB6_SKIP_SKIP_EN;
        *relu_en            = BB6_SKIP_RELU_EN;
        *max_pool_en        = BB6_SKIP_MAX_POOL_EN;
        *avg_pool_en        = BB6_SKIP_AVG_POOL_EN;
        *fc_en             = BB6_SKIP_FC_EN;
        *base_addr_in       = BB6_SKIP_BASE_ADDR_IN;
        *base_addr_out      = BB6_SKIP_BASE_ADDR_OUT;
        *base_addr_add      = BB6_SKIP_BASE_ADDR_ADD;
        *weight_base        = BB6_SKIP_WEIGHT_BASE;
        *weight_size        = BB6_SKIP_WEIGHT_SIZE;
        *bn_weight_base     = BB6_SKIP_BN_WEIGHT_BASE;
        *bn_weight_size     = BB6_SKIP_BN_WEIGHT_SIZE;
        *in_size            = BB6_SKIP_IN_SIZE;
        *out_size           = BB6_SKIP_OUT_SIZE;
    }
    else if (*layer_cnt == 20) {
        *nif                = BB6_SKIP_C;
        *nof                = BB7_CONV1_C;
        *noy                = BB7_CONV1_H;
        *nox                = BB7_CONV1_W;
        *nkx                = BB7_CONV1_K;
        *nky                = BB7_CONV1_K;
        *stride             = BB7_CONV1_S;
        *pad                = BB7_CONV1_PAD;
        *bb_en              = BB7_CONV1_BB_EN;
        *conv_en            = BB7_CONV1_CONV_EN;
        *bn_en              = BB7_CONV1_BN_EN;
        *skip_en            = BB7_CONV1_SKIP_EN;
        *relu_en            = BB7_CONV1_RELU_EN;
        *max_pool_en        = BB7_CONV1_MAX_POOL_EN;
        *avg_pool_en        = BB7_CONV1_AVG_POOL_EN;
        *fc_en             = BB7_CONV1_FC_EN;
        *base_addr_in       = BB7_CONV1_BASE_ADDR_IN;
        *base_addr_out      = BB7_CONV1_BASE_ADDR_OUT;
        *base_addr_add      = BB7_CONV1_BASE_ADDR_ADD;
        *weight_base        = BB7_CONV1_WEIGHT_BASE;
        *weight_size        = BB7_CONV1_WEIGHT_SIZE;
        *bn_weight_base     = BB7_CONV1_BN_WEIGHT_BASE;
        *bn_weight_size     = BB7_CONV1_BN_WEIGHT_SIZE;
        *in_size            = BB7_CONV1_IN_SIZE;
        *out_size           = BB7_CONV1_OUT_SIZE;
    }
    else if (*layer_cnt == 21) {
        *nif                = BB7_CONV1_C;
        *nof                = BB7_CONV2_C;
        *noy                = BB7_CONV2_H;
        *nox                = BB7_CONV2_W;
        *nkx                = BB7_CONV2_K;
        *nky                = BB7_CONV2_K;
        *stride             = BB7_CONV2_S;
        *pad                = BB7_CONV2_PAD;
        *bb_en              = BB7_CONV2_BB_EN;
        *conv_en            = BB7_CONV2_CONV_EN;
        *bn_en              = BB7_CONV2_BN_EN;
        *skip_en            = BB7_CONV2_SKIP_EN;
        *relu_en            = BB7_CONV2_RELU_EN;
        *max_pool_en        = BB7_CONV2_MAX_POOL_EN;
        *avg_pool_en        = BB7_CONV2_AVG_POOL_EN;
        *fc_en             = BB7_CONV2_FC_EN;
        *base_addr_in       = BB7_CONV2_BASE_ADDR_IN;
        *base_addr_out      = BB7_CONV2_BASE_ADDR_OUT;
        *base_addr_add      = BB7_CONV2_BASE_ADDR_ADD;
        *weight_base        = BB7_CONV2_WEIGHT_BASE;
        *weight_size        = BB7_CONV2_WEIGHT_SIZE;
        *bn_weight_base     = BB7_CONV2_BN_WEIGHT_BASE;
        *bn_weight_size     = BB7_CONV2_BN_WEIGHT_SIZE;
        *in_size            = BB7_CONV2_IN_SIZE;
        *out_size           = BB7_CONV2_OUT_SIZE;
    }
    else if (*layer_cnt == 22) {
        *nif                = BB7_CONV2_C;
        *nof                = BB7_SKIP_C;
        *noy                = BB7_SKIP_H;
        *nox                = BB7_SKIP_W;
        *nkx                = BB7_SKIP_K;
        *nky                = BB7_SKIP_K;
        *stride             = BB7_SKIP_S;
        *pad                = BB7_SKIP_PAD;
        *bb_en              = BB7_SKIP_BB_EN;
        *conv_en            = BB7_SKIP_CONV_EN;
        *bn_en              = BB7_SKIP_BN_EN;
        *skip_en            = BB7_SKIP_SKIP_EN;
        *relu_en            = BB7_SKIP_RELU_EN;
        *max_pool_en        = BB7_SKIP_MAX_POOL_EN;
        *avg_pool_en        = BB7_SKIP_AVG_POOL_EN;
        *fc_en             = BB7_SKIP_FC_EN;
        *base_addr_in       = BB7_SKIP_BASE_ADDR_IN;
        *base_addr_out      = BB7_SKIP_BASE_ADDR_OUT;
        *base_addr_add      = BB7_SKIP_BASE_ADDR_ADD;
        *weight_base        = BB7_SKIP_WEIGHT_BASE;
        *weight_size        = BB7_SKIP_WEIGHT_SIZE;
        *bn_weight_base     = BB7_SKIP_BN_WEIGHT_BASE;
        *bn_weight_size     = BB7_SKIP_BN_WEIGHT_SIZE;
        *in_size            = BB7_SKIP_IN_SIZE;
        *out_size           = BB7_SKIP_OUT_SIZE;
    }
    else if (*layer_cnt == 23) {
        *nif                = BB7_SKIP_C;
        *nof                = BB8_CONV1_C;
        *noy                = BB8_CONV1_H;
        *nox                = BB8_CONV1_W;
        *nkx                = BB8_CONV1_K;
        *nky                = BB8_CONV1_K;
        *stride             = BB8_CONV1_S;
        *pad                = BB8_CONV1_PAD;
        *bb_en              = BB8_CONV1_BB_EN;
        *conv_en            = BB8_CONV1_CONV_EN;
        *bn_en              = BB8_CONV1_BN_EN;
        *skip_en            = BB8_CONV1_SKIP_EN;
        *relu_en            = BB8_CONV1_RELU_EN;
        *max_pool_en        = BB8_CONV1_MAX_POOL_EN;
        *avg_pool_en        = BB8_CONV1_AVG_POOL_EN;
        *fc_en             = BB8_CONV1_FC_EN;
        *base_addr_in       = BB8_CONV1_BASE_ADDR_IN;
        *base_addr_out      = BB8_CONV1_BASE_ADDR_OUT;
        *base_addr_add      = BB8_CONV1_BASE_ADDR_ADD;
        *weight_base        = BB8_CONV1_WEIGHT_BASE;
        *weight_size        = BB8_CONV1_WEIGHT_SIZE;
        *bn_weight_base     = BB8_CONV1_BN_WEIGHT_BASE;
        *bn_weight_size     = BB8_CONV1_BN_WEIGHT_SIZE;
        *in_size            = BB8_CONV1_IN_SIZE;
        *out_size           = BB8_CONV1_OUT_SIZE;
    }
    else if (*layer_cnt == 24) {
        *nif                = BB8_CONV1_C;
        *nof                = BB8_CONV2_C;
        *noy                = BB8_CONV2_H;
        *nox                = BB8_CONV2_W;
        *nkx                = BB8_CONV2_K;
        *nky                = BB8_CONV2_K;
        *stride             = BB8_CONV2_S;
        *pad                = BB8_CONV2_PAD;
        *bb_en              = BB8_CONV2_BB_EN;
        *conv_en            = BB8_CONV2_CONV_EN;
        *bn_en              = BB8_CONV2_BN_EN;
        *skip_en            = BB8_CONV2_SKIP_EN;
        *relu_en            = BB8_CONV2_RELU_EN;
        *max_pool_en        = BB8_CONV2_MAX_POOL_EN;
        *avg_pool_en        = BB8_CONV2_AVG_POOL_EN;
        *fc_en             = BB8_CONV2_FC_EN;
        *base_addr_in       = BB8_CONV2_BASE_ADDR_IN;
        *base_addr_out      = BB8_CONV2_BASE_ADDR_OUT;
        *base_addr_add      = BB8_CONV2_BASE_ADDR_ADD;
        *weight_base        = BB8_CONV2_WEIGHT_BASE;
        *weight_size        = BB8_CONV2_WEIGHT_SIZE;
        *bn_weight_base     = BB8_CONV2_BN_WEIGHT_BASE;
        *bn_weight_size     = BB8_CONV2_BN_WEIGHT_SIZE;
        *in_size            = BB8_CONV2_IN_SIZE;
        *out_size           = BB8_CONV2_OUT_SIZE;
    }
    else if (*layer_cnt == 25) {
        *nif                = BB8_CONV2_C;
        *nof                = BB8_SKIP_C;
        *noy                = BB8_SKIP_H;
        *nox                = BB8_SKIP_W;
        *nkx                = BB8_SKIP_K;
        *nky                = BB8_SKIP_K;
        *stride             = BB8_SKIP_S;
        *pad                = BB8_SKIP_PAD;
        *bb_en              = BB8_SKIP_BB_EN;
        *conv_en            = BB8_SKIP_CONV_EN;
        *bn_en              = BB8_SKIP_BN_EN;
        *skip_en            = BB8_SKIP_SKIP_EN;
        *relu_en            = BB8_SKIP_RELU_EN;
        *max_pool_en        = BB8_SKIP_MAX_POOL_EN;
        *avg_pool_en        = BB8_SKIP_AVG_POOL_EN;
        *fc_en             = BB8_SKIP_FC_EN;
        *base_addr_in       = BB8_SKIP_BASE_ADDR_IN;
        *base_addr_out      = BB8_SKIP_BASE_ADDR_OUT;
        *base_addr_add      = BB8_SKIP_BASE_ADDR_ADD;
        *weight_base        = BB8_SKIP_WEIGHT_BASE;
        *weight_size        = BB8_SKIP_WEIGHT_SIZE;
        *bn_weight_base     = BB8_SKIP_BN_WEIGHT_BASE;
        *bn_weight_size     = BB8_SKIP_BN_WEIGHT_SIZE;
        *in_size            = BB8_SKIP_IN_SIZE;
        *out_size           = BB8_SKIP_OUT_SIZE;
    }
    else if (*layer_cnt == 26) {
        *nif                = BB8_SKIP_C;
        *nof                = AVG_POOL_C;
        *noy                = AVG_POOL_H;
        *nox                = AVG_POOL_W;
        *nkx                = AVG_POOL_K;
        *nky                = AVG_POOL_K;
        *stride             = AVG_POOL_S;
        *pad                = AVG_POOL_PAD;
        *bb_en              = AVG_POOL_BB_EN;
        *conv_en            = AVG_POOL_CONV_EN;
        *bn_en              = AVG_POOL_BN_EN;
        *skip_en            = AVG_POOL_SKIP_EN;
        *relu_en            = AVG_POOL_RELU_EN;
        *max_pool_en        = AVG_POOL_MAX_POOL_EN;
        *avg_pool_en        = AVG_POOL_AVG_POOL_EN;
        *fc_en             = AVG_POOL_FC_EN;
        *base_addr_in       = AVG_POOL_BASE_ADDR_IN;
        *base_addr_out      = AVG_POOL_BASE_ADDR_OUT;
        *base_addr_add      = AVG_POOL_BASE_ADDR_ADD;
        *weight_base        = AVG_POOL_WEIGHT_BASE;
        *weight_size        = AVG_POOL_WEIGHT_SIZE;
        *bn_weight_base     = AVG_POOL_BN_WEIGHT_BASE;
        *bn_weight_size     = AVG_POOL_BN_WEIGHT_SIZE;
        *in_size            = AVG_POOL_IN_SIZE;
        *out_size           = AVG_POOL_OUT_SIZE;
    }
    else if (*layer_cnt == 27) {
        *nif                = AVG_POOL_C;
        *nof                = FC_C;
        *noy                = FC_H;
        *nox                = FC_W;
        *nkx                = FC_K;
        *nky                = FC_K;
        *stride             = FC_S;
        *pad                = FC_PAD;
        *bb_en              = FC_BB_EN;
        *conv_en            = FC_CONV_EN;
        *bn_en              = FC_BN_EN;
        *skip_en            = FC_SKIP_EN;
        *relu_en            = FC_RELU_EN;
        *max_pool_en        = FC_MAX_POOL_EN;
        *avg_pool_en        = FC_AVG_POOL_EN;
        *fc_en             = FC_FC_EN;
        *base_addr_in       = FC_BASE_ADDR_IN;
        *base_addr_out      = FC_BASE_ADDR_OUT;
        *base_addr_add      = FC_BASE_ADDR_ADD;
        *weight_base        = FC_WEIGHT_BASE;
        *weight_size        = FC_WEIGHT_SIZE;
        *bn_weight_base     = FC_BN_WEIGHT_BASE;
        *bn_weight_size     = FC_BN_WEIGHT_SIZE;
        *in_size            = FC_IN_SIZE;
        *out_size           = FC_OUT_SIZE;
    }
    
}

void conv_kernel(
    DTYPE_ACT *act_in_host,
    DTYPE_ACT *act_out_host,
    DTYPE_FIL *weight_mem,
    float *bn_weight_mem,
    unsigned *start_layer,
    unsigned *end_layer
) {
    // interface
    // #pragma HLS INTERFACE s_axilite port=return bundle=control
    // #pragma HLS INTERFACE mode=m_axi port=act_mem offset=slave bundle=gmem0 depth = 10000000
    // #pragma HLS INTERFACE mode=m_axi port=weight_mem offset=slave bundle=gmem0 depth = 10000000
    // #pragma HLS INTERFACE mode=m_axi port=bn_weight_mem offset=slave bundle=gmem0 depth = 10000000
    // #pragma HLS INTERFACE mode=s_axilite port=act_mem bundle=control
    // #pragma HLS INTERFACE mode=s_axilite port=weight_mem bundle=control
    // #pragma HLS INTERFACE mode=s_axilite port=bn_weight_mem bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS INTERFACE mode=m_axi port=act_in_host offset=slave bundle=gmem0 depth = MAX_ACT_MEM_SIZE
    #pragma HLS INTERFACE mode=m_axi port=weight_mem offset=slave bundle=gmem0 depth = WEIGHT_MEM_SIZE
    #pragma HLS INTERFACE mode=m_axi port=bn_weight_mem offset=slave bundle=gmem0 depth = BN_WEIGHT_MEM_SIZE
    #pragma HLS INTERFACE mode=m_axi port=act_out_host offset=slave bundle=gmem0 depth = MAX_ACT_MEM_SIZE
    #pragma HLS INTERFACE mode=s_axilite port=act_in_host bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=weight_mem bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=bn_weight_mem bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=act_out_host bundle=control
    #pragma HLS INTERFACE mode=m_axi port=start_layer offset=slave bundle=gmem0 depth = 1
    #pragma HLS INTERFACE mode=m_axi port=end_layer offset=slave bundle=gmem0 depth = 1
    #pragma HLS INTERFACE mode=s_axilite port=start_layer bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=end_layer bundle=control
    
    DTYPE_MEM_ACT act_mem[ACT_MEM_SIZE];
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
    
    unsigned layer_cnt;
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
    bool fc_en;
    unsigned base_addr_in;
    unsigned base_addr_out;
    unsigned base_addr_add;
    unsigned weight_base;
    unsigned weight_size;
    unsigned bn_weight_base;
    unsigned bn_weight_size;
    unsigned in_size;
    unsigned out_size;

    // for (layer_cnt = *start_layer; layer_cnt <= *end_layer; layer_cnt++) {
    //     controller (
    //         &layer_cnt,
    //         &nif,
    //         &nof,
    //         &noy,
    //         &nox,
    //         &nkx,
    //         &nky,
    //         &stride,
    //         &pad,
    //         &bb_en,
    //         &conv_en,
    //         &bn_en,
    //         &skip_en,
    //         &relu_en,
    //         &max_pool_en,
    //         &avg_pool_en,
    //         &fc_en,
    //         &base_addr_in,
    //         &base_addr_out,
    //         &base_addr_add,
    //         &weight_base,
    //         &weight_size,
    //         &bn_weight_base,
    //         &bn_weight_size,
    //         &in_size,
    //         &out_size
    //     );
    //     // initial input
    //     if (layer_cnt == *start_layer) {
    //         // load input
    //         for (int idx = 0; idx < in_size/ACT_PACK; idx++){
    //             DTYPE_MEM_ACT pack;
    //             for (int jdx = 0; jdx < ACT_PACK; jdx++) {
    //                 pack.range(W_ACT*(jdx+1)-1, W_ACT*jdx) = act_in_host[idx*ACT_PACK+jdx];
    //             }
    //             act_mem[base_addr_in+idx] = pack;
    //         }
    //     }
//
//        // conv
//        load_input(act_mem, load_input_fifo, base_addr_in,
//                nky, nkx, nof, nif, noy, nox, stride, pad, bb_en, conv_en);
//        load_weight(weight_mem, load_weight_fifo, weight_base,
//                nky, nkx, nof, nif, noy, nox, bb_en, conv_en);
//        PE(load_input_fifo, load_weight_fifo, pe_out_fifo,
//                nky, nkx, nof, nif, noy, nox, bb_en, conv_en);
//        batch_norm(bn_weight_mem, pe_out_fifo, bn_out_fifo, bn_weight_base,
//                nof, noy, nox, bb_en, bn_en);
//        skip_conn(act_mem, bn_out_fifo, skip_out_fifo, base_addr_add,
//                nof, noy, nox, bb_en, skip_en, relu_en);
//        store_output(act_mem, skip_out_fifo, base_addr_out, 
//                nky, nkx, nof, nif, noy, nox, bb_en);
//                
//        // max pool
//        max_pool(act_mem, base_addr_in, base_addr_out, 
//                nky, nkx, nof, nif, noy, nox, stride, pad, max_pool_en);
//        
//        // avg pool
//        avg_pool(act_mem, base_addr_in, base_addr_out, 
//                nky, nkx, nof, nif, noy, nox, stride, pad, avg_pool_en);
//
//        // fc
//        fc(act_mem, bn_weight_mem, base_addr_in, base_addr_out, bn_weight_base,
//                nof, nif, fc_en);
//
//        // output back to host
//        if (layer_cnt == *end_layer) {
//            for (int idx = 0; idx < out_size/ACT_PACK; idx++){
//                for (int jdx = 0; jdx < ACT_PACK; jdx++) {
//                    act_out_host[idx*ACT_PACK+jdx] = act_mem[base_addr_out+idx].range(W_ACT*(jdx+1)-1, W_ACT*jdx);
//                }
//            }
//        }
//
}

#endif
