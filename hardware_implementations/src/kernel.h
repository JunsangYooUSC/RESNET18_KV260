#ifndef KERNEL_H
#define KERNEL_H

// Include Vitis HLS headers
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_vector.h>
#include <hls_stream.h>
#include <algorithm>

// Include project headers
#include "conv_config.h"

void conv_kernel(
    DTYPE_ACT *act_mem,
    DTYPE_ACT *act_in,
    DTYPE_ACT *act_out,
    DTYPE_FIL *weight_mem,
    float *bn_weight_mem,
    unsigned *start_layer,
    unsigned *end_layer
);

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
    bool *lin_en,
    unsigned *base_addr_in,
    unsigned *base_addr_out,
    unsigned *base_addr_add,
    unsigned *weight_base,
    unsigned *weight_size,
    unsigned *bn_weight_base,
    unsigned *bn_weight_size,
    unsigned *in_size,
    unsigned *out_size
);

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
    unsigned int pad,
    unsigned int bb_en,
    unsigned int conv_en
);

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
);

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
);

void store_output(
    DTYPE_ACT *act_mem,
    hls::stream<float> &out_fifo,
    unsigned int base_addr,
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox,
    unsigned int bb_en
);

void max_pool(
    DTYPE_ACT *act_mem,
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
);

void avg_pool(
    DTYPE_ACT *act_mem,
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
);

void fc(
    DTYPE_ACT *act_mem,
    float *bn_weight_mem,
    unsigned int in_base_addr,
    unsigned int out_base_addr,
    unsigned int bn_weight_base_addr,
    unsigned int nof,
    unsigned int nif,
    unsigned int fc_en
);



#endif
