#ifndef KERNEL_H
#define KERNEL_H

// Include Vitis HLS headers
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_vector.h>
#include <hls_stream.h>
// Include project headers
#include "conv_config.h"

void kernel(
    DTYPE_ACT *act_mem,
    DTYPE_FIL *weight_mem,
    float *bn_weight_mem
);


void kernel_func(DTYPE_ACT *in_host, DTYPE_FIL *weight_mem, float *bn_weight_mem, DTYPE_ACT *out_host);

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
);

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
);

#endif
