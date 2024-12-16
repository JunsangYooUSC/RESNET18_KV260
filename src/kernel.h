#ifndef KERNEL_H
#define KERNEL_H

// Include Vitis HLS headers
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_vector.h>
#include <hls_stream.h>
// Include project headers
#include "conv_config.h"

void kernel_func(DTYPE_ACT *in_host, DTYPE_FIL *weight_mem, float *bn_weight_mem, DTYPE_ACT *out_host);

void PE(
    hls::stream<DTYPE_ACT> mac_in_fifo_arr[POY][POX],
    hls::stream<DTYPE_FIL> weight_in_fifo_arr[POF],
    hls::stream<float> out_fifo_arr[POF][POY][POX],
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox
);

void BUF2PE_stride(
    DTYPE_MEM_ACT *mem,
    hls::stream<DTYPE_ACT> mac_in_fifo_arr[POY][POX],
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox,
    unsigned int s,
    unsigned int pad
);

void load_weight_fifo(
    DTYPE_FIL *mem_fil,
    hls::stream<DTYPE_FIL> weight_in_fifo_arr[POF],
    unsigned int base_addr,
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox
);

void store_output_fifo(
    DTYPE_MEM_ACT *mem,
    hls::stream<float> out_fifo_arr[POF][POY][POX],
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox
);

void conv_pass(
    DTYPE_MEM_ACT *mem_in,
    hls::stream<float> out_fifo_arr[POF][POY][POX],
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox,
    unsigned int stride,
    unsigned int pad
);

void conv(
    DTYPE_MEM_ACT *mem_in,
    DTYPE_FIL *mem_fil,
    hls::stream<float> out_fifo_arr[POF][POY][POX],
    unsigned int weight_base_addr,
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

void batch_norm(
    float *bn_weight_mem,
    hls::stream<float> in_fifo_arr[POF][POY][POX],
    hls::stream<float> out_fifo_arr[POF][POY][POX],
    unsigned int bn_weight_base_addr,
    unsigned int nof,
    unsigned int noy,
    unsigned int nox,
    unsigned int bb_en,
    unsigned int bn_en
);

#endif
