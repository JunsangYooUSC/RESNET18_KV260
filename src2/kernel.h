#ifndef KERNEL_H
#define KERNEL_H

// Include Vitis HLS headers
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_vector.h>
#include <hls_stream.h>
// Include project headers
#include "conv_config.h"

// Declare functions
void BUF2PE(
    // DTYPE_ACT *input_buffer, 
    DTYPE_ACT input_buffer[2][POY+PAD*2][POX+PAD*2],
    hls::stream<DTYPE_ACT> mac_in_fifo_arr[POY][POX],
    unsigned int nkx,
    unsigned int nky,
    unsigned int total_loops,       // 
    unsigned int db_idx     // double buffering index) 
);

void kernel_func(DTYPE_ACT *in_offchip, DTYPE_ACT *filter_offchip, DTYPE_ACT *out_offchip);

#endif
