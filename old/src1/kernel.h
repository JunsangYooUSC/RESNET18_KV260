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
template<typename DTYPE, unsigned int LEN>
void BUF2PE(DTYPE_ACT *input_buffer, hls::stream<BUF2PEVEC> fifo_arr[POF][POY-1], hls::stream<BUF2PEVEC> mac_in_fifo_arr[POF][POY]);

void kernel_func(DTYPE_ACT *in_offchip, DTYPE_ACT *filter_offchip, DTYPE_ACT *out_offchip);

#endif
