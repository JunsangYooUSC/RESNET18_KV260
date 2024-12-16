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

#endif
