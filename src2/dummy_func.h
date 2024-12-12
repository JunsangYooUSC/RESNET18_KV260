#ifndef DUMMY_FUNC_H
#define DUMMY_FUNC_H

// Include Vitis HLS headers
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_vector.h>
#include <hls_stream.h>
// Include project headers
#include "conv_config.h"

void dummy_fill_input_buffer(DTYPE_ACT input_buffer[2][POY+PAD*2][POX+PAD*2]){
    DTYPE_ACT val = 0;
    DTYPE_ACT step = 1;
    step = step >> (W_ACT-I_ACT);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < POY+PAD*2; j++) {
            for (int k = 0; k < POX+PAD*2; k++) {
                input_buffer[i][j][k] = val;
                val += step;
            }
        }
    }
}
#endif
