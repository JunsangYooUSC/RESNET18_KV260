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

// Include Vitis HLS headers
#include "ap_fixed.h"
// Include project headers
#include "conv_config.h"
#include "kernel.h"



// kernel function
void kernel_func(DTYPE_ACT *in_host,
                DTYPE_ACT *filter_offchip,
                DTYPE_ACT *out_host
) {
    // on-chip memory
    DTYPE_ACT input_buffer[Nif][Noy][Nox];
	#pragma HLS bind_storage variable=input_buffer type=RAM_1P impl=uram
    // #pragma HLS array_partition type=complete variable=input_buffer
    DTYPE_ACT output_buffer[Nof][Noy][Nox];
    #pragma HLS bind_storage variable=output_buffer type=RAM_1P impl=uram
    // #pragma HLS array_partition type=complete variable=output_buffer
    DTYPE_ACT weight_buffer[Pof][Nif][Tky][Tkx];
	// #pragma HLS bind_storage variable=weight_buffer type=RAM_1P impl=uram
	// #pragma HLS array_partition type=complete variable=weight_buffer
    DTYPE_MAC mac_buffer[Pof][Poy][Pox];

    // off-chip memory
    #pragma HLS INTERFACE mode=m_axi port=filter_offchip offset=slave bundle=gmem0
    #pragma HLS INTERFACE mode=s_axilite port=filter_offchip bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    for (int i = 0; i < Nif; i++) {
        for (int j = 0; j < Noy; j++) {
            for (int k = 0; k < Nox; k++) {
                input_buffer[i][j][k] = in_host[i*Noy*Nox + j*Nox + k];
            }
        }
    }

    // weight is stationary
    #pragma HLS pipeline off
    kernelFuncLoop1:
    for (int nof_idx = 0; nof_idx < Nof; nof_idx += Tof) {
        // #pragma HLS pipeline II=1
        weightLoadLoop1:
        for (int tof_idx = 0; tof_idx < Tof; tof_idx++) {
            // #pragma HLS pipeline II=1
            weightLoadLoop2:
            for (int nif_idx = 0; nif_idx < Nif; nif_idx++) {
                // #pragma HLS pipeline II=1
                weightLoadLoop3:
                for (int nky_idx = 0; nky_idx < Nky; nky_idx++) {
                    // #pragma HLS pipeline II=1
                    weightLoadLoop4:
                    for (int nkx_idx = 0; nkx_idx < Nkx; nkx_idx++) {
                        weight_buffer[tof_idx][nif_idx][nky_idx][nkx_idx] = filter_offchip[(nof_idx+tof_idx)*Nif*Nky*Nkx + nif_idx*Nky*Nkx + nky_idx*Nkx + nkx_idx];
                    }
                }
            }
        }
        kernelFuncLoop2:
        for (int noy_idx = 0; noy_idx < Noy; noy_idx += Toy) {
            kernelFuncLoop3:
            for (int nox_idx = 0; nox_idx < Nox; nox_idx += Tox) {
                // mac buffer init
                macBufferInitLoop1:
                for (int pof_idx = 0; pof_idx < Pof; pof_idx++) {
                    macBufferInitLoop2:
                    for (int poy_idx = 0; poy_idx < Poy; poy_idx++) {
                        macBufferInitLoop3:
                        for (int pox_idx = 0; pox_idx < Pox; pox_idx++) {
                            mac_buffer[pof_idx][poy_idx][pox_idx] = 0;
                        }
                    }
                }
                
                // pipelined computation
                kernelFuncLoop4:
                for (int nif_idx = 0; nif_idx < Nif; nif_idx += Tif) {
                    kernelFuncLoop5:
                    for (int tif_idx = 0; tif_idx < Tif; tif_idx += 1) {
                        kernelFuncLoop6:
                        for (int tky_idx = 0; tky_idx < Tky; tky_idx++) {
                            kernelFuncLoop7:
                            for (int tkx_idx = 0; tkx_idx < Tkx; tkx_idx++) {
                                // parallel computation
                                // #pragma HLS unroll
                                kernelFuncLoop8:
                                for (int pof_idx = 0; pof_idx < Pof; pof_idx++) {
                                    // #pragma HLS unroll
                                    kernelFuncLoop9:
                                    for (int poy_idx = 0; poy_idx < Poy; poy_idx++) {
                                        // #pragma HLS unroll
                                        kernelFuncLoop10:
                                        for (int pox_idx = 0; pox_idx < Pox; pox_idx++) {
                                            int inx_idx = nox_idx + pox_idx + tkx_idx;
                                            int iny_idx = noy_idx + poy_idx + tky_idx;
                                            // input value
                                            DTYPE_ACT in_val;
                                            if ( (inx_idx < pad) || (inx_idx >= Nox + pad) || (iny_idx < pad) || (iny_idx >= Nox + pad) )
                                                in_val = 0;
                                            else
                                                in_val = input_buffer[nif_idx+tif_idx][iny_idx - pad][inx_idx - pad];
                                            // weight
                                            DTYPE_FILTER w_val = weight_buffer[pof_idx][nif_idx+tif_idx][tky_idx][tkx_idx];
                                            // DTYPE_MULT mult_val = in_val * w_val;
                                            DTYPE_MULT mult_val = in_val;
                                            mac_buffer[pof_idx][poy_idx][pox_idx] += mult_val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // update output buffer
                outBufferLoop1:
                for (int pof_idx = 0; pof_idx < Pof; pof_idx++) {
                    outBufferLoop2:
                    for (int poy_idx = 0; poy_idx < Poy; poy_idx++) {
                        outBufferLoop3:
                        for (int pox_idx = 0; pox_idx < Pox; pox_idx++) {
                            output_buffer[nof_idx + pof_idx][noy_idx + poy_idx][nox_idx + pox_idx] = mac_buffer[pof_idx][poy_idx][pox_idx];
                        }
                    }
                }
            }
        }
    }
    for (int idx = 0; idx < Nof; idx++) {
        for (int jdx = 0; jdx < Noy; jdx++) {
            for (int kdx = 0; kdx < Nox; kdx++) {
                out_host[idx*Noy*Nox + jdx*Nox + kdx] = output_buffer[idx][jdx][kdx];
            }
        }
    }
}

#endif
