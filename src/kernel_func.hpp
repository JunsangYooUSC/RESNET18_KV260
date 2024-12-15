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

#ifndef KERNEL_FUNC_HPP
#define KERNEL_FUNC_HPP

#include <iostream>
#include <iomanip> // For std::setw

// Include Vitis HLS headers
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_vector.h>
#include <hls_stream.h>
// Include project headers
#include "conv_config.h"
#include "dummy_func.h"

void PE(
    hls::stream<DTYPE_ACT> mac_in_fifo_arr[POY][POX],
    hls::stream<DTYPE_FIL> weight_in_fifo_arr[POF],
    hls::stream<DTYPE_MAC> out_fifo_arr[POF][POY][POX],
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox
) {
    DTYPE_MUL mul_vals[POF][POY][POX];
    DTYPE_MAC mac_vals[POF][POY][POX];
    DTYPE_ACT in_vals[POY][POX];
    DTYPE_FIL fil_vals[POF];

    for (int i = 0; i < (nof*noy*nox/POF/POY/POX); i++) {
        // initialize mac
        for (int f = 0; f < POF; f++) {
#pragma HLS unroll
            for (int y = 0; y < POY; y++) {
#pragma HLS unroll
                for (int x = 0; x < POX; x++) {
#pragma HLS unroll
                    mac_vals[f][y][x] = 0;
                }
            }
        }

        for (int loop = 0; loop < nky*nkx*nif; loop++) {
            // read input
            for (int y = 0; y < POY; y++) {
#pragma HLS unroll
                for (int x = 0; x < POX; x++) {
#pragma HLS unroll
                    in_vals[y][x] = mac_in_fifo_arr[y][x].read();
                }
            }
            // read weight
            for (int f = 0; f < POF; f++) {
#pragma HLS unroll
                fil_vals[f] = weight_in_fifo_arr[f].read();
            }
            // compute
            for (int f = 0; f < POF; f++) {
#pragma HLS unroll
                for (int y = 0; y < POY; y++) {
#pragma HLS unroll
                    for (int x = 0; x < POX; x++) {
#pragma HLS unroll
                        mul_vals[f][y][x] = in_vals[y][x] * fil_vals[f];
                        mac_vals[f][y][x] += mul_vals[f][y][x];
                    }
                }
            }
        }

        // 
        for (int f = 0; f < POF; f++) {
#pragma HLS unroll
            for (int y = 0; y < POY; y++) {
#pragma HLS unroll
                for (int x = 0; x < POX; x++) {
#pragma HLS unroll
                    out_fifo_arr[f][y][x].write(mac_vals[f][y][x]);
                }
            }
        }
    }
}

// BUF2PE with support of stride
void BUF2PE_stride(
    // DTYPE_ACT *input_buffer, 
    // DTYPE_ACT input_buffer_stride[2][POY*MAX_STRIDE+PAD*2][POX*MAX_STRIDE+PAD*2],
    DTYPE_MEM act_mem[2][ACT_MEM_SIZE],
    hls::stream<DTYPE_ACT> mac_in_fifo_arr[POY][POX],
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox,
    unsigned int s,
    unsigned int pad,
    unsigned int db_idx     // double buffering index) 
) {
    DTYPE_ACT input_buffer_stride[2][POY*MAX_STRIDE+PAD*2][POX*MAX_STRIDE+PAD*2];
    // register R* in fig13 of Optimizing_the_Convolution_Operation_to_Accelerate_Deep_Neural_Networks_on_FPGA
    DTYPE_ACT buf2pe_reg_stride[POY*MAX_STRIDE][POX*MAX_STRIDE+1];
    // #pragma HLS BIND_STORAGE variable=buf2pe_reg type=register
    // fifo in fig13 of Optimizing_the_Convolution_Operation_to_Accelerate_Deep_Neural_Networks_on_FPGA
    hls::stream<DTYPE_ACT> fifo_arr_stride[POY*MAX_STRIDE-1][POX*MAX_STRIDE];
    #pragma HLS STREAM variable=fifo_arr_stride depth=FIFO_ARR_DEPTH
    for (int f_out = 0; f_out < nof; f_out += POF) {
        for (int y0 = 0; y0 < noy*s; y0 += POY*s) {
            for (int x0 = 0; x0 < nox*s; x0 += POX*s) {
                for (int f_in = 0; f_in < nif; f_in ++) {
                    // load input buffer
                    // unsigned int act_mem_base_idx = f_in*noy*s*nox*s + y0*nox*s + x0;
                    for (int y = 0; y < POY*s+pad*2; y++) {
                        for (int x = 0; x < POX*s+pad*2; x++) {
                            // zero padding cond
                            bool zero_pad_cond = (y0 + y < pad) || (y0 + y >= noy*s + pad) || (x0 + x < pad) || (x0 + x >= nox*s + pad);
                            DTYPE_ACT data;
                            if (zero_pad_cond) {
                                data = 0;
                            } 
                            else {
                                unsigned int mem_idx = f_in*noy*s*nox*s + (y0+y-pad)*nox*s + (x0+x-pad);
                                unsigned int idx1 = mem_idx / MEM_PACK;
                                unsigned int idx2 = mem_idx % MEM_PACK;
                                DTYPE_MEM block = act_mem[0][idx1];
                                data.range() = block.range(W_ACT*(idx2+1)-1,W_ACT*idx2);
                            }
                            input_buffer_stride[0][y][x] = data;
                        }
                    }
                    // intra input tile
                    for (unsigned int cnt = 0; cnt < nky*nkx; cnt++) {
                        // last poy
                        // copy input buffer content to register
                        unsigned int last_y_idx = cnt / nky;
                        unsigned int last_x_idx = cnt % nkx;
                        if (cnt % nkx == 0) {
                            for (int x = 0; x < POX*s+1; x++) {
#pragma HLS unroll
                                // first row of input buffer
                                buf2pe_reg_stride[POY*s-1][x] = input_buffer_stride[db_idx][POY*s-1+last_y_idx][x];
                            }
                        }
                        // last register gets new val. rest values are fed from adjacent reg
                        else if (cnt % nkx < nkx-1) {
                            for (int x = 0; x < POX*s; x++) {
#pragma HLS unroll
                                buf2pe_reg_stride[POY*s-1][x] = buf2pe_reg_stride[POY*s-1][x+1];
                            }
                            buf2pe_reg_stride[POY*s-1][POX*s] = input_buffer_stride[db_idx][POY*s-1+last_y_idx][POX*s+last_x_idx];
                        }
                        // reg values are fed from adjacent reg
                        else if (cnt % nkx == nkx-1) {
                            for (int x = 0; x < POX*s; x++) {
#pragma HLS unroll
                                buf2pe_reg_stride[POY*s-1][x] = buf2pe_reg_stride[POY*s-1][x+1];
                            }
                        }
                        if (cnt < nkx*nky-nkx) {
                            // feed into previous fifo
                            for (int x = 0; x < POX*s; x++) {
#pragma HLS unroll
                                fifo_arr_stride[POY*s-2][x].write(buf2pe_reg_stride[POY*s-1][x]);
                            }
                        }

                        // middle poy
                        #pragma HLS unroll
                        for (int y = 1; y < POY*s-1; y++) {
                            // copy input buffer content to register at first cycle
                            if (cnt == 0) {
                                for (int x = 0; x < POX*s+1; x++) {
#pragma HLS unroll
                                    // first row of input buffer
                                    buf2pe_reg_stride[y][x] = input_buffer_stride[db_idx][y][x];
                                }
                            }
                            // last register gets new val. rest values are fed from adjacent reg
                            else if (cnt < nkx-1) {
                                for (int x = 0; x < POX*s; x++) {
#pragma HLS unroll
                                    buf2pe_reg_stride[y][x] = buf2pe_reg_stride[y][x+1];
                                }
                                buf2pe_reg_stride[y][POX*s] = input_buffer_stride[db_idx][y][POX*s+cnt];
                            }
                            // reg values are fed from adjacent reg
                            else if (cnt == nkx-1) {
                                for (int x = 0; x < POX*s; x++) {
#pragma HLS unroll
                                    buf2pe_reg_stride[y][x] = buf2pe_reg_stride[y][x+1];
                                }
                            }
                            // get values from FIFO
                            if (cnt >= nkx) {
                                for (int x = 0; x < POX*s; x++) {
#pragma HLS unroll
                                    buf2pe_reg_stride[y][x] = fifo_arr_stride[y][x].read();
                                    //std::cout << "read x: " << x << ", y: " << y << std::endl;
                                }
                            }
                            // feed into previous fifo
                            if (cnt < nkx*nky-nkx) {
                                for (int x = 0; x < POX*s; x++) {
#pragma HLS unroll
                                    fifo_arr_stride[y-1][x].write(buf2pe_reg_stride[y][x]);
                                    //std::cout << "write x: " << x << ", y: " << y << std::endl;
                                }
                            }
                        }

                        // first poy
                        // copy input buffer content to register at first cycle
                        if (cnt == 0) {
                            for (int x = 0; x < POX*s+1; x++) {
#pragma HLS unroll
                                // first row of input buffer
                                buf2pe_reg_stride[0][x] = input_buffer_stride[db_idx][0][x];
                            }
                        }
                        // last register gets new val. rest values are fed from adjacent reg
                        else if (cnt < nkx-1) {
                            for (int x = 0; x < POX*s; x++) {
#pragma HLS unroll
                                buf2pe_reg_stride[0][x] = buf2pe_reg_stride[0][x+1];
                            }
                            buf2pe_reg_stride[0][POX*s] = input_buffer_stride[db_idx][0][POX*s+cnt];
                        }
                        // reg values are fed from adjacent reg
                        else if (cnt == nkx-1) {
                            for (int x = 0; x < POX*s; x++) {
#pragma HLS unroll
                                buf2pe_reg_stride[0][x] = buf2pe_reg_stride[0][x+1];
                            }
                        }
                        // get values from FIFO
                        if (cnt >= nkx) {
                            for (int x = 0; x < POX*s; x++) {
#pragma HLS unroll
                                buf2pe_reg_stride[0][x] = fifo_arr_stride[0][x].read();
                            }
                        }

                        // feed mac unit
                        for (int y = 0; y < POY; y++) {
#pragma HLS unroll
                            for (int x = 0; x < POX; x++) {
#pragma HLS unroll
                                mac_in_fifo_arr[y][x].write(buf2pe_reg_stride[y*s][x*s]);
                            }
                        }
                    }
                    
                }
            }
        }
    }
}

void load_weight_fifo(
    DTYPE_FIL fil_mem[FIL_MEM_SIZE],
    hls::stream<DTYPE_FIL> weight_in_fifo_arr[POF],
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox

) {
    DTYPE_FIL filter_buffer[2][POF][NIF][NKY][NKX];  // todo: use double buffer
    for (int f_out = 0; f_out < nof; f_out += POF) {
        for (int y0 = 0; y0 < noy; y0+=POY) {
            for (int x0 = 0; x0 < nox; x0+=POX) {
                // 
                if ( (y0 == 0) && (x0 == 0) ) {
                    for (int f_in = 0; f_in < nif; f_in++){
                        for (int f = 0; f < POF; f++) {
                            for (int y = 0; y < nky; y++) {
                                for (int x = 0; x < nkx; x++) {
                                    unsigned int fil_idx = (f+f_out)*nif*nky*nkx + f_in*nky*nkx + y*nkx + x;
                                    filter_buffer[0][f][f_in][y][x] = fil_mem[fil_idx];
                                }
                            }
                        }
                    }
                }
                // reuse loop times
                for (int f_in = 0; f_in < nif; f_in++){
                    for (int y = 0; y < nky; y++) {
                        for (int x = 0; x < nkx; x++) {
                            load_weight_loop1:
                            for (int f = 0; f < POF; f++) {
#pragma unroll
                                weight_in_fifo_arr[f].write(filter_buffer[0][f][f_in][y][x]);
                            }
                        }
                    }
                }
            }
        }
    }
}

void store_output_fifo(
    DTYPE_MEM act_mem[2][ACT_MEM_SIZE],
    hls::stream<DTYPE_MAC> out_fifo_arr[POF][POY][POX],
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox,
    unsigned int db_mem     // double bufferingn index
) {
    DTYPE_ACT output_buffer[2][POF][POY][POX];
    for (int out_f = 0; out_f < nof; out_f+=POF) {
        for (int y0 = 0; y0 < noy; y0+=POY) {
            for (int x0 = 0; x0 < nox; x0+=POX) {
                // parallel
                for (int f = 0; f < POF; f++) {
#pragma HLS unroll
                    for (int y = 0; y < POY; y++) {
#pragma HLS unroll
                        for (int x = 0; x < POX; x++) {
#pragma HLS unroll
                            output_buffer[0][f][y][x] = out_fifo_arr[f][y][x].read();
                        }
                    }
                }
                for (int f = 0; f < POF; f++) {
                    for (int y = 0; y < POY; y++) {
                        unsigned int act_mem_idx = ((out_f+f)*noy*nox + (y0+y)*nox + x0) / MEM_PACK;
                        DTYPE_MEM block;
                        for (int x = 0; x < POX; x++) {
                            block.range(W_ACT*(x+1)-1, W_ACT*x) = output_buffer[0][f][y][x].range();
                        }
                        act_mem[db_mem][act_mem_idx] = block;
                    }
                }
            }
        }
    }
}

#endif
