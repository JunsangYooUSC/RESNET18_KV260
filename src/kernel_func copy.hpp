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
            for (int y = 0; y < POY; y++) {
                for (int x = 0; x < POX; x++) {
                    mac_vals[f][y][x] = 0;
                }
            }
        }

        for (int loop = 0; loop < nky*nkx*nif; loop++) {
            // read input
            for (int y = 0; y < POY; y++) {
                for (int x = 0; x < POX; x++) {
                    in_vals[y][x] = mac_in_fifo_arr[y][x].read();
                }
            }
            // read weight
            for (int f = 0; f < POF; f++) {
                fil_vals[f] = weight_in_fifo_arr[f].read();
            }
            // compute
            for (int f = 0; f < POF; f++) {
                for (int y = 0; y < POY; y++) {
                    for (int x = 0; x < POX; x++) {
                        mul_vals[f][y][x] = in_vals[y][x] * fil_vals[f];
                        mac_vals[f][y][x] += mul_vals[f][y][x];
                    }
                }
            }
        }

        // 
        for (int f = 0; f < POF; f++) {
            for (int y = 0; y < POY; y++) {
                for (int x = 0; x < POX; x++) {
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
    DTYPE_MEM *act_mem[2][ACT_MEM_SIZE],
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
        for (int f_in = 0; f_in < nif; f_in ++) {
            // inter input tile
            for (int y0 = 0; y0 < noy; y0 += POY) {
                for (int x0 = 0; x0 < nox; x0 += POX) {
                    // load input buffer
                    for (int y = 0; y < POY*s+pad*2; y++) {
                        for (int x = 0; x < POX*s+pad*2; x++) {
                            // zero padding
                            if ( (y0 + y < pad) || (y0 + y >= noy*s + pad) || (x0 + x < pad) || (x0 + x >= nox*s + pad) ) {
                                input_buffer_stride[0][y][x] = 0;
                            }
                            else {
                                unsigned int act_mem_idx = act_mem_base_idx + y * nox*s + x;
                                unsigned int idx1 = act_mem_idx / MEM_PACK;
                                unsigned int idx2 = act_mem_idx % MEM_PACK;
                                DTYPE_MEM block = act_mem[0][idx1];
                                DTYPE_ACT data;
                                data.range() = block.range(W_ACT*(idx2+1)-1,W_ACT*idx2);
                                input_buffer_stride[0][y][x] = data;
                            }
                        }
                    }
                    // intra input tile
                    for (unsigned int cnt = 0; cnt < nky*nkx; cnt++) {
                        // last poy
                        // copy input buffer content to register
                        unsigned int last_y_idx = cnt / nky;
                        unsigned int last_x_idx = cnt % nkx;
                        if (cnt % nkx == 0) {
                            #pragma HLS unroll
                            for (int x = 0; x < POX*s+1; x++) {
                                // first row of input buffer
                                buf2pe_reg_stride[POY*s-1][x] = input_buffer_stride[db_idx][POY*s-1+last_y_idx][x];
                            }
                        }
                        // last register gets new val. rest values are fed from adjacent reg
                        else if (cnt % nkx < nkx-1) {
                            #pragma HLS unroll
                            for (int x = 0; x < POX*s; x++) {
                                buf2pe_reg_stride[POY*s-1][x] = buf2pe_reg_stride[POY*s-1][x+1];
                            }
                            buf2pe_reg_stride[POY*s-1][POX*s] = input_buffer_stride[db_idx][POY*s-1+last_y_idx][POX*s+last_x_idx];
                        }
                        // reg values are fed from adjacent reg
                        else if (cnt % nkx == nkx-1) {
                            #pragma HLS unroll
                            for (int x = 0; x < POX*s; x++) {
                                buf2pe_reg_stride[POY*s-1][x] = buf2pe_reg_stride[POY*s-1][x+1];
                            }
                        }
                        if (cnt < nkx*nky-nkx) {
                            // feed into previous fifo
                            #pragma HLS unroll
                            for (int x = 0; x < POX*s; x++) {
                                fifo_arr_stride[POY*s-2][x].write(buf2pe_reg_stride[POY*s-1][x]);
                            }
                        }

                        // middle poy
                        #pragma HLS unroll
                        for (int y = 1; y < POY*s-1; y++) {
                            // copy input buffer content to register at first cycle
                            if (cnt == 0) {
                                #pragma HLS unroll
                                for (int x = 0; x < POX*s+1; x++) {
                                    // first row of input buffer
                                    buf2pe_reg_stride[y][x] = input_buffer_stride[db_idx][y][x];
                                }
                            }
                            // last register gets new val. rest values are fed from adjacent reg
                            else if (cnt < nkx-1) {
                                #pragma HLS unroll
                                for (int x = 0; x < POX*s; x++) {
                                    buf2pe_reg_stride[y][x] = buf2pe_reg_stride[y][x+1];
                                }
                                buf2pe_reg_stride[y][POX*s] = input_buffer_stride[db_idx][y][POX*s+cnt];
                            }
                            // reg values are fed from adjacent reg
                            else if (cnt == nkx-1) {
                                #pragma HLS unroll
                                for (int x = 0; x < POX*s; x++) {
                                    buf2pe_reg_stride[y][x] = buf2pe_reg_stride[y][x+1];
                                }
                            }
                            // get values from FIFO
                            if (cnt >= nkx) {
                                #pragma HLS unroll
                                for (int x = 0; x < POX*s; x++) {
                                    buf2pe_reg_stride[y][x] = fifo_arr_stride[y][x].read();
                                    //std::cout << "read x: " << x << ", y: " << y << std::endl;
                                }
                            }
                            // feed into previous fifo
                            if (cnt < nkx*nky-nkx) {
                                #pragma HLS unroll
                                for (int x = 0; x < POX*s; x++) {
                                    fifo_arr_stride[y-1][x].write(buf2pe_reg_stride[y][x]);
                                    //std::cout << "write x: " << x << ", y: " << y << std::endl;
                                }
                            }
                        }

                        // first poy
                        // copy input buffer content to register at first cycle
                        if (cnt == 0) {
                            #pragma HLS unroll
                            for (int x = 0; x < POX*s+1; x++) {
                                // first row of input buffer
                                buf2pe_reg_stride[0][x] = input_buffer_stride[db_idx][0][x];
                            }
                        }
                        // last register gets new val. rest values are fed from adjacent reg
                        else if (cnt < nkx-1) {
                            #pragma HLS unroll
                            for (int x = 0; x < POX*s; x++) {
                                buf2pe_reg_stride[0][x] = buf2pe_reg_stride[0][x+1];
                            }
                            buf2pe_reg_stride[0][POX*s] = input_buffer_stride[db_idx][0][POX*s+cnt];
                        }
                        // reg values are fed from adjacent reg
                        else if (cnt == nkx-1) {
                            #pragma HLS unroll
                            for (int x = 0; x < POX*s; x++) {
                                buf2pe_reg_stride[0][x] = buf2pe_reg_stride[0][x+1];
                            }
                        }
                        // get values from FIFO
                        if (cnt >= nkx) {
                            #pragma HLS unroll
                            for (int x = 0; x < POX*s; x++) {
                                buf2pe_reg_stride[0][x] = fifo_arr_stride[0][x].read();
                            }
                        }

                        // feed mac unit
                        #pragma HLS unroll
                        for (int y = 0; y < POY; y++) {
                            #pragma HLS unroll
                            for (int x = 0; x < POX; x++) {
                                mac_in_fifo_arr[y][x].write(buf2pe_reg_stride[y*s][x*s]);
                            }
                        }
                    }
                    
                }
            }
        }
    }
}

// BUF2PE with support of stride
void BUF2PE_stride_old(
    // DTYPE_ACT *input_buffer, 
    DTYPE_ACT input_buffer_stride[2][POY*MAX_STRIDE+PAD*2][POX*MAX_STRIDE+PAD*2],
    hls::stream<DTYPE_ACT> mac_in_fifo_arr[POY][POX],
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox
    unsigned int s,
    unsigned int db_idx     // double buffering index) 
) {
    // register R* in fig13 of Optimizing_the_Convolution_Operation_to_Accelerate_Deep_Neural_Networks_on_FPGA
    DTYPE_ACT buf2pe_reg_stride[POY*MAX_STRIDE][POX*MAX_STRIDE+1];
    // #pragma HLS BIND_STORAGE variable=buf2pe_reg type=register
    // fifo in fig13 of Optimizing_the_Convolution_Operation_to_Accelerate_Deep_Neural_Networks_on_FPGA
    hls::stream<DTYPE_ACT> fifo_arr_stride[POY*MAX_STRIDE-1][POX*MAX_STRIDE];
    #pragma HLS STREAM variable=fifo_arr_stride depth=FIFO_ARR_DEPTH
    
    // load for single kernel
    for (unsigned int cnt = 0; cnt < nky*nkx*s*s; cnt++) {
        // last poy
        // copy input buffer content to register
        unsigned int last_y_idx = cnt / nky;
        unsigned int last_x_idx = cnt % nkx;
        if (cnt % nkx == 0) {
            #pragma HLS unroll
            for (int x = 0; x < POX*s+1; x++) {
                // first row of input buffer
                buf2pe_reg_stride[POY*s-1][x] = input_buffer_stride[db_idx][POY*s-1+last_y_idx][x];
            }
        }
        // last register gets new val. rest values are fed from adjacent reg
        else if (cnt % nkx < nkx-1) {
            #pragma HLS unroll
            for (int x = 0; x < POX*s; x++) {
                buf2pe_reg_stride[POY*s-1][x] = buf2pe_reg_stride[POY*s-1][x+1];
            }
            buf2pe_reg_stride[POY*s-1][POX*s] = input_buffer_stride[db_idx][POY*s-1+last_y_idx][POX*s+last_x_idx];
        }
        // reg values are fed from adjacent reg
        else if (cnt % nkx == nkx-1) {
            #pragma HLS unroll
            for (int x = 0; x < POX*s; x++) {
                buf2pe_reg_stride[POY*s-1][x] = buf2pe_reg_stride[POY*s-1][x+1];
            }
        }
        if (cnt < nkx*nky-nkx) {
            // feed into previous fifo
            #pragma HLS unroll
            for (int x = 0; x < POX*s; x++) {
                fifo_arr_stride[POY*s-2][x].write(buf2pe_reg_stride[POY*s-1][x]);
            }
        }

        // middle poy
        #pragma HLS unroll
        for (int y = 1; y < POY*s-1; y++) {
            // copy input buffer content to register at first cycle
            if (cnt == 0) {
                #pragma HLS unroll
                for (int x = 0; x < POX*s+1; x++) {
                    // first row of input buffer
                    buf2pe_reg_stride[y][x] = input_buffer_stride[db_idx][y][x];
                }
            }
            // last register gets new val. rest values are fed from adjacent reg
            else if (cnt < nkx-1) {
                #pragma HLS unroll
                for (int x = 0; x < POX*s; x++) {
                    buf2pe_reg_stride[y][x] = buf2pe_reg_stride[y][x+1];
                }
                buf2pe_reg_stride[y][POX*s] = input_buffer_stride[db_idx][y][POX*s+cnt];
            }
            // reg values are fed from adjacent reg
            else if (cnt == nkx-1) {
                #pragma HLS unroll
                for (int x = 0; x < POX*s; x++) {
                    buf2pe_reg_stride[y][x] = buf2pe_reg_stride[y][x+1];
                }
            }
            // get values from FIFO
            if (cnt >= nkx) {
                #pragma HLS unroll
                for (int x = 0; x < POX*s; x++) {
                    buf2pe_reg_stride[y][x] = fifo_arr_stride[y][x].read();
                    //std::cout << "read x: " << x << ", y: " << y << std::endl;
                }
            }
            // feed into previous fifo
            if (cnt < nkx*nky-nkx) {
                #pragma HLS unroll
                for (int x = 0; x < POX*s; x++) {
                    fifo_arr_stride[y-1][x].write(buf2pe_reg_stride[y][x]);
                    //std::cout << "write x: " << x << ", y: " << y << std::endl;
                }
            }
        }

        // first poy
        // copy input buffer content to register at first cycle
        if (cnt == 0) {
            #pragma HLS unroll
            for (int x = 0; x < POX*s+1; x++) {
                // first row of input buffer
                buf2pe_reg_stride[0][x] = input_buffer_stride[db_idx][0][x];
            }
        }
        // last register gets new val. rest values are fed from adjacent reg
        else if (cnt < nkx-1) {
            #pragma HLS unroll
            for (int x = 0; x < POX*s; x++) {
                buf2pe_reg_stride[0][x] = buf2pe_reg_stride[0][x+1];
            }
            buf2pe_reg_stride[0][POX*s] = input_buffer_stride[db_idx][0][POX*s+cnt];
        }
        // reg values are fed from adjacent reg
        else if (cnt == nkx-1) {
            #pragma HLS unroll
            for (int x = 0; x < POX*s; x++) {
                buf2pe_reg_stride[0][x] = buf2pe_reg_stride[0][x+1];
            }
        }
        // get values from FIFO
        if (cnt >= nkx) {
            #pragma HLS unroll
            for (int x = 0; x < POX*s; x++) {
                buf2pe_reg_stride[0][x] = fifo_arr_stride[0][x].read();
            }
        }

        // feed mac unit
        #pragma HLS unroll
        for (int y = 0; y < POY; y++) {
            #pragma HLS unroll
            for (int x = 0; x < POX; x++) {
                mac_in_fifo_arr[y][x].write(buf2pe_reg_stride[y*s][x*s]);
            }
        }
    }
}

void load_input_buffer_stride(
    DTYPE_ACT input_buffer_stride[2][POY*MAX_STRIDE+PAD*2][POX*MAX_STRIDE+PAD*2],
    DTYPE_MEM act_mem[2][ACT_MEM_SIZE],
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox,
    unsigned int s,
    unsigned int act_fidx,  // input filter idx
    unsigned int act_yidx,  // input height starting idx
    unsigned int act_xidx,  // input width starting idx
    unsigned int db_read,   // double buffering index
    unsigned int db_write   // double bufferingn index
){
    #pragma HLS INLINE
    unsigned int nix = nox*s;
    unsigned int niy = noy*s;
    unsigned int act_mem_base_idx = act_fidx*niy*nix + act_yidx*nix + act_xidx;
    for (int y = 0; y < niy; y++) {
        for (int x = 0; x < nix; x++) {
            if ( (act_yidx + y < PAD) || (act_yidx + y >= niy + PAD) || (act_xidx + x < PAD) || (act_xidx + x >= nix + PAD)) {
                input_buffer_stride[db_write][y][x] = 0;
            }
            else {
                unsigned int act_mem_idx = act_mem_base_idx + y * nix + x;
                unsigned int idx1 = act_mem_idx / MEM_PACK;
                unsigned int idx2 = act_mem_idx % MEM_PACK;
                DTYPE_MEM block = act_mem[db_read][idx1];
                DTYPE_ACT data;
                data.range() = block.range(W_ACT*(idx2+1)-1,W_ACT*idx2);
                input_buffer_stride[db_write][y][x] = data;
            }
        }
    }
}

void load_weight_fifo(
    DTYPE_FIL *offchip_fil,
    hls::stream<DTYPE_FIL> weight_in_fifo_arr[POF],
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox

) {
    DTYPE_FIL filter_buffer[2][POF][NKY][NKX];  // todo: use double buffer

    for (int f_out = 0; f_out < nof; f_out += POF) {
        for (int f_in = 0; f_in < nif; f_in++){
            // load filter_buffer
            for (int f = 0; f < POF; f++) {
                for (int y = 0; y < nky; y++) {
                    for (int x = 0; x < nkx; x++) {
                        unsigned int fil_idx = (f+f_out)*nif*noy*nox + f_in*noy*nox y*nox + x;
                        filter_buffer[0][f][y][x] = offchip_fil[fil_idx];
                    }
                }
            }
            // reuse loop times
            for (int loop = 0; loop < noy*nox/POY/POX; loop++) {
                for (int y = 0; y < nky; y++) {
                    for (int x = 0; x < nkx; x++) {
                        #pragma unroll
                        load_weight_loop1:
                        for (int f = 0; f < POF; f++) {
                            weight_in_fifo_arr[f].write(filter_buffer[0][y][x]);
                        }
                    }
                }
            }
        }
    }
}

void store_output_buffer(
    DTYPE_ACT output_buffer[2][POF][POY][POX],
    DTYPE_MEM act_mem[2][ACT_MEM_SIZE],
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox,
    unsigned int act_fidx,  // output filter idx
    unsigned int act_yidx,  // output height starting idx
    unsigned int act_xidx,  // output width starting idx
    unsigned int db_buf,    // double buffering index
    unsigned int db_mem     // double bufferingn index
) {
    #pragma HLS INLINE
    unsigned int act_mem_base_idx = act_fidx*noy*nox + act_yidx*nox + act_xidx;
    for (int y = 0; y < noy; y++) {
        for (int x = 0; x < nox; x++) {
            unsigned int act_mem_idx = act_mem_base_idx + y*nox + x;
            unsigned int idx1 = act_mem_idx / MEM_PACK;
            unsigned int idx2 = act_mem_idx % MEM_PACK;
            DTYPE_MEM block = act_mem[db_mem][idx1];
            DTYPE_ACT data;
            
    for (int out_f = 0; out_f < nof; out_f+=POF) {
        for (int y0 = 0; y0 < noy; y0+=POY) {
            for (int x0 = 0; x0 < nox; x0+=POX) {
                // parallel
                for (int f = 0; f < POF; f++) {
                    for (int y = 0; y < POY; y++) {
                        for (int x = 0; x < POX; x++) {
                            
                            out_fifo_arr[f][y][x].write(mac_vals[f][y][x]);
                        }
                    }
                }
            }
        }
    }
}

#endif
