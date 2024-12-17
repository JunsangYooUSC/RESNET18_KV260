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

#include <iomanip> // For std::setw

// Include Vitis HLS headers
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_vector.h>
#include <hls_stream.h>
// Include project headers
#include "conv_config.h"
#include "kernel.h"
#include "kernel_func.hpp"

// BUF2PE with support of stride
void BUF2PE_stride(
    // DTYPE_ACT *input_buffer, 
    // DTYPE_ACT input_buffer_stride[2][POY*MAX_STRIDE+PAD*2][POX*MAX_STRIDE+PAD*2],
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
) {
    unsigned int db_idx = 0;
    DTYPE_ACT input_buffer_stride[2][POY*MAX_STRIDE+MAX_PAD*2][POX*MAX_STRIDE+MAX_PAD*2];
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
                                unsigned int idx1 = mem_idx / ACT_PACK;
                                unsigned int idx2 = mem_idx % ACT_PACK;
                                DTYPE_MEM_ACT block = mem[idx1];
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
    DTYPE_FIL *mem_fil,
    hls::stream<DTYPE_FIL> weight_in_fifo_arr[POF],
    unsigned int base_addr,
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox

) {
    DTYPE_FIL filter_buffer[2][POF][nif][nky][nkx];  // todo: use double buffer
    for (int f_out = 0; f_out < nof; f_out += POF) {
        for (int y0 = 0; y0 < noy; y0+=POY) {
            for (int x0 = 0; x0 < nox; x0+=POX) {
                // 
                if ( (y0 == 0) && (x0 == 0) ) {
                    for (int f_in = 0; f_in < nif; f_in++){
                        for (int f = 0; f < POF; f++) {
                            for (int y = 0; y < nky; y++) {
                                for (int x = 0; x < nkx; x++) {
#pragma HLS pipeline
                                    unsigned int fil_idx = base_addr + (f+f_out)*nif*nky*nkx + f_in*nky*nkx + y*nkx + x;
                                    filter_buffer[0][f][f_in][y][x] = mem_fil[fil_idx];
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
#pragma HLS unroll
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
    DTYPE_MEM_ACT *mem,
    hls::stream<float> out_fifo_arr[POF][POY][POX],
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox
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
#pragma HLS pipeline
                        unsigned int mem_idx = ((out_f+f)*noy*nox + (y0+y)*nox + x0) / ACT_PACK;
                        DTYPE_MEM_ACT block;
                        for (int x = 0; x < POX; x++) {
                            block.range(W_ACT*(x+1)-1, W_ACT*x) = output_buffer[0][f][y][x].range();
                        }
                        mem[mem_idx] = block;
                    }
                }
            }
        }
    }
}

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

/*
void conv(
    DTYPE_MEM_ACT *mem_in,
    DTYPE_FIL *mem_fil,
    DTYPE_MEM_ACT *mem_out,
    unsigned int nky,
    unsigned int nkx,
    unsigned int nof,
    unsigned int nif,
    unsigned int noy,
    unsigned int nox,
    unsigned int stride,
    unsigned int pad,
    unsigned int en
) {
    if (!en) return;

    // fifo
    hls::stream<DTYPE_ACT> mac_in_fifo_arr[POY][POX];
    #pragma HLS STREAM variable=mac_in_fifo_arr depth=FIFO_ARR_DEPTH
    hls::stream<DTYPE_FIL> weight_in_fifo_arr[POF];
    #pragma HLS STREAM variable=weight_in_fifo_arr depth=FIFO_ARR_DEPTH
    hls::stream<DTYPE_MAC> out_fifo_arr[POF][POY][POX];
    #pragma HLS STREAM variable=out_fifo_arr depth=FIFO_ARR_DEPTH

// #pragma HLS DATAFLOW
    BUF2PE_stride(mem_in, mac_in_fifo_arr,
            nky, nkx, nof, nif, noy, nox, stride, pad);
    load_weight_fifo(mem_fil, weight_in_fifo_arr,
            nky, nkx, nof, nif, noy, nox);
    PE(mac_in_fifo_arr, weight_in_fifo_arr, out_fifo_arr,
            nky, nkx, nof, nif, noy, nox);
    store_output_fifo(mem_out, out_fifo_arr,
            nky, nkx, nof, nif, noy, nox);

}
*/


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
) {
    for (int out_f = 0; out_f < nof; out_f+=POF) {
        for (int y0 = 0; y0 < noy; y0+=POY) {
            for (int x0 = 0; x0 < nox; x0+=POX) {
                for (int f = 0; f < POF; f++) {
#pragma HLS unroll
                    for (int y = 0; y < POY; y++) {
#pragma HLS unroll
                        DTYPE_MEM_ACT block;
                        unsigned mem_idx = (out_f+f)*noy*nox + (y0+y)*nox + x0;
                        block = mem_in[mem_idx];
                        for (int x = 0; x < POX; x++) {
// #pragma HLS unroll
                            DTYPE_ACT val;
                            val.range() = block.range(W_ACT*(x+1)-1, W_ACT*x);
                            out_fifo_arr[f][y][x].write(val);
                        }
                    }
                }
            }
        }
    }

}

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
) {
    if (!bb_en) return;

    // pass convolution
    if (!conv_en) {
        conv_pass(mem_in, out_fifo_arr,
                nky, nkx, nof, nif, noy, nox, stride, pad);
    }
    // convolution
    else {
        // fifo
        hls::stream<DTYPE_ACT> mac_in_fifo_arr[POY][POX];
        #pragma HLS STREAM variable=mac_in_fifo_arr depth=FIFO_ARR_DEPTH
        hls::stream<DTYPE_FIL> weight_in_fifo_arr[POF];
        #pragma HLS STREAM variable=weight_in_fifo_arr depth=FIFO_ARR_DEPTH

// #pragma HLS DATAFLOW
        BUF2PE_stride(mem_in, mac_in_fifo_arr,
                nky, nkx, nof, nif, noy, nox, stride, pad);
        load_weight_fifo(mem_fil, weight_in_fifo_arr, weight_base_addr,
                nky, nkx, nof, nif, noy, nox);
        PE(mac_in_fifo_arr, weight_in_fifo_arr, out_fifo_arr,
                nky, nkx, nof, nif, noy, nox);
    }
}

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
) {
    if (!bb_en) return;
    
    float mean[POF];
    float var_mult[POF];
    float gamma[POF];
    float beta[POF];
    for (int f_out = 0; f_out < nof; f_out+=POF) {
        for (int f = 0; f < POF; f++) {
            mean[f] = bn_weight_mem[f_out+f];
            var_mult[f] = bn_weight_mem[f_out+f+nof];
            gamma[f] = bn_weight_mem[f_out+f+nof*2];
            beta[f] = bn_weight_mem[f_out+f+nof*3];
        }
        for (int y0 = 0; y0 < noy; y0+=POY) {
            for (int x0 = 0; x0 < nox; x0+=POX) {
                // parallel
                for (int f = 0; f < POF; f++) {
#pragma HLS unroll
                    for (int y = 0; y < POY; y++) {
#pragma HLS unroll                        
                        for (int x = 0; x < POX; x++) {
#pragma HLS unroll
                            float val;
                            val = in_fifo_arr[f][y][x].read();
                            // batch norm when enabled
                            if (bn_en) {
                                val = (val-mean[f])*var_mult[f]*gamma[f]+beta[f];
                            }
                            out_fifo_arr[f][y][x].write(val);
                        }
                    }
                }

            }
        }
    }
}

void skip_conn(
    DTYPE_MEM_ACT *mem_add,
    hls::stream<float> in_fifo_arr[POF][POY][POX],
    hls::stream<float> out_fifo_arr[POF][POY][POX],
    unsigned int nof,
    unsigned int noy,
    unsigned int nox,
    unsigned int bb_en,
    unsigned int skip_en,
    unsigned int relu_en
) {
    if (!bb_en) return;

    for (int f_out = 0; f_out < nof; f_out+=POF) {
        for (int y0 = 0; y0 < noy; y0+=POY) {
            for (int x0 = 0; x0 < nox; x0+=POX) {
                for (int f = 0; f < POF; f++) {
#pragma HLS unroll
                    for (int y = 0; y < POY; y++) {
#pragma HLS unroll 
                        for (int x = 0; x < POX; x++) {
#pragma HLS unroll
                            float val = in_fifo_arr[f][y][x].read();
                            if (skip_en) {
                                unsigned add_addr = ((f_out+f)*noy*nox + (y0+y)*nox + x0) / POX;
                                DTYPE_ACT add_val;
                                add_val.range() = mem_add[add_addr].range(W_ACT*(x+1)-1, W_ACT*(x));
                                val = val + (float)add_val;
                            }
                            if (relu_en) {
                                val = (val > 0) ? val : 0;
                            }
                            out_fifo_arr[f][y][x].write(val);
                        }
                    }
                }
            }
        }
    }
}

// kernel function
void kernel_func(
    DTYPE_ACT *in_host,
    DTYPE_FIL *weight_mem,
    float *bn_weight_mem,
    // DTYPE_MEM_WEIGHT *weight_mem,
    DTYPE_ACT *out_host
) {
    // on-chip buffers
    // DTYPE_ACT input_buffer[2*INPUT_BUFFER_SIZE];
    // DTYPE_ACT input_buffer[2][POY+PAD*2][POX+PAD*2];
    // #pragma HLS BIND_STORAGE variable=input_buffer type=register
    // DTYPE_ACT input_buffer_stride[2][POY*MAX_STRIDE+PAD*2][POX*MAX_STRIDE+PAD*2];
    // #pragma HLS BIND_STORAGE variable=input_buffer_stride type=register
    // DTYPE_FIL filter_buffer[2*FILTER_BUFFER_SIZE];
    // #pragma HLS BIND_STORAGE variable=filter_buffer type=register
    // DTYPE_ACT output_buffer[2*OUTPUT_BUFFER_SIZE];
    // #pragma HLS BIND_STORAGE variable=output_buffer type=register

    // on-chip memory
    DTYPE_MEM_ACT mem0[MEM0_SIZE];
    #pragma HLS bind_storage variable=mem0 impl=uram
    #pragma HLS ARRAY_PARTITION variable=mem0 dim=1 complete
    DTYPE_MEM_ACT mem1[MEM1_SIZE];
    #pragma HLS bind_storage variable=mem1 impl=uram
    #pragma HLS ARRAY_PARTITION variable=mem1 dim=1 complete
    DTYPE_MEM_ACT mem2[MEM2_SIZE];
    #pragma HLS bind_storage variable=mem2 impl=uram
    #pragma HLS ARRAY_PARTITION variable=mem2 dim=1 complete
    DTYPE_MEM_ACT mem3[16];
    // off-chip memory
    // DTYPE_MEM_WEIGHT weight_mem[WEIGHT_MEM_SIZE];
    #pragma HLS INTERFACE mode=m_axi port=weight_mem offset=slave bundle=gmem0
    #pragma HLS INTERFACE mode=s_axilite port=weight_mem bundle=control

    // interface
    #pragma HLS INTERFACE mode=m_axi port=in_host offset=slave bundle=gmem0
    #pragma HLS INTERFACE mode=m_axi port=out_host offset=slave bundle=gmem0
    #pragma HLS INTERFACE mode=s_axilite port=in_host bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=out_host bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // parameters for kernel functions
    DTYPE_MEM_ACT *mem_in;
    DTYPE_MEM_ACT *mem_out;
    DTYPE_MEM_ACT *mem_add;
    unsigned nif;
    unsigned nof;
    unsigned noy;
    unsigned nox;
    unsigned nkx;
    unsigned nky;
    unsigned stride;
    unsigned pad;
    bool bb_en;
    bool conv_en;
    bool bn_en;
    bool skip_en;
    bool relu_en;
    bool max_pool_en;
    bool avg_pool_en;
    bool lin_en;
    unsigned weight_base;
    unsigned weight_size;
    unsigned bn_weight_base;
    unsigned bn_weight_size;

    // global pipes
    hls::stream<float> fifo1[POF][POY][POX];
    #pragma HLS STREAM variable=fifo1 depth=FIFO_ARR_DEPTH
    hls::stream<float> fifo2[POF][POY][POX];
    #pragma HLS STREAM variable=fifo2 depth=FIFO_ARR_DEPTH
    hls::stream<float> fifo3[POF][POY][POX];
    #pragma HLS STREAM variable=fifo3 depth=FIFO_ARR_DEPTH

    for (int opcnt = 0; opcnt < 3; opcnt++) {
        if (opcnt == 0) {
            nif             = BB6_SKIP_C;
            nof             = BB7_CONV1_C;
            noy             = BB7_CONV1_H;
            nox             = BB7_CONV1_W;
            nkx             = BB7_CONV1_K;
            nky             = BB7_CONV1_K;
            stride          = BB7_CONV1_S;
            pad             = BB7_CONV1_PAD;
            bb_en           = BB7_CONV1_BB_EN;
            conv_en         = BB7_CONV1_CONV_EN;
            bn_en           = BB7_CONV1_BN_EN;
            skip_en         = BB7_CONV1_SKIP_EN;
            relu_en         = BB7_CONV1_RELU_EN;
            max_pool_en     = BB7_CONV1_MAX_POOL;
            avg_pool_en     = BB7_CONV1_AVG_POOL;
            lin_en          = BB7_CONV1_LIN_EN;
            mem_in          = mem0;
            mem_out         = mem1;
            mem_add         = mem3;
            weight_base     = BB7_CONV1_WEIGHT_BASE;
            weight_size     = BB7_CONV1_CONV_WEIGHT_SIZE;
            bn_weight_base  = BB7_CONV1_BN_WEIGHT_BASE;
            bn_weight_size  = BB7_CONV1_BN_WEIGHT_SIZE;
        } 
        else if (opcnt == 1) {
            nif             = BB7_CONV1_C;
            nof             = BB7_CONV2_C;
            noy             = BB7_CONV2_H;
            nox             = BB7_CONV2_W;
            nkx             = BB7_CONV2_K;
            nky             = BB7_CONV2_K;
            stride          = BB7_CONV2_S;
            pad             = BB7_CONV2_PAD;
            bb_en           = BB7_CONV2_BB_EN;
            conv_en         = BB7_CONV2_CONV_EN;
            bn_en           = BB7_CONV2_BN_EN;
            skip_en         = BB7_CONV2_SKIP_EN;
            relu_en         = BB7_CONV2_RELU_EN;
            max_pool_en     = BB7_CONV2_MAX_POOL;
            avg_pool_en     = BB7_CONV2_AVG_POOL;
            lin_en          = BB7_CONV2_LIN;
            mem_in          = mem1;
            mem_out         = mem2;
            mem_add         = mem3;
            weight_base     = BB7_CONV2_WEIGHT_BASE;
            weight_size     = BB7_CONV2_CONV_WEIGHT_SIZE;
            bn_weight_base  = BB7_CONV2_BN_WEIGHT_BASE;
            bn_weight_size  = BB7_CONV2_BN_WEIGHT_SIZE;
        }
        else if (opcnt == 2) {
            nif             = BB7_CONV2_C;
            nof             = BB7_SKIP_C;
            noy             = BB7_SKIP_H;
            nox             = BB7_SKIP_W;
            nkx             = BB7_SKIP_K;
            nky             = BB7_SKIP_K;
            stride          = BB7_SKIP_S;
            pad             = BB7_SKIP_PAD;
            bb_en           = BB7_SKIP_BB_EN;
            conv_en         = BB7_SKIP_CONV_EN;
            bn_en           = BB7_SKIP_BN_EN;
            skip_en         = BB7_SKIP_SKIP_EN;
            relu_en         = BB7_SKIP_RELU_EN;
            max_pool_en     = BB7_SKIP_MAX_POOL;
            avg_pool_en     = BB7_SKIP_AVG_POOL;
            lin_en          = BB7_SKIP_LIN;
            mem_in          = mem0;
            mem_out         = mem1;
            mem_add         = mem2;
            weight_base     = BB7_SKIP_WEIGHT_BASE;
            weight_size     = BB7_SKIP_CONV_WEIGHT_SIZE;
            bn_weight_base  = BB7_SKIP_BN_WEIGHT_BASE;
            bn_weight_size  = BB7_SKIP_BN_WEIGHT_SIZE;
        }

        // initial input
        if (opcnt == 0) {
            int niy = noy*stride;
            int nix = nox*stride;
            // load mem_in with input
            for (int idx = 0; idx < nif*niy*nix/POX; idx++) {
                DTYPE_MEM_ACT block;
                for (int x = 0; x < POX; x++) {
                    DTYPE_ACT val;
                    block.range(W_ACT*(x+1)-1, W_ACT*(x)) = in_host[idx*POX+x].range();
                }
                mem_in[idx] = block;
            }
        }

        // conv
        conv(mem_in, weight_mem, fifo1,
                weight_base, nky, nkx, nof, nif, noy, nox, stride, pad, bb_en, conv_en);
        batch_norm(bn_weight_mem, fifo1, fifo2,
                bn_weight_base, nof, noy, nox, bb_en, bn_en);
        skip_conn(mem_add, fifo2, fifo3,
                nof, noy, nox, bb_en, skip_en, relu_en);
        store_output_fifo(mem_out, fifo3,
                nky, nkx, nof, nif, noy, nox);
        
        // output back to host
        if (opcnt == 2) {
            for (int idx = 0; idx < nof*noy*nox/POX; idx++) {
                DTYPE_MEM_ACT block;
                block = mem1[idx];
                for (int x = 0; x < POX; x++) {
                    DTYPE_ACT val;
                    out_host[idx*POX+x].range() = block.range(W_ACT*(x+1)-1, W_ACT*(x));
                }
            }
        }
    }
}

#endif
