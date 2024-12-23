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
#include "dummy_func.h"

// BUF2PE
void BUF2PE(
    // DTYPE_ACT *input_buffer, 
    DTYPE_ACT input_buffer[2][POY+PAD*2][POX+PAD*2],
    hls::stream<DTYPE_ACT> mac_in_fifo_arr[POY][POX],
    unsigned int nkx,
    unsigned int nky,
    unsigned int total_loops,       // 
    unsigned int db_idx     // double buffering index) 
) {
    // register R* in fig13 of Optimizing_the_Convolution_Operation_to_Accelerate_Deep_Neural_Networks_on_FPGA
    DTYPE_ACT buf2pe_reg[POY][POX+1];
    // #pragma HLS BIND_STORAGE variable=buf2pe_reg type=register
    // fifo in fig13 of Optimizing_the_Convolution_Operation_to_Accelerate_Deep_Neural_Networks_on_FPGA
    hls::stream<DTYPE_ACT> fifo_arr[POY-1][POX];
    #pragma HLS STREAM variable=fifo_arr depth=FIFO_ARR_DEPTH
    
    for (unsigned int cnt = 0; cnt < total_loops; cnt++) {
        // last poy
        // copy input buffer content to register
        unsigned int last_y_idx = cnt / nky;
        unsigned int last_x_idx = cnt % nkx;
        if (cnt % nkx == 0) {
            #pragma HLS unroll
            for (int x = 0; x < POX+1; x++) {
                // first row of input buffer
                buf2pe_reg[POY-1][x] = input_buffer[db_idx][POY-1+last_y_idx][x];
            }
        }
        // last register gets new val. rest values are fed from adjacent reg
        else if (cnt % nkx < nkx-1) {
            #pragma HLS unroll
            for (int x = 0; x < POX; x++) {
                buf2pe_reg[POY-1][x] = buf2pe_reg[POY-1][x+1];
            }
            buf2pe_reg[POY-1][POX] = input_buffer[db_idx][POY-1+last_y_idx][POX+last_x_idx];
        }
        // reg values are fed from adjacent reg
        else if (cnt % nkx == nkx-1) {
            #pragma HLS unroll
            for (int x = 0; x < POX; x++) {
                buf2pe_reg[POY-1][x] = buf2pe_reg[POY-1][x+1];
            }
        }
        if (cnt < nkx*nky-nkx) {
            // feed into previous fifo
            #pragma HLS unroll
            for (int x = 0; x < POX; x++) {
                fifo_arr[POY-2][x].write(buf2pe_reg[POY-1][x]);
            }
        }

        // middle poy
        #pragma HLS unroll
        for (int y = 1; y < POY-1; y++) {
            // copy input buffer content to register at first cycle
            if (cnt == 0) {
                #pragma HLS unroll
                for (int x = 0; x < POX+1; x++) {
                    // first row of input buffer
                    buf2pe_reg[y][x] = input_buffer[db_idx][y][x];
                }
            }
            // last register gets new val. rest values are fed from adjacent reg
            else if (cnt < nkx-1) {
                #pragma HLS unroll
                for (int x = 0; x < POX; x++) {
                    buf2pe_reg[y][x] = buf2pe_reg[y][x+1];
                }
                buf2pe_reg[y][POX] = input_buffer[db_idx][y][POX+cnt];
            }
            // reg values are fed from adjacent reg
            else if (cnt == nkx-1) {
                #pragma HLS unroll
                for (int x = 0; x < POX; x++) {
                    buf2pe_reg[y][x] = buf2pe_reg[y][x+1];
                }
            }
            // get values from FIFO
            if (cnt >= nkx) {
                #pragma HLS unroll
                for (int x = 0; x < POX; x++) {
                    buf2pe_reg[y][x] = fifo_arr[y][x].read();
                    //std::cout << "read x: " << x << ", y: " << y << std::endl;
                }
            }
            // feed into previous fifo
            if (cnt < nkx*nky-nkx) {
                #pragma HLS unroll
                for (int x = 0; x < POX; x++) {
                    fifo_arr[y-1][x].write(buf2pe_reg[y][x]);
                    //std::cout << "write x: " << x << ", y: " << y << std::endl;
                }
            }
        }

        // first poy
        // copy input buffer content to register at first cycle
        if (cnt == 0) {
            #pragma HLS unroll
            for (int x = 0; x < POX+1; x++) {
                // first row of input buffer
                buf2pe_reg[0][x] = input_buffer[db_idx][0][x];
            }
        }
        // last register gets new val. rest values are fed from adjacent reg
        else if (cnt < nkx-1) {
            #pragma HLS unroll
            for (int x = 0; x < POX; x++) {
                buf2pe_reg[0][x] = buf2pe_reg[0][x+1];
            }
            buf2pe_reg[0][POX] = input_buffer[db_idx][0][POX+cnt];
        }
        // reg values are fed from adjacent reg
        else if (cnt == nkx-1) {
            #pragma HLS unroll
            for (int x = 0; x < POX; x++) {
                buf2pe_reg[0][x] = buf2pe_reg[0][x+1];
            }
        }
        // get values from FIFO
        if (cnt >= nkx) {
            #pragma HLS unroll
            for (int x = 0; x < POX; x++) {
                buf2pe_reg[0][x] = fifo_arr[0][x].read();
            }
        }

        // feed mac unit
        #pragma HLS unroll
        for (int y = 0; y < POY; y++) {
            #pragma HLS unroll
            for (int x = 0; x < POX; x++) {
                mac_in_fifo_arr[y][x].write(buf2pe_reg[y][x]);
            }
        }
    }
}

// BUF2PE
void BUF2PE_stride(
    // DTYPE_ACT *input_buffer, 
    DTYPE_ACT input_buffer_stride[2][POY*MAX_STRIDE+PAD*2][POX*MAX_STRIDE+PAD*2],
    hls::stream<DTYPE_ACT> mac_in_fifo_arr[POY][POX],
    unsigned int nkx,
    unsigned int nky,
    unsigned int total_loops,
    unsigned int s,
    unsigned int db_idx     // double buffering index) 
) {
    // register R* in fig13 of Optimizing_the_Convolution_Operation_to_Accelerate_Deep_Neural_Networks_on_FPGA
    DTYPE_ACT buf2pe_reg_stride[POY*MAX_STRIDE][POX*MAX_STRIDE+1];
    // #pragma HLS BIND_STORAGE variable=buf2pe_reg type=register
    // fifo in fig13 of Optimizing_the_Convolution_Operation_to_Accelerate_Deep_Neural_Networks_on_FPGA
    hls::stream<DTYPE_ACT> fifo_arr_stride[POY*MAX_STRIDE-1][POX*MAX_STRIDE];
    #pragma HLS STREAM variable=fifo_arr_stride depth=FIFO_ARR_DEPTH
    
    for (unsigned int cnt = 0; cnt < total_loops; cnt++) {
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


void load_input_buffer(
    DTYPE_ACT input_buffer[2][POY+PAD*2][POX+PAD*2],
    DTYPE_MEM act_mem[ACT_MEM_SIZE],
    unsigned int act_fidx,
    unsigned int act_yidx,
    unsigned int act_xidx,
    unsigned int db_idx     // double buffering index) 
){
    unsigned int act_mem_base_idx = act_fidx*NIY*NIX + act_yidx*NIX + act_xidx;
    for (int y = 0; y < POY+PAD*2; y++) {
        for (int x = 0; x < POX+PAD*2; x++) {
            if ( (act_yidx + y < PAD) || (act_yidx + y >= NIY + PAD) || (act_xidx + x < PAD) || (act_xidx + x >= NIX + PAD)) {
                input_buffer[db_idx][y][x] = 0;
            }
            else {
                unsigned int act_mem_idx = act_mem_base_idx + y * NIX + x;
                unsigned int idx1 = act_mem_idx / MEM_PACK;
                unsigned int idx2 = act_mem_idx % MEM_PACK;
                DTYPE_MEM block = act_mem[idx1];
                DTYPE_ACT data;
                data.range() = block.range(W_ACT*(idx2+1)-1,W_ACT*idx2);
                input_buffer[db_idx][y][x] = data;
            }
        }
    }
}

void load_input_buffer_stride(
    DTYPE_ACT input_buffer_stride[2][POY*MAX_STRIDE+PAD*2][POX*MAX_STRIDE+PAD*2],
    DTYPE_MEM act_mem[ACT_MEM_SIZE],
    unsigned int act_fidx,  // input filter idx
    unsigned int act_yidx,  // input height starting idx
    unsigned int act_xidx,  // input width starting idx
    unsigned int y_size,    // size of input buffer mat height
    unsigned int x_size,    // size of input buffer mat width
    unsigned int db_idx     // double buffering index
){
    unsigned int act_mem_base_idx = act_fidx*NIY*NIX + act_yidx*NIX + act_xidx;
    for (int y = 0; y < y_size; y++) {
        for (int x = 0; x < x_size; x++) {
            if ( (act_yidx + y < PAD) || (act_yidx + y >= NIY + PAD) || (act_xidx + x < PAD) || (act_xidx + x >= NIX + PAD)) {
                input_buffer_stride[db_idx][y][x] = 0;
            }
            else {
                unsigned int act_mem_idx = act_mem_base_idx + y * NIX + x;
                unsigned int idx1 = act_mem_idx / MEM_PACK;
                unsigned int idx2 = act_mem_idx % MEM_PACK;
                DTYPE_MEM block = act_mem[idx1];
                DTYPE_ACT data;
                data.range() = block.range(W_ACT*(idx2+1)-1,W_ACT*idx2);
                input_buffer_stride[db_idx][y][x] = data;
            }
        }
    }
}


// kernel function
void kernel_func(DTYPE_ACT *in_host,
                DTYPE_ACT *filter_offchip,
                DTYPE_ACT *out_host
) {
    #pragma HLS DATAFLOW

    // on-chip buffers
    // DTYPE_ACT input_buffer[2*INPUT_BUFFER_SIZE];
    DTYPE_ACT input_buffer[2][POY+PAD*2][POX+PAD*2];
    #pragma HLS BIND_STORAGE variable=input_buffer type=register
    DTYPE_ACT input_buffer_stride[2][POY*MAX_STRIDE+PAD*2][POX*MAX_STRIDE+PAD*2];
    #pragma HLS BIND_STORAGE variable=input_buffer_stride type=register
    DTYPE_FIL filter_buffer[2*FILTER_BUFFER_SIZE];
    #pragma HLS BIND_STORAGE variable=filter_buffer type=register
    DTYPE_ACT output_buffer[2*OUTPUT_BUFFER_SIZE];
    #pragma HLS BIND_STORAGE variable=output_buffer type=register

    // fifo
    hls::stream<DTYPE_ACT> mac_in_fifo_arr[POY][POX];
    #pragma HLS STREAM variable=mac_in_fifo_arr depth=FIFO_ARR_DEPTH

    // on-chip memory
    DTYPE_MEM act_mem1[ACT_MEM_SIZE];
    #pragma HLS bind_storage variable=act_mem1 impl=uram
    DTYPE_MEM act_mem2[ACT_MEM_SIZE];
    #pragma HLS bind_storage variable=act_mem2 impl=uram
    DTYPE_MEM fil_mem[FIL_MEM_SIZE];
    #pragma HLS bind_storage variable=fil_mem impl=bram

    // dummy function to fill input buffer
    // dummy_fill_input_buffer(input_buffer);

    std::cout << "in_host[0]: " << in_host[0] << std::endl;
    // load in_host to act_mem1
    DTYPE_MEM block;
    for (int idx = 0; idx < TOTAL_IN_LEN; idx++) {
        unsigned int idx2 = idx % MEM_PACK;
        block.range(W_ACT*(idx2+1)-1, W_ACT*(idx2)) = in_host[idx].range();
        if (idx2 == MEM_PACK-1) {
            act_mem1[idx/MEM_PACK] = block;
        }
    }

    // load input buffer
    // load_input_buffer(input_buffer, act_mem1, 0, 0, 0, 0);
    load_input_buffer_stride(input_buffer_stride, act_mem1, 0, 0, 0, POY*MAX_STRIDE+PAD*2, POX*MAX_STRIDE+PAD*2, 0);
    
    // DTYPE_ACT step = 1;
    // step = step >> 8;
    // DTYPE_ACT i = 0;
    // for (int idx = 0; idx < 2; idx++) {
    //     for (int jdx = 0; jdx < POY*MAX_STRIDE+PAD*2; jdx++) {
    //         for (int kdx = 0; kdx < POX*MAX_STRIDE+PAD*2; kdx++) {
    //             input_buffer_stride[idx][jdx][kdx] = i;
    //             i += step;
    //         }
    //     }
    // }

    unsigned int total_loops = NKX*NKY;
    // BUF2PE(input_buffer, mac_in_fifo_arr, NKX, NKY, total_loops, 0);
    BUF2PE_stride(input_buffer_stride, mac_in_fifo_arr, NKX, NKY, total_loops, STRIDE, 0);

    // dummy consumer
    for (int kdx = 0; kdx < NKX*NKY; kdx++) {
        for (int y = 0; y < POY; y++) {
            for (int x = 0; x < POX; x++) {

            }
        }
    }
    for (int idx = 0; idx < 9; idx++) {
        for (int jdx = 0; jdx < POY; jdx++) {
            for (int kdx = 0; kdx < POX; kdx++) {
                std::cout << std::setw(5) << (mac_in_fifo_arr[jdx][kdx].read() << 8) << " ";
            }
        }
        std::cout << std::endl;
    }

}

#endif
