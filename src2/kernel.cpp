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
    unsigned int cnta,       // 
    unsigned int db_idx     // double buffering index) 
) {
    // register R* in fig13 of Optimizing_the_Convolution_Operation_to_Accelerate_Deep_Neural_Networks_on_FPGA
    DTYPE_ACT buf2pe_reg[POY][POX+1];
    #pragma HLS BIND_STORAGE variable=buf2pe_reg type=register
    // fifo in fig13 of Optimizing_the_Convolution_Operation_to_Accelerate_Deep_Neural_Networks_on_FPGA
    hls::stream<DTYPE_ACT> fifo_arr[POY-1][POX];
    #pragma HLS STREAM variable=fifo_arr depth=FIFO_ARR_DEPTH
    
    for (unsigned int cnt = 0; cnt < nkx*nky; cnt++) {
        // last poy
        // copy input buffer content to register
        unsigned int last_y_idx = cnt / nkx;
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
            buf2pe_reg[POY-1][POX] = input_buffer[db_idx][POY-1+last_y_idx][POX+last_y_idx];
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
    DTYPE_FIL filter_buffer[2*FILTER_BUFFER_SIZE];
    #pragma HLS BIND_STORAGE variable=filter_buffer type=register
    DTYPE_ACT output_buffer[2*OUTPUT_BUFFER_SIZE];
    #pragma HLS BIND_STORAGE variable=output_buffer type=register

    // fifo
    hls::stream<DTYPE_ACT> mac_in_fifo_arr[POY][POX];
    #pragma HLS STREAM variable=mac_in_fifo_arr depth=FIFO_ARR_DEPTH

    // on-chip memory
    DTYPE_MEM act_mem[ACT_MEM_SIZE];
    #pragma HLS bind_storage variable=act_mem impl=uram
    DTYPE_MEM fil_mem[FIL_MEM_SIZE];
    #pragma HLS bind_storage variable=fil_mem impl=bram

    // dummy function to fill input buffer
    dummy_fill_input_buffer(input_buffer);

    for (int idx = 0; idx < 9; idx++) {
        BUF2PE(input_buffer, mac_in_fifo_arr, NKX, NKY, idx, 0);
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
