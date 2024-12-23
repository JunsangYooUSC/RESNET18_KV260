/******************************************************************************
 * Filename: conv_config.h
 * Author: Junsang Yoo
 *
 * Description:
 * All the hyperparameters for the convolutional layer design are defined here.
 *
 * Features:
 * - Macros to define the size of input, output, filter
 * - Data types that will be used in kerenel code
 * - Load / Store / PE configuration
 ******************************************************************************/

#ifndef CONV_CONFIG_H
#define CONV_CONFIG_H

#include <iostream>
#include <iomanip> // For std::setw

// Include Vitis HLS headers
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_vector.h>
#include <hls_stream.h>
// Include C++ headers
#include <cmath>
#include <cassert>

#define STRIDE              2
#define MAX_STRIDE          2
// N
#define NKX                 3
#define NKY                 3
#define PAD                 ((NKX-1)/2)
// #define NIF                 64
#define NIF                 2
// #define NOX                 56
#define NOX                 14
// #define NOY                 56
#define NOY                 14
// #define NOF                 64
#define NOF                 8
#define NIX                 NOX*STRIDE
#define NIY                 NOY*STRIDE
// parallel
#define PKX                 1
#define PKY                 1
#define PIF                 NIF
#define POX                 7
#define POY                 7
// #define POF                 16
#define POF                 4
// tiling
#define TKX                 NKX
#define TKY                 NKY
// #define TIF                 8
#define TIF                 PIF
#define TOX                 POX
#define TOY                 POY
#define TOF                 POF
//

#define PIX                 POX*STRIDE
#define PIY                 POY*STRIDE

// Bit widths needed for convolution calculation
#define W_ACT               16
#define I_ACT               8
#define W_FIL               16
#define I_FIL               8
#define W_MUL               (W_ACT + W_FIL)
#define I_MUL               (I_ACT + I_FIL)
#define W_MAC               (W_MUL + MAC_EXTRA_BITS)
#define I_MAC               (I_MUL + MAC_EXTRA_BITS)
// Calculation of extra bit widths for MAC is done at compile time
constexpr unsigned int log2_ceil(unsigned int n) {
    return (n <= 1) ? 0 : 1 + log2_ceil((n + 1) / 2);
}
constexpr unsigned int MAC_EXTRA_BITS = log2_ceil(NOF * NIF * NOX * NOY) + 1;

// Data type definition
typedef ap_fixed<W_ACT, I_ACT> DTYPE_ACT;  // data type used for input / output activation
typedef ap_fixed<W_FIL, I_FIL> DTYPE_FIL;
typedef ap_fixed<W_MUL, I_MUL> DTYPE_MUL;
typedef ap_fixed<W_MAC, I_MAC> DTYPE_MAC;

// size of mem blocks
#define MAX_ACT_SIZE        ((TOTAL_OUT_LEN > TOTAL_IN_LEN) ? TOTAL_OUT_LEN : TOTAL_IN_LEN)
#define TOTAL_OUT_LEN       NOF*NOX*NOY
#define TOTAL_IN_LEN        NIF*NIX*NIY
#define TOTAL_FIL_LEN       NIF*NOF*NKX*NKY
#define INPUT_BUFFER_SIZE   (PIY+PAD*2)*(PIX+PAD*2)     // without double buffering
#define FILTER_BUFFER_SIZE  (POF*NKX*NKY)               // without double buffering
#define OUTPUT_BUFFER_SIZE  (POF*POX*POY)

#define FIL_MEM_SIZE        NOF*NIF*NKX*NKY
#define MEM_PACK            7
typedef ap_uint<MEM_PACK*W_ACT> DTYPE_MEM;
constexpr unsigned int ACT_MEM_SIZE = MAX_ACT_SIZE/MEM_PACK;

// BUF2PE vectors
constexpr unsigned int FIFO_ARR_DEPTH = NKX*NKY;

#endif
