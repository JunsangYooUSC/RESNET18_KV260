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

// Include Vitis HLS headers
#include <ap_fixed.h>
// Include C++ headers
#include <cmath>
#include <cassert>

// N
#define NIX                 28
#define NIY                 28
#define NKX                 3
#define NKY                 3
// #define NIF                 64
#define NIF                 4
// #define NOX                 56
#define NOX                 28
// #define NOY                 56
#define NOY                 28
// #define NOF                 64
#define NOF                 12
// parallel
#define PKX                 1
#define PKY                 1
#define PIF                 1
#define POX                 7
#define POY                 7
// #define POF                 16
#define POF                 2
// tiling
#define TKX                 NKX
#define TKY                 NKY
// #define TIF                 8
#define TIF                 PIF
#define TOX                 POX
#define TOY                 POY
#define TOF                 POF
//
#define PAD                 1
#define STRIDE              1

#define PIX                 (POX+PAD*2)
#define PIY                 (POY+PAD*2)

// Bit widths needed for convolution calculation
#define W_ACT               8
#define I_ACT               3
#define W_FILTER            8
#define I_FILTER            3
#define W_MULT              (W_ACT + W_FILTER)
#define I_MULT              (I_ACT + I_FILTER)
#define W_MAC               (W_MULT + MAC_EXTRA_BITS)
#define I_MAC               (I_MULT + MAC_EXTRA_BITS)
// Calculation of extra bit widths for MAC is done at compile time
constexpr unsigned int log2_ceil(unsigned int n) {
    return (n <= 1) ? 0 : 1 + log2_ceil((n + 1) / 2);
}
constexpr unsigned int MAC_EXTRA_BITS = log2_ceil(Nof * Nif * Nox * Noy) + 1;

// Data type definition
typedef ap_fixed<W_ACT, I_ACT> DTYPE_ACT;  // data type used for input / output activation
typedef ap_fixed<W_FILTER, I_FILTER> DTYPE_FILTER;
typedef ap_fixed<W_MULT, I_MULT> DTYPE_MULT;
typedef ap_fixed<W_MAC, I_MAC> DTYPE_MAC;

// size of mem blocks
#define TOTAL_OUT_LEN       NOF*NOX*NOY
#define TOTAL_IN_LEN        NIF*NIX*NIY
#define TOTAL_FILTER_LEN    NOF*NKX*NKY
#endif
