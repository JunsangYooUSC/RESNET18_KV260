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

// kernel parallel parameters
#define PKX                 1
#define PKY                 1
#define PIF                 1
#define POX                 7
#define POY                 7
#define POF                 4       // change_to: if necessary
#define MAX_STRIDE          2
#define MAX_PAD             3

#define WEIGHT_PACK         1       // change_to: 8
#define ACT_PACK            7

// Bit widths
constexpr unsigned W_ACT = 8;
constexpr unsigned I_ACT = 3;
constexpr unsigned W_FIL = 8;
constexpr unsigned I_FIL = 1;
constexpr unsigned W_MUL = (W_ACT + W_FIL);
constexpr unsigned I_MUL = (I_ACT + I_FIL);
constexpr unsigned W_MAC = 32;
constexpr unsigned I_MAC = 16;
// constexpr unsigned log2_ceil(unsigned int n) {
//     return (n <= 1) ? 0 : 1 + log2_ceil((n + 1) / 2);
// }
// constexpr unsigned MAC_EXTRA_BITS = log2_ceil(NOF * NIF * NOX * NOY) + 1;

// Data type definition
typedef ap_fixed<W_ACT, I_ACT> DTYPE_ACT;  // data type used for input / output activation
typedef ap_fixed<W_FIL, I_FIL> DTYPE_FIL;
typedef ap_fixed<W_MUL, I_MUL> DTYPE_MUL;
typedef ap_fixed<W_MAC, I_MAC> DTYPE_MAC;

typedef ap_uint<ACT_PACK*W_ACT> DTYPE_MEM_ACT;
typedef ap_uint<WEIGHT_PACK*W_FIL> DTYPE_MEM_WEIGHT;

// BUF2PE vectors
constexpr unsigned int FIFO_ARR_DEPTH = 3*3*2;

// layer sizes
// input
#define IN_H                224
#define IN_W                224
#define IN_C                3

// enable signals
// 0: off
// 1: on
// 2: passed
// mem blocks
// 0: mem0
// 1: mem1
// 2: mem2
// 3: fake mem block mem3

// CONV1
#define CONV1_C                 64
#define CONV1_H                 112
#define CONV1_W                 112
#define CONV1_K                 7
#define CONV1_S                 2
#define CONV1_PAD               3
#define CONV1_CONV_EN           1
#define CONV1_BN_EN             1
#define CONV1_SKIP_EN           2
#define CONV1_RELU_EN           1
#define CONV1_MAX_POOL          0
#define CONV1_AVG_POOL          0
#define CONV1_LIN               0
#define CONV1_IN_MEM            0
#define CONV1_OUT_MEM           1
#define CONV1_SKIP_MEM          3
constexpr unsigned CONV1_WEIGHT_BASE = 0;
constexpr unsigned CONV1_CONV_WEIGHT_SIZE = IN_C * CONV1_C * CONV1_K * CONV1_K / WEIGHT_PACK;
constexpr unsigned CONV1_BN_WEIGHT_BASE = CONV1_WEIGHT_BASE + CONV1_CONV_WEIGHT_SIZE;
constexpr unsigned CONV1_BN_WEIGHT_SIZE = 4 * CONV1_C / WEIGHT_PACK;

// MAXPOOL
#define MAXPOOL_C               64
#define MAXPOOL_H               56
#define MAXPOOL_W               56
#define MAXPOOL_K               3
#define MAXPOOL_S               2
#define MAXPOOL_PAD             1
#define MAXPOOL_CONV_EN         0
#define MAXPOOL_BN_EN           0
#define MAXPOOL_SKIP_EN         0
#define MAXPOOL_RELU_EN         0
#define MAXPOOL_MAX_POOL        1
#define MAXPOOL_AVG_POOL        0
#define MAXPOOL_LIN             0
#define MAXPOOL_IN_MEM          1
#define MAXPOOL_OUT_MEM         0
#define MAXPOOL_SKIP_MEM        3

// BB1_CONV1
#define BB1_CONV1_C             64
#define BB1_CONV1_H             56
#define BB1_CONV1_W             56
#define BB1_CONV1_K             3
#define BB1_CONV1_S             1
#define BB1_CONV1_PAD           1
#define BB1_CONV1_CONV_EN       1
#define BB1_CONV1_BN_EN         1
#define BB1_CONV1_SKIP_EN       2
#define BB1_CONV1_RELU_EN       1
#define BB1_CONV1_MAX_POOL      0
#define BB1_CONV1_AVG_POOL      0
#define BB1_CONV1_LIN           0
#define BB1_CONV1_IN_MEM        0
#define BB1_CONV1_OUT_MEM       1
#define BB1_CONV1_SKIP_MEM      3
constexpr unsigned BB1_CONV1_WEIGHT_BASE = CONV1_BN_WEIGHT_BASE + CONV1_BN_WEIGHT_SIZE;
constexpr unsigned BB1_CONV1_CONV_WEIGHT_SIZE = MAXPOOL_C * BB1_CONV1_C * BB1_CONV1_K * BB1_CONV1_K / WEIGHT_PACK;
constexpr unsigned BB1_CONV1_BN_WEIGHT_BASE = BB1_CONV1_WEIGHT_BASE + BB1_CONV1_CONV_WEIGHT_SIZE;
constexpr unsigned BB1_CONV1_BN_WEIGHT_SIZE = 4 * BB1_CONV1_C / WEIGHT_PACK;

// BB1_CONV2   
#define BB1_CONV2_C             64
#define BB1_CONV2_H             56
#define BB1_CONV2_W             56
#define BB1_CONV2_K             3
#define BB1_CONV2_S             1
#define BB1_CONV2_PAD           1
#define BB1_CONV2_CONV_EN       1
#define BB1_CONV2_BN_EN         1
#define BB1_CONV2_SKIP_EN       2
#define BB1_CONV2_RELU_EN       2
#define BB1_CONV2_MAX_POOL      0
#define BB1_CONV2_AVG_POOL      0
#define BB1_CONV2_LIN           0
#define BB1_CONV2_IN_MEM        1
#define BB1_CONV2_OUT_MEM       2
#define BB1_CONV2_SKIP_MEM      3
constexpr unsigned BB1_CONV2_WEIGHT_BASE = BB1_CONV1_WEIGHT_BASE + BB1_CONV1_CONV_WEIGHT_SIZE;
constexpr unsigned BB1_CONV2_CONV_WEIGHT_SIZE = BB1_CONV1_C * BB1_CONV2_C * BB1_CONV2_K * BB1_CONV2_K / WEIGHT_PACK;
constexpr unsigned BB1_CONV2_BN_WEIGHT_BASE = BB1_CONV2_WEIGHT_BASE + BB1_CONV2_CONV_WEIGHT_SIZE;
constexpr unsigned BB1_CONV2_BN_WEIGHT_SIZE = 4 * BB1_CONV2_C / WEIGHT_PACK;

// BB1_SKIP
#define BB1_SKIP_C              64
#define BB1_SKIP_H              56
#define BB1_SKIP_W              56
#define BB1_SKIP_K              0
#define BB1_SKIP_S              0
#define BB1_SKIP_PAD            0
#define BB1_SKIP_CONV_EN        2
#define BB1_SKIP_BN_EN          2
#define BB1_SKIP_SKIP_EN        1
#define BB1_SKIP_RELU_EN        1
#define BB1_SKIP_MAX_POOL       0
#define BB1_SKIP_AVG_POOL       0
#define BB1_SKIP_LIN            0
#define BB1_SKIP_IN_MEM         2
#define BB1_SKIP_OUT_MEM        1
#define BB1_SKIP_SKIP_MEM       0
constexpr unsigned BB1_SKIP_WEIGHT_BASE = BB1_CONV2_BN_WEIGHT_BASE + BB1_CONV2_BN_WEIGHT_SIZE;
constexpr unsigned BB1_SKIP_CONV_WEIGHT_SIZE = BB1_CONV2_C * BB1_SKIP_C * BB1_SKIP_K * BB1_SKIP_K / WEIGHT_PACK;
constexpr unsigned BB1_SKIP_BN_WEIGHT_BASE = BB1_SKIP_WEIGHT_BASE + BB1_SKIP_CONV_WEIGHT_SIZE;
constexpr unsigned BB1_SKIP_BN_WEIGHT_SIZE = 4 * BB1_CONV2_C / WEIGHT_PACK;

// BB2_CONV1
#define BB2_CONV1_C             64
#define BB2_CONV1_H             56
#define BB2_CONV1_W             56
#define BB2_CONV1_K             3
#define BB2_CONV1_S             1
#define BB2_CONV1_PAD           1
#define BB2_CONV1_CONV_EN       1
#define BB2_CONV1_BN_EN         1
#define BB2_CONV1_SKIP_EN       2
#define BB2_CONV1_RELU_EN       1
#define BB2_CONV1_MAX_POOL      0
#define BB2_CONV1_AVG_POOL      0
#define BB2_CONV1_LIN           0
#define BB2_CONV1_IN_MEM        1
#define BB2_CONV1_OUT_MEM       2
#define BB2_CONV1_SKIP_MEM      3
// BB2_CONV2
#define BB2_CONV2_C             64
#define BB2_CONV2_H             56
#define BB2_CONV2_W             56
#define BB2_CONV2_K             3
#define BB2_CONV2_S             1
#define BB2_CONV2_PAD           1
#define BB2_CONV2_CONV_EN       1
#define BB2_CONV2_BN_EN         1
#define BB2_CONV2_SKIP_EN       2
#define BB2_CONV2_RELU_EN       2
#define BB2_CONV2_MAX_POOL      0
#define BB2_CONV2_AVG_POOL      0
#define BB2_CONV2_LIN           0
#define BB2_CONV2_IN_MEM        2
#define BB2_CONV2_OUT_MEM       0
#define BB2_CONV2_SKIP_MEM      3
// BB2_SKIP
#define BB2_SKIP_C              64
#define BB2_SKIP_H              56
#define BB2_SKIP_W              56
#define BB2_SKIP_K              0
#define BB2_SKIP_S              0
#define BB2_SKIP_PAD            0
#define BB2_SKIP_CONV_EN        2
#define BB2_SKIP_BN_EN          2
#define BB2_SKIP_SKIP_EN        1
#define BB2_SKIP_RELU_EN        1
#define BB2_SKIP_MAX_POOL       0
#define BB2_SKIP_AVG_POOL       0
#define BB2_SKIP_LIN            0
#define BB2_SKIP_IN_MEM         0
#define BB2_SKIP_OUT_MEM        2
#define BB2_SKIP_SKIP_MEM       1

// BB3_CONV1
#define BB3_CONV1_C             128
#define BB3_CONV1_H             28
#define BB3_CONV1_W             28
#define BB3_CONV1_K             3
#define BB3_CONV1_S             2
#define BB3_CONV1_PAD           1
#define BB3_CONV1_CONV_EN       1
#define BB3_CONV1_BN_EN         1
#define BB3_CONV1_SKIP_EN       2
#define BB3_CONV1_RELU_EN       1
#define BB3_CONV1_MAX_POOL      0
#define BB3_CONV1_AVG_POOL      0
#define BB3_CONV1_LIN           0
#define BB3_CONV1_IN_MEM        2
#define BB3_CONV1_OUT_MEM       0
#define BB3_CONV1_SKIP_MEM      3
// BB3_CONV2
#define BB3_CONV2_C             128
#define BB3_CONV2_H             28
#define BB3_CONV2_W             28
#define BB3_CONV2_K             3
#define BB3_CONV2_S             1
#define BB3_CONV2_PAD           1
#define BB3_CONV2_CONV_EN       1
#define BB3_CONV2_BN_EN         1
#define BB3_CONV2_SKIP_EN       2
#define BB3_CONV2_RELU_EN       2
#define BB3_CONV2_MAX_POOL      0
#define BB3_CONV2_AVG_POOL      0
#define BB3_CONV2_LIN           0
#define BB3_CONV2_IN_MEM        0
#define BB3_CONV2_OUT_MEM       1
#define BB3_CONV2_SKIP_MEM      3
// BB3_SKIP
#define BB3_SKIP_C              128
#define BB3_SKIP_H              28
#define BB3_SKIP_W              28
#define BB3_SKIP_K              1
#define BB3_SKIP_S              2
#define BB3_SKIP_PAD            0
#define BB3_SKIP_CONV_EN        1
#define BB3_SKIP_BN_EN          1
#define BB3_SKIP_SKIP_EN        1
#define BB3_SKIP_RELU_EN        1
#define BB3_SKIP_MAX_POOL       0
#define BB3_SKIP_AVG_POOL       0
#define BB3_SKIP_LIN            0
#define BB3_SKIP_IN_MEM         1
#define BB3_SKIP_OUT_MEM        0
#define BB3_SKIP_SKIP_MEM       2

// BB4_CONV1
#define BB4_CONV1_C             128
#define BB4_CONV1_H             28
#define BB4_CONV1_W             28
#define BB4_CONV1_K             3
#define BB4_CONV1_S             1
#define BB4_CONV1_PAD           1
#define BB4_CONV1_CONV_EN       1
#define BB4_CONV1_BN_EN         1
#define BB4_CONV1_SKIP_EN       2
#define BB4_CONV1_RELU_EN       1
#define BB4_CONV1_MAX_POOL      0
#define BB4_CONV1_AVG_POOL      0
#define BB4_CONV1_LIN           0
#define BB4_CONV1_IN_MEM        0
#define BB4_CONV1_OUT_MEM       1
#define BB4_CONV1_SKIP_MEM      3
// BB4_CONV2
#define BB4_CONV2_C             128
#define BB4_CONV2_H             28
#define BB4_CONV2_W             28
#define BB4_CONV2_K             3
#define BB4_CONV2_S             1
#define BB4_CONV2_PAD           1
#define BB4_CONV2_CONV_EN       1
#define BB4_CONV2_BN_EN         1
#define BB4_CONV2_SKIP_EN       2
#define BB4_CONV2_RELU_EN       2
#define BB4_CONV2_MAX_POOL      0
#define BB4_CONV2_AVG_POOL      0
#define BB4_CONV2_LIN           0
#define BB4_CONV2_IN_MEM        1
#define BB4_CONV2_OUT_MEM       2
#define BB4_CONV2_SKIP_MEM      3
// BB4_SKIP
#define BB4_SKIP_C              128
#define BB4_SKIP_H              28
#define BB4_SKIP_W              28
#define BB4_SKIP_K              0
#define BB4_SKIP_S              0
#define BB4_SKIP_PAD            0
#define BB4_SKIP_CONV_EN        2
#define BB4_SKIP_BN_EN          2
#define BB4_SKIP_SKIP_EN        1
#define BB4_SKIP_RELU_EN        1
#define BB4_SKIP_MAX_POOL       0
#define BB4_SKIP_AVG_POOL       0
#define BB4_SKIP_LIN            0
#define BB4_SKIP_IN_MEM         2
#define BB4_SKIP_OUT_MEM        1
#define BB4_SKIP_SKIP_MEM       0

// BB5_CONV1
#define BB5_CONV1_C             256
#define BB5_CONV1_H             14
#define BB5_CONV1_W             14
#define BB5_CONV1_K             3
#define BB5_CONV1_S             2
#define BB5_CONV1_PAD           1
#define BB5_CONV1_CONV_EN       1
#define BB5_CONV1_BN_EN         1
#define BB5_CONV1_SKIP_EN       2
#define BB5_CONV1_RELU_EN       1
#define BB5_CONV1_MAX_POOL      0
#define BB5_CONV1_AVG_POOL      0
#define BB5_CONV1_LIN           0
#define BB5_CONV1_IN_MEM        1
#define BB5_CONV1_OUT_MEM       2
#define BB5_CONV1_SKIP_MEM      3
// BB5_CONV2
#define BB5_CONV2_C             256
#define BB5_CONV2_H             14
#define BB5_CONV2_W             14
#define BB5_CONV2_K             3
#define BB5_CONV2_S             1
#define BB5_CONV2_PAD           1
#define BB5_CONV2_CONV_EN       1
#define BB5_CONV2_BN_EN         1
#define BB5_CONV2_SKIP_EN       2
#define BB5_CONV2_RELU_EN       2
#define BB5_CONV2_MAX_POOL      0
#define BB5_CONV2_AVG_POOL      0
#define BB5_CONV2_LIN           0
#define BB5_CONV2_IN_MEM        2
#define BB5_CONV2_OUT_MEM       0
#define BB5_CONV2_SKIP_MEM      3
// BB5_SKIP
#define BB5_SKIP_C              256
#define BB5_SKIP_H              14
#define BB5_SKIP_W              14
#define BB5_SKIP_K              1
#define BB5_SKIP_S              2
#define BB5_SKIP_PAD            0
#define BB5_SKIP_CONV_EN        1
#define BB5_SKIP_BN_EN          1
#define BB5_SKIP_SKIP_EN        1
#define BB5_SKIP_RELU_EN        1
#define BB5_SKIP_MAX_POOL       0
#define BB5_SKIP_AVG_POOL       0
#define BB5_SKIP_LIN            0
#define BB5_SKIP_IN_MEM         0
#define BB5_SKIP_OUT_MEM        2
#define BB5_SKIP_SKIP_MEM       1

// BB6_CONV1
#define BB6_CONV1_C             256
#define BB6_CONV1_H             14
#define BB6_CONV1_W             14
#define BB6_CONV1_K             3
#define BB6_CONV1_S             1
#define BB6_CONV1_PAD           1
#define BB6_CONV1_CONV_EN       1
#define BB6_CONV1_BN_EN         1
#define BB6_CONV1_SKIP_EN       2
#define BB6_CONV1_RELU_EN       1
#define BB6_CONV1_MAX_POOL      0
#define BB6_CONV1_AVG_POOL      0
#define BB6_CONV1_LIN           0
#define BB6_CONV1_IN_MEM        2
#define BB6_CONV1_OUT_MEM       0
#define BB6_CONV1_SKIP_MEM      3
// BB6_CONV2    
#define BB6_CONV2_C             256
#define BB6_CONV2_H             14
#define BB6_CONV2_W             14
#define BB6_CONV2_K             3
#define BB6_CONV2_S             1
#define BB6_CONV2_PAD           1
#define BB6_CONV2_CONV_EN       1
#define BB6_CONV2_BN_EN         1
#define BB6_CONV2_SKIP_EN       2
#define BB6_CONV2_RELU_EN       2
#define BB6_CONV2_MAX_POOL      0
#define BB6_CONV2_AVG_POOL      0
#define BB6_CONV2_LIN           0
#define BB6_CONV2_IN_MEM        0
#define BB6_CONV2_OUT_MEM       1
#define BB6_CONV2_SKIP_MEM      3
// BB6_SKIP 
#define BB6_SKIP_C              256
#define BB6_SKIP_H              14
#define BB6_SKIP_W              14
#define BB6_SKIP_K              0
#define BB6_SKIP_S              0
#define BB6_SKIP_PAD            0
#define BB6_SKIP_CONV_EN        2
#define BB6_SKIP_BN_EN          2
#define BB6_SKIP_SKIP_EN        1
#define BB6_SKIP_RELU_EN        1
#define BB6_SKIP_MAX_POOL       0
#define BB6_SKIP_AVG_POOL       0
#define BB6_SKIP_LIN            0
#define BB6_SKIP_IN_MEM         1
#define BB6_SKIP_OUT_MEM        0
#define BB6_SKIP_SKIP_MEM       2

// BB7_CONV1    
#define BB7_CONV1_C             512
#define BB7_CONV1_H             7
#define BB7_CONV1_W             7
#define BB7_CONV1_K             3
#define BB7_CONV1_S             2
#define BB7_CONV1_PAD           1
#define BB7_CONV1_CONV_EN       1
#define BB7_CONV1_BN_EN         1
#define BB7_CONV1_SKIP_EN       2
#define BB7_CONV1_RELU_EN       1
#define BB7_CONV1_MAX_POOL      0
#define BB7_CONV1_AVG_POOL      0
#define BB7_CONV1_LIN           0
#define BB7_CONV1_IN_MEM        0
#define BB7_CONV1_OUT_MEM       1
#define BB7_CONV1_SKIP_MEM      3
constexpr unsigned BB7_CONV1_WEIGHT_BASE = 0;
constexpr unsigned BB7_CONV1_CONV_WEIGHT_SIZE = BB6_SKIP_C * BB7_CONV1_C * BB7_CONV1_K * BB7_CONV1_K / WEIGHT_PACK;
constexpr unsigned BB7_CONV1_BN_WEIGHT_BASE = 0;
constexpr unsigned BB7_CONV1_BN_WEIGHT_SIZE = 4 * BB7_CONV1_C;
// BB7_CONV2    
#define BB7_CONV2_C             512
#define BB7_CONV2_H             7
#define BB7_CONV2_W             7
#define BB7_CONV2_K             3
#define BB7_CONV2_S             1
#define BB7_CONV2_PAD           1
#define BB7_CONV2_CONV_EN       1
#define BB7_CONV2_BN_EN         1
#define BB7_CONV2_SKIP_EN       2
#define BB7_CONV2_RELU_EN       2
#define BB7_CONV2_MAX_POOL      0
#define BB7_CONV2_AVG_POOL      0
#define BB7_CONV2_LIN           0
#define BB7_CONV2_IN_MEM        1
#define BB7_CONV2_OUT_MEM       2
#define BB7_CONV2_SKIP_MEM      3
constexpr unsigned BB7_CONV2_WEIGHT_BASE = BB7_CONV1_WEIGHT_BASE + BB7_CONV1_CONV_WEIGHT_SIZE;
constexpr unsigned BB7_CONV2_CONV_WEIGHT_SIZE = BB7_CONV1_C * BB7_CONV2_C * BB7_CONV2_K * BB7_CONV2_K / WEIGHT_PACK;
constexpr unsigned BB7_CONV2_BN_WEIGHT_BASE = BB7_CONV1_BN_WEIGHT_BASE + BB7_CONV1_BN_WEIGHT_SIZE;
constexpr unsigned BB7_CONV2_BN_WEIGHT_SIZE = 4 * BB7_CONV2_C;
// BB7_SKIP 
#define BB7_SKIP_C              512
#define BB7_SKIP_H              7
#define BB7_SKIP_W              7
#define BB7_SKIP_K              1
#define BB7_SKIP_S              2
#define BB7_SKIP_PAD            0
#define BB7_SKIP_CONV_EN        1
#define BB7_SKIP_BN_EN          1
#define BB7_SKIP_SKIP_EN        1
#define BB7_SKIP_RELU_EN        1
#define BB7_SKIP_MAX_POOL       0
#define BB7_SKIP_AVG_POOL       0
#define BB7_SKIP_LIN            0
#define BB7_SKIP_IN_MEM         2
#define BB7_SKIP_OUT_MEM        1
#define BB7_SKIP_SKIP_MEM       0
constexpr unsigned BB7_SKIP_WEIGHT_BASE = BB7_CONV2_WEIGHT_BASE + BB7_CONV2_CONV_WEIGHT_SIZE;
constexpr unsigned BB7_SKIP_CONV_WEIGHT_SIZE = BB7_CONV2_C * BB7_SKIP_C * BB7_SKIP_K * BB7_SKIP_K / WEIGHT_PACK;
constexpr unsigned BB7_SKIP_BN_WEIGHT_BASE = BB7_CONV2_BN_WEIGHT_BASE + BB7_CONV2_BN_WEIGHT_SIZE;
constexpr unsigned BB7_SKIP_BN_WEIGHT_SIZE = 4 * BB7_CONV2_C;

// BB8_CONV1    
#define BB8_CONV1_C             512
#define BB8_CONV1_H             7
#define BB8_CONV1_W             7
#define BB8_CONV1_K             3
#define BB8_CONV1_S             1
#define BB8_CONV1_PAD           1
#define BB8_CONV1_CONV_EN       1
#define BB8_CONV1_BN_EN         1
#define BB8_CONV1_SKIP_EN       2
#define BB8_CONV1_RELU_EN       1
#define BB8_CONV1_MAX_POOL      0
#define BB8_CONV1_AVG_POOL      0
#define BB8_CONV1_LIN           0
#define BB8_CONV1_IN_MEM        1
#define BB8_CONV1_OUT_MEM       2
#define BB8_CONV1_SKIP_MEM      3
// BB8_CONV2    
#define BB8_CONV2_C             512
#define BB8_CONV2_H             7
#define BB8_CONV2_W             7
#define BB8_CONV2_K             3
#define BB8_CONV2_S             1
#define BB8_CONV2_PAD           1
#define BB8_CONV2_CONV_EN       1
#define BB8_CONV2_BN_EN         1
#define BB8_CONV2_SKIP_EN       2
#define BB8_CONV2_RELU_EN       2
#define BB8_CONV2_MAX_POOL      0
#define BB8_CONV2_AVG_POOL      0
#define BB8_CONV2_LIN           0
#define BB8_CONV2_IN_MEM        2
#define BB8_CONV2_OUT_MEM       0
#define BB8_CONV2_SKIP_MEM      3
// BB8_SKIP 
#define BB8_SKIP_C              512
#define BB8_SKIP_H              7
#define BB8_SKIP_W              7
#define BB8_SKIP_K              0
#define BB8_SKIP_S              0
#define BB8_SKIP_PAD            0
#define BB8_SKIP_CONV_EN        2
#define BB8_SKIP_BN_EN          2
#define BB8_SKIP_SKIP_EN        1
#define BB8_SKIP_RELU_EN        1
#define BB8_SKIP_MAX_POOL       0
#define BB8_SKIP_AVG_POOL       0
#define BB8_SKIP_LIN            0
#define BB8_SKIP_IN_MEM         0
#define BB8_SKIP_OUT_MEM        2
#define BB8_SKIP_SKIP_MEM       1

// AVG_POOL 
#define AVG_POOL_C              512
#define AVG_POOL_H              1
#define AVG_POOL_W              1
#define AVG_POOL_K              0
#define AVG_POOL_S              0
#define AVG_POOL_PAD            0
#define AVG_POOL_CONV_EN        0
#define AVG_POOL_BN_EN          0
#define AVG_POOL_SKIP_EN        0
#define AVG_POOL_RELU_EN        0
#define AVG_POOL_MAX_POOL       0
#define AVG_POOL_AVG_POOL       1
#define AVG_POOL_LIN            0
#define AVG_POOL_IN_MEM         2
#define AVG_POOL_OUT_MEM        0
#define AVG_POOL_SKIP_MEM       3

// LIN
#define LIN_C                   1000
#define LIN_H                   1
#define LIN_W                   1
#define LIN_K                   0
#define LIN_S                   0
#define LIN_PAD                 0
#define LIN_CONV_EN             0
#define LIN_BN_EN               0
#define LIN_SKIP_EN             0
#define LIN_RELU_EN             0
#define LIN_MAX_POOL            0
#define LIN_AVG_POOL            0
#define LIN_LIN                 1
#define LIN_IN_MEM              0
#define LIN_OUT_MEM             1
#define LIN_SKIP_MEM            3

// size of mem blocks
constexpr unsigned MEM0_SIZE = CONV1_C  * CONV1_H * CONV1_W / POX;
constexpr unsigned MEM1_SIZE = MAXPOOL_C  * MAXPOOL_H * MAXPOOL_W / POX;
constexpr unsigned MEM2_SIZE = BB1_CONV2_C  * BB1_CONV2_H * BB1_CONV2_W / POX;
constexpr unsigned WEIGHT_MEM_SIZE = 1200000/WEIGHT_PACK;  // todo: temporary for now
constexpr unsigned BN_WEIGHT_MEM_SIZE = 100000;
#endif
