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
#include <algorithm>

// Include C++ headers
#include <cmath>
#include <cassert>

// *****************************************
// simulation settings vs synthesis settings
#define SIM_MODE            1
// *****************************************

// kernel parallel parameters
#define PKX                 1
#define PKY                 1
#define PIF                 1
#define POX                 7
#define POY                 7
#define POF                 1       // change_to: if necessary
#define MAX_STRIDE          2
#define MAX_PAD             3

#define WEIGHT_PACK         1       // change_to: 8
#define ACT_PACK            1       // change to: 7?

// Bit widths
#if SIM_MODE
constexpr unsigned W_ACT = 16;
constexpr unsigned I_ACT = 8;
constexpr unsigned W_FIL = 16;
constexpr unsigned I_FIL = 8;
#else
constexpr unsigned W_ACT = 8;
constexpr unsigned I_ACT = 3;
constexpr unsigned W_FIL = 8;
constexpr unsigned I_FIL = 2;
#endif
constexpr unsigned W_MUL = (W_ACT + W_FIL);
constexpr unsigned I_MUL = (I_ACT + I_FIL);
constexpr unsigned W_MAC = 32;      // todo: 24?
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
constexpr unsigned int FIFO_ARR_DEPTH = 9;  // todo: reduce if unnecessary

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

#if SIM_MODE
// CONV1 layer cnt 0
#define CONV1_C                     2
#define CONV1_H                     112
#define CONV1_W                     112
#define CONV1_K                     7
#define CONV1_S                     2
#define CONV1_PAD                   3
#define CONV1_BB_EN                 1
#define CONV1_CONV_EN               1
#define CONV1_BN_EN                 1
#define CONV1_SKIP_EN               0
#define CONV1_RELU_EN               1
#define CONV1_MAX_POOL_EN           0
#define CONV1_AVG_POOL_EN           0
#define CONV1_FC_EN                0
#define CONV1_BASE_ADDR_IN          MEM0_BASE_ADDR
#define CONV1_BASE_ADDR_OUT         MEM1_BASE_ADDR
#define CONV1_BASE_ADDR_ADD         NOMEM_BASE_ADDR
constexpr unsigned CONV1_WEIGHT_BASE = 0;
constexpr unsigned CONV1_WEIGHT_SIZE = IN_C * CONV1_C * CONV1_K * CONV1_K / WEIGHT_PACK;
constexpr unsigned CONV1_BN_WEIGHT_BASE = 0;
constexpr unsigned CONV1_BN_WEIGHT_SIZE = 3 * CONV1_C;
constexpr unsigned CONV1_IN_SIZE = IN_C * IN_H * IN_W;
constexpr unsigned CONV1_OUT_SIZE = CONV1_C * CONV1_H * CONV1_W;

// MAX_POOL layer cnt 1
#define MAX_POOL_C                   2
#define MAX_POOL_H                   56
#define MAX_POOL_W                   56
#define MAX_POOL_K                   3
#define MAX_POOL_S                   2
#define MAX_POOL_PAD                 1
#define MAX_POOL_BB_EN               0
#define MAX_POOL_CONV_EN             0
#define MAX_POOL_BN_EN               0
#define MAX_POOL_SKIP_EN             0
#define MAX_POOL_RELU_EN             0
#define MAX_POOL_MAX_POOL_EN         1
#define MAX_POOL_AVG_POOL_EN         0
#define MAX_POOL_FC_EN              0
#define MAX_POOL_BASE_ADDR_IN        MEM1_BASE_ADDR
#define MAX_POOL_BASE_ADDR_OUT       MEM0_BASE_ADDR
#define MAX_POOL_BASE_ADDR_ADD       NOMEM_BASE_ADDR
constexpr unsigned MAX_POOL_WEIGHT_BASE = 0;
constexpr unsigned MAX_POOL_WEIGHT_SIZE = 0;
constexpr unsigned MAX_POOL_BN_WEIGHT_BASE = 0;
constexpr unsigned MAX_POOL_BN_WEIGHT_SIZE = 0;
constexpr unsigned MAX_POOL_IN_SIZE = CONV1_C * CONV1_H * CONV1_W;
constexpr unsigned MAX_POOL_OUT_SIZE = MAX_POOL_C * MAX_POOL_H * MAX_POOL_W;

// BB1_CONV1 layer cnt 2
#define BB1_CONV1_C                 2
#define BB1_CONV1_H                 56
#define BB1_CONV1_W                 56
#define BB1_CONV1_K                 3
#define BB1_CONV1_S                 1
#define BB1_CONV1_PAD               1
#define BB1_CONV1_BB_EN             1
#define BB1_CONV1_CONV_EN           1
#define BB1_CONV1_BN_EN             1
#define BB1_CONV1_SKIP_EN           0
#define BB1_CONV1_RELU_EN           1
#define BB1_CONV1_MAX_POOL_EN       0
#define BB1_CONV1_AVG_POOL_EN       0
#define BB1_CONV1_FC_EN            0
#define BB1_CONV1_BASE_ADDR_IN      MEM0_BASE_ADDR
#define BB1_CONV1_BASE_ADDR_OUT     MEM1_BASE_ADDR
#define BB1_CONV1_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB1_CONV1_WEIGHT_BASE = CONV1_WEIGHT_BASE + CONV1_WEIGHT_SIZE;
constexpr unsigned BB1_CONV1_WEIGHT_SIZE = MAX_POOL_C * BB1_CONV1_C * BB1_CONV1_K * BB1_CONV1_K / WEIGHT_PACK;
constexpr unsigned BB1_CONV1_BN_WEIGHT_BASE = CONV1_BN_WEIGHT_BASE + CONV1_BN_WEIGHT_SIZE;
constexpr unsigned BB1_CONV1_BN_WEIGHT_SIZE = 3 * BB1_CONV1_C;
constexpr unsigned BB1_CONV1_IN_SIZE = MAX_POOL_C * MAX_POOL_H * MAX_POOL_W;
constexpr unsigned BB1_CONV1_OUT_SIZE = BB1_CONV1_C * BB1_CONV1_H * BB1_CONV1_W;

// BB1_CONV2 layer cnt 3
#define BB1_CONV2_C                 2
#define BB1_CONV2_H                 56
#define BB1_CONV2_W                 56
#define BB1_CONV2_K                 3
#define BB1_CONV2_S                 1
#define BB1_CONV2_PAD               1
#define BB1_CONV2_BB_EN             1
#define BB1_CONV2_CONV_EN           1
#define BB1_CONV2_BN_EN             1
#define BB1_CONV2_SKIP_EN           0
#define BB1_CONV2_RELU_EN           0
#define BB1_CONV2_MAX_POOL_EN       0
#define BB1_CONV2_AVG_POOL_EN       0
#define BB1_CONV2_FC_EN            0
#define BB1_CONV2_BASE_ADDR_IN      MEM1_BASE_ADDR
#define BB1_CONV2_BASE_ADDR_OUT     MEM2_BASE_ADDR
#define BB1_CONV2_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB1_CONV2_WEIGHT_BASE = BB1_CONV1_WEIGHT_BASE + BB1_CONV1_WEIGHT_SIZE;
constexpr unsigned BB1_CONV2_WEIGHT_SIZE = BB1_CONV1_C * BB1_CONV2_C * BB1_CONV2_K * BB1_CONV2_K / WEIGHT_PACK;
constexpr unsigned BB1_CONV2_BN_WEIGHT_BASE = BB1_CONV1_BN_WEIGHT_BASE + BB1_CONV1_BN_WEIGHT_SIZE;
constexpr unsigned BB1_CONV2_BN_WEIGHT_SIZE = 3 * BB1_CONV2_C;
constexpr unsigned BB1_CONV2_IN_SIZE = BB1_CONV1_C * BB1_CONV1_H * BB1_CONV1_W;
constexpr unsigned BB1_CONV2_OUT_SIZE = BB1_CONV2_C * BB1_CONV2_H * BB1_CONV2_W;

// BB1_SKIP layer cnt 4
#define BB1_SKIP_C                  2
#define BB1_SKIP_H                  56
#define BB1_SKIP_W                  56
#define BB1_SKIP_K                  0
#define BB1_SKIP_S                  0
#define BB1_SKIP_PAD                0
#define BB1_SKIP_BB_EN              1
#define BB1_SKIP_CONV_EN            0
#define BB1_SKIP_BN_EN              0
#define BB1_SKIP_SKIP_EN            1
#define BB1_SKIP_RELU_EN            1
#define BB1_SKIP_MAX_POOL_EN        0
#define BB1_SKIP_AVG_POOL_EN        0
#define BB1_SKIP_FC_EN             0
#define BB1_SKIP_BASE_ADDR_IN       MEM0_BASE_ADDR
#define BB1_SKIP_BASE_ADDR_OUT      MEM1_BASE_ADDR
#define BB1_SKIP_BASE_ADDR_ADD      MEM2_BASE_ADDR
constexpr unsigned BB1_SKIP_WEIGHT_BASE = 0;
constexpr unsigned BB1_SKIP_WEIGHT_SIZE = 0;
constexpr unsigned BB1_SKIP_BN_WEIGHT_BASE = 0;
constexpr unsigned BB1_SKIP_BN_WEIGHT_SIZE = 0;
constexpr unsigned BB1_SKIP_IN_SIZE = BB1_CONV2_C * BB1_CONV2_H * BB1_CONV2_W;
constexpr unsigned BB1_SKIP_OUT_SIZE = BB1_SKIP_C * BB1_SKIP_H * BB1_SKIP_W;

// BB2_CONV1 layer cnt 5
#define BB2_CONV1_C                 2
#define BB2_CONV1_H                 56
#define BB2_CONV1_W                 56
#define BB2_CONV1_K                 3
#define BB2_CONV1_S                 1
#define BB2_CONV1_PAD               1
#define BB2_CONV1_BB_EN             1
#define BB2_CONV1_CONV_EN           1
#define BB2_CONV1_BN_EN             1
#define BB2_CONV1_SKIP_EN           0
#define BB2_CONV1_RELU_EN           1
#define BB2_CONV1_MAX_POOL_EN       0
#define BB2_CONV1_AVG_POOL_EN       0
#define BB2_CONV1_FC_EN            0
#define BB2_CONV1_BASE_ADDR_IN      MEM1_BASE_ADDR
#define BB2_CONV1_BASE_ADDR_OUT     MEM2_BASE_ADDR
#define BB2_CONV1_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB2_CONV1_WEIGHT_BASE = BB1_CONV2_WEIGHT_BASE + BB1_CONV2_WEIGHT_SIZE;
constexpr unsigned BB2_CONV1_WEIGHT_SIZE = BB1_SKIP_C * BB2_CONV1_C * BB2_CONV1_K * BB2_CONV1_K / WEIGHT_PACK;
constexpr unsigned BB2_CONV1_BN_WEIGHT_BASE = BB1_CONV2_BN_WEIGHT_BASE + BB1_CONV2_BN_WEIGHT_SIZE;
constexpr unsigned BB2_CONV1_BN_WEIGHT_SIZE = 3 * BB2_CONV1_C;
constexpr unsigned BB2_CONV1_IN_SIZE = BB1_SKIP_C * BB1_SKIP_H * BB1_SKIP_W;
constexpr unsigned BB2_CONV1_OUT_SIZE = BB2_CONV1_C * BB2_CONV1_H * BB2_CONV1_W;

// BB2_CONV2 layer cnt 6
#define BB2_CONV2_C                 2
#define BB2_CONV2_H                 56
#define BB2_CONV2_W                 56
#define BB2_CONV2_K                 3
#define BB2_CONV2_S                 1
#define BB2_CONV2_PAD               1
#define BB2_CONV2_BB_EN             1
#define BB2_CONV2_CONV_EN           1
#define BB2_CONV2_BN_EN             1
#define BB2_CONV2_SKIP_EN           0
#define BB2_CONV2_RELU_EN           0
#define BB2_CONV2_MAX_POOL_EN       0
#define BB2_CONV2_AVG_POOL_EN       0
#define BB2_CONV2_FC_EN            0
#define BB2_CONV2_BASE_ADDR_IN      MEM2_BASE_ADDR
#define BB2_CONV2_BASE_ADDR_OUT     MEM0_BASE_ADDR
#define BB2_CONV2_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB2_CONV2_WEIGHT_BASE = BB2_CONV1_WEIGHT_BASE + BB2_CONV1_WEIGHT_SIZE;
constexpr unsigned BB2_CONV2_WEIGHT_SIZE = BB2_CONV1_C * BB2_CONV2_C * BB2_CONV2_K * BB2_CONV2_K / WEIGHT_PACK;
constexpr unsigned BB2_CONV2_BN_WEIGHT_BASE = BB2_CONV1_BN_WEIGHT_BASE + BB2_CONV1_BN_WEIGHT_SIZE;
constexpr unsigned BB2_CONV2_BN_WEIGHT_SIZE = 3 * BB2_CONV2_C;
constexpr unsigned BB2_CONV2_IN_SIZE = BB2_CONV1_C * BB2_CONV1_H * BB2_CONV1_W;
constexpr unsigned BB2_CONV2_OUT_SIZE = BB2_CONV2_C * BB2_CONV2_H * BB2_CONV2_W;

// BB2_SKIP layer cnt 7
#define BB2_SKIP_C                  2
#define BB2_SKIP_H                  56
#define BB2_SKIP_W                  56
#define BB2_SKIP_K                  0
#define BB2_SKIP_S                  0
#define BB2_SKIP_PAD                0
#define BB2_SKIP_BB_EN              1
#define BB2_SKIP_CONV_EN            0
#define BB2_SKIP_BN_EN              0
#define BB2_SKIP_SKIP_EN            1
#define BB2_SKIP_RELU_EN            1
#define BB2_SKIP_MAX_POOL_EN        0
#define BB2_SKIP_AVG_POOL_EN        0
#define BB2_SKIP_FC_EN             0
#define BB2_SKIP_BASE_ADDR_IN       MEM1_BASE_ADDR
#define BB2_SKIP_BASE_ADDR_OUT      MEM2_BASE_ADDR
#define BB2_SKIP_BASE_ADDR_ADD      MEM0_BASE_ADDR
constexpr unsigned BB2_SKIP_WEIGHT_BASE = 0;
constexpr unsigned BB2_SKIP_WEIGHT_SIZE = 0;
constexpr unsigned BB2_SKIP_BN_WEIGHT_BASE = 0;
constexpr unsigned BB2_SKIP_BN_WEIGHT_SIZE = 0;
constexpr unsigned BB2_SKIP_IN_SIZE = BB2_CONV2_C * BB2_CONV2_H * BB2_CONV2_W;
constexpr unsigned BB2_SKIP_OUT_SIZE = BB2_SKIP_C * BB2_SKIP_H * BB2_SKIP_W;

// BB3_CONV1 layer cnt 8
#define BB3_CONV1_C                 4
#define BB3_CONV1_H                 28
#define BB3_CONV1_W                 28
#define BB3_CONV1_K                 3
#define BB3_CONV1_S                 2
#define BB3_CONV1_PAD               1
#define BB3_CONV1_BB_EN             1
#define BB3_CONV1_CONV_EN           1
#define BB3_CONV1_BN_EN             1
#define BB3_CONV1_SKIP_EN           0
#define BB3_CONV1_RELU_EN           1
#define BB3_CONV1_MAX_POOL_EN       0
#define BB3_CONV1_AVG_POOL_EN       0
#define BB3_CONV1_FC_EN            0
#define BB3_CONV1_BASE_ADDR_IN      MEM2_BASE_ADDR
#define BB3_CONV1_BASE_ADDR_OUT     MEM0_BASE_ADDR
#define BB3_CONV1_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB3_CONV1_WEIGHT_BASE = BB2_CONV2_WEIGHT_BASE + BB2_CONV2_WEIGHT_SIZE;
constexpr unsigned BB3_CONV1_WEIGHT_SIZE = BB2_SKIP_C * BB3_CONV1_C * BB3_CONV1_K * BB3_CONV1_K / WEIGHT_PACK;
constexpr unsigned BB3_CONV1_BN_WEIGHT_BASE = BB2_CONV2_BN_WEIGHT_BASE + BB2_CONV2_BN_WEIGHT_SIZE;
constexpr unsigned BB3_CONV1_BN_WEIGHT_SIZE = 3 * BB3_CONV1_C;
constexpr unsigned BB3_CONV1_IN_SIZE = BB2_SKIP_C * BB2_SKIP_H * BB2_SKIP_W;
constexpr unsigned BB3_CONV1_OUT_SIZE = BB3_CONV1_C * BB3_CONV1_H * BB3_CONV1_W;

// BB3_CONV2 layer cnt 9
#define BB3_CONV2_C                 4
#define BB3_CONV2_H                 28
#define BB3_CONV2_W                 28
#define BB3_CONV2_K                 3
#define BB3_CONV2_S                 1
#define BB3_CONV2_PAD               1
#define BB3_CONV2_BB_EN             1
#define BB3_CONV2_CONV_EN           1
#define BB3_CONV2_BN_EN             1
#define BB3_CONV2_SKIP_EN           0
#define BB3_CONV2_RELU_EN           0
#define BB3_CONV2_MAX_POOL_EN       0
#define BB3_CONV2_AVG_POOL_EN       0
#define BB3_CONV2_FC_EN            0
#define BB3_CONV2_BASE_ADDR_IN      MEM0_BASE_ADDR
#define BB3_CONV2_BASE_ADDR_OUT     MEM1_BASE_ADDR
#define BB3_CONV2_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB3_CONV2_WEIGHT_BASE = BB3_CONV1_WEIGHT_BASE + BB3_CONV1_WEIGHT_SIZE;
constexpr unsigned BB3_CONV2_WEIGHT_SIZE = BB3_CONV1_C * BB3_CONV2_C * BB3_CONV2_K * BB3_CONV2_K / WEIGHT_PACK;
constexpr unsigned BB3_CONV2_BN_WEIGHT_BASE = BB3_CONV1_BN_WEIGHT_BASE + BB3_CONV1_BN_WEIGHT_SIZE;
constexpr unsigned BB3_CONV2_BN_WEIGHT_SIZE = 3 * BB3_CONV2_C;
constexpr unsigned BB3_CONV2_IN_SIZE = BB3_CONV1_C * BB3_CONV1_H * BB3_CONV1_W;
constexpr unsigned BB3_CONV2_OUT_SIZE = BB3_CONV2_C * BB3_CONV2_H * BB3_CONV2_W;

// BB3_SKIP layer (PROJ) cnt 10
#define BB3_SKIP_C                  2
#define BB3_SKIP_H                  28
#define BB3_SKIP_W                  28
#define BB3_SKIP_K                  1
#define BB3_SKIP_S                  2
#define BB3_SKIP_PAD                0
#define BB3_SKIP_BB_EN              1
#define BB3_SKIP_CONV_EN            1
#define BB3_SKIP_BN_EN              1
#define BB3_SKIP_SKIP_EN            1
#define BB3_SKIP_RELU_EN            1
#define BB3_SKIP_MAX_POOL_EN        0
#define BB3_SKIP_AVG_POOL_EN        0
#define BB3_SKIP_FC_EN             0
#define BB3_SKIP_BASE_ADDR_IN       MEM2_BASE_ADDR
#define BB3_SKIP_BASE_ADDR_OUT      MEM0_BASE_ADDR
#define BB3_SKIP_BASE_ADDR_ADD      MEM1_BASE_ADDR
constexpr unsigned BB3_SKIP_WEIGHT_BASE = BB3_CONV2_WEIGHT_BASE + BB3_CONV2_WEIGHT_SIZE;
constexpr unsigned BB3_SKIP_WEIGHT_SIZE = BB3_CONV2_C * BB3_SKIP_C * BB3_SKIP_K * BB3_SKIP_K / WEIGHT_PACK;
constexpr unsigned BB3_SKIP_BN_WEIGHT_BASE = BB3_CONV2_BN_WEIGHT_BASE + BB3_CONV2_BN_WEIGHT_SIZE;
constexpr unsigned BB3_SKIP_BN_WEIGHT_SIZE = 3 * BB3_SKIP_C;
constexpr unsigned BB3_SKIP_IN_SIZE = BB3_CONV2_C * BB3_CONV2_H * BB3_CONV2_W;
constexpr unsigned BB3_SKIP_OUT_SIZE = BB3_SKIP_C * BB3_SKIP_H * BB3_SKIP_W;

// BB4_CONV1 layer cnt 11
#define BB4_CONV1_C                 4
#define BB4_CONV1_H                 28
#define BB4_CONV1_W                 28
#define BB4_CONV1_K                 3
#define BB4_CONV1_S                 1
#define BB4_CONV1_PAD               1
#define BB4_CONV1_BB_EN             1
#define BB4_CONV1_CONV_EN           1
#define BB4_CONV1_BN_EN             1
#define BB4_CONV1_SKIP_EN           0
#define BB4_CONV1_RELU_EN           1
#define BB4_CONV1_MAX_POOL_EN       0
#define BB4_CONV1_AVG_POOL_EN       0
#define BB4_CONV1_FC_EN            0
#define BB4_CONV1_BASE_ADDR_IN      MEM0_BASE_ADDR
#define BB4_CONV1_BASE_ADDR_OUT     MEM1_BASE_ADDR
#define BB4_CONV1_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB4_CONV1_WEIGHT_BASE = BB3_SKIP_WEIGHT_BASE + BB3_SKIP_WEIGHT_SIZE;
constexpr unsigned BB4_CONV1_WEIGHT_SIZE = BB3_SKIP_C * BB4_CONV1_C * BB4_CONV1_K * BB4_CONV1_K / WEIGHT_PACK;
constexpr unsigned BB4_CONV1_BN_WEIGHT_BASE = BB3_SKIP_BN_WEIGHT_BASE + BB3_SKIP_BN_WEIGHT_SIZE;
constexpr unsigned BB4_CONV1_BN_WEIGHT_SIZE = 3 * BB4_CONV1_C;
constexpr unsigned BB4_CONV1_IN_SIZE = BB3_SKIP_C * BB3_SKIP_H * BB3_SKIP_W;
constexpr unsigned BB4_CONV1_OUT_SIZE = BB4_CONV1_C * BB4_CONV1_H * BB4_CONV1_W;

// BB4_CONV2 layer cnt 12
#define BB4_CONV2_C                 4
#define BB4_CONV2_H                 28
#define BB4_CONV2_W                 28
#define BB4_CONV2_K                 3
#define BB4_CONV2_S                 1
#define BB4_CONV2_PAD               1
#define BB4_CONV2_BB_EN             1
#define BB4_CONV2_CONV_EN           1
#define BB4_CONV2_BN_EN             1
#define BB4_CONV2_SKIP_EN           0
#define BB4_CONV2_RELU_EN           0
#define BB4_CONV2_MAX_POOL_EN       0
#define BB4_CONV2_AVG_POOL_EN       0
#define BB4_CONV2_FC_EN            0
#define BB4_CONV2_BASE_ADDR_IN      MEM1_BASE_ADDR
#define BB4_CONV2_BASE_ADDR_OUT     MEM2_BASE_ADDR
#define BB4_CONV2_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB4_CONV2_WEIGHT_BASE = BB4_CONV1_WEIGHT_BASE + BB4_CONV1_WEIGHT_SIZE;
constexpr unsigned BB4_CONV2_WEIGHT_SIZE = BB4_CONV1_C * BB4_CONV2_C * BB4_CONV2_K * BB4_CONV2_K / WEIGHT_PACK;
constexpr unsigned BB4_CONV2_BN_WEIGHT_BASE = BB4_CONV1_BN_WEIGHT_BASE + BB4_CONV1_BN_WEIGHT_SIZE;
constexpr unsigned BB4_CONV2_BN_WEIGHT_SIZE = 3 * BB4_CONV2_C;
constexpr unsigned BB4_CONV2_IN_SIZE = BB4_CONV1_C * BB4_CONV1_H * BB4_CONV1_W;
constexpr unsigned BB4_CONV2_OUT_SIZE = BB4_CONV2_C * BB4_CONV2_H * BB4_CONV2_W;

// BB4_SKIP layer cnt 13
#define BB4_SKIP_C                  4
#define BB4_SKIP_H                  28
#define BB4_SKIP_W                  28
#define BB4_SKIP_K                  0
#define BB4_SKIP_S                  0
#define BB4_SKIP_PAD                0
#define BB4_SKIP_BB_EN              1
#define BB4_SKIP_CONV_EN            0
#define BB4_SKIP_BN_EN              0
#define BB4_SKIP_SKIP_EN            1
#define BB4_SKIP_RELU_EN            1
#define BB4_SKIP_MAX_POOL_EN        0
#define BB4_SKIP_AVG_POOL_EN        0
#define BB4_SKIP_FC_EN             0
#define BB4_SKIP_BASE_ADDR_IN       MEM0_BASE_ADDR
#define BB4_SKIP_BASE_ADDR_OUT      MEM1_BASE_ADDR
#define BB4_SKIP_BASE_ADDR_ADD      MEM2_BASE_ADDR
constexpr unsigned BB4_SKIP_WEIGHT_BASE = 0;
constexpr unsigned BB4_SKIP_WEIGHT_SIZE = 0;
constexpr unsigned BB4_SKIP_BN_WEIGHT_BASE = 0;
constexpr unsigned BB4_SKIP_BN_WEIGHT_SIZE = 0;
constexpr unsigned BB4_SKIP_IN_SIZE = BB4_CONV2_C * BB4_CONV2_H * BB4_CONV2_W;
constexpr unsigned BB4_SKIP_OUT_SIZE = BB4_SKIP_C * BB4_SKIP_H * BB4_SKIP_W;

// BB5_CONV1 layer cnt 14
#define BB5_CONV1_C                 8
#define BB5_CONV1_H                 14
#define BB5_CONV1_W                 14
#define BB5_CONV1_K                 3
#define BB5_CONV1_S                 2
#define BB5_CONV1_PAD               1
#define BB5_CONV1_BB_EN             1
#define BB5_CONV1_CONV_EN           1
#define BB5_CONV1_BN_EN             1
#define BB5_CONV1_SKIP_EN           0
#define BB5_CONV1_RELU_EN           1
#define BB5_CONV1_MAX_POOL_EN       0
#define BB5_CONV1_AVG_POOL_EN       0
#define BB5_CONV1_FC_EN            0
#define BB5_CONV1_BASE_ADDR_IN      MEM1_BASE_ADDR
#define BB5_CONV1_BASE_ADDR_OUT     MEM2_BASE_ADDR
#define BB5_CONV1_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB5_CONV1_WEIGHT_BASE = BB4_CONV2_WEIGHT_BASE + BB4_CONV2_WEIGHT_SIZE;
constexpr unsigned BB5_CONV1_WEIGHT_SIZE = BB4_SKIP_C * BB5_CONV1_C * BB5_CONV1_K * BB5_CONV1_K / WEIGHT_PACK;
constexpr unsigned BB5_CONV1_BN_WEIGHT_BASE = BB4_CONV2_BN_WEIGHT_BASE + BB4_CONV2_BN_WEIGHT_SIZE;
constexpr unsigned BB5_CONV1_BN_WEIGHT_SIZE = 3 * BB5_CONV1_C;
constexpr unsigned BB5_CONV1_IN_SIZE = BB4_SKIP_C * BB4_SKIP_H * BB4_SKIP_W;
constexpr unsigned BB5_CONV1_OUT_SIZE = BB5_CONV1_C * BB5_CONV1_H * BB5_CONV1_W;

// BB5_CONV2 layer cnt 15
#define BB5_CONV2_C                 8
#define BB5_CONV2_H                 14
#define BB5_CONV2_W                 14
#define BB5_CONV2_K                 3
#define BB5_CONV2_S                 1
#define BB5_CONV2_PAD               1
#define BB5_CONV2_BB_EN             1
#define BB5_CONV2_CONV_EN           1
#define BB5_CONV2_BN_EN             1
#define BB5_CONV2_SKIP_EN           0
#define BB5_CONV2_RELU_EN           0
#define BB5_CONV2_MAX_POOL_EN       0
#define BB5_CONV2_AVG_POOL_EN       0
#define BB5_CONV2_FC_EN            0
#define BB5_CONV2_BASE_ADDR_IN      MEM2_BASE_ADDR
#define BB5_CONV2_BASE_ADDR_OUT     MEM0_BASE_ADDR
#define BB5_CONV2_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB5_CONV2_WEIGHT_BASE = BB5_CONV1_WEIGHT_BASE + BB5_CONV1_WEIGHT_SIZE;
constexpr unsigned BB5_CONV2_WEIGHT_SIZE = BB5_CONV1_C * BB5_CONV2_C * BB5_CONV2_K * BB5_CONV2_K / WEIGHT_PACK;
constexpr unsigned BB5_CONV2_BN_WEIGHT_BASE = BB5_CONV1_BN_WEIGHT_BASE + BB5_CONV1_BN_WEIGHT_SIZE;
constexpr unsigned BB5_CONV2_BN_WEIGHT_SIZE = 3 * BB5_CONV2_C;
constexpr unsigned BB5_CONV2_IN_SIZE = BB5_CONV1_C * BB5_CONV1_H * BB5_CONV1_W;
constexpr unsigned BB5_CONV2_OUT_SIZE = BB5_CONV2_C * BB5_CONV2_H * BB5_CONV2_W;

// BB5_SKIP layer (PROJ) cnt 16
#define BB5_SKIP_C                  4
#define BB5_SKIP_H                  14
#define BB5_SKIP_W                  14
#define BB5_SKIP_K                  1
#define BB5_SKIP_S                  2
#define BB5_SKIP_PAD                0
#define BB5_SKIP_BB_EN              1
#define BB5_SKIP_CONV_EN            1
#define BB5_SKIP_BN_EN              1
#define BB5_SKIP_SKIP_EN            1
#define BB5_SKIP_RELU_EN            1
#define BB5_SKIP_MAX_POOL_EN        0
#define BB5_SKIP_AVG_POOL_EN        0
#define BB5_SKIP_FC_EN             0
#define BB5_SKIP_BASE_ADDR_IN       MEM1_BASE_ADDR
#define BB5_SKIP_BASE_ADDR_OUT      MEM2_BASE_ADDR
#define BB5_SKIP_BASE_ADDR_ADD      MEM0_BASE_ADDR
constexpr unsigned BB5_SKIP_WEIGHT_BASE = BB5_CONV2_WEIGHT_BASE + BB5_CONV2_WEIGHT_SIZE;
constexpr unsigned BB5_SKIP_WEIGHT_SIZE = BB5_CONV2_C * BB5_SKIP_C * BB5_SKIP_K * BB5_SKIP_K / WEIGHT_PACK;
constexpr unsigned BB5_SKIP_BN_WEIGHT_BASE = BB5_CONV2_BN_WEIGHT_BASE + BB5_CONV2_BN_WEIGHT_SIZE;
constexpr unsigned BB5_SKIP_BN_WEIGHT_SIZE = 3 * BB5_SKIP_C;
constexpr unsigned BB5_SKIP_IN_SIZE = BB5_CONV2_C * BB5_CONV2_H * BB5_CONV2_W;
constexpr unsigned BB5_SKIP_OUT_SIZE = BB5_SKIP_C * BB5_SKIP_H * BB5_SKIP_W;

// BB6_CONV1 layer cnt 17
#define BB6_CONV1_C                 8
#define BB6_CONV1_H                 14
#define BB6_CONV1_W                 14
#define BB6_CONV1_K                 3
#define BB6_CONV1_S                 1
#define BB6_CONV1_PAD               1
#define BB6_CONV1_BB_EN             1
#define BB6_CONV1_CONV_EN           1
#define BB6_CONV1_BN_EN             1
#define BB6_CONV1_SKIP_EN           0
#define BB6_CONV1_RELU_EN           1
#define BB6_CONV1_MAX_POOL_EN       0
#define BB6_CONV1_AVG_POOL_EN       0
#define BB6_CONV1_FC_EN            0
#define BB6_CONV1_BASE_ADDR_IN      MEM2_BASE_ADDR
#define BB6_CONV1_BASE_ADDR_OUT     MEM0_BASE_ADDR
#define BB6_CONV1_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB6_CONV1_WEIGHT_BASE = BB5_SKIP_WEIGHT_BASE + BB5_SKIP_WEIGHT_SIZE;
constexpr unsigned BB6_CONV1_WEIGHT_SIZE = BB5_SKIP_C * BB6_CONV1_C * BB6_CONV1_K * BB6_CONV1_K / WEIGHT_PACK;
constexpr unsigned BB6_CONV1_BN_WEIGHT_BASE = BB5_SKIP_BN_WEIGHT_BASE + BB5_SKIP_BN_WEIGHT_SIZE;
constexpr unsigned BB6_CONV1_BN_WEIGHT_SIZE = 3 * BB6_CONV1_C;
constexpr unsigned BB6_CONV1_IN_SIZE = BB5_SKIP_C * BB5_SKIP_H * BB5_SKIP_W;
constexpr unsigned BB6_CONV1_OUT_SIZE = BB6_CONV1_C * BB6_CONV1_H * BB6_CONV1_W;

// BB6_CONV2 layer cnt 18
#define BB6_CONV2_C                 8
#define BB6_CONV2_H                 14
#define BB6_CONV2_W                 14
#define BB6_CONV2_K                 3
#define BB6_CONV2_S                 1
#define BB6_CONV2_PAD               1
#define BB6_CONV2_BB_EN             1
#define BB6_CONV2_CONV_EN           1
#define BB6_CONV2_BN_EN             1
#define BB6_CONV2_SKIP_EN           0
#define BB6_CONV2_RELU_EN           0
#define BB6_CONV2_MAX_POOL_EN       0
#define BB6_CONV2_AVG_POOL_EN       0
#define BB6_CONV2_FC_EN            0
#define BB6_CONV2_BASE_ADDR_IN      MEM0_BASE_ADDR
#define BB6_CONV2_BASE_ADDR_OUT     MEM1_BASE_ADDR
#define BB6_CONV2_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB6_CONV2_WEIGHT_BASE = BB6_CONV1_WEIGHT_BASE + BB6_CONV1_WEIGHT_SIZE;
constexpr unsigned BB6_CONV2_WEIGHT_SIZE = BB6_CONV1_C * BB6_CONV2_C * BB6_CONV2_K * BB6_CONV2_K / WEIGHT_PACK;
constexpr unsigned BB6_CONV2_BN_WEIGHT_BASE = BB6_CONV1_BN_WEIGHT_BASE + BB6_CONV1_BN_WEIGHT_SIZE;
constexpr unsigned BB6_CONV2_BN_WEIGHT_SIZE = 3 * BB6_CONV2_C;
constexpr unsigned BB6_CONV2_IN_SIZE = BB6_CONV1_C * BB6_CONV1_H * BB6_CONV1_W;
constexpr unsigned BB6_CONV2_OUT_SIZE = BB6_CONV2_C * BB6_CONV2_H * BB6_CONV2_W;

// BB6_SKIP layer cnt 19
#define BB6_SKIP_C                  8
#define BB6_SKIP_H                  14
#define BB6_SKIP_W                  14
#define BB6_SKIP_K                  0
#define BB6_SKIP_S                  0
#define BB6_SKIP_PAD                0
#define BB6_SKIP_BB_EN              1
#define BB6_SKIP_CONV_EN            0
#define BB6_SKIP_BN_EN              0
#define BB6_SKIP_SKIP_EN            1
#define BB6_SKIP_RELU_EN            1
#define BB6_SKIP_MAX_POOL_EN        0
#define BB6_SKIP_AVG_POOL_EN        0
#define BB6_SKIP_FC_EN             0
#define BB6_SKIP_BASE_ADDR_IN       MEM2_BASE_ADDR
#define BB6_SKIP_BASE_ADDR_OUT      MEM0_BASE_ADDR
#define BB6_SKIP_BASE_ADDR_ADD      MEM1_BASE_ADDR
constexpr unsigned BB6_SKIP_WEIGHT_BASE = 0;
constexpr unsigned BB6_SKIP_WEIGHT_SIZE = 0;
constexpr unsigned BB6_SKIP_BN_WEIGHT_BASE = 0;
constexpr unsigned BB6_SKIP_BN_WEIGHT_SIZE = 0;
constexpr unsigned BB6_SKIP_IN_SIZE = BB6_CONV2_C * BB6_CONV2_H * BB6_CONV2_W;
constexpr unsigned BB6_SKIP_OUT_SIZE = BB6_SKIP_C * BB6_SKIP_H * BB6_SKIP_W;

// BB7_CONV1 layer cnt 20
#define BB7_CONV1_C                 16
#define BB7_CONV1_H                 7
#define BB7_CONV1_W                 7
#define BB7_CONV1_K                 3
#define BB7_CONV1_S                 2
#define BB7_CONV1_PAD               1
#define BB7_CONV1_BB_EN             1
#define BB7_CONV1_CONV_EN           1
#define BB7_CONV1_BN_EN             1
#define BB7_CONV1_SKIP_EN           0
#define BB7_CONV1_RELU_EN           1
#define BB7_CONV1_MAX_POOL_EN       0
#define BB7_CONV1_AVG_POOL_EN       0
#define BB7_CONV1_FC_EN            0
#define BB7_CONV1_BASE_ADDR_IN      MEM0_BASE_ADDR
#define BB7_CONV1_BASE_ADDR_OUT     MEM1_BASE_ADDR
#define BB7_CONV1_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB7_CONV1_WEIGHT_BASE = BB6_CONV2_WEIGHT_BASE + BB6_CONV2_WEIGHT_SIZE;
constexpr unsigned BB7_CONV1_WEIGHT_SIZE = BB6_SKIP_C * BB7_CONV1_C * BB7_CONV1_K * BB7_CONV1_K / WEIGHT_PACK;
constexpr unsigned BB7_CONV1_BN_WEIGHT_BASE = BB6_CONV2_BN_WEIGHT_BASE + BB6_CONV2_BN_WEIGHT_SIZE;
constexpr unsigned BB7_CONV1_BN_WEIGHT_SIZE = 3 * BB7_CONV1_C;
constexpr unsigned BB7_CONV1_IN_SIZE = BB6_SKIP_C * BB6_SKIP_H * BB6_SKIP_W;
constexpr unsigned BB7_CONV1_OUT_SIZE = BB7_CONV1_C * BB7_CONV1_H * BB7_CONV1_W;

// BB7_CONV2 layer cnt 21
#define BB7_CONV2_C                 16
#define BB7_CONV2_H                 7
#define BB7_CONV2_W                 7
#define BB7_CONV2_K                 3
#define BB7_CONV2_S                 1
#define BB7_CONV2_PAD               1
#define BB7_CONV2_BB_EN             1
#define BB7_CONV2_CONV_EN           1
#define BB7_CONV2_BN_EN             1
#define BB7_CONV2_SKIP_EN           0
#define BB7_CONV2_RELU_EN           0
#define BB7_CONV2_MAX_POOL_EN       0
#define BB7_CONV2_AVG_POOL_EN       0
#define BB7_CONV2_FC_EN            0
#define BB7_CONV2_BASE_ADDR_IN      MEM1_BASE_ADDR
#define BB7_CONV2_BASE_ADDR_OUT     MEM2_BASE_ADDR
#define BB7_CONV2_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB7_CONV2_WEIGHT_BASE = BB7_CONV1_WEIGHT_BASE + BB7_CONV1_WEIGHT_SIZE;
constexpr unsigned BB7_CONV2_WEIGHT_SIZE = BB7_CONV1_C * BB7_CONV2_C * BB7_CONV2_K * BB7_CONV2_K / WEIGHT_PACK;
constexpr unsigned BB7_CONV2_BN_WEIGHT_BASE = BB7_CONV1_BN_WEIGHT_BASE + BB7_CONV1_BN_WEIGHT_SIZE;
constexpr unsigned BB7_CONV2_BN_WEIGHT_SIZE = 3 * BB7_CONV2_C;
constexpr unsigned BB7_CONV2_IN_SIZE = BB7_CONV1_C * BB7_CONV1_H * BB7_CONV1_W;
constexpr unsigned BB7_CONV2_OUT_SIZE = BB7_CONV2_C * BB7_CONV2_H * BB7_CONV2_W;

// BB7_SKIP layer (PROJ) cnt 22
#define BB7_SKIP_C                  8
#define BB7_SKIP_H                  7
#define BB7_SKIP_W                  7
#define BB7_SKIP_K                  1
#define BB7_SKIP_S                  2
#define BB7_SKIP_PAD                0
#define BB7_SKIP_BB_EN              1
#define BB7_SKIP_CONV_EN            1
#define BB7_SKIP_BN_EN              1
#define BB7_SKIP_SKIP_EN            1
#define BB7_SKIP_RELU_EN            1
#define BB7_SKIP_MAX_POOL_EN        0
#define BB7_SKIP_AVG_POOL_EN        0
#define BB7_SKIP_FC_EN             0
#define BB7_SKIP_BASE_ADDR_IN       MEM0_BASE_ADDR
#define BB7_SKIP_BASE_ADDR_OUT      MEM1_BASE_ADDR
#define BB7_SKIP_BASE_ADDR_ADD      MEM2_BASE_ADDR
constexpr unsigned BB7_SKIP_WEIGHT_BASE = BB7_CONV2_WEIGHT_BASE + BB7_CONV2_WEIGHT_SIZE;
constexpr unsigned BB7_SKIP_WEIGHT_SIZE = BB7_CONV2_C * BB7_SKIP_C * BB7_SKIP_K * BB7_SKIP_K / WEIGHT_PACK;
constexpr unsigned BB7_SKIP_BN_WEIGHT_BASE = BB7_CONV2_BN_WEIGHT_BASE + BB7_CONV2_BN_WEIGHT_SIZE;
constexpr unsigned BB7_SKIP_BN_WEIGHT_SIZE = 3 * BB7_SKIP_C;
constexpr unsigned BB7_SKIP_IN_SIZE = BB7_CONV2_C * BB7_CONV2_H * BB7_CONV2_W;
constexpr unsigned BB7_SKIP_OUT_SIZE = BB7_SKIP_C * BB7_SKIP_H * BB7_SKIP_W;

// BB8_CONV1 layer cnt 23
#define BB8_CONV1_C                 16
#define BB8_CONV1_H                 7
#define BB8_CONV1_W                 7
#define BB8_CONV1_K                 3
#define BB8_CONV1_S                 1
#define BB8_CONV1_PAD               1
#define BB8_CONV1_BB_EN             1
#define BB8_CONV1_CONV_EN           1
#define BB8_CONV1_BN_EN             1
#define BB8_CONV1_SKIP_EN           0
#define BB8_CONV1_RELU_EN           1
#define BB8_CONV1_MAX_POOL_EN       0
#define BB8_CONV1_AVG_POOL_EN       0
#define BB8_CONV1_FC_EN            0
#define BB8_CONV1_BASE_ADDR_IN      MEM1_BASE_ADDR
#define BB8_CONV1_BASE_ADDR_OUT     MEM2_BASE_ADDR
#define BB8_CONV1_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB8_CONV1_WEIGHT_BASE = BB7_SKIP_WEIGHT_BASE + BB7_SKIP_WEIGHT_SIZE;
constexpr unsigned BB8_CONV1_WEIGHT_SIZE = BB7_SKIP_C * BB8_CONV1_C * BB8_CONV1_K * BB8_CONV1_K / WEIGHT_PACK;
constexpr unsigned BB8_CONV1_BN_WEIGHT_BASE = BB7_SKIP_BN_WEIGHT_BASE + BB7_SKIP_BN_WEIGHT_SIZE;
constexpr unsigned BB8_CONV1_BN_WEIGHT_SIZE = 3 * BB8_CONV1_C;
constexpr unsigned BB8_CONV1_IN_SIZE = BB7_SKIP_C * BB7_SKIP_H * BB7_SKIP_W;
constexpr unsigned BB8_CONV1_OUT_SIZE = BB8_CONV1_C * BB8_CONV1_H * BB8_CONV1_W;

// BB8_CONV2 layer cnt 24
#define BB8_CONV2_C                 16
#define BB8_CONV2_H                 7
#define BB8_CONV2_W                 7
#define BB8_CONV2_K                 3
#define BB8_CONV2_S                 1
#define BB8_CONV2_PAD               1
#define BB8_CONV2_BB_EN             1
#define BB8_CONV2_CONV_EN           1
#define BB8_CONV2_BN_EN             1
#define BB8_CONV2_SKIP_EN           0
#define BB8_CONV2_RELU_EN           0
#define BB8_CONV2_MAX_POOL_EN       0
#define BB8_CONV2_AVG_POOL_EN       0
#define BB8_CONV2_FC_EN            0
#define BB8_CONV2_BASE_ADDR_IN      MEM2_BASE_ADDR
#define BB8_CONV2_BASE_ADDR_OUT     MEM0_BASE_ADDR
#define BB8_CONV2_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB8_CONV2_WEIGHT_BASE = BB8_CONV1_WEIGHT_BASE + BB8_CONV1_WEIGHT_SIZE;
constexpr unsigned BB8_CONV2_WEIGHT_SIZE = BB8_CONV1_C * BB8_CONV2_C * BB8_CONV2_K * BB8_CONV2_K / WEIGHT_PACK;
constexpr unsigned BB8_CONV2_BN_WEIGHT_BASE = BB8_CONV1_BN_WEIGHT_BASE + BB8_CONV1_BN_WEIGHT_SIZE;
constexpr unsigned BB8_CONV2_BN_WEIGHT_SIZE = 3 * BB8_CONV2_C;
constexpr unsigned BB8_CONV2_IN_SIZE = BB8_CONV1_C * BB8_CONV1_H * BB8_CONV1_W;
constexpr unsigned BB8_CONV2_OUT_SIZE = BB8_CONV2_C * BB8_CONV2_H * BB8_CONV2_W;

// BB8_SKIP layer cnt 25
#define BB8_SKIP_C                  16
#define BB8_SKIP_H                  7
#define BB8_SKIP_W                  7
#define BB8_SKIP_K                  0
#define BB8_SKIP_S                  0
#define BB8_SKIP_PAD                0
#define BB8_SKIP_BB_EN              1
#define BB8_SKIP_CONV_EN            0
#define BB8_SKIP_BN_EN              0
#define BB8_SKIP_SKIP_EN            1
#define BB8_SKIP_RELU_EN            1
#define BB8_SKIP_MAX_POOL_EN        0
#define BB8_SKIP_AVG_POOL_EN        0
#define BB8_SKIP_FC_EN             0
#define BB8_SKIP_BASE_ADDR_IN       MEM1_BASE_ADDR
#define BB8_SKIP_BASE_ADDR_OUT      MEM2_BASE_ADDR
#define BB8_SKIP_BASE_ADDR_ADD      MEM0_BASE_ADDR
constexpr unsigned BB8_SKIP_WEIGHT_BASE = 0;
constexpr unsigned BB8_SKIP_WEIGHT_SIZE = 0;
constexpr unsigned BB8_SKIP_BN_WEIGHT_BASE = 0;
constexpr unsigned BB8_SKIP_BN_WEIGHT_SIZE = 0;
constexpr unsigned BB8_SKIP_IN_SIZE = BB8_CONV2_C * BB8_CONV2_H * BB8_CONV2_W;
constexpr unsigned BB8_SKIP_OUT_SIZE = BB8_SKIP_C * BB8_SKIP_H * BB8_SKIP_W;

// AVG_POOL layer cnt 26
#define AVG_POOL_C                   16
#define AVG_POOL_H                   1
#define AVG_POOL_W                   1
#define AVG_POOL_K                   0
#define AVG_POOL_S                   0
#define AVG_POOL_PAD                 0
#define AVG_POOL_BB_EN               0
#define AVG_POOL_CONV_EN             0
#define AVG_POOL_BN_EN               0
#define AVG_POOL_SKIP_EN             0
#define AVG_POOL_RELU_EN             0
#define AVG_POOL_MAX_POOL_EN         0
#define AVG_POOL_AVG_POOL_EN         1
#define AVG_POOL_FC_EN              0
#define AVG_POOL_BASE_ADDR_IN        MEM2_BASE_ADDR
#define AVG_POOL_BASE_ADDR_OUT       MEM0_BASE_ADDR
#define AVG_POOL_BASE_ADDR_ADD       NOMEM_BASE_ADDR
constexpr unsigned AVG_POOL_WEIGHT_BASE = 0;
constexpr unsigned AVG_POOL_WEIGHT_SIZE = 0;
constexpr unsigned AVG_POOL_BN_WEIGHT_BASE = 0;
constexpr unsigned AVG_POOL_BN_WEIGHT_SIZE = 0;
constexpr unsigned AVG_POOL_IN_SIZE = BB8_SKIP_C * BB8_SKIP_H * BB8_SKIP_W;
constexpr unsigned AVG_POOL_OUT_SIZE = AVG_POOL_C * AVG_POOL_H * AVG_POOL_W;

// FC layer cnt 27
#define FC_C                        10
#define FC_H                        1
#define FC_W                        1
#define FC_K                        0
#define FC_S                        0
#define FC_PAD                      0
#define FC_BB_EN                    0
#define FC_CONV_EN                  0
#define FC_BN_EN                    0
#define FC_SKIP_EN                  0
#define FC_RELU_EN                  0
#define FC_MAX_POOL_EN              0
#define FC_AVG_POOL_EN              0
#define FC_FC_EN                   1
#define FC_BASE_ADDR_IN             MEM0_BASE_ADDR
#define FC_BASE_ADDR_OUT            MEM1_BASE_ADDR
#define FC_BASE_ADDR_ADD            NOMEM_BASE_ADDR
constexpr unsigned FC_WEIGHT_BASE = 0;
constexpr unsigned FC_WEIGHT_SIZE = 0;
constexpr unsigned FC_BN_WEIGHT_BASE = BB8_CONV2_BN_WEIGHT_BASE + BB8_CONV2_BN_WEIGHT_SIZE;
constexpr unsigned FC_BN_WEIGHT_SIZE = AVG_POOL_C * FC_C + FC_C;
constexpr unsigned FC_IN_SIZE = AVG_POOL_C * AVG_POOL_H * AVG_POOL_W;
constexpr unsigned FC_OUT_SIZE = FC_C * FC_H * FC_W;
#else
// CONV1 layer cnt 0
#define CONV1_C                     64
#define CONV1_H                     112
#define CONV1_W                     112
#define CONV1_K                     7
#define CONV1_S                     2
#define CONV1_PAD                   3
#define CONV1_BB_EN                 1
#define CONV1_CONV_EN               1
#define CONV1_BN_EN                 1
#define CONV1_SKIP_EN               0
#define CONV1_RELU_EN               1
#define CONV1_MAX_POOL_EN           0
#define CONV1_AVG_POOL_EN           0
#define CONV1_FC_EN                0
#define CONV1_BASE_ADDR_IN          MEM0_BASE_ADDR
#define CONV1_BASE_ADDR_OUT         MEM1_BASE_ADDR
#define CONV1_BASE_ADDR_ADD         NOMEM_BASE_ADDR
constexpr unsigned CONV1_WEIGHT_BASE = 0;
constexpr unsigned CONV1_WEIGHT_SIZE = IN_C * CONV1_C * CONV1_K * CONV1_K / WEIGHT_PACK;
constexpr unsigned CONV1_BN_WEIGHT_BASE = 0;
constexpr unsigned CONV1_BN_WEIGHT_SIZE = 3 * CONV1_C;
constexpr unsigned CONV1_IN_SIZE = IN_C * IN_H * IN_W;
constexpr unsigned CONV1_OUT_SIZE = CONV1_C * CONV1_H * CONV1_W;

// MAX_POOL layer cnt 1
#define MAX_POOL_C                   64
#define MAX_POOL_H                   56
#define MAX_POOL_W                   56
#define MAX_POOL_K                   3
#define MAX_POOL_S                   2
#define MAX_POOL_PAD                 1
#define MAX_POOL_BB_EN               0
#define MAX_POOL_CONV_EN             0
#define MAX_POOL_BN_EN               0
#define MAX_POOL_SKIP_EN             0
#define MAX_POOL_RELU_EN             0
#define MAX_POOL_MAX_POOL_EN         1
#define MAX_POOL_AVG_POOL_EN         0
#define MAX_POOL_FC_EN              0
#define MAX_POOL_BASE_ADDR_IN        MEM1_BASE_ADDR
#define MAX_POOL_BASE_ADDR_OUT       MEM0_BASE_ADDR
#define MAX_POOL_BASE_ADDR_ADD       NOMEM_BASE_ADDR
constexpr unsigned MAX_POOL_WEIGHT_BASE = 0;
constexpr unsigned MAX_POOL_WEIGHT_SIZE = 0;
constexpr unsigned MAX_POOL_BN_WEIGHT_BASE = 0;
constexpr unsigned MAX_POOL_BN_WEIGHT_SIZE = 0;
constexpr unsigned MAX_POOL_IN_SIZE = CONV1_C * CONV1_H * CONV1_W;
constexpr unsigned MAX_POOL_OUT_SIZE = MAX_POOL_C * MAX_POOL_H * MAX_POOL_W;

// BB1_CONV1 layer cnt 2
#define BB1_CONV1_C                 64
#define BB1_CONV1_H                 56
#define BB1_CONV1_W                 56
#define BB1_CONV1_K                 3
#define BB1_CONV1_S                 1
#define BB1_CONV1_PAD               1
#define BB1_CONV1_BB_EN             1
#define BB1_CONV1_CONV_EN           1
#define BB1_CONV1_BN_EN             1
#define BB1_CONV1_SKIP_EN           0
#define BB1_CONV1_RELU_EN           1
#define BB1_CONV1_MAX_POOL_EN       0
#define BB1_CONV1_AVG_POOL_EN       0
#define BB1_CONV1_FC_EN            0
#define BB1_CONV1_BASE_ADDR_IN      MEM0_BASE_ADDR
#define BB1_CONV1_BASE_ADDR_OUT     MEM1_BASE_ADDR
#define BB1_CONV1_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB1_CONV1_WEIGHT_BASE = CONV1_WEIGHT_BASE + CONV1_WEIGHT_SIZE;
constexpr unsigned BB1_CONV1_WEIGHT_SIZE = MAX_POOL_C * BB1_CONV1_C * BB1_CONV1_K * BB1_CONV1_K / WEIGHT_PACK;
constexpr unsigned BB1_CONV1_BN_WEIGHT_BASE = CONV1_BN_WEIGHT_BASE + CONV1_BN_WEIGHT_SIZE;
constexpr unsigned BB1_CONV1_BN_WEIGHT_SIZE = 3 * BB1_CONV1_C;
constexpr unsigned BB1_CONV1_IN_SIZE = MAX_POOL_C * MAX_POOL_H * MAX_POOL_W;
constexpr unsigned BB1_CONV1_OUT_SIZE = BB1_CONV1_C * BB1_CONV1_H * BB1_CONV1_W;

// BB1_CONV2 layer cnt 3
#define BB1_CONV2_C                 64
#define BB1_CONV2_H                 56
#define BB1_CONV2_W                 56
#define BB1_CONV2_K                 3
#define BB1_CONV2_S                 1
#define BB1_CONV2_PAD               1
#define BB1_CONV2_BB_EN             1
#define BB1_CONV2_CONV_EN           1
#define BB1_CONV2_BN_EN             1
#define BB1_CONV2_SKIP_EN           0
#define BB1_CONV2_RELU_EN           0
#define BB1_CONV2_MAX_POOL_EN       0
#define BB1_CONV2_AVG_POOL_EN       0
#define BB1_CONV2_FC_EN            0
#define BB1_CONV2_BASE_ADDR_IN      MEM1_BASE_ADDR
#define BB1_CONV2_BASE_ADDR_OUT     MEM2_BASE_ADDR
#define BB1_CONV2_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB1_CONV2_WEIGHT_BASE = BB1_CONV1_WEIGHT_BASE + BB1_CONV1_WEIGHT_SIZE;
constexpr unsigned BB1_CONV2_WEIGHT_SIZE = BB1_CONV1_C * BB1_CONV2_C * BB1_CONV2_K * BB1_CONV2_K / WEIGHT_PACK;
constexpr unsigned BB1_CONV2_BN_WEIGHT_BASE = BB1_CONV1_BN_WEIGHT_BASE + BB1_CONV1_BN_WEIGHT_SIZE;
constexpr unsigned BB1_CONV2_BN_WEIGHT_SIZE = 3 * BB1_CONV2_C;
constexpr unsigned BB1_CONV2_IN_SIZE = BB1_CONV1_C * BB1_CONV1_H * BB1_CONV1_W;
constexpr unsigned BB1_CONV2_OUT_SIZE = BB1_CONV2_C * BB1_CONV2_H * BB1_CONV2_W;

// BB1_SKIP layer cnt 4
#define BB1_SKIP_C                  64
#define BB1_SKIP_H                  56
#define BB1_SKIP_W                  56
#define BB1_SKIP_K                  0
#define BB1_SKIP_S                  0
#define BB1_SKIP_PAD                0
#define BB1_SKIP_BB_EN              1
#define BB1_SKIP_CONV_EN            0
#define BB1_SKIP_BN_EN              0
#define BB1_SKIP_SKIP_EN            1
#define BB1_SKIP_RELU_EN            1
#define BB1_SKIP_MAX_POOL_EN        0
#define BB1_SKIP_AVG_POOL_EN        0
#define BB1_SKIP_FC_EN             0
#define BB1_SKIP_BASE_ADDR_IN       MEM0_BASE_ADDR
#define BB1_SKIP_BASE_ADDR_OUT      MEM1_BASE_ADDR
#define BB1_SKIP_BASE_ADDR_ADD      MEM2_BASE_ADDR
constexpr unsigned BB1_SKIP_WEIGHT_BASE = 0;
constexpr unsigned BB1_SKIP_WEIGHT_SIZE = 0;
constexpr unsigned BB1_SKIP_BN_WEIGHT_BASE = 0;
constexpr unsigned BB1_SKIP_BN_WEIGHT_SIZE = 0;
constexpr unsigned BB1_SKIP_IN_SIZE = BB1_CONV2_C * BB1_CONV2_H * BB1_CONV2_W;
constexpr unsigned BB1_SKIP_OUT_SIZE = BB1_SKIP_C * BB1_SKIP_H * BB1_SKIP_W;

// BB2_CONV1 layer cnt 5
#define BB2_CONV1_C                 64
#define BB2_CONV1_H                 56
#define BB2_CONV1_W                 56
#define BB2_CONV1_K                 3
#define BB2_CONV1_S                 1
#define BB2_CONV1_PAD               1
#define BB2_CONV1_BB_EN             1
#define BB2_CONV1_CONV_EN           1
#define BB2_CONV1_BN_EN             1
#define BB2_CONV1_SKIP_EN           0
#define BB2_CONV1_RELU_EN           1
#define BB2_CONV1_MAX_POOL_EN       0
#define BB2_CONV1_AVG_POOL_EN       0
#define BB2_CONV1_FC_EN            0
#define BB2_CONV1_BASE_ADDR_IN      MEM1_BASE_ADDR
#define BB2_CONV1_BASE_ADDR_OUT     MEM2_BASE_ADDR
#define BB2_CONV1_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB2_CONV1_WEIGHT_BASE = BB1_CONV2_WEIGHT_BASE + BB1_CONV2_WEIGHT_SIZE;
constexpr unsigned BB2_CONV1_WEIGHT_SIZE = BB1_SKIP_C * BB2_CONV1_C * BB2_CONV1_K * BB2_CONV1_K / WEIGHT_PACK;
constexpr unsigned BB2_CONV1_BN_WEIGHT_BASE = BB1_CONV2_BN_WEIGHT_BASE + BB1_CONV2_BN_WEIGHT_SIZE;
constexpr unsigned BB2_CONV1_BN_WEIGHT_SIZE = 3 * BB2_CONV1_C;
constexpr unsigned BB2_CONV1_IN_SIZE = BB1_SKIP_C * BB1_SKIP_H * BB1_SKIP_W;
constexpr unsigned BB2_CONV1_OUT_SIZE = BB2_CONV1_C * BB2_CONV1_H * BB2_CONV1_W;

// BB2_CONV2 layer cnt 6
#define BB2_CONV2_C                 64
#define BB2_CONV2_H                 56
#define BB2_CONV2_W                 56
#define BB2_CONV2_K                 3
#define BB2_CONV2_S                 1
#define BB2_CONV2_PAD               1
#define BB2_CONV2_BB_EN             1
#define BB2_CONV2_CONV_EN           1
#define BB2_CONV2_BN_EN             1
#define BB2_CONV2_SKIP_EN           0
#define BB2_CONV2_RELU_EN           0
#define BB2_CONV2_MAX_POOL_EN       0
#define BB2_CONV2_AVG_POOL_EN       0
#define BB2_CONV2_FC_EN            0
#define BB2_CONV2_BASE_ADDR_IN      MEM2_BASE_ADDR
#define BB2_CONV2_BASE_ADDR_OUT     MEM0_BASE_ADDR
#define BB2_CONV2_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB2_CONV2_WEIGHT_BASE = BB2_CONV1_WEIGHT_BASE + BB2_CONV1_WEIGHT_SIZE;
constexpr unsigned BB2_CONV2_WEIGHT_SIZE = BB2_CONV1_C * BB2_CONV2_C * BB2_CONV2_K * BB2_CONV2_K / WEIGHT_PACK;
constexpr unsigned BB2_CONV2_BN_WEIGHT_BASE = BB2_CONV1_BN_WEIGHT_BASE + BB2_CONV1_BN_WEIGHT_SIZE;
constexpr unsigned BB2_CONV2_BN_WEIGHT_SIZE = 3 * BB2_CONV2_C;
constexpr unsigned BB2_CONV2_IN_SIZE = BB2_CONV1_C * BB2_CONV1_H * BB2_CONV1_W;
constexpr unsigned BB2_CONV2_OUT_SIZE = BB2_CONV2_C * BB2_CONV2_H * BB2_CONV2_W;

// BB2_SKIP layer cnt 7
#define BB2_SKIP_C                  64
#define BB2_SKIP_H                  56
#define BB2_SKIP_W                  56
#define BB2_SKIP_K                  0
#define BB2_SKIP_S                  0
#define BB2_SKIP_PAD                0
#define BB2_SKIP_BB_EN              1
#define BB2_SKIP_CONV_EN            0
#define BB2_SKIP_BN_EN              0
#define BB2_SKIP_SKIP_EN            1
#define BB2_SKIP_RELU_EN            1
#define BB2_SKIP_MAX_POOL_EN        0
#define BB2_SKIP_AVG_POOL_EN        0
#define BB2_SKIP_FC_EN             0
#define BB2_SKIP_BASE_ADDR_IN       MEM1_BASE_ADDR
#define BB2_SKIP_BASE_ADDR_OUT      MEM2_BASE_ADDR
#define BB2_SKIP_BASE_ADDR_ADD      MEM0_BASE_ADDR
constexpr unsigned BB2_SKIP_WEIGHT_BASE = 0;
constexpr unsigned BB2_SKIP_WEIGHT_SIZE = 0;
constexpr unsigned BB2_SKIP_BN_WEIGHT_BASE = 0;
constexpr unsigned BB2_SKIP_BN_WEIGHT_SIZE = 0;
constexpr unsigned BB2_SKIP_IN_SIZE = BB2_CONV2_C * BB2_CONV2_H * BB2_CONV2_W;
constexpr unsigned BB2_SKIP_OUT_SIZE = BB2_SKIP_C * BB2_SKIP_H * BB2_SKIP_W;

// BB3_CONV1 layer cnt 8
#define BB3_CONV1_C                 128
#define BB3_CONV1_H                 28
#define BB3_CONV1_W                 28
#define BB3_CONV1_K                 3
#define BB3_CONV1_S                 2
#define BB3_CONV1_PAD               1
#define BB3_CONV1_BB_EN             1
#define BB3_CONV1_CONV_EN           1
#define BB3_CONV1_BN_EN             1
#define BB3_CONV1_SKIP_EN           0
#define BB3_CONV1_RELU_EN           1
#define BB3_CONV1_MAX_POOL_EN       0
#define BB3_CONV1_AVG_POOL_EN       0
#define BB3_CONV1_FC_EN            0
#define BB3_CONV1_BASE_ADDR_IN      MEM2_BASE_ADDR
#define BB3_CONV1_BASE_ADDR_OUT     MEM0_BASE_ADDR
#define BB3_CONV1_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB3_CONV1_WEIGHT_BASE = BB2_CONV2_WEIGHT_BASE + BB2_CONV2_WEIGHT_SIZE;
constexpr unsigned BB3_CONV1_WEIGHT_SIZE = BB2_SKIP_C * BB3_CONV1_C * BB3_CONV1_K * BB3_CONV1_K / WEIGHT_PACK;
constexpr unsigned BB3_CONV1_BN_WEIGHT_BASE = BB2_CONV2_BN_WEIGHT_BASE + BB2_CONV2_BN_WEIGHT_SIZE;
constexpr unsigned BB3_CONV1_BN_WEIGHT_SIZE = 3 * BB3_CONV1_C;
constexpr unsigned BB3_CONV1_IN_SIZE = BB2_SKIP_C * BB2_SKIP_H * BB2_SKIP_W;
constexpr unsigned BB3_CONV1_OUT_SIZE = BB3_CONV1_C * BB3_CONV1_H * BB3_CONV1_W;

// BB3_CONV2 layer cnt 9
#define BB3_CONV2_C                 128
#define BB3_CONV2_H                 28
#define BB3_CONV2_W                 28
#define BB3_CONV2_K                 3
#define BB3_CONV2_S                 1
#define BB3_CONV2_PAD               1
#define BB3_CONV2_BB_EN             1
#define BB3_CONV2_CONV_EN           1
#define BB3_CONV2_BN_EN             1
#define BB3_CONV2_SKIP_EN           0
#define BB3_CONV2_RELU_EN           0
#define BB3_CONV2_MAX_POOL_EN       0
#define BB3_CONV2_AVG_POOL_EN       0
#define BB3_CONV2_FC_EN            0
#define BB3_CONV2_BASE_ADDR_IN      MEM0_BASE_ADDR
#define BB3_CONV2_BASE_ADDR_OUT     MEM1_BASE_ADDR
#define BB3_CONV2_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB3_CONV2_WEIGHT_BASE = BB3_CONV1_WEIGHT_BASE + BB3_CONV1_WEIGHT_SIZE;
constexpr unsigned BB3_CONV2_WEIGHT_SIZE = BB3_CONV1_C * BB3_CONV2_C * BB3_CONV2_K * BB3_CONV2_K / WEIGHT_PACK;
constexpr unsigned BB3_CONV2_BN_WEIGHT_BASE = BB3_CONV1_BN_WEIGHT_BASE + BB3_CONV1_BN_WEIGHT_SIZE;
constexpr unsigned BB3_CONV2_BN_WEIGHT_SIZE = 3 * BB3_CONV2_C;
constexpr unsigned BB3_CONV2_IN_SIZE = BB3_CONV1_C * BB3_CONV1_H * BB3_CONV1_W;
constexpr unsigned BB3_CONV2_OUT_SIZE = BB3_CONV2_C * BB3_CONV2_H * BB3_CONV2_W;

// BB3_SKIP layer (PROJ) cnt 10
#define BB3_SKIP_C                  128
#define BB3_SKIP_H                  28
#define BB3_SKIP_W                  28
#define BB3_SKIP_K                  1
#define BB3_SKIP_S                  2
#define BB3_SKIP_PAD                0
#define BB3_SKIP_BB_EN              1
#define BB3_SKIP_CONV_EN            1
#define BB3_SKIP_BN_EN              1
#define BB3_SKIP_SKIP_EN            1
#define BB3_SKIP_RELU_EN            1
#define BB3_SKIP_MAX_POOL_EN        0
#define BB3_SKIP_AVG_POOL_EN        0
#define BB3_SKIP_FC_EN             0
#define BB3_SKIP_BASE_ADDR_IN       MEM2_BASE_ADDR
#define BB3_SKIP_BASE_ADDR_OUT      MEM0_BASE_ADDR
#define BB3_SKIP_BASE_ADDR_ADD      MEM1_BASE_ADDR
constexpr unsigned BB3_SKIP_WEIGHT_BASE = BB3_CONV2_WEIGHT_BASE + BB3_CONV2_WEIGHT_SIZE;
constexpr unsigned BB3_SKIP_WEIGHT_SIZE = BB3_CONV2_C * BB3_SKIP_C * BB3_SKIP_K * BB3_SKIP_K / WEIGHT_PACK;
constexpr unsigned BB3_SKIP_BN_WEIGHT_BASE = BB3_CONV2_BN_WEIGHT_BASE + BB3_CONV2_BN_WEIGHT_SIZE;
constexpr unsigned BB3_SKIP_BN_WEIGHT_SIZE = 3 * BB3_SKIP_C;
constexpr unsigned BB3_SKIP_IN_SIZE = BB3_CONV2_C * BB3_CONV2_H * BB3_CONV2_W;
constexpr unsigned BB3_SKIP_OUT_SIZE = BB3_SKIP_C * BB3_SKIP_H * BB3_SKIP_W;

// BB4_CONV1 layer cnt 11
#define BB4_CONV1_C                 128
#define BB4_CONV1_H                 28
#define BB4_CONV1_W                 28
#define BB4_CONV1_K                 3
#define BB4_CONV1_S                 1
#define BB4_CONV1_PAD               1
#define BB4_CONV1_BB_EN             1
#define BB4_CONV1_CONV_EN           1
#define BB4_CONV1_BN_EN             1
#define BB4_CONV1_SKIP_EN           0
#define BB4_CONV1_RELU_EN           1
#define BB4_CONV1_MAX_POOL_EN       0
#define BB4_CONV1_AVG_POOL_EN       0
#define BB4_CONV1_FC_EN            0
#define BB4_CONV1_BASE_ADDR_IN      MEM0_BASE_ADDR
#define BB4_CONV1_BASE_ADDR_OUT     MEM1_BASE_ADDR
#define BB4_CONV1_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB4_CONV1_WEIGHT_BASE = BB3_SKIP_WEIGHT_BASE + BB3_SKIP_WEIGHT_SIZE;
constexpr unsigned BB4_CONV1_WEIGHT_SIZE = BB3_SKIP_C * BB4_CONV1_C * BB4_CONV1_K * BB4_CONV1_K / WEIGHT_PACK;
constexpr unsigned BB4_CONV1_BN_WEIGHT_BASE = BB3_SKIP_BN_WEIGHT_BASE + BB3_SKIP_BN_WEIGHT_SIZE;
constexpr unsigned BB4_CONV1_BN_WEIGHT_SIZE = 3 * BB4_CONV1_C;
constexpr unsigned BB4_CONV1_IN_SIZE = BB3_SKIP_C * BB3_SKIP_H * BB3_SKIP_W;
constexpr unsigned BB4_CONV1_OUT_SIZE = BB4_CONV1_C * BB4_CONV1_H * BB4_CONV1_W;

// BB4_CONV2 layer cnt 12
#define BB4_CONV2_C                 128
#define BB4_CONV2_H                 28
#define BB4_CONV2_W                 28
#define BB4_CONV2_K                 3
#define BB4_CONV2_S                 1
#define BB4_CONV2_PAD               1
#define BB4_CONV2_BB_EN             1
#define BB4_CONV2_CONV_EN           1
#define BB4_CONV2_BN_EN             1
#define BB4_CONV2_SKIP_EN           0
#define BB4_CONV2_RELU_EN           0
#define BB4_CONV2_MAX_POOL_EN       0
#define BB4_CONV2_AVG_POOL_EN       0
#define BB4_CONV2_FC_EN            0
#define BB4_CONV2_BASE_ADDR_IN      MEM1_BASE_ADDR
#define BB4_CONV2_BASE_ADDR_OUT     MEM2_BASE_ADDR
#define BB4_CONV2_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB4_CONV2_WEIGHT_BASE = BB4_CONV1_WEIGHT_BASE + BB4_CONV1_WEIGHT_SIZE;
constexpr unsigned BB4_CONV2_WEIGHT_SIZE = BB4_CONV1_C * BB4_CONV2_C * BB4_CONV2_K * BB4_CONV2_K / WEIGHT_PACK;
constexpr unsigned BB4_CONV2_BN_WEIGHT_BASE = BB4_CONV1_BN_WEIGHT_BASE + BB4_CONV1_BN_WEIGHT_SIZE;
constexpr unsigned BB4_CONV2_BN_WEIGHT_SIZE = 3 * BB4_CONV2_C;
constexpr unsigned BB4_CONV2_IN_SIZE = BB4_CONV1_C * BB4_CONV1_H * BB4_CONV1_W;
constexpr unsigned BB4_CONV2_OUT_SIZE = BB4_CONV2_C * BB4_CONV2_H * BB4_CONV2_W;

// BB4_SKIP layer cnt 13
#define BB4_SKIP_C                  128
#define BB4_SKIP_H                  28
#define BB4_SKIP_W                  28
#define BB4_SKIP_K                  0
#define BB4_SKIP_S                  0
#define BB4_SKIP_PAD                0
#define BB4_SKIP_BB_EN              1
#define BB4_SKIP_CONV_EN            0
#define BB4_SKIP_BN_EN              0
#define BB4_SKIP_SKIP_EN            1
#define BB4_SKIP_RELU_EN            1
#define BB4_SKIP_MAX_POOL_EN        0
#define BB4_SKIP_AVG_POOL_EN        0
#define BB4_SKIP_FC_EN             0
#define BB4_SKIP_BASE_ADDR_IN       MEM0_BASE_ADDR
#define BB4_SKIP_BASE_ADDR_OUT      MEM1_BASE_ADDR
#define BB4_SKIP_BASE_ADDR_ADD      MEM2_BASE_ADDR
constexpr unsigned BB4_SKIP_WEIGHT_BASE = 0;
constexpr unsigned BB4_SKIP_WEIGHT_SIZE = 0;
constexpr unsigned BB4_SKIP_BN_WEIGHT_BASE = 0;
constexpr unsigned BB4_SKIP_BN_WEIGHT_SIZE = 0;
constexpr unsigned BB4_SKIP_IN_SIZE = BB4_CONV2_C * BB4_CONV2_H * BB4_CONV2_W;
constexpr unsigned BB4_SKIP_OUT_SIZE = BB4_SKIP_C * BB4_SKIP_H * BB4_SKIP_W;

// BB5_CONV1 layer cnt 14
#define BB5_CONV1_C                 256
#define BB5_CONV1_H                 14
#define BB5_CONV1_W                 14
#define BB5_CONV1_K                 3
#define BB5_CONV1_S                 2
#define BB5_CONV1_PAD               1
#define BB5_CONV1_BB_EN             1
#define BB5_CONV1_CONV_EN           1
#define BB5_CONV1_BN_EN             1
#define BB5_CONV1_SKIP_EN           0
#define BB5_CONV1_RELU_EN           1
#define BB5_CONV1_MAX_POOL_EN       0
#define BB5_CONV1_AVG_POOL_EN       0
#define BB5_CONV1_FC_EN            0
#define BB5_CONV1_BASE_ADDR_IN      MEM1_BASE_ADDR
#define BB5_CONV1_BASE_ADDR_OUT     MEM2_BASE_ADDR
#define BB5_CONV1_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB5_CONV1_WEIGHT_BASE = BB4_CONV2_WEIGHT_BASE + BB4_CONV2_WEIGHT_SIZE;
constexpr unsigned BB5_CONV1_WEIGHT_SIZE = BB4_SKIP_C * BB5_CONV1_C * BB5_CONV1_K * BB5_CONV1_K / WEIGHT_PACK;
constexpr unsigned BB5_CONV1_BN_WEIGHT_BASE = BB4_CONV2_BN_WEIGHT_BASE + BB4_CONV2_BN_WEIGHT_SIZE;
constexpr unsigned BB5_CONV1_BN_WEIGHT_SIZE = 3 * BB5_CONV1_C;
constexpr unsigned BB5_CONV1_IN_SIZE = BB4_SKIP_C * BB4_SKIP_H * BB4_SKIP_W;
constexpr unsigned BB5_CONV1_OUT_SIZE = BB5_CONV1_C * BB5_CONV1_H * BB5_CONV1_W;

// BB5_CONV2 layer cnt 15
#define BB5_CONV2_C                 256
#define BB5_CONV2_H                 14
#define BB5_CONV2_W                 14
#define BB5_CONV2_K                 3
#define BB5_CONV2_S                 1
#define BB5_CONV2_PAD               1
#define BB5_CONV2_BB_EN             1
#define BB5_CONV2_CONV_EN           1
#define BB5_CONV2_BN_EN             1
#define BB5_CONV2_SKIP_EN           0
#define BB5_CONV2_RELU_EN           0
#define BB5_CONV2_MAX_POOL_EN       0
#define BB5_CONV2_AVG_POOL_EN       0
#define BB5_CONV2_FC_EN            0
#define BB5_CONV2_BASE_ADDR_IN      MEM2_BASE_ADDR
#define BB5_CONV2_BASE_ADDR_OUT     MEM0_BASE_ADDR
#define BB5_CONV2_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB5_CONV2_WEIGHT_BASE = BB5_CONV1_WEIGHT_BASE + BB5_CONV1_WEIGHT_SIZE;
constexpr unsigned BB5_CONV2_WEIGHT_SIZE = BB5_CONV1_C * BB5_CONV2_C * BB5_CONV2_K * BB5_CONV2_K / WEIGHT_PACK;
constexpr unsigned BB5_CONV2_BN_WEIGHT_BASE = BB5_CONV1_BN_WEIGHT_BASE + BB5_CONV1_BN_WEIGHT_SIZE;
constexpr unsigned BB5_CONV2_BN_WEIGHT_SIZE = 3 * BB5_CONV2_C;
constexpr unsigned BB5_CONV2_IN_SIZE = BB5_CONV1_C * BB5_CONV1_H * BB5_CONV1_W;
constexpr unsigned BB5_CONV2_OUT_SIZE = BB5_CONV2_C * BB5_CONV2_H * BB5_CONV2_W;

// BB5_SKIP layer (PROJ) cnt 16
#define BB5_SKIP_C                  256
#define BB5_SKIP_H                  14
#define BB5_SKIP_W                  14
#define BB5_SKIP_K                  1
#define BB5_SKIP_S                  2
#define BB5_SKIP_PAD                0
#define BB5_SKIP_BB_EN              1
#define BB5_SKIP_CONV_EN            1
#define BB5_SKIP_BN_EN              1
#define BB5_SKIP_SKIP_EN            1
#define BB5_SKIP_RELU_EN            1
#define BB5_SKIP_MAX_POOL_EN        0
#define BB5_SKIP_AVG_POOL_EN        0
#define BB5_SKIP_FC_EN             0
#define BB5_SKIP_BASE_ADDR_IN       MEM1_BASE_ADDR
#define BB5_SKIP_BASE_ADDR_OUT      MEM2_BASE_ADDR
#define BB5_SKIP_BASE_ADDR_ADD      MEM0_BASE_ADDR
constexpr unsigned BB5_SKIP_WEIGHT_BASE = BB5_CONV2_WEIGHT_BASE + BB5_CONV2_WEIGHT_SIZE;
constexpr unsigned BB5_SKIP_WEIGHT_SIZE = BB5_CONV2_C * BB5_SKIP_C * BB5_SKIP_K * BB5_SKIP_K / WEIGHT_PACK;
constexpr unsigned BB5_SKIP_BN_WEIGHT_BASE = BB5_CONV2_BN_WEIGHT_BASE + BB5_CONV2_BN_WEIGHT_SIZE;
constexpr unsigned BB5_SKIP_BN_WEIGHT_SIZE = 3 * BB5_SKIP_C;
constexpr unsigned BB5_SKIP_IN_SIZE = BB5_CONV2_C * BB5_CONV2_H * BB5_CONV2_W;
constexpr unsigned BB5_SKIP_OUT_SIZE = BB5_SKIP_C * BB5_SKIP_H * BB5_SKIP_W;

// BB6_CONV1 layer cnt 17
#define BB6_CONV1_C                 256
#define BB6_CONV1_H                 14
#define BB6_CONV1_W                 14
#define BB6_CONV1_K                 3
#define BB6_CONV1_S                 1
#define BB6_CONV1_PAD               1
#define BB6_CONV1_BB_EN             1
#define BB6_CONV1_CONV_EN           1
#define BB6_CONV1_BN_EN             1
#define BB6_CONV1_SKIP_EN           0
#define BB6_CONV1_RELU_EN           1
#define BB6_CONV1_MAX_POOL_EN       0
#define BB6_CONV1_AVG_POOL_EN       0
#define BB6_CONV1_FC_EN            0
#define BB6_CONV1_BASE_ADDR_IN      MEM2_BASE_ADDR
#define BB6_CONV1_BASE_ADDR_OUT     MEM0_BASE_ADDR
#define BB6_CONV1_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB6_CONV1_WEIGHT_BASE = BB5_SKIP_WEIGHT_BASE + BB5_SKIP_WEIGHT_SIZE;
constexpr unsigned BB6_CONV1_WEIGHT_SIZE = BB5_SKIP_C * BB6_CONV1_C * BB6_CONV1_K * BB6_CONV1_K / WEIGHT_PACK;
constexpr unsigned BB6_CONV1_BN_WEIGHT_BASE = BB5_SKIP_BN_WEIGHT_BASE + BB5_SKIP_BN_WEIGHT_SIZE;
constexpr unsigned BB6_CONV1_BN_WEIGHT_SIZE = 3 * BB6_CONV1_C;
constexpr unsigned BB6_CONV1_IN_SIZE = BB5_SKIP_C * BB5_SKIP_H * BB5_SKIP_W;
constexpr unsigned BB6_CONV1_OUT_SIZE = BB6_CONV1_C * BB6_CONV1_H * BB6_CONV1_W;

// BB6_CONV2 layer cnt 18
#define BB6_CONV2_C                 256
#define BB6_CONV2_H                 14
#define BB6_CONV2_W                 14
#define BB6_CONV2_K                 3
#define BB6_CONV2_S                 1
#define BB6_CONV2_PAD               1
#define BB6_CONV2_BB_EN             1
#define BB6_CONV2_CONV_EN           1
#define BB6_CONV2_BN_EN             1
#define BB6_CONV2_SKIP_EN           0
#define BB6_CONV2_RELU_EN           0
#define BB6_CONV2_MAX_POOL_EN       0
#define BB6_CONV2_AVG_POOL_EN       0
#define BB6_CONV2_FC_EN            0
#define BB6_CONV2_BASE_ADDR_IN      MEM0_BASE_ADDR
#define BB6_CONV2_BASE_ADDR_OUT     MEM1_BASE_ADDR
#define BB6_CONV2_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB6_CONV2_WEIGHT_BASE = BB6_CONV1_WEIGHT_BASE + BB6_CONV1_WEIGHT_SIZE;
constexpr unsigned BB6_CONV2_WEIGHT_SIZE = BB6_CONV1_C * BB6_CONV2_C * BB6_CONV2_K * BB6_CONV2_K / WEIGHT_PACK;
constexpr unsigned BB6_CONV2_BN_WEIGHT_BASE = BB6_CONV1_BN_WEIGHT_BASE + BB6_CONV1_BN_WEIGHT_SIZE;
constexpr unsigned BB6_CONV2_BN_WEIGHT_SIZE = 3 * BB6_CONV2_C;
constexpr unsigned BB6_CONV2_IN_SIZE = BB6_CONV1_C * BB6_CONV1_H * BB6_CONV1_W;
constexpr unsigned BB6_CONV2_OUT_SIZE = BB6_CONV2_C * BB6_CONV2_H * BB6_CONV2_W;

// BB6_SKIP layer cnt 19
#define BB6_SKIP_C                  256
#define BB6_SKIP_H                  14
#define BB6_SKIP_W                  14
#define BB6_SKIP_K                  0
#define BB6_SKIP_S                  0
#define BB6_SKIP_PAD                0
#define BB6_SKIP_BB_EN              1
#define BB6_SKIP_CONV_EN            0
#define BB6_SKIP_BN_EN              0
#define BB6_SKIP_SKIP_EN            1
#define BB6_SKIP_RELU_EN            1
#define BB6_SKIP_MAX_POOL_EN        0
#define BB6_SKIP_AVG_POOL_EN        0
#define BB6_SKIP_FC_EN             0
#define BB6_SKIP_BASE_ADDR_IN       MEM2_BASE_ADDR
#define BB6_SKIP_BASE_ADDR_OUT      MEM0_BASE_ADDR
#define BB6_SKIP_BASE_ADDR_ADD      MEM1_BASE_ADDR
constexpr unsigned BB6_SKIP_WEIGHT_BASE = 0;
constexpr unsigned BB6_SKIP_WEIGHT_SIZE = 0;
constexpr unsigned BB6_SKIP_BN_WEIGHT_BASE = 0;
constexpr unsigned BB6_SKIP_BN_WEIGHT_SIZE = 0;
constexpr unsigned BB6_SKIP_IN_SIZE = BB6_CONV2_C * BB6_CONV2_H * BB6_CONV2_W;
constexpr unsigned BB6_SKIP_OUT_SIZE = BB6_SKIP_C * BB6_SKIP_H * BB6_SKIP_W;

// BB7_CONV1 layer cnt 20
#define BB7_CONV1_C                 512
#define BB7_CONV1_H                 7
#define BB7_CONV1_W                 7
#define BB7_CONV1_K                 3
#define BB7_CONV1_S                 2
#define BB7_CONV1_PAD               1
#define BB7_CONV1_BB_EN             1
#define BB7_CONV1_CONV_EN           1
#define BB7_CONV1_BN_EN             1
#define BB7_CONV1_SKIP_EN           0
#define BB7_CONV1_RELU_EN           1
#define BB7_CONV1_MAX_POOL_EN       0
#define BB7_CONV1_AVG_POOL_EN       0
#define BB7_CONV1_FC_EN            0
#define BB7_CONV1_BASE_ADDR_IN      MEM0_BASE_ADDR
#define BB7_CONV1_BASE_ADDR_OUT     MEM1_BASE_ADDR
#define BB7_CONV1_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB7_CONV1_WEIGHT_BASE = BB6_CONV2_WEIGHT_BASE + BB6_CONV2_WEIGHT_SIZE;
constexpr unsigned BB7_CONV1_WEIGHT_SIZE = BB6_SKIP_C * BB7_CONV1_C * BB7_CONV1_K * BB7_CONV1_K / WEIGHT_PACK;
constexpr unsigned BB7_CONV1_BN_WEIGHT_BASE = BB6_CONV2_BN_WEIGHT_BASE + BB6_CONV2_BN_WEIGHT_SIZE;
constexpr unsigned BB7_CONV1_BN_WEIGHT_SIZE = 3 * BB7_CONV1_C;
constexpr unsigned BB7_CONV1_IN_SIZE = BB6_SKIP_C * BB6_SKIP_H * BB6_SKIP_W;
constexpr unsigned BB7_CONV1_OUT_SIZE = BB7_CONV1_C * BB7_CONV1_H * BB7_CONV1_W;

// BB7_CONV2 layer cnt 21
#define BB7_CONV2_C                 512
#define BB7_CONV2_H                 7
#define BB7_CONV2_W                 7
#define BB7_CONV2_K                 3
#define BB7_CONV2_S                 1
#define BB7_CONV2_PAD               1
#define BB7_CONV2_BB_EN             1
#define BB7_CONV2_CONV_EN           1
#define BB7_CONV2_BN_EN             1
#define BB7_CONV2_SKIP_EN           0
#define BB7_CONV2_RELU_EN           0
#define BB7_CONV2_MAX_POOL_EN       0
#define BB7_CONV2_AVG_POOL_EN       0
#define BB7_CONV2_FC_EN            0
#define BB7_CONV2_BASE_ADDR_IN      MEM1_BASE_ADDR
#define BB7_CONV2_BASE_ADDR_OUT     MEM2_BASE_ADDR
#define BB7_CONV2_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB7_CONV2_WEIGHT_BASE = BB7_CONV1_WEIGHT_BASE + BB7_CONV1_WEIGHT_SIZE;
constexpr unsigned BB7_CONV2_WEIGHT_SIZE = BB7_CONV1_C * BB7_CONV2_C * BB7_CONV2_K * BB7_CONV2_K / WEIGHT_PACK;
constexpr unsigned BB7_CONV2_BN_WEIGHT_BASE = BB7_CONV1_BN_WEIGHT_BASE + BB7_CONV1_BN_WEIGHT_SIZE;
constexpr unsigned BB7_CONV2_BN_WEIGHT_SIZE = 3 * BB7_CONV2_C;
constexpr unsigned BB7_CONV2_IN_SIZE = BB7_CONV1_C * BB7_CONV1_H * BB7_CONV1_W;
constexpr unsigned BB7_CONV2_OUT_SIZE = BB7_CONV2_C * BB7_CONV2_H * BB7_CONV2_W;

// BB7_SKIP layer (PROJ) cnt 22
#define BB7_SKIP_C                  512
#define BB7_SKIP_H                  7
#define BB7_SKIP_W                  7
#define BB7_SKIP_K                  1
#define BB7_SKIP_S                  2
#define BB7_SKIP_PAD                0
#define BB7_SKIP_BB_EN              1
#define BB7_SKIP_CONV_EN            1
#define BB7_SKIP_BN_EN              1
#define BB7_SKIP_SKIP_EN            1
#define BB7_SKIP_RELU_EN            1
#define BB7_SKIP_MAX_POOL_EN        0
#define BB7_SKIP_AVG_POOL_EN        0
#define BB7_SKIP_FC_EN             0
#define BB7_SKIP_BASE_ADDR_IN       MEM0_BASE_ADDR
#define BB7_SKIP_BASE_ADDR_OUT      MEM1_BASE_ADDR
#define BB7_SKIP_BASE_ADDR_ADD      MEM2_BASE_ADDR
constexpr unsigned BB7_SKIP_WEIGHT_BASE = BB7_CONV2_WEIGHT_BASE + BB7_CONV2_WEIGHT_SIZE;
constexpr unsigned BB7_SKIP_WEIGHT_SIZE = BB7_CONV2_C * BB7_SKIP_C * BB7_SKIP_K * BB7_SKIP_K / WEIGHT_PACK;
constexpr unsigned BB7_SKIP_BN_WEIGHT_BASE = BB7_CONV2_BN_WEIGHT_BASE + BB7_CONV2_BN_WEIGHT_SIZE;
constexpr unsigned BB7_SKIP_BN_WEIGHT_SIZE = 3 * BB7_SKIP_C;
constexpr unsigned BB7_SKIP_IN_SIZE = BB7_CONV2_C * BB7_CONV2_H * BB7_CONV2_W;
constexpr unsigned BB7_SKIP_OUT_SIZE = BB7_SKIP_C * BB7_SKIP_H * BB7_SKIP_W;

// BB8_CONV1 layer cnt 23
#define BB8_CONV1_C                 512
#define BB8_CONV1_H                 7
#define BB8_CONV1_W                 7
#define BB8_CONV1_K                 3
#define BB8_CONV1_S                 1
#define BB8_CONV1_PAD               1
#define BB8_CONV1_BB_EN             1
#define BB8_CONV1_CONV_EN           1
#define BB8_CONV1_BN_EN             1
#define BB8_CONV1_SKIP_EN           0
#define BB8_CONV1_RELU_EN           1
#define BB8_CONV1_MAX_POOL_EN       0
#define BB8_CONV1_AVG_POOL_EN       0
#define BB8_CONV1_FC_EN            0
#define BB8_CONV1_BASE_ADDR_IN      MEM1_BASE_ADDR
#define BB8_CONV1_BASE_ADDR_OUT     MEM2_BASE_ADDR
#define BB8_CONV1_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB8_CONV1_WEIGHT_BASE = BB7_SKIP_WEIGHT_BASE + BB7_SKIP_WEIGHT_SIZE;
constexpr unsigned BB8_CONV1_WEIGHT_SIZE = BB7_SKIP_C * BB8_CONV1_C * BB8_CONV1_K * BB8_CONV1_K / WEIGHT_PACK;
constexpr unsigned BB8_CONV1_BN_WEIGHT_BASE = BB7_SKIP_BN_WEIGHT_BASE + BB7_SKIP_BN_WEIGHT_SIZE;
constexpr unsigned BB8_CONV1_BN_WEIGHT_SIZE = 3 * BB8_CONV1_C;
constexpr unsigned BB8_CONV1_IN_SIZE = BB7_SKIP_C * BB7_SKIP_H * BB7_SKIP_W;
constexpr unsigned BB8_CONV1_OUT_SIZE = BB8_CONV1_C * BB8_CONV1_H * BB8_CONV1_W;

// BB8_CONV2 layer cnt 24
#define BB8_CONV2_C                 512
#define BB8_CONV2_H                 7
#define BB8_CONV2_W                 7
#define BB8_CONV2_K                 3
#define BB8_CONV2_S                 1
#define BB8_CONV2_PAD               1
#define BB8_CONV2_BB_EN             1
#define BB8_CONV2_CONV_EN           1
#define BB8_CONV2_BN_EN             1
#define BB8_CONV2_SKIP_EN           0
#define BB8_CONV2_RELU_EN           0
#define BB8_CONV2_MAX_POOL_EN       0
#define BB8_CONV2_AVG_POOL_EN       0
#define BB8_CONV2_FC_EN            0
#define BB8_CONV2_BASE_ADDR_IN      MEM2_BASE_ADDR
#define BB8_CONV2_BASE_ADDR_OUT     MEM0_BASE_ADDR
#define BB8_CONV2_BASE_ADDR_ADD     NOMEM_BASE_ADDR
constexpr unsigned BB8_CONV2_WEIGHT_BASE = BB8_CONV1_WEIGHT_BASE + BB8_CONV1_WEIGHT_SIZE;
constexpr unsigned BB8_CONV2_WEIGHT_SIZE = BB8_CONV1_C * BB8_CONV2_C * BB8_CONV2_K * BB8_CONV2_K / WEIGHT_PACK;
constexpr unsigned BB8_CONV2_BN_WEIGHT_BASE = BB8_CONV1_BN_WEIGHT_BASE + BB8_CONV1_BN_WEIGHT_SIZE;
constexpr unsigned BB8_CONV2_BN_WEIGHT_SIZE = 3 * BB8_CONV2_C;
constexpr unsigned BB8_CONV2_IN_SIZE = BB8_CONV1_C * BB8_CONV1_H * BB8_CONV1_W;
constexpr unsigned BB8_CONV2_OUT_SIZE = BB8_CONV2_C * BB8_CONV2_H * BB8_CONV2_W;

// BB8_SKIP layer cnt 25
#define BB8_SKIP_C                  512
#define BB8_SKIP_H                  7
#define BB8_SKIP_W                  7
#define BB8_SKIP_K                  0
#define BB8_SKIP_S                  0
#define BB8_SKIP_PAD                0
#define BB8_SKIP_BB_EN              1
#define BB8_SKIP_CONV_EN            0
#define BB8_SKIP_BN_EN              0
#define BB8_SKIP_SKIP_EN            1
#define BB8_SKIP_RELU_EN            1
#define BB8_SKIP_MAX_POOL_EN        0
#define BB8_SKIP_AVG_POOL_EN        0
#define BB8_SKIP_FC_EN             0
#define BB8_SKIP_BASE_ADDR_IN       MEM1_BASE_ADDR
#define BB8_SKIP_BASE_ADDR_OUT      MEM2_BASE_ADDR
#define BB8_SKIP_BASE_ADDR_ADD      MEM0_BASE_ADDR
constexpr unsigned BB8_SKIP_WEIGHT_BASE = 0;
constexpr unsigned BB8_SKIP_WEIGHT_SIZE = 0;
constexpr unsigned BB8_SKIP_BN_WEIGHT_BASE = 0;
constexpr unsigned BB8_SKIP_BN_WEIGHT_SIZE = 0;
constexpr unsigned BB8_SKIP_IN_SIZE = BB8_CONV2_C * BB8_CONV2_H * BB8_CONV2_W;
constexpr unsigned BB8_SKIP_OUT_SIZE = BB8_SKIP_C * BB8_SKIP_H * BB8_SKIP_W;

// AVG_POOL layer cnt 26
#define AVG_POOL_C                   512
#define AVG_POOL_H                   1
#define AVG_POOL_W                   1
#define AVG_POOL_K                   0
#define AVG_POOL_S                   0
#define AVG_POOL_PAD                 0
#define AVG_POOL_BB_EN               0
#define AVG_POOL_CONV_EN             0
#define AVG_POOL_BN_EN               0
#define AVG_POOL_SKIP_EN             0
#define AVG_POOL_RELU_EN             0
#define AVG_POOL_MAX_POOL_EN         0
#define AVG_POOL_AVG_POOL_EN         1
#define AVG_POOL_FC_EN              0
#define AVG_POOL_BASE_ADDR_IN        MEM2_BASE_ADDR
#define AVG_POOL_BASE_ADDR_OUT       MEM0_BASE_ADDR
#define AVG_POOL_BASE_ADDR_ADD       NOMEM_BASE_ADDR
constexpr unsigned AVG_POOL_WEIGHT_BASE = 0;
constexpr unsigned AVG_POOL_WEIGHT_SIZE = 0;
constexpr unsigned AVG_POOL_BN_WEIGHT_BASE = 0;
constexpr unsigned AVG_POOL_BN_WEIGHT_SIZE = 0;
constexpr unsigned AVG_POOL_IN_SIZE = BB8_SKIP_C * BB8_SKIP_H * BB8_SKIP_W;
constexpr unsigned AVG_POOL_OUT_SIZE = AVG_POOL_C * AVG_POOL_H * AVG_POOL_W;

// FC layer cnt 27
#define FC_C                        100
#define FC_H                        1
#define FC_W                        1
#define FC_K                        0
#define FC_S                        0
#define FC_PAD                      0
#define FC_BB_EN                    0
#define FC_CONV_EN                  0
#define FC_BN_EN                    0
#define FC_SKIP_EN                  0
#define FC_RELU_EN                  0
#define FC_MAX_POOL_EN              0
#define FC_AVG_POOL_EN              0
#define FC_FC_EN                   1
#define FC_BASE_ADDR_IN             MEM0_BASE_ADDR
#define FC_BASE_ADDR_OUT            MEM1_BASE_ADDR
#define FC_BASE_ADDR_ADD            NOMEM_BASE_ADDR
constexpr unsigned FC_WEIGHT_BASE = 0;
constexpr unsigned FC_WEIGHT_SIZE = 0;
constexpr unsigned FC_BN_WEIGHT_BASE = BB8_CONV2_BN_WEIGHT_BASE + BB8_CONV2_BN_WEIGHT_SIZE;
constexpr unsigned FC_BN_WEIGHT_SIZE = AVG_POOL_C * FC_C + FC_C;
constexpr unsigned FC_IN_SIZE = AVG_POOL_C * AVG_POOL_H * AVG_POOL_W;
constexpr unsigned FC_OUT_SIZE = FC_C * FC_H * FC_W;

#endif

// size of mem blocks
#define MEM0_BASE_ADDR          0
#define MEM1_BASE_ADDR          MEM0_SIZE
#define MEM2_BASE_ADDR          MEM0_SIZE + MEM1_SIZE
#define NOMEM_BASE_ADDR         0

// #if SIM_MODE
// constexpr unsigned MEM0_SIZE = 100000;
// constexpr unsigned MEM1_SIZE = 100000;
// constexpr unsigned MEM2_SIZE = 100000;
// constexpr unsigned WEIGHT_MEM_SIZE = 100000;  // todo: temporary for now
// // constexpr unsigned WEIGHT_MEM_SIZE = BB7_SKIP_WEIGHT_BASE;  // todo: temporary for now
// 
// constexpr unsigned BN_WEIGHT_MEM_SIZE = 100000;
// constexpr unsigned BN_WEIGHT_MEM_SIZE = BB7_SKIP_BN_WEIGHT_BASE;
// #else
// todo:
// constexpr unsigned MEM0_SIZE = CONV1_C  * CONV1_H * CONV1_W / ACT_PACK;
// constexpr unsigned MEM1_SIZE = MAX_POOL_C  * MAX_POOL_H * MAX_POOL_W / ACT_PACK;
// constexpr unsigned MEM2_SIZE = BB1_CONV2_C  * BB1_CONV2_H * BB1_CONV2_W / ACT_PACK;
constexpr unsigned MEM0_SIZE = MAX_POOL_OUT_SIZE / ACT_PACK;
constexpr unsigned MEM1_SIZE = CONV1_OUT_SIZE / ACT_PACK;
constexpr unsigned MEM2_SIZE = MAX_POOL_OUT_SIZE / ACT_PACK;

constexpr unsigned WEIGHT_MEM_SIZE = BB8_CONV2_WEIGHT_BASE+BB8_CONV2_WEIGHT_SIZE;  // todo: temporary for now
// constexpr unsigned WEIGHT_MEM_SIZE = BB7_SKIP_WEIGHT_BASE;  // todo: temporary for now
constexpr unsigned BN_WEIGHT_MEM_SIZE = FC_BN_WEIGHT_BASE+FC_BN_WEIGHT_SIZE;
// constexpr unsigned BN_WEIGHT_MEM_SIZE = BB7_SKIP_BN_WEIGHT_BASE;
// #endif

constexpr unsigned MAX_ACT_MEM_SIZE = std::max({MEM0_SIZE, MEM1_SIZE, MEM2_SIZE});
constexpr unsigned ACT_MEM_SIZE = MEM0_SIZE + MEM1_SIZE + MEM2_SIZE;
constexpr unsigned MAX_WEIGHT_MEM_SIZE = std::max({
        CONV1_WEIGHT_SIZE, MAX_POOL_WEIGHT_SIZE, 
        BB1_CONV1_WEIGHT_SIZE, BB1_CONV2_WEIGHT_SIZE, BB1_SKIP_WEIGHT_SIZE,
        BB2_CONV1_WEIGHT_SIZE, BB2_CONV2_WEIGHT_SIZE, BB2_SKIP_WEIGHT_SIZE,
        BB3_CONV1_WEIGHT_SIZE, BB3_CONV2_WEIGHT_SIZE, BB3_SKIP_WEIGHT_SIZE,
        BB4_CONV1_WEIGHT_SIZE, BB4_CONV2_WEIGHT_SIZE, BB4_SKIP_WEIGHT_SIZE,
        BB5_CONV1_WEIGHT_SIZE, BB5_CONV2_WEIGHT_SIZE, BB5_SKIP_WEIGHT_SIZE,
        BB6_CONV1_WEIGHT_SIZE, BB6_CONV2_WEIGHT_SIZE, BB6_SKIP_WEIGHT_SIZE,
        BB7_CONV1_WEIGHT_SIZE, BB7_CONV2_WEIGHT_SIZE, BB7_SKIP_WEIGHT_SIZE,
        BB8_CONV1_WEIGHT_SIZE, BB8_CONV2_WEIGHT_SIZE, BB8_SKIP_WEIGHT_SIZE
        });
constexpr unsigned MAX_BN_WEIGHT_MEM_SIZE = std::max({
        CONV1_BN_WEIGHT_SIZE, MAX_POOL_BN_WEIGHT_SIZE, 
        BB1_CONV1_BN_WEIGHT_SIZE, BB1_CONV2_BN_WEIGHT_SIZE, BB1_SKIP_BN_WEIGHT_SIZE,
        BB2_CONV1_BN_WEIGHT_SIZE, BB2_CONV2_BN_WEIGHT_SIZE, BB2_SKIP_BN_WEIGHT_SIZE,
        BB3_CONV1_BN_WEIGHT_SIZE, BB3_CONV2_BN_WEIGHT_SIZE, BB3_SKIP_BN_WEIGHT_SIZE,
        BB4_CONV1_BN_WEIGHT_SIZE, BB4_CONV2_BN_WEIGHT_SIZE, BB4_SKIP_BN_WEIGHT_SIZE,
        BB5_CONV1_BN_WEIGHT_SIZE, BB5_CONV2_BN_WEIGHT_SIZE, BB5_SKIP_BN_WEIGHT_SIZE,
        BB6_CONV1_BN_WEIGHT_SIZE, BB6_CONV2_BN_WEIGHT_SIZE, BB6_SKIP_BN_WEIGHT_SIZE,
        BB7_CONV1_BN_WEIGHT_SIZE, BB7_CONV2_BN_WEIGHT_SIZE, BB7_SKIP_BN_WEIGHT_SIZE,
        BB8_CONV1_BN_WEIGHT_SIZE, BB8_CONV2_BN_WEIGHT_SIZE, BB8_SKIP_BN_WEIGHT_SIZE
    });

#endif
