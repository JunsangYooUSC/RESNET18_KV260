// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Tool Version Limit: 2022.04
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
// control
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read/COR)
//        bit 7  - auto_restart (Read/Write)
//        bit 9  - interrupt (Read)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0 - enable ap_done interrupt (Read/Write)
//        bit 1 - enable ap_ready interrupt (Read/Write)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/COR)
//        bit 0 - ap_done (Read/COR)
//        bit 1 - ap_ready (Read/COR)
//        others - reserved
// 0x10 : Data signal of act_mem
//        bit 31~0 - act_mem[31:0] (Read/Write)
// 0x14 : Data signal of act_mem
//        bit 31~0 - act_mem[63:32] (Read/Write)
// 0x18 : reserved
// 0x1c : Data signal of act_in
//        bit 31~0 - act_in[31:0] (Read/Write)
// 0x20 : Data signal of act_in
//        bit 31~0 - act_in[63:32] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of act_out
//        bit 31~0 - act_out[31:0] (Read/Write)
// 0x2c : Data signal of act_out
//        bit 31~0 - act_out[63:32] (Read/Write)
// 0x30 : reserved
// 0x34 : Data signal of weight_mem
//        bit 31~0 - weight_mem[31:0] (Read/Write)
// 0x38 : Data signal of weight_mem
//        bit 31~0 - weight_mem[63:32] (Read/Write)
// 0x3c : reserved
// 0x40 : Data signal of bn_weight_mem
//        bit 31~0 - bn_weight_mem[31:0] (Read/Write)
// 0x44 : Data signal of bn_weight_mem
//        bit 31~0 - bn_weight_mem[63:32] (Read/Write)
// 0x48 : reserved
// 0x4c : Data signal of start_layer
//        bit 31~0 - start_layer[31:0] (Read/Write)
// 0x50 : Data signal of start_layer
//        bit 31~0 - start_layer[63:32] (Read/Write)
// 0x54 : reserved
// 0x58 : Data signal of end_layer
//        bit 31~0 - end_layer[31:0] (Read/Write)
// 0x5c : Data signal of end_layer
//        bit 31~0 - end_layer[63:32] (Read/Write)
// 0x60 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XCONV_KERNEL_CONTROL_ADDR_AP_CTRL            0x00
#define XCONV_KERNEL_CONTROL_ADDR_GIE                0x04
#define XCONV_KERNEL_CONTROL_ADDR_IER                0x08
#define XCONV_KERNEL_CONTROL_ADDR_ISR                0x0c
#define XCONV_KERNEL_CONTROL_ADDR_ACT_MEM_DATA       0x10
#define XCONV_KERNEL_CONTROL_BITS_ACT_MEM_DATA       64
#define XCONV_KERNEL_CONTROL_ADDR_ACT_IN_DATA        0x1c
#define XCONV_KERNEL_CONTROL_BITS_ACT_IN_DATA        64
#define XCONV_KERNEL_CONTROL_ADDR_ACT_OUT_DATA       0x28
#define XCONV_KERNEL_CONTROL_BITS_ACT_OUT_DATA       64
#define XCONV_KERNEL_CONTROL_ADDR_WEIGHT_MEM_DATA    0x34
#define XCONV_KERNEL_CONTROL_BITS_WEIGHT_MEM_DATA    64
#define XCONV_KERNEL_CONTROL_ADDR_BN_WEIGHT_MEM_DATA 0x40
#define XCONV_KERNEL_CONTROL_BITS_BN_WEIGHT_MEM_DATA 64
#define XCONV_KERNEL_CONTROL_ADDR_START_LAYER_DATA   0x4c
#define XCONV_KERNEL_CONTROL_BITS_START_LAYER_DATA   64
#define XCONV_KERNEL_CONTROL_ADDR_END_LAYER_DATA     0x58
#define XCONV_KERNEL_CONTROL_BITS_END_LAYER_DATA     64

