// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Tool Version Limit: 2022.04
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
/***************************** Include Files *********************************/
#include "xconv_kernel.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XConv_kernel_CfgInitialize(XConv_kernel *InstancePtr, XConv_kernel_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XConv_kernel_Start(XConv_kernel *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_AP_CTRL) & 0x80;
    XConv_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XConv_kernel_IsDone(XConv_kernel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XConv_kernel_IsIdle(XConv_kernel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XConv_kernel_IsReady(XConv_kernel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XConv_kernel_EnableAutoRestart(XConv_kernel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XConv_kernel_DisableAutoRestart(XConv_kernel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_AP_CTRL, 0);
}

void XConv_kernel_Set_act_mem(XConv_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_ACT_MEM_DATA, (u32)(Data));
    XConv_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_ACT_MEM_DATA + 4, (u32)(Data >> 32));
}

u64 XConv_kernel_Get_act_mem(XConv_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_ACT_MEM_DATA);
    Data += (u64)XConv_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_ACT_MEM_DATA + 4) << 32;
    return Data;
}

void XConv_kernel_Set_act_in(XConv_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_ACT_IN_DATA, (u32)(Data));
    XConv_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_ACT_IN_DATA + 4, (u32)(Data >> 32));
}

u64 XConv_kernel_Get_act_in(XConv_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_ACT_IN_DATA);
    Data += (u64)XConv_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_ACT_IN_DATA + 4) << 32;
    return Data;
}

void XConv_kernel_Set_act_out(XConv_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_ACT_OUT_DATA, (u32)(Data));
    XConv_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_ACT_OUT_DATA + 4, (u32)(Data >> 32));
}

u64 XConv_kernel_Get_act_out(XConv_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_ACT_OUT_DATA);
    Data += (u64)XConv_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_ACT_OUT_DATA + 4) << 32;
    return Data;
}

void XConv_kernel_Set_weight_mem(XConv_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_WEIGHT_MEM_DATA, (u32)(Data));
    XConv_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_WEIGHT_MEM_DATA + 4, (u32)(Data >> 32));
}

u64 XConv_kernel_Get_weight_mem(XConv_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_WEIGHT_MEM_DATA);
    Data += (u64)XConv_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_WEIGHT_MEM_DATA + 4) << 32;
    return Data;
}

void XConv_kernel_Set_bn_weight_mem(XConv_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_BN_WEIGHT_MEM_DATA, (u32)(Data));
    XConv_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_BN_WEIGHT_MEM_DATA + 4, (u32)(Data >> 32));
}

u64 XConv_kernel_Get_bn_weight_mem(XConv_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_BN_WEIGHT_MEM_DATA);
    Data += (u64)XConv_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_BN_WEIGHT_MEM_DATA + 4) << 32;
    return Data;
}

void XConv_kernel_Set_start_layer(XConv_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_START_LAYER_DATA, (u32)(Data));
    XConv_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_START_LAYER_DATA + 4, (u32)(Data >> 32));
}

u64 XConv_kernel_Get_start_layer(XConv_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_START_LAYER_DATA);
    Data += (u64)XConv_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_START_LAYER_DATA + 4) << 32;
    return Data;
}

void XConv_kernel_Set_end_layer(XConv_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_END_LAYER_DATA, (u32)(Data));
    XConv_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_END_LAYER_DATA + 4, (u32)(Data >> 32));
}

u64 XConv_kernel_Get_end_layer(XConv_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_END_LAYER_DATA);
    Data += (u64)XConv_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_END_LAYER_DATA + 4) << 32;
    return Data;
}

void XConv_kernel_InterruptGlobalEnable(XConv_kernel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_GIE, 1);
}

void XConv_kernel_InterruptGlobalDisable(XConv_kernel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_GIE, 0);
}

void XConv_kernel_InterruptEnable(XConv_kernel *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XConv_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_IER);
    XConv_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_IER, Register | Mask);
}

void XConv_kernel_InterruptDisable(XConv_kernel *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XConv_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_IER);
    XConv_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_IER, Register & (~Mask));
}

void XConv_kernel_InterruptClear(XConv_kernel *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    //XConv_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_ISR, Mask);
}

u32 XConv_kernel_InterruptGetEnabled(XConv_kernel *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XConv_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_IER);
}

u32 XConv_kernel_InterruptGetStatus(XConv_kernel *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    // Current Interrupt Clear Behavior is Clear on Read(COR).
    return XConv_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV_KERNEL_CONTROL_ADDR_ISR);
}

