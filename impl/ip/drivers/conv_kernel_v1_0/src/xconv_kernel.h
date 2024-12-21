// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Tool Version Limit: 2022.04
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef XCONV_KERNEL_H
#define XCONV_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xconv_kernel_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
    u16 DeviceId;
    u64 Control_BaseAddress;
} XConv_kernel_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XConv_kernel;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XConv_kernel_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XConv_kernel_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XConv_kernel_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XConv_kernel_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
int XConv_kernel_Initialize(XConv_kernel *InstancePtr, u16 DeviceId);
XConv_kernel_Config* XConv_kernel_LookupConfig(u16 DeviceId);
int XConv_kernel_CfgInitialize(XConv_kernel *InstancePtr, XConv_kernel_Config *ConfigPtr);
#else
int XConv_kernel_Initialize(XConv_kernel *InstancePtr, const char* InstanceName);
int XConv_kernel_Release(XConv_kernel *InstancePtr);
#endif

void XConv_kernel_Start(XConv_kernel *InstancePtr);
u32 XConv_kernel_IsDone(XConv_kernel *InstancePtr);
u32 XConv_kernel_IsIdle(XConv_kernel *InstancePtr);
u32 XConv_kernel_IsReady(XConv_kernel *InstancePtr);
void XConv_kernel_EnableAutoRestart(XConv_kernel *InstancePtr);
void XConv_kernel_DisableAutoRestart(XConv_kernel *InstancePtr);

void XConv_kernel_Set_act_mem(XConv_kernel *InstancePtr, u64 Data);
u64 XConv_kernel_Get_act_mem(XConv_kernel *InstancePtr);
void XConv_kernel_Set_act_in(XConv_kernel *InstancePtr, u64 Data);
u64 XConv_kernel_Get_act_in(XConv_kernel *InstancePtr);
void XConv_kernel_Set_act_out(XConv_kernel *InstancePtr, u64 Data);
u64 XConv_kernel_Get_act_out(XConv_kernel *InstancePtr);
void XConv_kernel_Set_weight_mem(XConv_kernel *InstancePtr, u64 Data);
u64 XConv_kernel_Get_weight_mem(XConv_kernel *InstancePtr);
void XConv_kernel_Set_bn_weight_mem(XConv_kernel *InstancePtr, u64 Data);
u64 XConv_kernel_Get_bn_weight_mem(XConv_kernel *InstancePtr);
void XConv_kernel_Set_start_layer(XConv_kernel *InstancePtr, u64 Data);
u64 XConv_kernel_Get_start_layer(XConv_kernel *InstancePtr);
void XConv_kernel_Set_end_layer(XConv_kernel *InstancePtr, u64 Data);
u64 XConv_kernel_Get_end_layer(XConv_kernel *InstancePtr);

void XConv_kernel_InterruptGlobalEnable(XConv_kernel *InstancePtr);
void XConv_kernel_InterruptGlobalDisable(XConv_kernel *InstancePtr);
void XConv_kernel_InterruptEnable(XConv_kernel *InstancePtr, u32 Mask);
void XConv_kernel_InterruptDisable(XConv_kernel *InstancePtr, u32 Mask);
void XConv_kernel_InterruptClear(XConv_kernel *InstancePtr, u32 Mask);
u32 XConv_kernel_InterruptGetEnabled(XConv_kernel *InstancePtr);
u32 XConv_kernel_InterruptGetStatus(XConv_kernel *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
