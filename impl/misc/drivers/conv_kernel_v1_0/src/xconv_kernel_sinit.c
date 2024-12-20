// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Tool Version Limit: 2022.04
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#include "xparameters.h"
#include "xconv_kernel.h"

extern XConv_kernel_Config XConv_kernel_ConfigTable[];

XConv_kernel_Config *XConv_kernel_LookupConfig(u16 DeviceId) {
	XConv_kernel_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XCONV_KERNEL_NUM_INSTANCES; Index++) {
		if (XConv_kernel_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XConv_kernel_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XConv_kernel_Initialize(XConv_kernel *InstancePtr, u16 DeviceId) {
	XConv_kernel_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XConv_kernel_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XConv_kernel_CfgInitialize(InstancePtr, ConfigPtr);
}

#endif

