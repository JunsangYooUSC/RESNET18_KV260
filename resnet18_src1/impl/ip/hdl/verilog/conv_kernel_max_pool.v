// ==============================================================
// RTL generated by Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Version: 2022.1
// Copyright (C) Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module conv_kernel_max_pool (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        m_axi_gmem0_AWVALID,
        m_axi_gmem0_AWREADY,
        m_axi_gmem0_AWADDR,
        m_axi_gmem0_AWID,
        m_axi_gmem0_AWLEN,
        m_axi_gmem0_AWSIZE,
        m_axi_gmem0_AWBURST,
        m_axi_gmem0_AWLOCK,
        m_axi_gmem0_AWCACHE,
        m_axi_gmem0_AWPROT,
        m_axi_gmem0_AWQOS,
        m_axi_gmem0_AWREGION,
        m_axi_gmem0_AWUSER,
        m_axi_gmem0_WVALID,
        m_axi_gmem0_WREADY,
        m_axi_gmem0_WDATA,
        m_axi_gmem0_WSTRB,
        m_axi_gmem0_WLAST,
        m_axi_gmem0_WID,
        m_axi_gmem0_WUSER,
        m_axi_gmem0_ARVALID,
        m_axi_gmem0_ARREADY,
        m_axi_gmem0_ARADDR,
        m_axi_gmem0_ARID,
        m_axi_gmem0_ARLEN,
        m_axi_gmem0_ARSIZE,
        m_axi_gmem0_ARBURST,
        m_axi_gmem0_ARLOCK,
        m_axi_gmem0_ARCACHE,
        m_axi_gmem0_ARPROT,
        m_axi_gmem0_ARQOS,
        m_axi_gmem0_ARREGION,
        m_axi_gmem0_ARUSER,
        m_axi_gmem0_RVALID,
        m_axi_gmem0_RREADY,
        m_axi_gmem0_RDATA,
        m_axi_gmem0_RLAST,
        m_axi_gmem0_RID,
        m_axi_gmem0_RFIFONUM,
        m_axi_gmem0_RUSER,
        m_axi_gmem0_RRESP,
        m_axi_gmem0_BVALID,
        m_axi_gmem0_BREADY,
        m_axi_gmem0_BRESP,
        m_axi_gmem0_BID,
        m_axi_gmem0_BUSER,
        act_mem,
        in_base_addr,
        out_base_addr,
        nif,
        noy,
        nox,
        stride,
        pad,
        max_pool_en,
        grp_fu_2957_p_din0,
        grp_fu_2957_p_din1,
        grp_fu_2957_p_opcode,
        grp_fu_2957_p_dout0,
        grp_fu_2957_p_ce
);

parameter    ap_ST_fsm_state1 = 4'd1;
parameter    ap_ST_fsm_state2 = 4'd2;
parameter    ap_ST_fsm_state3 = 4'd4;
parameter    ap_ST_fsm_state4 = 4'd8;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
output   m_axi_gmem0_AWVALID;
input   m_axi_gmem0_AWREADY;
output  [63:0] m_axi_gmem0_AWADDR;
output  [0:0] m_axi_gmem0_AWID;
output  [31:0] m_axi_gmem0_AWLEN;
output  [2:0] m_axi_gmem0_AWSIZE;
output  [1:0] m_axi_gmem0_AWBURST;
output  [1:0] m_axi_gmem0_AWLOCK;
output  [3:0] m_axi_gmem0_AWCACHE;
output  [2:0] m_axi_gmem0_AWPROT;
output  [3:0] m_axi_gmem0_AWQOS;
output  [3:0] m_axi_gmem0_AWREGION;
output  [0:0] m_axi_gmem0_AWUSER;
output   m_axi_gmem0_WVALID;
input   m_axi_gmem0_WREADY;
output  [31:0] m_axi_gmem0_WDATA;
output  [3:0] m_axi_gmem0_WSTRB;
output   m_axi_gmem0_WLAST;
output  [0:0] m_axi_gmem0_WID;
output  [0:0] m_axi_gmem0_WUSER;
output   m_axi_gmem0_ARVALID;
input   m_axi_gmem0_ARREADY;
output  [63:0] m_axi_gmem0_ARADDR;
output  [0:0] m_axi_gmem0_ARID;
output  [31:0] m_axi_gmem0_ARLEN;
output  [2:0] m_axi_gmem0_ARSIZE;
output  [1:0] m_axi_gmem0_ARBURST;
output  [1:0] m_axi_gmem0_ARLOCK;
output  [3:0] m_axi_gmem0_ARCACHE;
output  [2:0] m_axi_gmem0_ARPROT;
output  [3:0] m_axi_gmem0_ARQOS;
output  [3:0] m_axi_gmem0_ARREGION;
output  [0:0] m_axi_gmem0_ARUSER;
input   m_axi_gmem0_RVALID;
output   m_axi_gmem0_RREADY;
input  [31:0] m_axi_gmem0_RDATA;
input   m_axi_gmem0_RLAST;
input  [0:0] m_axi_gmem0_RID;
input  [8:0] m_axi_gmem0_RFIFONUM;
input  [0:0] m_axi_gmem0_RUSER;
input  [1:0] m_axi_gmem0_RRESP;
input   m_axi_gmem0_BVALID;
output   m_axi_gmem0_BREADY;
input  [1:0] m_axi_gmem0_BRESP;
input  [0:0] m_axi_gmem0_BID;
input  [0:0] m_axi_gmem0_BUSER;
input  [63:0] act_mem;
input  [31:0] in_base_addr;
input  [31:0] out_base_addr;
input  [31:0] nif;
input  [31:0] noy;
input  [31:0] nox;
input  [31:0] stride;
input  [31:0] pad;
input  [0:0] max_pool_en;
output  [31:0] grp_fu_2957_p_din0;
output  [31:0] grp_fu_2957_p_din1;
output  [4:0] grp_fu_2957_p_opcode;
input  [0:0] grp_fu_2957_p_dout0;
output   grp_fu_2957_p_ce;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg m_axi_gmem0_AWVALID;
reg m_axi_gmem0_WVALID;
reg m_axi_gmem0_ARVALID;
reg m_axi_gmem0_RREADY;
reg m_axi_gmem0_BREADY;

(* fsm_encoding = "none" *) reg   [3:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
wire   [0:0] max_pool_en_read_read_fu_58_p2;
wire   [63:0] mul_ln410_fu_142_p2;
reg   [63:0] mul_ln410_reg_319;
wire   [31:0] mul35_fu_148_p2;
reg   [31:0] mul35_reg_325;
wire    ap_CS_fsm_state2;
wire   [95:0] mul_ln410_1_fu_158_p2;
reg   [95:0] mul_ln410_1_reg_331;
wire   [31:0] add26_fu_168_p2;
reg   [31:0] add26_reg_336;
wire    ap_CS_fsm_state3;
wire   [31:0] add36_fu_174_p2;
reg   [31:0] add36_reg_341;
wire   [31:0] sub_ln435_fu_179_p2;
reg   [31:0] sub_ln435_reg_346;
wire   [0:0] brmerge_mid132_fu_195_p2;
reg   [0:0] brmerge_mid132_reg_351;
wire   [0:0] brmerge46_mid144_fu_233_p2;
reg   [0:0] brmerge46_mid144_reg_356;
wire   [0:0] brmerge49_mid156_fu_251_p2;
reg   [0:0] brmerge49_mid156_reg_361;
wire   [0:0] icmp_ln422_fu_258_p2;
reg   [0:0] icmp_ln422_reg_366;
wire    grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_ap_start;
wire    grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_ap_done;
wire    grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_ap_idle;
wire    grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_ap_ready;
wire    grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWVALID;
wire   [63:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWADDR;
wire   [0:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWID;
wire   [31:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWLEN;
wire   [2:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWSIZE;
wire   [1:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWBURST;
wire   [1:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWLOCK;
wire   [3:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWCACHE;
wire   [2:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWPROT;
wire   [3:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWQOS;
wire   [3:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWREGION;
wire   [0:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWUSER;
wire    grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_WVALID;
wire   [31:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_WDATA;
wire   [3:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_WSTRB;
wire    grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_WLAST;
wire   [0:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_WID;
wire   [0:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_WUSER;
wire    grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARVALID;
wire   [63:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARADDR;
wire   [0:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARID;
wire   [31:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARLEN;
wire   [2:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARSIZE;
wire   [1:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARBURST;
wire   [1:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARLOCK;
wire   [3:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARCACHE;
wire   [2:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARPROT;
wire   [3:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARQOS;
wire   [3:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARREGION;
wire   [0:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARUSER;
wire    grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_RREADY;
wire    grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_BREADY;
wire   [31:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_grp_fu_371_p_din0;
wire   [31:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_grp_fu_371_p_din1;
wire   [4:0] grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_grp_fu_371_p_opcode;
wire    grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_grp_fu_371_p_ce;
reg    grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_ap_start_reg;
wire    ap_CS_fsm_state4;
wire  signed [31:0] zext_ln410_fu_134_p0;
wire  signed [31:0] zext_ln410_1_fu_138_p0;
wire   [31:0] mul_ln410_fu_142_p0;
wire   [31:0] mul_ln410_fu_142_p1;
wire   [31:0] mul_ln410_1_fu_158_p0;
wire   [63:0] mul_ln410_1_fu_158_p1;
wire   [31:0] mul25_fu_164_p2;
wire   [0:0] cmp22_mid124_fu_184_p2;
wire   [0:0] cmp27_mid126_fu_189_p2;
wire   [30:0] tmp_fu_202_p4;
wire   [30:0] tmp_1_fu_217_p4;
wire   [0:0] icmp_fu_211_p2;
wire   [0:0] icmp125_fu_227_p2;
wire   [0:0] cmp22_2_mid148_fu_240_p2;
wire   [0:0] cmp27_2_mid150_fu_245_p2;
wire  signed [31:0] icmp_ln422_fu_258_p0;
reg    grp_fu_371_ce;
reg    ap_block_state4_on_subcall_done;
reg   [3:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
wire    ap_ST_fsm_state2_blk;
wire    ap_ST_fsm_state3_blk;
reg    ap_ST_fsm_state4_blk;
wire   [95:0] mul_ln410_1_fu_158_p00;
wire   [95:0] mul_ln410_1_fu_158_p10;
wire   [63:0] mul_ln410_fu_142_p00;
wire   [63:0] mul_ln410_fu_142_p10;
wire    ap_ce_reg;

// power-on initialization
initial begin
#0 ap_CS_fsm = 4'd1;
#0 grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_ap_start_reg = 1'b0;
end

conv_kernel_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3 grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_ap_start),
    .ap_done(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_ap_done),
    .ap_idle(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_ap_idle),
    .ap_ready(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_ap_ready),
    .m_axi_gmem0_AWVALID(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWVALID),
    .m_axi_gmem0_AWREADY(m_axi_gmem0_AWREADY),
    .m_axi_gmem0_AWADDR(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWADDR),
    .m_axi_gmem0_AWID(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWID),
    .m_axi_gmem0_AWLEN(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWLEN),
    .m_axi_gmem0_AWSIZE(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWSIZE),
    .m_axi_gmem0_AWBURST(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWBURST),
    .m_axi_gmem0_AWLOCK(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWLOCK),
    .m_axi_gmem0_AWCACHE(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWCACHE),
    .m_axi_gmem0_AWPROT(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWPROT),
    .m_axi_gmem0_AWQOS(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWQOS),
    .m_axi_gmem0_AWREGION(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWREGION),
    .m_axi_gmem0_AWUSER(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWUSER),
    .m_axi_gmem0_WVALID(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_WVALID),
    .m_axi_gmem0_WREADY(m_axi_gmem0_WREADY),
    .m_axi_gmem0_WDATA(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_WDATA),
    .m_axi_gmem0_WSTRB(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_WSTRB),
    .m_axi_gmem0_WLAST(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_WLAST),
    .m_axi_gmem0_WID(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_WID),
    .m_axi_gmem0_WUSER(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_WUSER),
    .m_axi_gmem0_ARVALID(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARVALID),
    .m_axi_gmem0_ARREADY(m_axi_gmem0_ARREADY),
    .m_axi_gmem0_ARADDR(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARADDR),
    .m_axi_gmem0_ARID(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARID),
    .m_axi_gmem0_ARLEN(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARLEN),
    .m_axi_gmem0_ARSIZE(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARSIZE),
    .m_axi_gmem0_ARBURST(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARBURST),
    .m_axi_gmem0_ARLOCK(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARLOCK),
    .m_axi_gmem0_ARCACHE(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARCACHE),
    .m_axi_gmem0_ARPROT(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARPROT),
    .m_axi_gmem0_ARQOS(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARQOS),
    .m_axi_gmem0_ARREGION(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARREGION),
    .m_axi_gmem0_ARUSER(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARUSER),
    .m_axi_gmem0_RVALID(m_axi_gmem0_RVALID),
    .m_axi_gmem0_RREADY(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_RREADY),
    .m_axi_gmem0_RDATA(m_axi_gmem0_RDATA),
    .m_axi_gmem0_RLAST(m_axi_gmem0_RLAST),
    .m_axi_gmem0_RID(m_axi_gmem0_RID),
    .m_axi_gmem0_RFIFONUM(m_axi_gmem0_RFIFONUM),
    .m_axi_gmem0_RUSER(m_axi_gmem0_RUSER),
    .m_axi_gmem0_RRESP(m_axi_gmem0_RRESP),
    .m_axi_gmem0_BVALID(m_axi_gmem0_BVALID),
    .m_axi_gmem0_BREADY(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_BREADY),
    .m_axi_gmem0_BRESP(m_axi_gmem0_BRESP),
    .m_axi_gmem0_BID(m_axi_gmem0_BID),
    .m_axi_gmem0_BUSER(m_axi_gmem0_BUSER),
    .noy(noy),
    .stride(stride),
    .pad(pad),
    .add26(add26_reg_336),
    .mul_ln410_1(mul_ln410_1_reg_331),
    .out_base_addr(out_base_addr),
    .act_mem(act_mem),
    .add36(add36_reg_341),
    .mul_ln410(mul_ln410_reg_319),
    .brmerge_mid132(brmerge_mid132_reg_351),
    .brmerge46_mid144(brmerge46_mid144_reg_356),
    .brmerge49_mid156(brmerge49_mid156_reg_361),
    .nox(nox),
    .icmp_ln422_1(icmp_ln422_reg_366),
    .mul35(mul35_reg_325),
    .sub_ln435(sub_ln435_reg_346),
    .grp_fu_371_p_din0(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_grp_fu_371_p_din0),
    .grp_fu_371_p_din1(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_grp_fu_371_p_din1),
    .grp_fu_371_p_opcode(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_grp_fu_371_p_opcode),
    .grp_fu_371_p_dout0(grp_fu_2957_p_dout0),
    .grp_fu_371_p_ce(grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_grp_fu_371_p_ce)
);

conv_kernel_mul_32ns_32ns_64_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 32 ),
    .din1_WIDTH( 32 ),
    .dout_WIDTH( 64 ))
mul_32ns_32ns_64_1_1_U375(
    .din0(mul_ln410_fu_142_p0),
    .din1(mul_ln410_fu_142_p1),
    .dout(mul_ln410_fu_142_p2)
);

conv_kernel_mul_32s_32s_32_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 32 ),
    .din1_WIDTH( 32 ),
    .dout_WIDTH( 32 ))
mul_32s_32s_32_1_1_U376(
    .din0(stride),
    .din1(nox),
    .dout(mul35_fu_148_p2)
);

conv_kernel_mul_32ns_64ns_96_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 32 ),
    .din1_WIDTH( 64 ),
    .dout_WIDTH( 96 ))
mul_32ns_64ns_96_1_1_U377(
    .din0(mul_ln410_1_fu_158_p0),
    .din1(mul_ln410_1_fu_158_p1),
    .dout(mul_ln410_1_fu_158_p2)
);

conv_kernel_mul_32s_32s_32_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 32 ),
    .din1_WIDTH( 32 ),
    .dout_WIDTH( 32 ))
mul_32s_32s_32_1_1_U378(
    .din0(stride),
    .din1(noy),
    .dout(mul25_fu_164_p2)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state3)) begin
            grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_ap_start_reg <= 1'b1;
        end else if ((grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_ap_ready == 1'b1)) begin
            grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        add26_reg_336 <= add26_fu_168_p2;
        add36_reg_341 <= add36_fu_174_p2;
        brmerge46_mid144_reg_356 <= brmerge46_mid144_fu_233_p2;
        brmerge49_mid156_reg_361 <= brmerge49_mid156_fu_251_p2;
        brmerge_mid132_reg_351 <= brmerge_mid132_fu_195_p2;
        icmp_ln422_reg_366 <= icmp_ln422_fu_258_p2;
        sub_ln435_reg_346 <= sub_ln435_fu_179_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        mul35_reg_325 <= mul35_fu_148_p2;
        mul_ln410_1_reg_331 <= mul_ln410_1_fu_158_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((max_pool_en == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        mul_ln410_reg_319 <= mul_ln410_fu_142_p2;
    end
end

always @ (*) begin
    if ((ap_start == 1'b0)) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_fsm_state2_blk = 1'b0;

assign ap_ST_fsm_state3_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state4_on_subcall_done)) begin
        ap_ST_fsm_state4_blk = 1'b1;
    end else begin
        ap_ST_fsm_state4_blk = 1'b0;
    end
end

always @ (*) begin
    if ((((1'b0 == ap_block_state4_on_subcall_done) & (1'b1 == ap_CS_fsm_state4)) | ((ap_start == 1'b0) & (1'b1 == ap_CS_fsm_state1)))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = 1'b0;
    end
end

always @ (*) begin
    if (((ap_start == 1'b0) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state4_on_subcall_done) & (1'b1 == ap_CS_fsm_state4))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        grp_fu_371_ce = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_grp_fu_371_p_ce;
    end else begin
        grp_fu_371_ce = 1'b1;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | ((max_pool_en == 1'd1) & (1'b1 == ap_CS_fsm_state4)))) begin
        m_axi_gmem0_ARVALID = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARVALID;
    end else begin
        m_axi_gmem0_ARVALID = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | ((max_pool_en == 1'd1) & (1'b1 == ap_CS_fsm_state4)))) begin
        m_axi_gmem0_AWVALID = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWVALID;
    end else begin
        m_axi_gmem0_AWVALID = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | ((max_pool_en == 1'd1) & (1'b1 == ap_CS_fsm_state4)))) begin
        m_axi_gmem0_BREADY = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_BREADY;
    end else begin
        m_axi_gmem0_BREADY = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | ((max_pool_en == 1'd1) & (1'b1 == ap_CS_fsm_state4)))) begin
        m_axi_gmem0_RREADY = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_RREADY;
    end else begin
        m_axi_gmem0_RREADY = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | ((max_pool_en == 1'd1) & (1'b1 == ap_CS_fsm_state4)))) begin
        m_axi_gmem0_WVALID = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_WVALID;
    end else begin
        m_axi_gmem0_WVALID = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((ap_start == 1'b1) & (max_pool_en_read_read_fu_58_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end else if (((ap_start == 1'b1) & (max_pool_en == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            ap_NS_fsm = ap_ST_fsm_state3;
        end
        ap_ST_fsm_state3 : begin
            ap_NS_fsm = ap_ST_fsm_state4;
        end
        ap_ST_fsm_state4 : begin
            if (((1'b0 == ap_block_state4_on_subcall_done) & (1'b1 == ap_CS_fsm_state4))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign add26_fu_168_p2 = (mul25_fu_164_p2 + pad);

assign add36_fu_174_p2 = (mul35_reg_325 + pad);

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

always @ (*) begin
    ap_block_state4_on_subcall_done = ((max_pool_en == 1'd1) & (grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_ap_done == 1'b0));
end

assign brmerge46_mid144_fu_233_p2 = (icmp_fu_211_p2 | icmp125_fu_227_p2);

assign brmerge49_mid156_fu_251_p2 = (cmp27_2_mid150_fu_245_p2 | cmp22_2_mid148_fu_240_p2);

assign brmerge_mid132_fu_195_p2 = (cmp27_mid126_fu_189_p2 | cmp22_mid124_fu_184_p2);

assign cmp22_2_mid148_fu_240_p2 = ((pad > 32'd2) ? 1'b1 : 1'b0);

assign cmp22_mid124_fu_184_p2 = ((pad != 32'd0) ? 1'b1 : 1'b0);

assign cmp27_2_mid150_fu_245_p2 = ((add26_fu_168_p2 < 32'd3) ? 1'b1 : 1'b0);

assign cmp27_mid126_fu_189_p2 = ((add26_fu_168_p2 == 32'd0) ? 1'b1 : 1'b0);

assign grp_fu_2957_p_ce = grp_fu_371_ce;

assign grp_fu_2957_p_din0 = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_grp_fu_371_p_din0;

assign grp_fu_2957_p_din1 = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_grp_fu_371_p_din1;

assign grp_fu_2957_p_opcode = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_grp_fu_371_p_opcode;

assign grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_ap_start = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_ap_start_reg;

assign icmp125_fu_227_p2 = ((tmp_1_fu_217_p4 == 31'd0) ? 1'b1 : 1'b0);

assign icmp_fu_211_p2 = ((tmp_fu_202_p4 != 31'd0) ? 1'b1 : 1'b0);

assign icmp_ln422_fu_258_p0 = nox;

assign icmp_ln422_fu_258_p2 = ((icmp_ln422_fu_258_p0 == 32'd0) ? 1'b1 : 1'b0);

assign m_axi_gmem0_ARADDR = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARADDR;

assign m_axi_gmem0_ARBURST = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARBURST;

assign m_axi_gmem0_ARCACHE = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARCACHE;

assign m_axi_gmem0_ARID = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARID;

assign m_axi_gmem0_ARLEN = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARLEN;

assign m_axi_gmem0_ARLOCK = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARLOCK;

assign m_axi_gmem0_ARPROT = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARPROT;

assign m_axi_gmem0_ARQOS = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARQOS;

assign m_axi_gmem0_ARREGION = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARREGION;

assign m_axi_gmem0_ARSIZE = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARSIZE;

assign m_axi_gmem0_ARUSER = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_ARUSER;

assign m_axi_gmem0_AWADDR = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWADDR;

assign m_axi_gmem0_AWBURST = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWBURST;

assign m_axi_gmem0_AWCACHE = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWCACHE;

assign m_axi_gmem0_AWID = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWID;

assign m_axi_gmem0_AWLEN = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWLEN;

assign m_axi_gmem0_AWLOCK = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWLOCK;

assign m_axi_gmem0_AWPROT = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWPROT;

assign m_axi_gmem0_AWQOS = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWQOS;

assign m_axi_gmem0_AWREGION = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWREGION;

assign m_axi_gmem0_AWSIZE = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWSIZE;

assign m_axi_gmem0_AWUSER = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_AWUSER;

assign m_axi_gmem0_WDATA = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_WDATA;

assign m_axi_gmem0_WID = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_WID;

assign m_axi_gmem0_WLAST = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_WLAST;

assign m_axi_gmem0_WSTRB = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_WSTRB;

assign m_axi_gmem0_WUSER = grp_max_pool_Pipeline_max_pool_loop1_max_pool_loop2_max_pool_loop3_fu_112_m_axi_gmem0_WUSER;

assign max_pool_en_read_read_fu_58_p2 = max_pool_en;

assign mul_ln410_1_fu_158_p0 = mul_ln410_1_fu_158_p00;

assign mul_ln410_1_fu_158_p00 = nif;

assign mul_ln410_1_fu_158_p1 = mul_ln410_1_fu_158_p10;

assign mul_ln410_1_fu_158_p10 = mul_ln410_reg_319;

assign mul_ln410_fu_142_p0 = mul_ln410_fu_142_p00;

assign mul_ln410_fu_142_p00 = $unsigned(zext_ln410_fu_134_p0);

assign mul_ln410_fu_142_p1 = mul_ln410_fu_142_p10;

assign mul_ln410_fu_142_p10 = $unsigned(zext_ln410_1_fu_138_p0);

assign sub_ln435_fu_179_p2 = (in_base_addr - pad);

assign tmp_1_fu_217_p4 = {{add26_fu_168_p2[31:1]}};

assign tmp_fu_202_p4 = {{pad[31:1]}};

assign zext_ln410_1_fu_138_p0 = nox;

assign zext_ln410_fu_134_p0 = noy;

endmodule //conv_kernel_max_pool
