// ==============================================================
// RTL generated by Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Version: 2022.1
// Copyright (C) Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module conv_kernel_conv_kernel_Pipeline_fc_loop1 (
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
        sext_ln514,
        nof,
        trunc_ln507_cast,
        base_addr_out,
        act_mem,
        nif,
        cmp21_i83,
        add_ln507,
        bn_weight_base,
        bn_weight_mem,
        grp_fu_2953_p_din0,
        grp_fu_2953_p_din1,
        grp_fu_2953_p_opcode,
        grp_fu_2953_p_dout0,
        grp_fu_2953_p_ce,
        grp_fu_2949_p_din0,
        grp_fu_2949_p_din1,
        grp_fu_2949_p_dout0,
        grp_fu_2949_p_ce
);

parameter    ap_ST_fsm_pp0_stage0 = 3'd1;
parameter    ap_ST_fsm_pp0_stage1 = 3'd2;
parameter    ap_ST_fsm_pp0_stage2 = 3'd4;

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
input  [61:0] sext_ln514;
input  [31:0] nof;
input  [61:0] trunc_ln507_cast;
input  [31:0] base_addr_out;
input  [63:0] act_mem;
input  [31:0] nif;
input  [0:0] cmp21_i83;
input  [31:0] add_ln507;
input  [31:0] bn_weight_base;
input  [63:0] bn_weight_mem;
output  [31:0] grp_fu_2953_p_din0;
output  [31:0] grp_fu_2953_p_din1;
output  [0:0] grp_fu_2953_p_opcode;
input  [31:0] grp_fu_2953_p_dout0;
output   grp_fu_2953_p_ce;
output  [31:0] grp_fu_2949_p_din0;
output  [31:0] grp_fu_2949_p_din1;
input  [31:0] grp_fu_2949_p_dout0;
output   grp_fu_2949_p_ce;

reg ap_idle;
reg m_axi_gmem0_AWVALID;
reg m_axi_gmem0_WVALID;
reg m_axi_gmem0_ARVALID;
reg[63:0] m_axi_gmem0_ARADDR;
reg m_axi_gmem0_RREADY;
reg m_axi_gmem0_BREADY;

(* fsm_encoding = "none" *) reg   [2:0] ap_CS_fsm;
wire    ap_CS_fsm_pp0_stage0;
reg    ap_enable_reg_pp0_iter0;
reg    ap_enable_reg_pp0_iter1;
reg    ap_enable_reg_pp0_iter2;
reg    ap_enable_reg_pp0_iter3;
reg    ap_enable_reg_pp0_iter4;
reg    ap_enable_reg_pp0_iter5;
reg    ap_enable_reg_pp0_iter6;
reg    ap_enable_reg_pp0_iter7;
reg    ap_idle_pp0;
wire    ap_CS_fsm_pp0_stage2;
wire    ap_block_state3_pp0_stage2_iter0;
reg   [0:0] icmp_ln507_reg_466;
reg    ap_predicate_op68_readreq_state3;
reg    ap_block_state3_io;
wire    ap_block_state6_pp0_stage2_iter1;
reg   [0:0] icmp_ln507_reg_466_pp0_iter2_reg;
reg    ap_predicate_op94_read_state9;
reg    ap_block_state9_pp0_stage2_iter2;
wire    ap_block_state12_pp0_stage2_iter3;
wire    ap_block_state15_pp0_stage2_iter4;
wire    ap_block_state18_pp0_stage2_iter5;
wire    ap_block_state21_pp0_stage2_iter6;
reg    ap_block_pp0_stage2_subdone;
reg    ap_condition_exit_pp0_iter0_stage2;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    gmem0_blk_n_AR;
wire    ap_CS_fsm_pp0_stage1;
wire    ap_block_pp0_stage1;
reg    gmem0_blk_n_R;
wire    ap_block_pp0_stage2;
wire    ap_block_pp0_stage0;
reg    gmem0_blk_n_AW;
reg    gmem0_blk_n_W;
reg    gmem0_blk_n_B;
wire    ap_block_state1_pp0_stage0_iter0;
wire    ap_block_state4_pp0_stage0_iter1;
wire    ap_block_state7_pp0_stage0_iter2;
reg    ap_predicate_op97_read_state10;
reg    ap_block_state10_pp0_stage0_iter3;
wire    ap_block_state13_pp0_stage0_iter4;
wire    ap_block_state16_pp0_stage0_iter5;
wire    ap_block_state19_pp0_stage0_iter6;
wire    ap_block_state22_pp0_stage0_iter7;
reg    ap_block_pp0_stage0_11001;
wire  signed [62:0] trunc_ln507_cast_cast_fu_223_p1;
reg  signed [62:0] trunc_ln507_cast_cast_reg_449;
wire  signed [63:0] sext_ln514_cast_fu_227_p1;
reg  signed [63:0] sext_ln514_cast_reg_454;
reg   [31:0] f_out_8_reg_459;
wire   [0:0] icmp_ln507_fu_244_p2;
reg   [0:0] icmp_ln507_reg_466_pp0_iter1_reg;
reg   [0:0] icmp_ln507_reg_466_pp0_iter3_reg;
reg   [0:0] icmp_ln507_reg_466_pp0_iter4_reg;
reg   [0:0] icmp_ln507_reg_466_pp0_iter5_reg;
reg   [0:0] icmp_ln507_reg_466_pp0_iter6_reg;
wire    ap_block_state2_pp0_stage1_iter0;
reg    ap_predicate_op53_readreq_state2;
reg    ap_block_state2_io;
wire    ap_block_state5_pp0_stage1_iter1;
wire    ap_block_state8_pp0_stage1_iter2;
reg    ap_block_state11_pp0_stage1_iter3;
wire    ap_block_state14_pp0_stage1_iter4;
wire    ap_block_state17_pp0_stage1_iter5;
wire    ap_block_state20_pp0_stage1_iter6;
reg    ap_block_state23_pp0_stage1_iter7;
reg    ap_block_pp0_stage1_11001;
reg   [63:0] gmem0_addr_12_reg_476;
reg   [63:0] gmem0_addr_13_reg_482;
reg    ap_block_pp0_stage2_11001;
reg   [63:0] gmem0_addr_14_reg_488;
reg   [63:0] gmem0_addr_14_reg_488_pp0_iter1_reg;
reg   [63:0] gmem0_addr_14_reg_488_pp0_iter2_reg;
reg   [63:0] gmem0_addr_14_reg_488_pp0_iter3_reg;
reg   [63:0] gmem0_addr_14_reg_488_pp0_iter4_reg;
reg   [31:0] gmem0_addr_read_reg_494;
reg   [31:0] gmem0_addr_12_read_reg_499;
wire   [31:0] bitcast_ln514_fu_385_p1;
wire   [31:0] bitcast_ln514_1_fu_389_p1;
reg   [31:0] gmem0_addr_13_read_reg_514;
reg   [31:0] mul9_le_i_reg_519;
wire   [31:0] bitcast_ln516_fu_393_p1;
reg   [31:0] add14_i1_reg_529;
reg    ap_enable_reg_pp0_iter0_reg;
reg    ap_block_pp0_stage1_subdone;
reg   [31:0] ap_phi_mux_sum_0_lcssa_i_phi_fu_206_p4;
wire  signed [63:0] sext_ln514_1_fu_306_p1;
wire  signed [63:0] sext_ln516_1_fu_334_p1;
wire  signed [63:0] sext_ln516_fu_375_p1;
reg    ap_block_pp0_stage2_01001;
reg   [31:0] phi_mul_fu_82;
wire   [31:0] next_mul_fu_264_p2;
wire    ap_loop_init;
reg   [31:0] f_out_fu_86;
wire   [31:0] add_ln507_1_fu_256_p2;
reg   [31:0] ap_sig_allocacmp_f_out_8;
wire   [31:0] add_ln514_2_fu_269_p2;
wire   [31:0] add_ln514_fu_274_p2;
wire   [33:0] shl_ln7_fu_279_p3;
wire   [63:0] zext_ln514_fu_287_p1;
wire   [63:0] add_ln514_1_fu_291_p2;
wire   [61:0] trunc_ln514_1_fu_296_p4;
wire   [62:0] zext_ln507_fu_326_p1;
wire   [62:0] add_ln516_fu_329_p2;
wire   [31:0] add_ln516_1_fu_344_p2;
wire   [33:0] shl_ln8_fu_348_p3;
wire   [63:0] zext_ln516_fu_356_p1;
wire   [63:0] add_ln516_2_fu_360_p2;
wire   [61:0] trunc_ln_fu_365_p4;
reg    ap_block_pp0_stage1_00001;
reg    grp_fu_214_ce;
reg    grp_fu_219_ce;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter1_reg;
reg    ap_condition_exit_pp0_iter6_stage1;
reg    ap_idle_pp0_0to5;
reg    ap_loop_exit_ready_pp0_iter2_reg;
reg    ap_loop_exit_ready_pp0_iter3_reg;
reg    ap_loop_exit_ready_pp0_iter4_reg;
reg    ap_loop_exit_ready_pp0_iter5_reg;
reg    ap_loop_exit_ready_pp0_iter6_reg;
reg   [2:0] ap_NS_fsm;
reg    ap_block_pp0_stage0_subdone;
reg    ap_idle_pp0_1to7;
reg    ap_done_pending_pp0;
wire    ap_enable_pp0;
wire    ap_start_int;
wire    ap_ce_reg;

// power-on initialization
initial begin
#0 ap_CS_fsm = 3'd1;
#0 ap_enable_reg_pp0_iter1 = 1'b0;
#0 ap_enable_reg_pp0_iter2 = 1'b0;
#0 ap_enable_reg_pp0_iter3 = 1'b0;
#0 ap_enable_reg_pp0_iter4 = 1'b0;
#0 ap_enable_reg_pp0_iter5 = 1'b0;
#0 ap_enable_reg_pp0_iter6 = 1'b0;
#0 ap_enable_reg_pp0_iter7 = 1'b0;
#0 ap_enable_reg_pp0_iter0_reg = 1'b0;
#0 ap_done_reg = 1'b0;
end

conv_kernel_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(ap_start),
    .ap_ready(ap_ready),
    .ap_done(ap_done),
    .ap_start_int(ap_start_int),
    .ap_loop_init(ap_loop_init),
    .ap_ready_int(ap_ready_int),
    .ap_loop_exit_ready(ap_condition_exit_pp0_iter0_stage2),
    .ap_loop_exit_done(ap_done_int),
    .ap_continue_int(ap_continue_int),
    .ap_done_int(ap_done_int)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_pp0_stage0;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue_int == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((ap_loop_exit_ready_pp0_iter6_reg == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage1) & (1'b0 == ap_block_pp0_stage1_subdone))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter0_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_pp0_stage0)) begin
            ap_enable_reg_pp0_iter0_reg <= ap_start_int;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter1 <= 1'b0;
    end else begin
        if ((1'b1 == ap_condition_exit_pp0_iter0_stage2)) begin
            ap_enable_reg_pp0_iter1 <= 1'b0;
        end else if (((1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_subdone))) begin
            ap_enable_reg_pp0_iter1 <= ap_enable_reg_pp0_iter0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter2 <= 1'b0;
    end else begin
        if (((1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_subdone))) begin
            ap_enable_reg_pp0_iter2 <= ap_enable_reg_pp0_iter1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter3 <= 1'b0;
    end else begin
        if (((1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_subdone))) begin
            ap_enable_reg_pp0_iter3 <= ap_enable_reg_pp0_iter2;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter4 <= 1'b0;
    end else begin
        if (((1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_subdone))) begin
            ap_enable_reg_pp0_iter4 <= ap_enable_reg_pp0_iter3;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter5 <= 1'b0;
    end else begin
        if (((1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_subdone))) begin
            ap_enable_reg_pp0_iter5 <= ap_enable_reg_pp0_iter4;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter6 <= 1'b0;
    end else begin
        if (((1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_subdone))) begin
            ap_enable_reg_pp0_iter6 <= ap_enable_reg_pp0_iter5;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter7 <= 1'b0;
    end else begin
        if (((ap_enable_reg_pp0_iter7 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage1) & (1'b0 == ap_block_pp0_stage1_subdone))) begin
            ap_enable_reg_pp0_iter7 <= 1'b0;
        end else if (((1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_subdone))) begin
            ap_enable_reg_pp0_iter7 <= ap_enable_reg_pp0_iter6;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((ap_idle_pp0_0to5 == 1'b1) & (1'b1 == ap_condition_exit_pp0_iter6_stage1))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= 1'b0;
    end else if (((1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_11001))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
    end
end

always @ (posedge ap_clk) begin
    if (((ap_idle_pp0_0to5 == 1'b1) & (1'b1 == ap_condition_exit_pp0_iter6_stage1))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= 1'b0;
    end else if (((1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_11001))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
    end
end

always @ (posedge ap_clk) begin
    if (((ap_idle_pp0_0to5 == 1'b1) & (1'b1 == ap_condition_exit_pp0_iter6_stage1))) begin
        ap_loop_exit_ready_pp0_iter3_reg <= 1'b0;
    end else if (((1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_11001))) begin
        ap_loop_exit_ready_pp0_iter3_reg <= ap_loop_exit_ready_pp0_iter2_reg;
    end
end

always @ (posedge ap_clk) begin
    if (((ap_idle_pp0_0to5 == 1'b1) & (1'b1 == ap_condition_exit_pp0_iter6_stage1))) begin
        ap_loop_exit_ready_pp0_iter4_reg <= 1'b0;
    end else if (((1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_11001))) begin
        ap_loop_exit_ready_pp0_iter4_reg <= ap_loop_exit_ready_pp0_iter3_reg;
    end
end

always @ (posedge ap_clk) begin
    if (((ap_idle_pp0_0to5 == 1'b1) & (1'b1 == ap_condition_exit_pp0_iter6_stage1))) begin
        ap_loop_exit_ready_pp0_iter5_reg <= 1'b0;
    end else if (((1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_11001))) begin
        ap_loop_exit_ready_pp0_iter5_reg <= ap_loop_exit_ready_pp0_iter4_reg;
    end
end

always @ (posedge ap_clk) begin
    if (((ap_idle_pp0_0to5 == 1'b1) & (1'b1 == ap_condition_exit_pp0_iter6_stage1))) begin
        ap_loop_exit_ready_pp0_iter6_reg <= 1'b0;
    end else if (((1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_11001))) begin
        ap_loop_exit_ready_pp0_iter6_reg <= ap_loop_exit_ready_pp0_iter5_reg;
    end
end

always @ (posedge ap_clk) begin
    if (((ap_loop_init == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        f_out_fu_86 <= 32'd0;
    end else if (((ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage1) & (1'b0 == ap_block_pp0_stage1_11001) & (icmp_ln507_reg_466 == 1'd0))) begin
        f_out_fu_86 <= add_ln507_1_fu_256_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((ap_loop_init == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        phi_mul_fu_82 <= 32'd0;
    end else if (((ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage1) & (1'b0 == ap_block_pp0_stage1_11001) & (icmp_ln507_reg_466 == 1'd0))) begin
        phi_mul_fu_82 <= next_mul_fu_264_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_pp0_stage1) & (1'b0 == ap_block_pp0_stage1_11001))) begin
        add14_i1_reg_529 <= grp_fu_2953_p_dout0;
        gmem0_addr_13_read_reg_514 <= m_axi_gmem0_RDATA;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        f_out_8_reg_459 <= ap_sig_allocacmp_f_out_8;
        icmp_ln507_reg_466 <= icmp_ln507_fu_244_p2;
        icmp_ln507_reg_466_pp0_iter1_reg <= icmp_ln507_reg_466;
        icmp_ln507_reg_466_pp0_iter2_reg <= icmp_ln507_reg_466_pp0_iter1_reg;
        icmp_ln507_reg_466_pp0_iter3_reg <= icmp_ln507_reg_466_pp0_iter2_reg;
        icmp_ln507_reg_466_pp0_iter4_reg <= icmp_ln507_reg_466_pp0_iter3_reg;
        icmp_ln507_reg_466_pp0_iter5_reg <= icmp_ln507_reg_466_pp0_iter4_reg;
        icmp_ln507_reg_466_pp0_iter6_reg <= icmp_ln507_reg_466_pp0_iter5_reg;
        sext_ln514_cast_reg_454 <= sext_ln514_cast_fu_227_p1;
        trunc_ln507_cast_cast_reg_449 <= trunc_ln507_cast_cast_fu_223_p1;
    end
end

always @ (posedge ap_clk) begin
    if (((ap_predicate_op97_read_state10 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        gmem0_addr_12_read_reg_499 <= m_axi_gmem0_RDATA;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_pp0_stage1) & (1'b0 == ap_block_pp0_stage1_11001) & (cmp21_i83 == 1'd0) & (icmp_ln507_reg_466 == 1'd0))) begin
        gmem0_addr_12_reg_476 <= sext_ln514_1_fu_306_p1;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_11001) & (icmp_ln507_reg_466 == 1'd0))) begin
        gmem0_addr_13_reg_482 <= sext_ln516_1_fu_334_p1;
        gmem0_addr_14_reg_488 <= sext_ln516_fu_375_p1;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_11001))) begin
        gmem0_addr_14_reg_488_pp0_iter1_reg <= gmem0_addr_14_reg_488;
        gmem0_addr_14_reg_488_pp0_iter2_reg <= gmem0_addr_14_reg_488_pp0_iter1_reg;
        gmem0_addr_14_reg_488_pp0_iter3_reg <= gmem0_addr_14_reg_488_pp0_iter2_reg;
        gmem0_addr_14_reg_488_pp0_iter4_reg <= gmem0_addr_14_reg_488_pp0_iter3_reg;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_11001) & (ap_predicate_op94_read_state9 == 1'b1))) begin
        gmem0_addr_read_reg_494 <= m_axi_gmem0_RDATA;
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln507_reg_466_pp0_iter3_reg == 1'd0) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001) & (cmp21_i83 == 1'd0))) begin
        mul9_le_i_reg_519 <= grp_fu_2949_p_dout0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_subdone) & (icmp_ln507_reg_466 == 1'd1))) begin
        ap_condition_exit_pp0_iter0_stage2 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage2 = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter6 == 1'b1) & (icmp_ln507_reg_466_pp0_iter6_reg == 1'd1) & (1'b1 == ap_CS_fsm_pp0_stage1) & (1'b0 == ap_block_pp0_stage1_subdone))) begin
        ap_condition_exit_pp0_iter6_stage1 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter6_stage1 = 1'b0;
    end
end

always @ (*) begin
    if (((ap_loop_exit_ready_pp0_iter6_reg == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage1) & (1'b0 == ap_block_pp0_stage1_subdone))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (~((ap_loop_exit_ready_pp0_iter6_reg == 1'b0) & (ap_loop_exit_ready_pp0_iter5_reg == 1'b0) & (ap_loop_exit_ready_pp0_iter4_reg == 1'b0) & (ap_loop_exit_ready_pp0_iter3_reg == 1'b0) & (ap_loop_exit_ready_pp0_iter2_reg == 1'b0) & (ap_loop_exit_ready_pp0_iter1_reg == 1'b0) & (ap_loop_exit_ready == 1'b0))) begin
        ap_done_pending_pp0 = 1'b1;
    end else begin
        ap_done_pending_pp0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_pp0_stage0)) begin
        ap_enable_reg_pp0_iter0 = ap_start_int;
    end else begin
        ap_enable_reg_pp0_iter0 = ap_enable_reg_pp0_iter0_reg;
    end
end

always @ (*) begin
    if (((ap_start_int == 1'b0) & (ap_idle_pp0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter7 == 1'b0) & (ap_enable_reg_pp0_iter6 == 1'b0) & (ap_enable_reg_pp0_iter5 == 1'b0) & (ap_enable_reg_pp0_iter4 == 1'b0) & (ap_enable_reg_pp0_iter3 == 1'b0) & (ap_enable_reg_pp0_iter2 == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b0) & (ap_enable_reg_pp0_iter0 == 1'b0))) begin
        ap_idle_pp0 = 1'b1;
    end else begin
        ap_idle_pp0 = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter5 == 1'b0) & (ap_enable_reg_pp0_iter4 == 1'b0) & (ap_enable_reg_pp0_iter3 == 1'b0) & (ap_enable_reg_pp0_iter2 == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b0) & (ap_enable_reg_pp0_iter0 == 1'b0))) begin
        ap_idle_pp0_0to5 = 1'b1;
    end else begin
        ap_idle_pp0_0to5 = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter7 == 1'b0) & (ap_enable_reg_pp0_iter6 == 1'b0) & (ap_enable_reg_pp0_iter5 == 1'b0) & (ap_enable_reg_pp0_iter4 == 1'b0) & (ap_enable_reg_pp0_iter3 == 1'b0) & (ap_enable_reg_pp0_iter2 == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b0))) begin
        ap_idle_pp0_1to7 = 1'b1;
    end else begin
        ap_idle_pp0_1to7 = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln507_reg_466_pp0_iter4_reg == 1'd0) & (cmp21_i83 == 1'd0))) begin
        ap_phi_mux_sum_0_lcssa_i_phi_fu_206_p4 = mul9_le_i_reg_519;
    end else begin
        ap_phi_mux_sum_0_lcssa_i_phi_fu_206_p4 = 32'd0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_subdone))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((ap_loop_init == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0))) begin
        ap_sig_allocacmp_f_out_8 = 32'd0;
    end else begin
        ap_sig_allocacmp_f_out_8 = f_out_fu_86;
    end
end

always @ (*) begin
    if ((((ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0)) | ((ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2) & (ap_predicate_op68_readreq_state3 == 1'b1)) | ((ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage1) & (1'b0 == ap_block_pp0_stage1) & (cmp21_i83 == 1'd0) & (icmp_ln507_reg_466 == 1'd0)))) begin
        gmem0_blk_n_AR = m_axi_gmem0_ARREADY;
    end else begin
        gmem0_blk_n_AR = 1'b1;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter5 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage1) & (1'b0 == ap_block_pp0_stage1))) begin
        gmem0_blk_n_AW = m_axi_gmem0_AWREADY;
    end else begin
        gmem0_blk_n_AW = 1'b1;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter7 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage1) & (1'b0 == ap_block_pp0_stage1))) begin
        gmem0_blk_n_B = m_axi_gmem0_BVALID;
    end else begin
        gmem0_blk_n_B = 1'b1;
    end
end

always @ (*) begin
    if ((((ap_enable_reg_pp0_iter3 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0) & (icmp_ln507_reg_466_pp0_iter2_reg == 1'd0) & (cmp21_i83 == 1'd0)) | ((ap_enable_reg_pp0_iter3 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage1) & (1'b0 == ap_block_pp0_stage1)) | ((ap_enable_reg_pp0_iter2 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2) & (ap_predicate_op94_read_state9 == 1'b1)))) begin
        gmem0_blk_n_R = m_axi_gmem0_RVALID;
    end else begin
        gmem0_blk_n_R = 1'b1;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter5 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2))) begin
        gmem0_blk_n_W = m_axi_gmem0_WREADY;
    end else begin
        gmem0_blk_n_W = 1'b1;
    end
end

always @ (*) begin
    if ((((1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_11001)) | ((1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001)) | ((1'b1 == ap_CS_fsm_pp0_stage1) & (1'b0 == ap_block_pp0_stage1_11001)))) begin
        grp_fu_214_ce = 1'b1;
    end else begin
        grp_fu_214_ce = 1'b0;
    end
end

always @ (*) begin
    if ((((1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_11001)) | ((1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001)) | ((1'b1 == ap_CS_fsm_pp0_stage1) & (1'b0 == ap_block_pp0_stage1_11001)))) begin
        grp_fu_219_ce = 1'b1;
    end else begin
        grp_fu_219_ce = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        m_axi_gmem0_ARADDR = gmem0_addr_13_reg_482;
    end else if (((ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_11001) & (ap_predicate_op68_readreq_state3 == 1'b1))) begin
        m_axi_gmem0_ARADDR = gmem0_addr_12_reg_476;
    end else if (((ap_predicate_op53_readreq_state2 == 1'b1) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage1) & (1'b0 == ap_block_pp0_stage1_11001))) begin
        m_axi_gmem0_ARADDR = sext_ln514_cast_reg_454;
    end else begin
        m_axi_gmem0_ARADDR = 'bx;
    end
end

always @ (*) begin
    if ((((ap_predicate_op53_readreq_state2 == 1'b1) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage1) & (1'b0 == ap_block_pp0_stage1_11001)) | ((ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001)) | ((ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_11001) & (ap_predicate_op68_readreq_state3 == 1'b1)))) begin
        m_axi_gmem0_ARVALID = 1'b1;
    end else begin
        m_axi_gmem0_ARVALID = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter5 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage1) & (1'b0 == ap_block_pp0_stage1_11001))) begin
        m_axi_gmem0_AWVALID = 1'b1;
    end else begin
        m_axi_gmem0_AWVALID = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter7 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage1) & (1'b0 == ap_block_pp0_stage1_11001))) begin
        m_axi_gmem0_BREADY = 1'b1;
    end else begin
        m_axi_gmem0_BREADY = 1'b0;
    end
end

always @ (*) begin
    if ((((ap_enable_reg_pp0_iter3 == 1'b1) & (ap_predicate_op97_read_state10 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001)) | ((ap_enable_reg_pp0_iter3 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage1) & (1'b0 == ap_block_pp0_stage1_11001)) | ((ap_enable_reg_pp0_iter2 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_11001) & (ap_predicate_op94_read_state9 == 1'b1)))) begin
        m_axi_gmem0_RREADY = 1'b1;
    end else begin
        m_axi_gmem0_RREADY = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter5 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_11001))) begin
        m_axi_gmem0_WVALID = 1'b1;
    end else begin
        m_axi_gmem0_WVALID = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_pp0_stage0 : begin
            if ((~((ap_start_int == 1'b0) & (ap_done_pending_pp0 == 1'b0) & (ap_idle_pp0_1to7 == 1'b1)) & (1'b0 == ap_block_pp0_stage0_subdone))) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end
        end
        ap_ST_fsm_pp0_stage1 : begin
            if (((ap_idle_pp0_0to5 == 1'b1) & (1'b1 == ap_condition_exit_pp0_iter6_stage1))) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end else if ((1'b0 == ap_block_pp0_stage1_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage1;
            end
        end
        ap_ST_fsm_pp0_stage2 : begin
            if ((1'b0 == ap_block_pp0_stage2_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage2;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign add_ln507_1_fu_256_p2 = (f_out_8_reg_459 + 32'd1);

assign add_ln514_1_fu_291_p2 = (zext_ln514_fu_287_p1 + bn_weight_mem);

assign add_ln514_2_fu_269_p2 = (add_ln507 + phi_mul_fu_82);

assign add_ln514_fu_274_p2 = (add_ln514_2_fu_269_p2 + bn_weight_base);

assign add_ln516_1_fu_344_p2 = (f_out_8_reg_459 + base_addr_out);

assign add_ln516_2_fu_360_p2 = (zext_ln516_fu_356_p1 + act_mem);

assign add_ln516_fu_329_p2 = ($signed(zext_ln507_fu_326_p1) + $signed(trunc_ln507_cast_cast_reg_449));

assign ap_CS_fsm_pp0_stage0 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_pp0_stage1 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_pp0_stage2 = ap_CS_fsm[32'd2];

assign ap_block_pp0_stage0 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_pp0_stage0_11001 = (((m_axi_gmem0_RVALID == 1'b0) & (ap_enable_reg_pp0_iter3 == 1'b1) & (ap_predicate_op97_read_state10 == 1'b1)) | ((m_axi_gmem0_ARREADY == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b1)));
end

always @ (*) begin
    ap_block_pp0_stage0_subdone = (((m_axi_gmem0_RVALID == 1'b0) & (ap_enable_reg_pp0_iter3 == 1'b1) & (ap_predicate_op97_read_state10 == 1'b1)) | ((m_axi_gmem0_ARREADY == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b1)));
end

assign ap_block_pp0_stage1 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_pp0_stage1_00001 = (((ap_enable_reg_pp0_iter7 == 1'b1) & (m_axi_gmem0_BVALID == 1'b0)) | ((m_axi_gmem0_RVALID == 1'b0) & (ap_enable_reg_pp0_iter3 == 1'b1)));
end

always @ (*) begin
    ap_block_pp0_stage1_11001 = (((m_axi_gmem0_AWREADY == 1'b0) & (ap_enable_reg_pp0_iter5 == 1'b1)) | ((ap_enable_reg_pp0_iter7 == 1'b1) & (m_axi_gmem0_BVALID == 1'b0)) | ((m_axi_gmem0_RVALID == 1'b0) & (ap_enable_reg_pp0_iter3 == 1'b1)) | ((ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_block_state2_io)));
end

always @ (*) begin
    ap_block_pp0_stage1_subdone = (((m_axi_gmem0_AWREADY == 1'b0) & (ap_enable_reg_pp0_iter5 == 1'b1)) | ((ap_enable_reg_pp0_iter7 == 1'b1) & (m_axi_gmem0_BVALID == 1'b0)) | ((m_axi_gmem0_RVALID == 1'b0) & (ap_enable_reg_pp0_iter3 == 1'b1)) | ((ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_block_state2_io)));
end

assign ap_block_pp0_stage2 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_pp0_stage2_01001 = ((m_axi_gmem0_RVALID == 1'b0) & (ap_enable_reg_pp0_iter2 == 1'b1) & (ap_predicate_op94_read_state9 == 1'b1));
end

always @ (*) begin
    ap_block_pp0_stage2_11001 = (((m_axi_gmem0_WREADY == 1'b0) & (ap_enable_reg_pp0_iter5 == 1'b1)) | ((m_axi_gmem0_RVALID == 1'b0) & (ap_enable_reg_pp0_iter2 == 1'b1) & (ap_predicate_op94_read_state9 == 1'b1)) | ((ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_block_state3_io)));
end

always @ (*) begin
    ap_block_pp0_stage2_subdone = (((m_axi_gmem0_WREADY == 1'b0) & (ap_enable_reg_pp0_iter5 == 1'b1)) | ((m_axi_gmem0_RVALID == 1'b0) & (ap_enable_reg_pp0_iter2 == 1'b1) & (ap_predicate_op94_read_state9 == 1'b1)) | ((ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_block_state3_io)));
end

always @ (*) begin
    ap_block_state10_pp0_stage0_iter3 = ((m_axi_gmem0_RVALID == 1'b0) & (ap_predicate_op97_read_state10 == 1'b1));
end

always @ (*) begin
    ap_block_state11_pp0_stage1_iter3 = (m_axi_gmem0_RVALID == 1'b0);
end

assign ap_block_state12_pp0_stage2_iter3 = ~(1'b1 == 1'b1);

assign ap_block_state13_pp0_stage0_iter4 = ~(1'b1 == 1'b1);

assign ap_block_state14_pp0_stage1_iter4 = ~(1'b1 == 1'b1);

assign ap_block_state15_pp0_stage2_iter4 = ~(1'b1 == 1'b1);

assign ap_block_state16_pp0_stage0_iter5 = ~(1'b1 == 1'b1);

assign ap_block_state17_pp0_stage1_iter5 = ~(1'b1 == 1'b1);

assign ap_block_state18_pp0_stage2_iter5 = ~(1'b1 == 1'b1);

assign ap_block_state19_pp0_stage0_iter6 = ~(1'b1 == 1'b1);

assign ap_block_state1_pp0_stage0_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state20_pp0_stage1_iter6 = ~(1'b1 == 1'b1);

assign ap_block_state21_pp0_stage2_iter6 = ~(1'b1 == 1'b1);

assign ap_block_state22_pp0_stage0_iter7 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_state23_pp0_stage1_iter7 = (m_axi_gmem0_BVALID == 1'b0);
end

always @ (*) begin
    ap_block_state2_io = ((m_axi_gmem0_ARREADY == 1'b0) & (ap_predicate_op53_readreq_state2 == 1'b1));
end

assign ap_block_state2_pp0_stage1_iter0 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_state3_io = ((m_axi_gmem0_ARREADY == 1'b0) & (ap_predicate_op68_readreq_state3 == 1'b1));
end

assign ap_block_state3_pp0_stage2_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state4_pp0_stage0_iter1 = ~(1'b1 == 1'b1);

assign ap_block_state5_pp0_stage1_iter1 = ~(1'b1 == 1'b1);

assign ap_block_state6_pp0_stage2_iter1 = ~(1'b1 == 1'b1);

assign ap_block_state7_pp0_stage0_iter2 = ~(1'b1 == 1'b1);

assign ap_block_state8_pp0_stage1_iter2 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_state9_pp0_stage2_iter2 = ((m_axi_gmem0_RVALID == 1'b0) & (ap_predicate_op94_read_state9 == 1'b1));
end

assign ap_enable_pp0 = (ap_idle_pp0 ^ 1'b1);

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage2;

always @ (*) begin
    ap_predicate_op53_readreq_state2 = ((cmp21_i83 == 1'd0) & (icmp_ln507_reg_466 == 1'd0));
end

always @ (*) begin
    ap_predicate_op68_readreq_state3 = ((cmp21_i83 == 1'd0) & (icmp_ln507_reg_466 == 1'd0));
end

always @ (*) begin
    ap_predicate_op94_read_state9 = ((icmp_ln507_reg_466_pp0_iter2_reg == 1'd0) & (cmp21_i83 == 1'd0));
end

always @ (*) begin
    ap_predicate_op97_read_state10 = ((icmp_ln507_reg_466_pp0_iter2_reg == 1'd0) & (cmp21_i83 == 1'd0));
end

assign bitcast_ln514_1_fu_389_p1 = gmem0_addr_12_read_reg_499;

assign bitcast_ln514_fu_385_p1 = gmem0_addr_read_reg_494;

assign bitcast_ln516_fu_393_p1 = gmem0_addr_13_read_reg_514;

assign grp_fu_2949_p_ce = grp_fu_219_ce;

assign grp_fu_2949_p_din0 = bitcast_ln514_fu_385_p1;

assign grp_fu_2949_p_din1 = bitcast_ln514_1_fu_389_p1;

assign grp_fu_2953_p_ce = grp_fu_214_ce;

assign grp_fu_2953_p_din0 = ap_phi_mux_sum_0_lcssa_i_phi_fu_206_p4;

assign grp_fu_2953_p_din1 = bitcast_ln516_fu_393_p1;

assign grp_fu_2953_p_opcode = 2'd0;

assign icmp_ln507_fu_244_p2 = ((ap_sig_allocacmp_f_out_8 == nof) ? 1'b1 : 1'b0);

assign m_axi_gmem0_ARBURST = 2'd0;

assign m_axi_gmem0_ARCACHE = 4'd0;

assign m_axi_gmem0_ARID = 1'd0;

assign m_axi_gmem0_ARLEN = 32'd1;

assign m_axi_gmem0_ARLOCK = 2'd0;

assign m_axi_gmem0_ARPROT = 3'd0;

assign m_axi_gmem0_ARQOS = 4'd0;

assign m_axi_gmem0_ARREGION = 4'd0;

assign m_axi_gmem0_ARSIZE = 3'd0;

assign m_axi_gmem0_ARUSER = 1'd0;

assign m_axi_gmem0_AWADDR = gmem0_addr_14_reg_488_pp0_iter4_reg;

assign m_axi_gmem0_AWBURST = 2'd0;

assign m_axi_gmem0_AWCACHE = 4'd0;

assign m_axi_gmem0_AWID = 1'd0;

assign m_axi_gmem0_AWLEN = 32'd1;

assign m_axi_gmem0_AWLOCK = 2'd0;

assign m_axi_gmem0_AWPROT = 3'd0;

assign m_axi_gmem0_AWQOS = 4'd0;

assign m_axi_gmem0_AWREGION = 4'd0;

assign m_axi_gmem0_AWSIZE = 3'd0;

assign m_axi_gmem0_AWUSER = 1'd0;

assign m_axi_gmem0_WDATA = add14_i1_reg_529;

assign m_axi_gmem0_WID = 1'd0;

assign m_axi_gmem0_WLAST = 1'b0;

assign m_axi_gmem0_WSTRB = 4'd15;

assign m_axi_gmem0_WUSER = 1'd0;

assign next_mul_fu_264_p2 = (phi_mul_fu_82 + nif);

assign sext_ln514_1_fu_306_p1 = $signed(trunc_ln514_1_fu_296_p4);

assign sext_ln514_cast_fu_227_p1 = $signed(sext_ln514);

assign sext_ln516_1_fu_334_p1 = $signed(add_ln516_fu_329_p2);

assign sext_ln516_fu_375_p1 = $signed(trunc_ln_fu_365_p4);

assign shl_ln7_fu_279_p3 = {{add_ln514_fu_274_p2}, {2'd0}};

assign shl_ln8_fu_348_p3 = {{add_ln516_1_fu_344_p2}, {2'd0}};

assign trunc_ln507_cast_cast_fu_223_p1 = $signed(trunc_ln507_cast);

assign trunc_ln514_1_fu_296_p4 = {{add_ln514_1_fu_291_p2[63:2]}};

assign trunc_ln_fu_365_p4 = {{add_ln516_2_fu_360_p2[63:2]}};

assign zext_ln507_fu_326_p1 = f_out_8_reg_459;

assign zext_ln514_fu_287_p1 = shl_ln7_fu_279_p3;

assign zext_ln516_fu_356_p1 = shl_ln8_fu_348_p3;

endmodule //conv_kernel_conv_kernel_Pipeline_fc_loop1
