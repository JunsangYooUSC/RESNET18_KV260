// ==============================================================
// RTL generated by Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Version: 2022.1
// Copyright (C) Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module conv_kernel_load_input_Pipeline_load_input_loop14_load_input_loop15 (
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
        load_input_fifo5_din,
        load_input_fifo5_full_n,
        load_input_fifo5_write,
        y0,
        y_2,
        pad,
        add97,
        bound158,
        nkx,
        sub,
        mul58,
        x0,
        x_1,
        add107,
        sub_ln92,
        act_mem
);

parameter    ap_ST_fsm_pp0_stage0 = 1'd1;

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
output  [31:0] load_input_fifo5_din;
input   load_input_fifo5_full_n;
output   load_input_fifo5_write;
input  [31:0] y0;
input  [31:0] y_2;
input  [31:0] pad;
input  [31:0] add97;
input  [63:0] bound158;
input  [31:0] nkx;
input  [31:0] sub;
input  [31:0] mul58;
input  [31:0] x0;
input  [31:0] x_1;
input  [31:0] add107;
input  [31:0] sub_ln92;
input  [63:0] act_mem;

reg ap_idle;
reg m_axi_gmem0_ARVALID;
reg m_axi_gmem0_RREADY;
reg load_input_fifo5_write;

(* fsm_encoding = "none" *) reg   [0:0] ap_CS_fsm;
wire    ap_CS_fsm_pp0_stage0;
wire    ap_enable_reg_pp0_iter0;
reg    ap_enable_reg_pp0_iter1;
reg    ap_enable_reg_pp0_iter2;
reg    ap_enable_reg_pp0_iter3;
reg    ap_enable_reg_pp0_iter4;
reg    ap_enable_reg_pp0_iter5;
reg    ap_enable_reg_pp0_iter6;
reg    ap_enable_reg_pp0_iter7;
reg    ap_enable_reg_pp0_iter8;
reg    ap_enable_reg_pp0_iter9;
reg    ap_enable_reg_pp0_iter10;
reg    ap_enable_reg_pp0_iter11;
reg    ap_idle_pp0;
wire    ap_block_state1_pp0_stage0_iter0;
wire    ap_block_state2_pp0_stage0_iter1;
wire    ap_block_state3_pp0_stage0_iter2;
wire    ap_block_state4_pp0_stage0_iter3;
reg   [0:0] icmp_ln82_reg_556;
reg   [0:0] icmp_ln82_reg_556_pp0_iter2_reg;
reg   [0:0] or_ln87_1_reg_560;
reg   [0:0] or_ln87_1_reg_560_pp0_iter2_reg;
reg    ap_predicate_op85_readreq_state4;
reg    ap_block_state4_io;
wire    ap_block_state5_pp0_stage0_iter4;
wire    ap_block_state6_pp0_stage0_iter5;
wire    ap_block_state7_pp0_stage0_iter6;
wire    ap_block_state8_pp0_stage0_iter7;
wire    ap_block_state9_pp0_stage0_iter8;
wire    ap_block_state10_pp0_stage0_iter9;
reg   [0:0] icmp_ln82_reg_556_pp0_iter9_reg;
reg   [0:0] or_ln87_1_reg_560_pp0_iter9_reg;
reg    ap_predicate_op92_read_state11;
reg    ap_block_state11_pp0_stage0_iter10;
reg    ap_block_state12_pp0_stage0_iter11;
reg    ap_block_pp0_stage0_subdone;
wire   [0:0] icmp_ln82_fu_266_p2;
reg    ap_condition_exit_pp0_iter1_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    load_input_fifo5_blk_n;
wire    ap_block_pp0_stage0;
reg    gmem0_blk_n_AR;
reg    gmem0_blk_n_R;
reg    ap_block_pp0_stage0_11001;
reg   [0:0] icmp_ln82_reg_556_pp0_iter3_reg;
reg   [0:0] icmp_ln82_reg_556_pp0_iter4_reg;
reg   [0:0] icmp_ln82_reg_556_pp0_iter5_reg;
reg   [0:0] icmp_ln82_reg_556_pp0_iter6_reg;
reg   [0:0] icmp_ln82_reg_556_pp0_iter7_reg;
reg   [0:0] icmp_ln82_reg_556_pp0_iter8_reg;
reg   [0:0] icmp_ln82_reg_556_pp0_iter10_reg;
wire   [0:0] or_ln87_1_fu_391_p2;
reg   [0:0] or_ln87_1_reg_560_pp0_iter3_reg;
reg   [0:0] or_ln87_1_reg_560_pp0_iter4_reg;
reg   [0:0] or_ln87_1_reg_560_pp0_iter5_reg;
reg   [0:0] or_ln87_1_reg_560_pp0_iter6_reg;
reg   [0:0] or_ln87_1_reg_560_pp0_iter7_reg;
reg   [0:0] or_ln87_1_reg_560_pp0_iter8_reg;
reg   [0:0] or_ln87_1_reg_560_pp0_iter10_reg;
wire   [31:0] add_ln93_fu_403_p2;
reg   [31:0] add_ln93_reg_564;
reg   [63:0] gmem0_addr_reg_569;
reg   [31:0] in_val_reg_575;
reg   [31:0] ap_phi_mux_p_0_phi_fu_199_p4;
reg   [31:0] ap_phi_reg_pp0_iter11_p_0_reg_195;
wire   [31:0] ap_phi_reg_pp0_iter0_p_0_reg_195;
reg   [31:0] ap_phi_reg_pp0_iter1_p_0_reg_195;
reg   [31:0] ap_phi_reg_pp0_iter2_p_0_reg_195;
reg   [31:0] ap_phi_reg_pp0_iter3_p_0_reg_195;
reg   [31:0] ap_phi_reg_pp0_iter4_p_0_reg_195;
reg   [31:0] ap_phi_reg_pp0_iter5_p_0_reg_195;
reg   [31:0] ap_phi_reg_pp0_iter6_p_0_reg_195;
reg   [31:0] ap_phi_reg_pp0_iter7_p_0_reg_195;
reg   [31:0] ap_phi_reg_pp0_iter8_p_0_reg_195;
reg   [31:0] ap_phi_reg_pp0_iter9_p_0_reg_195;
reg   [31:0] ap_phi_reg_pp0_iter10_p_0_reg_195;
wire  signed [63:0] sext_ln93_fu_455_p1;
reg   [31:0] j_fu_86;
wire   [31:0] add_ln84_fu_408_p2;
wire    ap_loop_init;
reg   [31:0] i_fu_90;
wire   [31:0] select_ln82_3_fu_351_p3;
reg   [63:0] indvar_flatten160_fu_94;
wire   [63:0] add_ln82_1_fu_271_p2;
reg    ap_block_pp0_stage0_01001;
wire   [31:0] tmp3_fu_228_p2;
wire   [31:0] add92_fu_233_p2;
wire   [0:0] ult_fu_243_p2;
wire   [0:0] cmp93_fu_238_p2;
wire   [0:0] rev18_fu_248_p2;
wire   [0:0] icmp_ln84_fu_280_p2;
wire   [31:0] i_3_fu_260_p2;
wire   [31:0] tmp3_mid1_fu_293_p2;
wire   [31:0] add92_mid1_fu_298_p2;
wire   [0:0] ult19_fu_308_p2;
wire   [31:0] select_ln82_1_fu_319_p3;
wire  signed [31:0] mul_ln82_fu_332_p0;
wire   [0:0] cmp93_mid1_fu_303_p2;
wire   [0:0] rev20_fu_313_p2;
wire   [0:0] or_ln87_3_fu_337_p2;
wire   [0:0] or_ln87_fu_254_p2;
wire   [31:0] add_ln87_1_fu_359_p2;
wire   [31:0] select_ln82_fu_285_p3;
wire   [31:0] add_ln87_fu_363_p2;
wire   [0:0] icmp_ln87_1_fu_374_p2;
wire   [0:0] xor_ln87_fu_379_p2;
wire   [0:0] icmp_ln87_fu_369_p2;
wire   [0:0] or_ln87_2_fu_385_p2;
wire   [0:0] select_ln82_2_fu_343_p3;
wire   [31:0] mul_ln82_fu_332_p2;
wire   [31:0] add_ln93_2_fu_397_p2;
wire   [33:0] shl_ln2_fu_429_p3;
wire   [63:0] zext_ln93_fu_436_p1;
wire   [63:0] add_ln93_1_fu_440_p2;
wire   [61:0] trunc_ln_fu_445_p4;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter2_reg;
reg    ap_loop_exit_ready_pp0_iter3_reg;
reg    ap_loop_exit_ready_pp0_iter4_reg;
reg    ap_loop_exit_ready_pp0_iter5_reg;
reg    ap_loop_exit_ready_pp0_iter6_reg;
reg    ap_loop_exit_ready_pp0_iter7_reg;
reg    ap_loop_exit_ready_pp0_iter8_reg;
reg    ap_loop_exit_ready_pp0_iter9_reg;
reg    ap_loop_exit_ready_pp0_iter10_reg;
reg   [0:0] ap_NS_fsm;
wire    ap_enable_pp0;
wire    ap_start_int;
reg    ap_condition_295;
wire    ap_ce_reg;

// power-on initialization
initial begin
#0 ap_CS_fsm = 1'd1;
#0 ap_enable_reg_pp0_iter1 = 1'b0;
#0 ap_enable_reg_pp0_iter2 = 1'b0;
#0 ap_enable_reg_pp0_iter3 = 1'b0;
#0 ap_enable_reg_pp0_iter4 = 1'b0;
#0 ap_enable_reg_pp0_iter5 = 1'b0;
#0 ap_enable_reg_pp0_iter6 = 1'b0;
#0 ap_enable_reg_pp0_iter7 = 1'b0;
#0 ap_enable_reg_pp0_iter8 = 1'b0;
#0 ap_enable_reg_pp0_iter9 = 1'b0;
#0 ap_enable_reg_pp0_iter10 = 1'b0;
#0 ap_enable_reg_pp0_iter11 = 1'b0;
#0 ap_done_reg = 1'b0;
end

conv_kernel_mul_32s_32s_32_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 32 ),
    .din1_WIDTH( 32 ),
    .dout_WIDTH( 32 ))
mul_32s_32s_32_1_1_U49(
    .din0(mul_ln82_fu_332_p0),
    .din1(mul58),
    .dout(mul_ln82_fu_332_p2)
);

conv_kernel_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(ap_start),
    .ap_ready(ap_ready),
    .ap_done(ap_done),
    .ap_start_int(ap_start_int),
    .ap_loop_init(ap_loop_init),
    .ap_ready_int(ap_ready_int),
    .ap_loop_exit_ready(ap_condition_exit_pp0_iter1_stage0),
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
        end else if (((ap_loop_exit_ready_pp0_iter10_reg == 1'b1) & (1'b0 == ap_block_pp0_stage0_subdone))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter1 <= 1'b0;
    end else begin
        if ((1'b1 == ap_condition_exit_pp0_iter1_stage0)) begin
            ap_enable_reg_pp0_iter1 <= 1'b0;
        end else if (((1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_subdone))) begin
            ap_enable_reg_pp0_iter1 <= ap_start_int;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter10 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter10 <= ap_enable_reg_pp0_iter9;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter11 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter11 <= ap_enable_reg_pp0_iter10;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter2 <= 1'b0;
    end else begin
        if ((1'b1 == ap_condition_exit_pp0_iter1_stage0)) begin
            ap_enable_reg_pp0_iter2 <= 1'b0;
        end else if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter2 <= ap_enable_reg_pp0_iter1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter3 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter3 <= ap_enable_reg_pp0_iter2;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter4 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter4 <= ap_enable_reg_pp0_iter3;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter5 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter5 <= ap_enable_reg_pp0_iter4;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter6 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter6 <= ap_enable_reg_pp0_iter5;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter7 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter7 <= ap_enable_reg_pp0_iter6;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter8 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter8 <= ap_enable_reg_pp0_iter7;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter9 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter9 <= ap_enable_reg_pp0_iter8;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_295)) begin
        if (((or_ln87_1_fu_391_p2 == 1'd1) & (icmp_ln82_fu_266_p2 == 1'd0))) begin
            ap_phi_reg_pp0_iter2_p_0_reg_195 <= 32'd0;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter2_p_0_reg_195 <= ap_phi_reg_pp0_iter1_p_0_reg_195;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        if ((ap_loop_init == 1'b1)) begin
            i_fu_90 <= 32'd0;
        end else if (((ap_enable_reg_pp0_iter1 == 1'b1) & (icmp_ln82_fu_266_p2 == 1'd0))) begin
            i_fu_90 <= select_ln82_3_fu_351_p3;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        if ((ap_loop_init == 1'b1)) begin
            indvar_flatten160_fu_94 <= 64'd0;
        end else if (((ap_enable_reg_pp0_iter1 == 1'b1) & (icmp_ln82_fu_266_p2 == 1'd0))) begin
            indvar_flatten160_fu_94 <= add_ln82_1_fu_271_p2;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        if ((ap_loop_init == 1'b1)) begin
            j_fu_86 <= 32'd0;
        end else if (((ap_enable_reg_pp0_iter1 == 1'b1) & (icmp_ln82_fu_266_p2 == 1'd0))) begin
            j_fu_86 <= add_ln84_fu_408_p2;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((or_ln87_1_fu_391_p2 == 1'd0) & (1'b1 == ap_CS_fsm_pp0_stage0) & (icmp_ln82_fu_266_p2 == 1'd0) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        add_ln93_reg_564 <= add_ln93_fu_403_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b0 == ap_block_pp0_stage0_11001)) begin
        ap_loop_exit_ready_pp0_iter10_reg <= ap_loop_exit_ready_pp0_iter9_reg;
        ap_loop_exit_ready_pp0_iter3_reg <= ap_loop_exit_ready_pp0_iter2_reg;
        ap_loop_exit_ready_pp0_iter4_reg <= ap_loop_exit_ready_pp0_iter3_reg;
        ap_loop_exit_ready_pp0_iter5_reg <= ap_loop_exit_ready_pp0_iter4_reg;
        ap_loop_exit_ready_pp0_iter6_reg <= ap_loop_exit_ready_pp0_iter5_reg;
        ap_loop_exit_ready_pp0_iter7_reg <= ap_loop_exit_ready_pp0_iter6_reg;
        ap_loop_exit_ready_pp0_iter8_reg <= ap_loop_exit_ready_pp0_iter7_reg;
        ap_loop_exit_ready_pp0_iter9_reg <= ap_loop_exit_ready_pp0_iter8_reg;
        icmp_ln82_reg_556_pp0_iter10_reg <= icmp_ln82_reg_556_pp0_iter9_reg;
        icmp_ln82_reg_556_pp0_iter2_reg <= icmp_ln82_reg_556;
        icmp_ln82_reg_556_pp0_iter3_reg <= icmp_ln82_reg_556_pp0_iter2_reg;
        icmp_ln82_reg_556_pp0_iter4_reg <= icmp_ln82_reg_556_pp0_iter3_reg;
        icmp_ln82_reg_556_pp0_iter5_reg <= icmp_ln82_reg_556_pp0_iter4_reg;
        icmp_ln82_reg_556_pp0_iter6_reg <= icmp_ln82_reg_556_pp0_iter5_reg;
        icmp_ln82_reg_556_pp0_iter7_reg <= icmp_ln82_reg_556_pp0_iter6_reg;
        icmp_ln82_reg_556_pp0_iter8_reg <= icmp_ln82_reg_556_pp0_iter7_reg;
        icmp_ln82_reg_556_pp0_iter9_reg <= icmp_ln82_reg_556_pp0_iter8_reg;
        or_ln87_1_reg_560_pp0_iter10_reg <= or_ln87_1_reg_560_pp0_iter9_reg;
        or_ln87_1_reg_560_pp0_iter2_reg <= or_ln87_1_reg_560;
        or_ln87_1_reg_560_pp0_iter3_reg <= or_ln87_1_reg_560_pp0_iter2_reg;
        or_ln87_1_reg_560_pp0_iter4_reg <= or_ln87_1_reg_560_pp0_iter3_reg;
        or_ln87_1_reg_560_pp0_iter5_reg <= or_ln87_1_reg_560_pp0_iter4_reg;
        or_ln87_1_reg_560_pp0_iter6_reg <= or_ln87_1_reg_560_pp0_iter5_reg;
        or_ln87_1_reg_560_pp0_iter7_reg <= or_ln87_1_reg_560_pp0_iter6_reg;
        or_ln87_1_reg_560_pp0_iter8_reg <= or_ln87_1_reg_560_pp0_iter7_reg;
        or_ln87_1_reg_560_pp0_iter9_reg <= or_ln87_1_reg_560_pp0_iter8_reg;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready;
        icmp_ln82_reg_556 <= icmp_ln82_fu_266_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((ap_enable_reg_pp0_iter9 == 1'b1) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        ap_phi_reg_pp0_iter10_p_0_reg_195 <= ap_phi_reg_pp0_iter9_p_0_reg_195;
    end
end

always @ (posedge ap_clk) begin
    if (((ap_enable_reg_pp0_iter10 == 1'b1) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        ap_phi_reg_pp0_iter11_p_0_reg_195 <= ap_phi_reg_pp0_iter10_p_0_reg_195;
    end
end

always @ (posedge ap_clk) begin
    if (((ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        ap_phi_reg_pp0_iter1_p_0_reg_195 <= ap_phi_reg_pp0_iter0_p_0_reg_195;
    end
end

always @ (posedge ap_clk) begin
    if (((ap_enable_reg_pp0_iter2 == 1'b1) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        ap_phi_reg_pp0_iter3_p_0_reg_195 <= ap_phi_reg_pp0_iter2_p_0_reg_195;
    end
end

always @ (posedge ap_clk) begin
    if (((ap_enable_reg_pp0_iter3 == 1'b1) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        ap_phi_reg_pp0_iter4_p_0_reg_195 <= ap_phi_reg_pp0_iter3_p_0_reg_195;
    end
end

always @ (posedge ap_clk) begin
    if (((ap_enable_reg_pp0_iter4 == 1'b1) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        ap_phi_reg_pp0_iter5_p_0_reg_195 <= ap_phi_reg_pp0_iter4_p_0_reg_195;
    end
end

always @ (posedge ap_clk) begin
    if (((ap_enable_reg_pp0_iter5 == 1'b1) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        ap_phi_reg_pp0_iter6_p_0_reg_195 <= ap_phi_reg_pp0_iter5_p_0_reg_195;
    end
end

always @ (posedge ap_clk) begin
    if (((ap_enable_reg_pp0_iter6 == 1'b1) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        ap_phi_reg_pp0_iter7_p_0_reg_195 <= ap_phi_reg_pp0_iter6_p_0_reg_195;
    end
end

always @ (posedge ap_clk) begin
    if (((ap_enable_reg_pp0_iter7 == 1'b1) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        ap_phi_reg_pp0_iter8_p_0_reg_195 <= ap_phi_reg_pp0_iter7_p_0_reg_195;
    end
end

always @ (posedge ap_clk) begin
    if (((ap_enable_reg_pp0_iter8 == 1'b1) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        ap_phi_reg_pp0_iter9_p_0_reg_195 <= ap_phi_reg_pp0_iter8_p_0_reg_195;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (or_ln87_1_reg_560 == 1'd0) & (icmp_ln82_reg_556 == 1'd0))) begin
        gmem0_addr_reg_569 <= sext_ln93_fu_455_p1;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (ap_predicate_op92_read_state11 == 1'b1))) begin
        in_val_reg_575 <= m_axi_gmem0_RDATA;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_pp0_stage0) & (icmp_ln82_fu_266_p2 == 1'd0) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        or_ln87_1_reg_560 <= or_ln87_1_fu_391_p2;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (icmp_ln82_fu_266_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_subdone))) begin
        ap_condition_exit_pp0_iter1_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter1_stage0 = 1'b0;
    end
end

always @ (*) begin
    if (((ap_loop_exit_ready_pp0_iter10_reg == 1'b1) & (1'b0 == ap_block_pp0_stage0_subdone))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((ap_idle_pp0 == 1'b1) & (ap_start_int == 1'b0) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter11 == 1'b0) & (ap_enable_reg_pp0_iter10 == 1'b0) & (ap_enable_reg_pp0_iter9 == 1'b0) & (ap_enable_reg_pp0_iter8 == 1'b0) & (ap_enable_reg_pp0_iter7 == 1'b0) & (ap_enable_reg_pp0_iter6 == 1'b0) & (ap_enable_reg_pp0_iter5 == 1'b0) & (ap_enable_reg_pp0_iter4 == 1'b0) & (ap_enable_reg_pp0_iter3 == 1'b0) & (ap_enable_reg_pp0_iter2 == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b0) & (ap_enable_reg_pp0_iter0 == 1'b0))) begin
        ap_idle_pp0 = 1'b1;
    end else begin
        ap_idle_pp0 = 1'b0;
    end
end

always @ (*) begin
    if (((or_ln87_1_reg_560_pp0_iter10_reg == 1'd0) & (icmp_ln82_reg_556_pp0_iter10_reg == 1'd0))) begin
        ap_phi_mux_p_0_phi_fu_199_p4 = in_val_reg_575;
    end else begin
        ap_phi_mux_p_0_phi_fu_199_p4 = ap_phi_reg_pp0_iter11_p_0_reg_195;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_subdone))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter3 == 1'b1) & (1'b0 == ap_block_pp0_stage0) & (ap_predicate_op85_readreq_state4 == 1'b1))) begin
        gmem0_blk_n_AR = m_axi_gmem0_ARREADY;
    end else begin
        gmem0_blk_n_AR = 1'b1;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter10 == 1'b1) & (1'b0 == ap_block_pp0_stage0) & (ap_predicate_op92_read_state11 == 1'b1))) begin
        gmem0_blk_n_R = m_axi_gmem0_RVALID;
    end else begin
        gmem0_blk_n_R = 1'b1;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter11 == 1'b1) & (1'b0 == ap_block_pp0_stage0))) begin
        load_input_fifo5_blk_n = load_input_fifo5_full_n;
    end else begin
        load_input_fifo5_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter11 == 1'b1) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        load_input_fifo5_write = 1'b1;
    end else begin
        load_input_fifo5_write = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter3 == 1'b1) & (1'b0 == ap_block_pp0_stage0_11001) & (ap_predicate_op85_readreq_state4 == 1'b1))) begin
        m_axi_gmem0_ARVALID = 1'b1;
    end else begin
        m_axi_gmem0_ARVALID = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter10 == 1'b1) & (1'b0 == ap_block_pp0_stage0_11001) & (ap_predicate_op92_read_state11 == 1'b1))) begin
        m_axi_gmem0_RREADY = 1'b1;
    end else begin
        m_axi_gmem0_RREADY = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_pp0_stage0 : begin
            ap_NS_fsm = ap_ST_fsm_pp0_stage0;
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign add92_fu_233_p2 = (tmp3_fu_228_p2 + y_2);

assign add92_mid1_fu_298_p2 = (tmp3_mid1_fu_293_p2 + y_2);

assign add_ln82_1_fu_271_p2 = (indvar_flatten160_fu_94 + 64'd1);

assign add_ln84_fu_408_p2 = (select_ln82_fu_285_p3 + 32'd1);

assign add_ln87_1_fu_359_p2 = (x_1 + x0);

assign add_ln87_fu_363_p2 = (add_ln87_1_fu_359_p2 + select_ln82_fu_285_p3);

assign add_ln93_1_fu_440_p2 = (zext_ln93_fu_436_p1 + act_mem);

assign add_ln93_2_fu_397_p2 = (mul_ln82_fu_332_p2 + add_ln87_fu_363_p2);

assign add_ln93_fu_403_p2 = (add_ln93_2_fu_397_p2 + sub_ln92);

assign ap_CS_fsm_pp0_stage0 = ap_CS_fsm[32'd0];

assign ap_block_pp0_stage0 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_pp0_stage0_01001 = (((ap_enable_reg_pp0_iter11 == 1'b1) & (load_input_fifo5_full_n == 1'b0)) | ((ap_enable_reg_pp0_iter10 == 1'b1) & (ap_predicate_op92_read_state11 == 1'b1) & (m_axi_gmem0_RVALID == 1'b0)));
end

always @ (*) begin
    ap_block_pp0_stage0_11001 = (((ap_enable_reg_pp0_iter11 == 1'b1) & (load_input_fifo5_full_n == 1'b0)) | ((ap_enable_reg_pp0_iter10 == 1'b1) & (ap_predicate_op92_read_state11 == 1'b1) & (m_axi_gmem0_RVALID == 1'b0)) | ((ap_enable_reg_pp0_iter3 == 1'b1) & (1'b1 == ap_block_state4_io)));
end

always @ (*) begin
    ap_block_pp0_stage0_subdone = (((ap_enable_reg_pp0_iter11 == 1'b1) & (load_input_fifo5_full_n == 1'b0)) | ((ap_enable_reg_pp0_iter10 == 1'b1) & (ap_predicate_op92_read_state11 == 1'b1) & (m_axi_gmem0_RVALID == 1'b0)) | ((ap_enable_reg_pp0_iter3 == 1'b1) & (1'b1 == ap_block_state4_io)));
end

assign ap_block_state10_pp0_stage0_iter9 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_state11_pp0_stage0_iter10 = ((ap_predicate_op92_read_state11 == 1'b1) & (m_axi_gmem0_RVALID == 1'b0));
end

always @ (*) begin
    ap_block_state12_pp0_stage0_iter11 = (load_input_fifo5_full_n == 1'b0);
end

assign ap_block_state1_pp0_stage0_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state2_pp0_stage0_iter1 = ~(1'b1 == 1'b1);

assign ap_block_state3_pp0_stage0_iter2 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_state4_io = ((m_axi_gmem0_ARREADY == 1'b0) & (ap_predicate_op85_readreq_state4 == 1'b1));
end

assign ap_block_state4_pp0_stage0_iter3 = ~(1'b1 == 1'b1);

assign ap_block_state5_pp0_stage0_iter4 = ~(1'b1 == 1'b1);

assign ap_block_state6_pp0_stage0_iter5 = ~(1'b1 == 1'b1);

assign ap_block_state7_pp0_stage0_iter6 = ~(1'b1 == 1'b1);

assign ap_block_state8_pp0_stage0_iter7 = ~(1'b1 == 1'b1);

assign ap_block_state9_pp0_stage0_iter8 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_condition_295 = ((ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001));
end

assign ap_enable_pp0 = (ap_idle_pp0 ^ 1'b1);

assign ap_enable_reg_pp0_iter0 = ap_start_int;

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter1_stage0;

assign ap_phi_reg_pp0_iter0_p_0_reg_195 = 'bx;

always @ (*) begin
    ap_predicate_op85_readreq_state4 = ((or_ln87_1_reg_560_pp0_iter2_reg == 1'd0) & (icmp_ln82_reg_556_pp0_iter2_reg == 1'd0));
end

always @ (*) begin
    ap_predicate_op92_read_state11 = ((or_ln87_1_reg_560_pp0_iter9_reg == 1'd0) & (icmp_ln82_reg_556_pp0_iter9_reg == 1'd0));
end

assign cmp93_fu_238_p2 = ((add92_fu_233_p2 < pad) ? 1'b1 : 1'b0);

assign cmp93_mid1_fu_303_p2 = ((add92_mid1_fu_298_p2 < pad) ? 1'b1 : 1'b0);

assign i_3_fu_260_p2 = (i_fu_90 + 32'd1);

assign icmp_ln82_fu_266_p2 = ((indvar_flatten160_fu_94 == bound158) ? 1'b1 : 1'b0);

assign icmp_ln84_fu_280_p2 = ((j_fu_86 == nkx) ? 1'b1 : 1'b0);

assign icmp_ln87_1_fu_374_p2 = ((add_ln87_fu_363_p2 < add107) ? 1'b1 : 1'b0);

assign icmp_ln87_fu_369_p2 = ((add_ln87_fu_363_p2 < pad) ? 1'b1 : 1'b0);

assign load_input_fifo5_din = ap_phi_mux_p_0_phi_fu_199_p4;

assign m_axi_gmem0_ARADDR = gmem0_addr_reg_569;

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

assign m_axi_gmem0_AWADDR = 64'd0;

assign m_axi_gmem0_AWBURST = 2'd0;

assign m_axi_gmem0_AWCACHE = 4'd0;

assign m_axi_gmem0_AWID = 1'd0;

assign m_axi_gmem0_AWLEN = 32'd0;

assign m_axi_gmem0_AWLOCK = 2'd0;

assign m_axi_gmem0_AWPROT = 3'd0;

assign m_axi_gmem0_AWQOS = 4'd0;

assign m_axi_gmem0_AWREGION = 4'd0;

assign m_axi_gmem0_AWSIZE = 3'd0;

assign m_axi_gmem0_AWUSER = 1'd0;

assign m_axi_gmem0_AWVALID = 1'b0;

assign m_axi_gmem0_BREADY = 1'b0;

assign m_axi_gmem0_WDATA = 32'd0;

assign m_axi_gmem0_WID = 1'd0;

assign m_axi_gmem0_WLAST = 1'b0;

assign m_axi_gmem0_WSTRB = 4'd0;

assign m_axi_gmem0_WUSER = 1'd0;

assign m_axi_gmem0_WVALID = 1'b0;

assign mul_ln82_fu_332_p0 = (sub + select_ln82_1_fu_319_p3);

assign or_ln87_1_fu_391_p2 = (select_ln82_2_fu_343_p3 | or_ln87_2_fu_385_p2);

assign or_ln87_2_fu_385_p2 = (xor_ln87_fu_379_p2 | icmp_ln87_fu_369_p2);

assign or_ln87_3_fu_337_p2 = (rev20_fu_313_p2 | cmp93_mid1_fu_303_p2);

assign or_ln87_fu_254_p2 = (rev18_fu_248_p2 | cmp93_fu_238_p2);

assign rev18_fu_248_p2 = (ult_fu_243_p2 ^ 1'd1);

assign rev20_fu_313_p2 = (ult19_fu_308_p2 ^ 1'd1);

assign select_ln82_1_fu_319_p3 = ((icmp_ln84_fu_280_p2[0:0] == 1'b1) ? add92_mid1_fu_298_p2 : add92_fu_233_p2);

assign select_ln82_2_fu_343_p3 = ((icmp_ln84_fu_280_p2[0:0] == 1'b1) ? or_ln87_3_fu_337_p2 : or_ln87_fu_254_p2);

assign select_ln82_3_fu_351_p3 = ((icmp_ln84_fu_280_p2[0:0] == 1'b1) ? i_3_fu_260_p2 : i_fu_90);

assign select_ln82_fu_285_p3 = ((icmp_ln84_fu_280_p2[0:0] == 1'b1) ? 32'd0 : j_fu_86);

assign sext_ln93_fu_455_p1 = $signed(trunc_ln_fu_445_p4);

assign shl_ln2_fu_429_p3 = {{add_ln93_reg_564}, {2'd0}};

assign tmp3_fu_228_p2 = (y0 + i_fu_90);

assign tmp3_mid1_fu_293_p2 = (y0 + i_3_fu_260_p2);

assign trunc_ln_fu_445_p4 = {{add_ln93_1_fu_440_p2[63:2]}};

assign ult19_fu_308_p2 = ((add92_mid1_fu_298_p2 < add97) ? 1'b1 : 1'b0);

assign ult_fu_243_p2 = ((add92_fu_233_p2 < add97) ? 1'b1 : 1'b0);

assign xor_ln87_fu_379_p2 = (icmp_ln87_1_fu_374_p2 ^ 1'd1);

assign zext_ln93_fu_436_p1 = shl_ln2_fu_429_p3;

endmodule //conv_kernel_load_input_Pipeline_load_input_loop14_load_input_loop15
