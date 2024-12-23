// ==============================================================
// RTL generated by Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Version: 2022.1
// Copyright (C) Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module conv_kernel_PE_Pipeline_pe_loop20_pe_loop21 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        pe_out_fifo7_din,
        pe_out_fifo7_full_n,
        pe_out_fifo7_write,
        mac_vals_98,
        mac_vals_99,
        mac_vals_100,
        mac_vals_101,
        mac_vals_102,
        mac_vals_103,
        mac_vals_104,
        mac_vals_105,
        mac_vals_106,
        mac_vals_107,
        mac_vals_108,
        mac_vals_109,
        mac_vals_110,
        mac_vals_111,
        mac_vals_112,
        mac_vals_113,
        mac_vals_114,
        mac_vals_115,
        mac_vals_116,
        mac_vals_117,
        mac_vals_118,
        mac_vals_119,
        mac_vals_120,
        mac_vals_121,
        mac_vals_122,
        mac_vals_123,
        mac_vals_124,
        mac_vals_125,
        mac_vals_126,
        mac_vals_127,
        mac_vals_128,
        mac_vals_129,
        mac_vals_130,
        mac_vals_131,
        mac_vals_132,
        mac_vals_133,
        mac_vals_134,
        mac_vals_135,
        mac_vals_136,
        mac_vals_137,
        mac_vals_138,
        mac_vals_139,
        mac_vals_140,
        mac_vals_141,
        mac_vals_142,
        mac_vals_143,
        mac_vals_144,
        mac_vals_145,
        mac_vals_146
);

parameter    ap_ST_fsm_pp0_stage0 = 1'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
output  [31:0] pe_out_fifo7_din;
input   pe_out_fifo7_full_n;
output   pe_out_fifo7_write;
input  [31:0] mac_vals_98;
input  [31:0] mac_vals_99;
input  [31:0] mac_vals_100;
input  [31:0] mac_vals_101;
input  [31:0] mac_vals_102;
input  [31:0] mac_vals_103;
input  [31:0] mac_vals_104;
input  [31:0] mac_vals_105;
input  [31:0] mac_vals_106;
input  [31:0] mac_vals_107;
input  [31:0] mac_vals_108;
input  [31:0] mac_vals_109;
input  [31:0] mac_vals_110;
input  [31:0] mac_vals_111;
input  [31:0] mac_vals_112;
input  [31:0] mac_vals_113;
input  [31:0] mac_vals_114;
input  [31:0] mac_vals_115;
input  [31:0] mac_vals_116;
input  [31:0] mac_vals_117;
input  [31:0] mac_vals_118;
input  [31:0] mac_vals_119;
input  [31:0] mac_vals_120;
input  [31:0] mac_vals_121;
input  [31:0] mac_vals_122;
input  [31:0] mac_vals_123;
input  [31:0] mac_vals_124;
input  [31:0] mac_vals_125;
input  [31:0] mac_vals_126;
input  [31:0] mac_vals_127;
input  [31:0] mac_vals_128;
input  [31:0] mac_vals_129;
input  [31:0] mac_vals_130;
input  [31:0] mac_vals_131;
input  [31:0] mac_vals_132;
input  [31:0] mac_vals_133;
input  [31:0] mac_vals_134;
input  [31:0] mac_vals_135;
input  [31:0] mac_vals_136;
input  [31:0] mac_vals_137;
input  [31:0] mac_vals_138;
input  [31:0] mac_vals_139;
input  [31:0] mac_vals_140;
input  [31:0] mac_vals_141;
input  [31:0] mac_vals_142;
input  [31:0] mac_vals_143;
input  [31:0] mac_vals_144;
input  [31:0] mac_vals_145;
input  [31:0] mac_vals_146;

reg ap_idle;
reg pe_out_fifo7_write;

(* fsm_encoding = "none" *) reg   [0:0] ap_CS_fsm;
wire    ap_CS_fsm_pp0_stage0;
wire    ap_enable_reg_pp0_iter0;
reg    ap_enable_reg_pp0_iter1;
reg    ap_idle_pp0;
wire    ap_block_state1_pp0_stage0_iter0;
reg    ap_block_state2_pp0_stage0_iter1;
reg    ap_block_pp0_stage0_subdone;
wire   [0:0] icmp_ln243_fu_475_p2;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    pe_out_fifo7_blk_n;
wire    ap_block_pp0_stage0;
wire   [31:0] tmp_s_fu_661_p9;
reg   [31:0] tmp_s_reg_730;
reg    ap_block_pp0_stage0_11001;
reg   [2:0] x_fu_144;
wire   [2:0] add_ln245_fu_681_p2;
wire    ap_loop_init;
reg   [2:0] ap_sig_allocacmp_x_load;
reg   [2:0] y_8_fu_148;
wire   [2:0] select_ln243_1_fu_513_p3;
reg   [2:0] ap_sig_allocacmp_y_8_load;
reg   [5:0] indvar_flatten357_fu_152;
wire   [5:0] add_ln243_fu_481_p2;
reg   [5:0] ap_sig_allocacmp_indvar_flatten357_load;
reg    ap_block_pp0_stage0_01001;
wire   [0:0] icmp_ln245_fu_493_p2;
wire   [2:0] add_ln243_1_fu_507_p2;
wire   [2:0] select_ln243_fu_499_p3;
wire   [31:0] tmp_3_fu_521_p9;
wire   [31:0] tmp_4_fu_541_p9;
wire   [31:0] tmp_5_fu_561_p9;
wire   [31:0] tmp_6_fu_581_p9;
wire   [31:0] tmp_7_fu_601_p9;
wire   [31:0] tmp_8_fu_621_p9;
wire   [31:0] tmp_9_fu_641_p9;
wire   [2:0] tmp_s_fu_661_p8;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg   [0:0] ap_NS_fsm;
wire    ap_enable_pp0;
wire    ap_start_int;
wire    ap_ce_reg;

// power-on initialization
initial begin
#0 ap_CS_fsm = 1'd1;
#0 ap_enable_reg_pp0_iter1 = 1'b0;
#0 ap_done_reg = 1'b0;
end

conv_kernel_mux_73_32_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 32 ),
    .din1_WIDTH( 32 ),
    .din2_WIDTH( 32 ),
    .din3_WIDTH( 32 ),
    .din4_WIDTH( 32 ),
    .din5_WIDTH( 32 ),
    .din6_WIDTH( 32 ),
    .din7_WIDTH( 3 ),
    .dout_WIDTH( 32 ))
mux_73_32_1_1_U205(
    .din0(mac_vals_98),
    .din1(mac_vals_99),
    .din2(mac_vals_100),
    .din3(mac_vals_101),
    .din4(mac_vals_102),
    .din5(mac_vals_103),
    .din6(mac_vals_104),
    .din7(select_ln243_fu_499_p3),
    .dout(tmp_3_fu_521_p9)
);

conv_kernel_mux_73_32_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 32 ),
    .din1_WIDTH( 32 ),
    .din2_WIDTH( 32 ),
    .din3_WIDTH( 32 ),
    .din4_WIDTH( 32 ),
    .din5_WIDTH( 32 ),
    .din6_WIDTH( 32 ),
    .din7_WIDTH( 3 ),
    .dout_WIDTH( 32 ))
mux_73_32_1_1_U206(
    .din0(mac_vals_105),
    .din1(mac_vals_106),
    .din2(mac_vals_107),
    .din3(mac_vals_108),
    .din4(mac_vals_109),
    .din5(mac_vals_110),
    .din6(mac_vals_111),
    .din7(select_ln243_fu_499_p3),
    .dout(tmp_4_fu_541_p9)
);

conv_kernel_mux_73_32_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 32 ),
    .din1_WIDTH( 32 ),
    .din2_WIDTH( 32 ),
    .din3_WIDTH( 32 ),
    .din4_WIDTH( 32 ),
    .din5_WIDTH( 32 ),
    .din6_WIDTH( 32 ),
    .din7_WIDTH( 3 ),
    .dout_WIDTH( 32 ))
mux_73_32_1_1_U207(
    .din0(mac_vals_112),
    .din1(mac_vals_113),
    .din2(mac_vals_114),
    .din3(mac_vals_115),
    .din4(mac_vals_116),
    .din5(mac_vals_117),
    .din6(mac_vals_118),
    .din7(select_ln243_fu_499_p3),
    .dout(tmp_5_fu_561_p9)
);

conv_kernel_mux_73_32_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 32 ),
    .din1_WIDTH( 32 ),
    .din2_WIDTH( 32 ),
    .din3_WIDTH( 32 ),
    .din4_WIDTH( 32 ),
    .din5_WIDTH( 32 ),
    .din6_WIDTH( 32 ),
    .din7_WIDTH( 3 ),
    .dout_WIDTH( 32 ))
mux_73_32_1_1_U208(
    .din0(mac_vals_119),
    .din1(mac_vals_120),
    .din2(mac_vals_121),
    .din3(mac_vals_122),
    .din4(mac_vals_123),
    .din5(mac_vals_124),
    .din6(mac_vals_125),
    .din7(select_ln243_fu_499_p3),
    .dout(tmp_6_fu_581_p9)
);

conv_kernel_mux_73_32_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 32 ),
    .din1_WIDTH( 32 ),
    .din2_WIDTH( 32 ),
    .din3_WIDTH( 32 ),
    .din4_WIDTH( 32 ),
    .din5_WIDTH( 32 ),
    .din6_WIDTH( 32 ),
    .din7_WIDTH( 3 ),
    .dout_WIDTH( 32 ))
mux_73_32_1_1_U209(
    .din0(mac_vals_126),
    .din1(mac_vals_127),
    .din2(mac_vals_128),
    .din3(mac_vals_129),
    .din4(mac_vals_130),
    .din5(mac_vals_131),
    .din6(mac_vals_132),
    .din7(select_ln243_fu_499_p3),
    .dout(tmp_7_fu_601_p9)
);

conv_kernel_mux_73_32_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 32 ),
    .din1_WIDTH( 32 ),
    .din2_WIDTH( 32 ),
    .din3_WIDTH( 32 ),
    .din4_WIDTH( 32 ),
    .din5_WIDTH( 32 ),
    .din6_WIDTH( 32 ),
    .din7_WIDTH( 3 ),
    .dout_WIDTH( 32 ))
mux_73_32_1_1_U210(
    .din0(mac_vals_133),
    .din1(mac_vals_134),
    .din2(mac_vals_135),
    .din3(mac_vals_136),
    .din4(mac_vals_137),
    .din5(mac_vals_138),
    .din6(mac_vals_139),
    .din7(select_ln243_fu_499_p3),
    .dout(tmp_8_fu_621_p9)
);

conv_kernel_mux_73_32_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 32 ),
    .din1_WIDTH( 32 ),
    .din2_WIDTH( 32 ),
    .din3_WIDTH( 32 ),
    .din4_WIDTH( 32 ),
    .din5_WIDTH( 32 ),
    .din6_WIDTH( 32 ),
    .din7_WIDTH( 3 ),
    .dout_WIDTH( 32 ))
mux_73_32_1_1_U211(
    .din0(mac_vals_140),
    .din1(mac_vals_141),
    .din2(mac_vals_142),
    .din3(mac_vals_143),
    .din4(mac_vals_144),
    .din5(mac_vals_145),
    .din6(mac_vals_146),
    .din7(select_ln243_fu_499_p3),
    .dout(tmp_9_fu_641_p9)
);

conv_kernel_mux_73_32_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 32 ),
    .din1_WIDTH( 32 ),
    .din2_WIDTH( 32 ),
    .din3_WIDTH( 32 ),
    .din4_WIDTH( 32 ),
    .din5_WIDTH( 32 ),
    .din6_WIDTH( 32 ),
    .din7_WIDTH( 3 ),
    .dout_WIDTH( 32 ))
mux_73_32_1_1_U212(
    .din0(tmp_3_fu_521_p9),
    .din1(tmp_4_fu_541_p9),
    .din2(tmp_5_fu_561_p9),
    .din3(tmp_6_fu_581_p9),
    .din4(tmp_7_fu_601_p9),
    .din5(tmp_8_fu_621_p9),
    .din6(tmp_9_fu_641_p9),
    .din7(tmp_s_fu_661_p8),
    .dout(tmp_s_fu_661_p9)
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
    .ap_loop_exit_ready(ap_condition_exit_pp0_iter0_stage0),
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
        end else if (((ap_loop_exit_ready == 1'b1) & (1'b0 == ap_block_pp0_stage0_subdone) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter1 <= 1'b0;
    end else begin
        if ((1'b1 == ap_condition_exit_pp0_iter0_stage0)) begin
            ap_enable_reg_pp0_iter1 <= 1'b0;
        end else if (((1'b0 == ap_block_pp0_stage0_subdone) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
            ap_enable_reg_pp0_iter1 <= ap_start_int;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if (((icmp_ln243_fu_475_p2 == 1'd0) & (ap_enable_reg_pp0_iter0 == 1'b1))) begin
            indvar_flatten357_fu_152 <= add_ln243_fu_481_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten357_fu_152 <= 6'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if (((icmp_ln243_fu_475_p2 == 1'd0) & (ap_enable_reg_pp0_iter0 == 1'b1))) begin
            x_fu_144 <= add_ln245_fu_681_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            x_fu_144 <= 3'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if (((icmp_ln243_fu_475_p2 == 1'd0) & (ap_enable_reg_pp0_iter0 == 1'b1))) begin
            y_8_fu_148 <= select_ln243_1_fu_513_p3;
        end else if ((ap_loop_init == 1'b1)) begin
            y_8_fu_148 <= 3'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln243_fu_475_p2 == 1'd0) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        tmp_s_reg_730 <= tmp_s_fu_661_p9;
    end
end

always @ (*) begin
    if (((icmp_ln243_fu_475_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_subdone) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if (((ap_loop_exit_ready == 1'b1) & (1'b0 == ap_block_pp0_stage0_subdone) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
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
    if (((ap_enable_reg_pp0_iter1 == 1'b0) & (ap_enable_reg_pp0_iter0 == 1'b0))) begin
        ap_idle_pp0 = 1'b1;
    end else begin
        ap_idle_pp0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_subdone) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0) & (1'b1 == ap_CS_fsm_pp0_stage0) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten357_load = 6'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten357_load = indvar_flatten357_fu_152;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0) & (1'b1 == ap_CS_fsm_pp0_stage0) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_x_load = 3'd0;
    end else begin
        ap_sig_allocacmp_x_load = x_fu_144;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0) & (1'b1 == ap_CS_fsm_pp0_stage0) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_y_8_load = 3'd0;
    end else begin
        ap_sig_allocacmp_y_8_load = y_8_fu_148;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        pe_out_fifo7_blk_n = pe_out_fifo7_full_n;
    end else begin
        pe_out_fifo7_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        pe_out_fifo7_write = 1'b1;
    end else begin
        pe_out_fifo7_write = 1'b0;
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

assign add_ln243_1_fu_507_p2 = (ap_sig_allocacmp_y_8_load + 3'd1);

assign add_ln243_fu_481_p2 = (ap_sig_allocacmp_indvar_flatten357_load + 6'd1);

assign add_ln245_fu_681_p2 = (select_ln243_fu_499_p3 + 3'd1);

assign ap_CS_fsm_pp0_stage0 = ap_CS_fsm[32'd0];

assign ap_block_pp0_stage0 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_pp0_stage0_01001 = ((pe_out_fifo7_full_n == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b1));
end

always @ (*) begin
    ap_block_pp0_stage0_11001 = ((pe_out_fifo7_full_n == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b1));
end

always @ (*) begin
    ap_block_pp0_stage0_subdone = ((pe_out_fifo7_full_n == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b1));
end

assign ap_block_state1_pp0_stage0_iter0 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_state2_pp0_stage0_iter1 = (pe_out_fifo7_full_n == 1'b0);
end

assign ap_enable_pp0 = (ap_idle_pp0 ^ 1'b1);

assign ap_enable_reg_pp0_iter0 = ap_start_int;

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign icmp_ln243_fu_475_p2 = ((ap_sig_allocacmp_indvar_flatten357_load == 6'd49) ? 1'b1 : 1'b0);

assign icmp_ln245_fu_493_p2 = ((ap_sig_allocacmp_x_load == 3'd7) ? 1'b1 : 1'b0);

assign pe_out_fifo7_din = tmp_s_reg_730;

assign select_ln243_1_fu_513_p3 = ((icmp_ln245_fu_493_p2[0:0] == 1'b1) ? add_ln243_1_fu_507_p2 : ap_sig_allocacmp_y_8_load);

assign select_ln243_fu_499_p3 = ((icmp_ln245_fu_493_p2[0:0] == 1'b1) ? 3'd0 : ap_sig_allocacmp_x_load);

assign tmp_s_fu_661_p8 = ((icmp_ln245_fu_493_p2[0:0] == 1'b1) ? add_ln243_1_fu_507_p2 : ap_sig_allocacmp_y_8_load);

endmodule //conv_kernel_PE_Pipeline_pe_loop20_pe_loop21
