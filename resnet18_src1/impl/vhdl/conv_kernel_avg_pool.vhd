-- ==============================================================
-- RTL generated by Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
-- Version: 2022.1
-- Copyright (C) Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- 
-- ===========================================================

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity conv_kernel_avg_pool is
port (
    ap_clk : IN STD_LOGIC;
    ap_rst : IN STD_LOGIC;
    ap_start : IN STD_LOGIC;
    ap_done : OUT STD_LOGIC;
    ap_idle : OUT STD_LOGIC;
    ap_ready : OUT STD_LOGIC;
    m_axi_gmem0_AWVALID : OUT STD_LOGIC;
    m_axi_gmem0_AWREADY : IN STD_LOGIC;
    m_axi_gmem0_AWADDR : OUT STD_LOGIC_VECTOR (63 downto 0);
    m_axi_gmem0_AWID : OUT STD_LOGIC_VECTOR (0 downto 0);
    m_axi_gmem0_AWLEN : OUT STD_LOGIC_VECTOR (31 downto 0);
    m_axi_gmem0_AWSIZE : OUT STD_LOGIC_VECTOR (2 downto 0);
    m_axi_gmem0_AWBURST : OUT STD_LOGIC_VECTOR (1 downto 0);
    m_axi_gmem0_AWLOCK : OUT STD_LOGIC_VECTOR (1 downto 0);
    m_axi_gmem0_AWCACHE : OUT STD_LOGIC_VECTOR (3 downto 0);
    m_axi_gmem0_AWPROT : OUT STD_LOGIC_VECTOR (2 downto 0);
    m_axi_gmem0_AWQOS : OUT STD_LOGIC_VECTOR (3 downto 0);
    m_axi_gmem0_AWREGION : OUT STD_LOGIC_VECTOR (3 downto 0);
    m_axi_gmem0_AWUSER : OUT STD_LOGIC_VECTOR (0 downto 0);
    m_axi_gmem0_WVALID : OUT STD_LOGIC;
    m_axi_gmem0_WREADY : IN STD_LOGIC;
    m_axi_gmem0_WDATA : OUT STD_LOGIC_VECTOR (31 downto 0);
    m_axi_gmem0_WSTRB : OUT STD_LOGIC_VECTOR (3 downto 0);
    m_axi_gmem0_WLAST : OUT STD_LOGIC;
    m_axi_gmem0_WID : OUT STD_LOGIC_VECTOR (0 downto 0);
    m_axi_gmem0_WUSER : OUT STD_LOGIC_VECTOR (0 downto 0);
    m_axi_gmem0_ARVALID : OUT STD_LOGIC;
    m_axi_gmem0_ARREADY : IN STD_LOGIC;
    m_axi_gmem0_ARADDR : OUT STD_LOGIC_VECTOR (63 downto 0);
    m_axi_gmem0_ARID : OUT STD_LOGIC_VECTOR (0 downto 0);
    m_axi_gmem0_ARLEN : OUT STD_LOGIC_VECTOR (31 downto 0);
    m_axi_gmem0_ARSIZE : OUT STD_LOGIC_VECTOR (2 downto 0);
    m_axi_gmem0_ARBURST : OUT STD_LOGIC_VECTOR (1 downto 0);
    m_axi_gmem0_ARLOCK : OUT STD_LOGIC_VECTOR (1 downto 0);
    m_axi_gmem0_ARCACHE : OUT STD_LOGIC_VECTOR (3 downto 0);
    m_axi_gmem0_ARPROT : OUT STD_LOGIC_VECTOR (2 downto 0);
    m_axi_gmem0_ARQOS : OUT STD_LOGIC_VECTOR (3 downto 0);
    m_axi_gmem0_ARREGION : OUT STD_LOGIC_VECTOR (3 downto 0);
    m_axi_gmem0_ARUSER : OUT STD_LOGIC_VECTOR (0 downto 0);
    m_axi_gmem0_RVALID : IN STD_LOGIC;
    m_axi_gmem0_RREADY : OUT STD_LOGIC;
    m_axi_gmem0_RDATA : IN STD_LOGIC_VECTOR (31 downto 0);
    m_axi_gmem0_RLAST : IN STD_LOGIC;
    m_axi_gmem0_RID : IN STD_LOGIC_VECTOR (0 downto 0);
    m_axi_gmem0_RFIFONUM : IN STD_LOGIC_VECTOR (8 downto 0);
    m_axi_gmem0_RUSER : IN STD_LOGIC_VECTOR (0 downto 0);
    m_axi_gmem0_RRESP : IN STD_LOGIC_VECTOR (1 downto 0);
    m_axi_gmem0_BVALID : IN STD_LOGIC;
    m_axi_gmem0_BREADY : OUT STD_LOGIC;
    m_axi_gmem0_BRESP : IN STD_LOGIC_VECTOR (1 downto 0);
    m_axi_gmem0_BID : IN STD_LOGIC_VECTOR (0 downto 0);
    m_axi_gmem0_BUSER : IN STD_LOGIC_VECTOR (0 downto 0);
    act_mem : IN STD_LOGIC_VECTOR (63 downto 0);
    in_base_addr : IN STD_LOGIC_VECTOR (31 downto 0);
    out_base_addr : IN STD_LOGIC_VECTOR (31 downto 0);
    nif : IN STD_LOGIC_VECTOR (31 downto 0);
    avg_pool_en : IN STD_LOGIC_VECTOR (0 downto 0);
    grp_fu_2953_p_din0 : OUT STD_LOGIC_VECTOR (31 downto 0);
    grp_fu_2953_p_din1 : OUT STD_LOGIC_VECTOR (31 downto 0);
    grp_fu_2953_p_opcode : OUT STD_LOGIC_VECTOR (0 downto 0);
    grp_fu_2953_p_dout0 : IN STD_LOGIC_VECTOR (31 downto 0);
    grp_fu_2953_p_ce : OUT STD_LOGIC );
end;


architecture behav of conv_kernel_avg_pool is 
    constant ap_const_logic_1 : STD_LOGIC := '1';
    constant ap_const_logic_0 : STD_LOGIC := '0';
    constant ap_ST_fsm_state1 : STD_LOGIC_VECTOR (1 downto 0) := "01";
    constant ap_ST_fsm_state2 : STD_LOGIC_VECTOR (1 downto 0) := "10";
    constant ap_const_boolean_1 : BOOLEAN := true;
    constant ap_const_lv32_0 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000000000";
    constant ap_const_lv1_1 : STD_LOGIC_VECTOR (0 downto 0) := "1";
    constant ap_const_lv32_1 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000000001";
    constant ap_const_boolean_0 : BOOLEAN := false;

attribute shreg_extract : string;
    signal ap_CS_fsm : STD_LOGIC_VECTOR (1 downto 0) := "01";
    attribute fsm_encoding : string;
    attribute fsm_encoding of ap_CS_fsm : signal is "none";
    signal ap_CS_fsm_state1 : STD_LOGIC;
    attribute fsm_encoding of ap_CS_fsm_state1 : signal is "none";
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_ap_start : STD_LOGIC;
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_ap_done : STD_LOGIC;
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_ap_idle : STD_LOGIC;
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_ap_ready : STD_LOGIC;
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWVALID : STD_LOGIC;
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWADDR : STD_LOGIC_VECTOR (63 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWID : STD_LOGIC_VECTOR (0 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWLEN : STD_LOGIC_VECTOR (31 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWSIZE : STD_LOGIC_VECTOR (2 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWBURST : STD_LOGIC_VECTOR (1 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWLOCK : STD_LOGIC_VECTOR (1 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWCACHE : STD_LOGIC_VECTOR (3 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWPROT : STD_LOGIC_VECTOR (2 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWQOS : STD_LOGIC_VECTOR (3 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWREGION : STD_LOGIC_VECTOR (3 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWUSER : STD_LOGIC_VECTOR (0 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_WVALID : STD_LOGIC;
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_WDATA : STD_LOGIC_VECTOR (31 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_WSTRB : STD_LOGIC_VECTOR (3 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_WLAST : STD_LOGIC;
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_WID : STD_LOGIC_VECTOR (0 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_WUSER : STD_LOGIC_VECTOR (0 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARVALID : STD_LOGIC;
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARADDR : STD_LOGIC_VECTOR (63 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARID : STD_LOGIC_VECTOR (0 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARLEN : STD_LOGIC_VECTOR (31 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARSIZE : STD_LOGIC_VECTOR (2 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARBURST : STD_LOGIC_VECTOR (1 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARLOCK : STD_LOGIC_VECTOR (1 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARCACHE : STD_LOGIC_VECTOR (3 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARPROT : STD_LOGIC_VECTOR (2 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARQOS : STD_LOGIC_VECTOR (3 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARREGION : STD_LOGIC_VECTOR (3 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARUSER : STD_LOGIC_VECTOR (0 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_RREADY : STD_LOGIC;
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_BREADY : STD_LOGIC;
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_grp_fu_106_p_din0 : STD_LOGIC_VECTOR (31 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_grp_fu_106_p_din1 : STD_LOGIC_VECTOR (31 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_grp_fu_106_p_opcode : STD_LOGIC_VECTOR (0 downto 0);
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_grp_fu_106_p_ce : STD_LOGIC;
    signal grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_ap_start_reg : STD_LOGIC := '0';
    signal ap_CS_fsm_state2 : STD_LOGIC;
    attribute fsm_encoding of ap_CS_fsm_state2 : signal is "none";
    signal grp_fu_106_ce : STD_LOGIC;
    signal ap_block_state2_on_subcall_done : BOOLEAN;
    signal ap_NS_fsm : STD_LOGIC_VECTOR (1 downto 0);
    signal ap_ST_fsm_state1_blk : STD_LOGIC;
    signal ap_ST_fsm_state2_blk : STD_LOGIC;
    signal ap_ce_reg : STD_LOGIC;

    component conv_kernel_avg_pool_Pipeline_avg_pool_loop1 IS
    port (
        ap_clk : IN STD_LOGIC;
        ap_rst : IN STD_LOGIC;
        ap_start : IN STD_LOGIC;
        ap_done : OUT STD_LOGIC;
        ap_idle : OUT STD_LOGIC;
        ap_ready : OUT STD_LOGIC;
        m_axi_gmem0_AWVALID : OUT STD_LOGIC;
        m_axi_gmem0_AWREADY : IN STD_LOGIC;
        m_axi_gmem0_AWADDR : OUT STD_LOGIC_VECTOR (63 downto 0);
        m_axi_gmem0_AWID : OUT STD_LOGIC_VECTOR (0 downto 0);
        m_axi_gmem0_AWLEN : OUT STD_LOGIC_VECTOR (31 downto 0);
        m_axi_gmem0_AWSIZE : OUT STD_LOGIC_VECTOR (2 downto 0);
        m_axi_gmem0_AWBURST : OUT STD_LOGIC_VECTOR (1 downto 0);
        m_axi_gmem0_AWLOCK : OUT STD_LOGIC_VECTOR (1 downto 0);
        m_axi_gmem0_AWCACHE : OUT STD_LOGIC_VECTOR (3 downto 0);
        m_axi_gmem0_AWPROT : OUT STD_LOGIC_VECTOR (2 downto 0);
        m_axi_gmem0_AWQOS : OUT STD_LOGIC_VECTOR (3 downto 0);
        m_axi_gmem0_AWREGION : OUT STD_LOGIC_VECTOR (3 downto 0);
        m_axi_gmem0_AWUSER : OUT STD_LOGIC_VECTOR (0 downto 0);
        m_axi_gmem0_WVALID : OUT STD_LOGIC;
        m_axi_gmem0_WREADY : IN STD_LOGIC;
        m_axi_gmem0_WDATA : OUT STD_LOGIC_VECTOR (31 downto 0);
        m_axi_gmem0_WSTRB : OUT STD_LOGIC_VECTOR (3 downto 0);
        m_axi_gmem0_WLAST : OUT STD_LOGIC;
        m_axi_gmem0_WID : OUT STD_LOGIC_VECTOR (0 downto 0);
        m_axi_gmem0_WUSER : OUT STD_LOGIC_VECTOR (0 downto 0);
        m_axi_gmem0_ARVALID : OUT STD_LOGIC;
        m_axi_gmem0_ARREADY : IN STD_LOGIC;
        m_axi_gmem0_ARADDR : OUT STD_LOGIC_VECTOR (63 downto 0);
        m_axi_gmem0_ARID : OUT STD_LOGIC_VECTOR (0 downto 0);
        m_axi_gmem0_ARLEN : OUT STD_LOGIC_VECTOR (31 downto 0);
        m_axi_gmem0_ARSIZE : OUT STD_LOGIC_VECTOR (2 downto 0);
        m_axi_gmem0_ARBURST : OUT STD_LOGIC_VECTOR (1 downto 0);
        m_axi_gmem0_ARLOCK : OUT STD_LOGIC_VECTOR (1 downto 0);
        m_axi_gmem0_ARCACHE : OUT STD_LOGIC_VECTOR (3 downto 0);
        m_axi_gmem0_ARPROT : OUT STD_LOGIC_VECTOR (2 downto 0);
        m_axi_gmem0_ARQOS : OUT STD_LOGIC_VECTOR (3 downto 0);
        m_axi_gmem0_ARREGION : OUT STD_LOGIC_VECTOR (3 downto 0);
        m_axi_gmem0_ARUSER : OUT STD_LOGIC_VECTOR (0 downto 0);
        m_axi_gmem0_RVALID : IN STD_LOGIC;
        m_axi_gmem0_RREADY : OUT STD_LOGIC;
        m_axi_gmem0_RDATA : IN STD_LOGIC_VECTOR (31 downto 0);
        m_axi_gmem0_RLAST : IN STD_LOGIC;
        m_axi_gmem0_RID : IN STD_LOGIC_VECTOR (0 downto 0);
        m_axi_gmem0_RFIFONUM : IN STD_LOGIC_VECTOR (8 downto 0);
        m_axi_gmem0_RUSER : IN STD_LOGIC_VECTOR (0 downto 0);
        m_axi_gmem0_RRESP : IN STD_LOGIC_VECTOR (1 downto 0);
        m_axi_gmem0_BVALID : IN STD_LOGIC;
        m_axi_gmem0_BREADY : OUT STD_LOGIC;
        m_axi_gmem0_BRESP : IN STD_LOGIC_VECTOR (1 downto 0);
        m_axi_gmem0_BID : IN STD_LOGIC_VECTOR (0 downto 0);
        m_axi_gmem0_BUSER : IN STD_LOGIC_VECTOR (0 downto 0);
        nif : IN STD_LOGIC_VECTOR (31 downto 0);
        in_base_addr : IN STD_LOGIC_VECTOR (31 downto 0);
        act_mem : IN STD_LOGIC_VECTOR (63 downto 0);
        out_base_addr : IN STD_LOGIC_VECTOR (31 downto 0);
        grp_fu_106_p_din0 : OUT STD_LOGIC_VECTOR (31 downto 0);
        grp_fu_106_p_din1 : OUT STD_LOGIC_VECTOR (31 downto 0);
        grp_fu_106_p_opcode : OUT STD_LOGIC_VECTOR (0 downto 0);
        grp_fu_106_p_dout0 : IN STD_LOGIC_VECTOR (31 downto 0);
        grp_fu_106_p_ce : OUT STD_LOGIC );
    end component;


    component conv_kernel_faddfsub_32ns_32ns_32_4_full_dsp_1 IS
    generic (
        ID : INTEGER;
        NUM_STAGE : INTEGER;
        din0_WIDTH : INTEGER;
        din1_WIDTH : INTEGER;
        dout_WIDTH : INTEGER );
    port (
        clk : IN STD_LOGIC;
        reset : IN STD_LOGIC;
        din0 : IN STD_LOGIC_VECTOR (31 downto 0);
        din1 : IN STD_LOGIC_VECTOR (31 downto 0);
        opcode : IN STD_LOGIC_VECTOR (1 downto 0);
        ce : IN STD_LOGIC;
        dout : OUT STD_LOGIC_VECTOR (31 downto 0) );
    end component;



begin
    grp_avg_pool_Pipeline_avg_pool_loop1_fu_68 : component conv_kernel_avg_pool_Pipeline_avg_pool_loop1
    port map (
        ap_clk => ap_clk,
        ap_rst => ap_rst,
        ap_start => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_ap_start,
        ap_done => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_ap_done,
        ap_idle => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_ap_idle,
        ap_ready => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_ap_ready,
        m_axi_gmem0_AWVALID => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWVALID,
        m_axi_gmem0_AWREADY => m_axi_gmem0_AWREADY,
        m_axi_gmem0_AWADDR => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWADDR,
        m_axi_gmem0_AWID => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWID,
        m_axi_gmem0_AWLEN => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWLEN,
        m_axi_gmem0_AWSIZE => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWSIZE,
        m_axi_gmem0_AWBURST => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWBURST,
        m_axi_gmem0_AWLOCK => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWLOCK,
        m_axi_gmem0_AWCACHE => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWCACHE,
        m_axi_gmem0_AWPROT => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWPROT,
        m_axi_gmem0_AWQOS => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWQOS,
        m_axi_gmem0_AWREGION => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWREGION,
        m_axi_gmem0_AWUSER => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWUSER,
        m_axi_gmem0_WVALID => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_WVALID,
        m_axi_gmem0_WREADY => m_axi_gmem0_WREADY,
        m_axi_gmem0_WDATA => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_WDATA,
        m_axi_gmem0_WSTRB => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_WSTRB,
        m_axi_gmem0_WLAST => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_WLAST,
        m_axi_gmem0_WID => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_WID,
        m_axi_gmem0_WUSER => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_WUSER,
        m_axi_gmem0_ARVALID => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARVALID,
        m_axi_gmem0_ARREADY => m_axi_gmem0_ARREADY,
        m_axi_gmem0_ARADDR => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARADDR,
        m_axi_gmem0_ARID => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARID,
        m_axi_gmem0_ARLEN => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARLEN,
        m_axi_gmem0_ARSIZE => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARSIZE,
        m_axi_gmem0_ARBURST => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARBURST,
        m_axi_gmem0_ARLOCK => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARLOCK,
        m_axi_gmem0_ARCACHE => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARCACHE,
        m_axi_gmem0_ARPROT => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARPROT,
        m_axi_gmem0_ARQOS => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARQOS,
        m_axi_gmem0_ARREGION => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARREGION,
        m_axi_gmem0_ARUSER => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARUSER,
        m_axi_gmem0_RVALID => m_axi_gmem0_RVALID,
        m_axi_gmem0_RREADY => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_RREADY,
        m_axi_gmem0_RDATA => m_axi_gmem0_RDATA,
        m_axi_gmem0_RLAST => m_axi_gmem0_RLAST,
        m_axi_gmem0_RID => m_axi_gmem0_RID,
        m_axi_gmem0_RFIFONUM => m_axi_gmem0_RFIFONUM,
        m_axi_gmem0_RUSER => m_axi_gmem0_RUSER,
        m_axi_gmem0_RRESP => m_axi_gmem0_RRESP,
        m_axi_gmem0_BVALID => m_axi_gmem0_BVALID,
        m_axi_gmem0_BREADY => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_BREADY,
        m_axi_gmem0_BRESP => m_axi_gmem0_BRESP,
        m_axi_gmem0_BID => m_axi_gmem0_BID,
        m_axi_gmem0_BUSER => m_axi_gmem0_BUSER,
        nif => nif,
        in_base_addr => in_base_addr,
        act_mem => act_mem,
        out_base_addr => out_base_addr,
        grp_fu_106_p_din0 => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_grp_fu_106_p_din0,
        grp_fu_106_p_din1 => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_grp_fu_106_p_din1,
        grp_fu_106_p_opcode => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_grp_fu_106_p_opcode,
        grp_fu_106_p_dout0 => grp_fu_2953_p_dout0,
        grp_fu_106_p_ce => grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_grp_fu_106_p_ce);





    ap_CS_fsm_assign_proc : process(ap_clk)
    begin
        if (ap_clk'event and ap_clk =  '1') then
            if (ap_rst = '1') then
                ap_CS_fsm <= ap_ST_fsm_state1;
            else
                ap_CS_fsm <= ap_NS_fsm;
            end if;
        end if;
    end process;


    grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_ap_start_reg_assign_proc : process(ap_clk)
    begin
        if (ap_clk'event and ap_clk =  '1') then
            if (ap_rst = '1') then
                grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_ap_start_reg <= ap_const_logic_0;
            else
                if (((ap_start = ap_const_logic_1) and (avg_pool_en = ap_const_lv1_1) and (ap_const_logic_1 = ap_CS_fsm_state1))) then 
                    grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_ap_start_reg <= ap_const_logic_1;
                elsif ((grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_ap_ready = ap_const_logic_1)) then 
                    grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_ap_start_reg <= ap_const_logic_0;
                end if; 
            end if;
        end if;
    end process;


    ap_NS_fsm_assign_proc : process (ap_start, ap_CS_fsm, ap_CS_fsm_state1, ap_CS_fsm_state2, ap_block_state2_on_subcall_done)
    begin
        case ap_CS_fsm is
            when ap_ST_fsm_state1 => 
                if (((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1))) then
                    ap_NS_fsm <= ap_ST_fsm_state2;
                else
                    ap_NS_fsm <= ap_ST_fsm_state1;
                end if;
            when ap_ST_fsm_state2 => 
                if (((ap_const_logic_1 = ap_CS_fsm_state2) and (ap_const_boolean_0 = ap_block_state2_on_subcall_done))) then
                    ap_NS_fsm <= ap_ST_fsm_state1;
                else
                    ap_NS_fsm <= ap_ST_fsm_state2;
                end if;
            when others =>  
                ap_NS_fsm <= "XX";
        end case;
    end process;
    ap_CS_fsm_state1 <= ap_CS_fsm(0);
    ap_CS_fsm_state2 <= ap_CS_fsm(1);

    ap_ST_fsm_state1_blk_assign_proc : process(ap_start)
    begin
        if ((ap_start = ap_const_logic_0)) then 
            ap_ST_fsm_state1_blk <= ap_const_logic_1;
        else 
            ap_ST_fsm_state1_blk <= ap_const_logic_0;
        end if; 
    end process;


    ap_ST_fsm_state2_blk_assign_proc : process(ap_block_state2_on_subcall_done)
    begin
        if ((ap_const_boolean_1 = ap_block_state2_on_subcall_done)) then 
            ap_ST_fsm_state2_blk <= ap_const_logic_1;
        else 
            ap_ST_fsm_state2_blk <= ap_const_logic_0;
        end if; 
    end process;


    ap_block_state2_on_subcall_done_assign_proc : process(avg_pool_en, grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_ap_done)
    begin
                ap_block_state2_on_subcall_done <= ((grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_ap_done = ap_const_logic_0) and (avg_pool_en = ap_const_lv1_1));
    end process;


    ap_done_assign_proc : process(ap_start, ap_CS_fsm_state1, ap_CS_fsm_state2, ap_block_state2_on_subcall_done)
    begin
        if ((((ap_start = ap_const_logic_0) and (ap_const_logic_1 = ap_CS_fsm_state1)) or ((ap_const_logic_1 = ap_CS_fsm_state2) and (ap_const_boolean_0 = ap_block_state2_on_subcall_done)))) then 
            ap_done <= ap_const_logic_1;
        else 
            ap_done <= ap_const_logic_0;
        end if; 
    end process;


    ap_idle_assign_proc : process(ap_start, ap_CS_fsm_state1)
    begin
        if (((ap_start = ap_const_logic_0) and (ap_const_logic_1 = ap_CS_fsm_state1))) then 
            ap_idle <= ap_const_logic_1;
        else 
            ap_idle <= ap_const_logic_0;
        end if; 
    end process;


    ap_ready_assign_proc : process(ap_CS_fsm_state2, ap_block_state2_on_subcall_done)
    begin
        if (((ap_const_logic_1 = ap_CS_fsm_state2) and (ap_const_boolean_0 = ap_block_state2_on_subcall_done))) then 
            ap_ready <= ap_const_logic_1;
        else 
            ap_ready <= ap_const_logic_0;
        end if; 
    end process;

    grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_ap_start <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_ap_start_reg;

    grp_fu_106_ce_assign_proc : process(grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_grp_fu_106_p_ce, ap_CS_fsm_state2)
    begin
        if ((ap_const_logic_1 = ap_CS_fsm_state2)) then 
            grp_fu_106_ce <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_grp_fu_106_p_ce;
        else 
            grp_fu_106_ce <= ap_const_logic_1;
        end if; 
    end process;

    grp_fu_2953_p_ce <= grp_fu_106_ce;
    grp_fu_2953_p_din0 <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_grp_fu_106_p_din0;
    grp_fu_2953_p_din1 <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_grp_fu_106_p_din1;
    grp_fu_2953_p_opcode <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_grp_fu_106_p_opcode;
    m_axi_gmem0_ARADDR <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARADDR;
    m_axi_gmem0_ARBURST <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARBURST;
    m_axi_gmem0_ARCACHE <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARCACHE;
    m_axi_gmem0_ARID <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARID;
    m_axi_gmem0_ARLEN <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARLEN;
    m_axi_gmem0_ARLOCK <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARLOCK;
    m_axi_gmem0_ARPROT <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARPROT;
    m_axi_gmem0_ARQOS <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARQOS;
    m_axi_gmem0_ARREGION <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARREGION;
    m_axi_gmem0_ARSIZE <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARSIZE;
    m_axi_gmem0_ARUSER <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARUSER;

    m_axi_gmem0_ARVALID_assign_proc : process(ap_CS_fsm_state1, avg_pool_en, grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARVALID, ap_CS_fsm_state2)
    begin
        if ((((avg_pool_en = ap_const_lv1_1) and (ap_const_logic_1 = ap_CS_fsm_state1)) or ((avg_pool_en = ap_const_lv1_1) and (ap_const_logic_1 = ap_CS_fsm_state2)))) then 
            m_axi_gmem0_ARVALID <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_ARVALID;
        else 
            m_axi_gmem0_ARVALID <= ap_const_logic_0;
        end if; 
    end process;

    m_axi_gmem0_AWADDR <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWADDR;
    m_axi_gmem0_AWBURST <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWBURST;
    m_axi_gmem0_AWCACHE <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWCACHE;
    m_axi_gmem0_AWID <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWID;
    m_axi_gmem0_AWLEN <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWLEN;
    m_axi_gmem0_AWLOCK <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWLOCK;
    m_axi_gmem0_AWPROT <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWPROT;
    m_axi_gmem0_AWQOS <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWQOS;
    m_axi_gmem0_AWREGION <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWREGION;
    m_axi_gmem0_AWSIZE <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWSIZE;
    m_axi_gmem0_AWUSER <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWUSER;

    m_axi_gmem0_AWVALID_assign_proc : process(ap_CS_fsm_state1, avg_pool_en, grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWVALID, ap_CS_fsm_state2)
    begin
        if ((((avg_pool_en = ap_const_lv1_1) and (ap_const_logic_1 = ap_CS_fsm_state1)) or ((avg_pool_en = ap_const_lv1_1) and (ap_const_logic_1 = ap_CS_fsm_state2)))) then 
            m_axi_gmem0_AWVALID <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_AWVALID;
        else 
            m_axi_gmem0_AWVALID <= ap_const_logic_0;
        end if; 
    end process;


    m_axi_gmem0_BREADY_assign_proc : process(ap_CS_fsm_state1, avg_pool_en, grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_BREADY, ap_CS_fsm_state2)
    begin
        if ((((avg_pool_en = ap_const_lv1_1) and (ap_const_logic_1 = ap_CS_fsm_state1)) or ((avg_pool_en = ap_const_lv1_1) and (ap_const_logic_1 = ap_CS_fsm_state2)))) then 
            m_axi_gmem0_BREADY <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_BREADY;
        else 
            m_axi_gmem0_BREADY <= ap_const_logic_0;
        end if; 
    end process;


    m_axi_gmem0_RREADY_assign_proc : process(ap_CS_fsm_state1, avg_pool_en, grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_RREADY, ap_CS_fsm_state2)
    begin
        if ((((avg_pool_en = ap_const_lv1_1) and (ap_const_logic_1 = ap_CS_fsm_state1)) or ((avg_pool_en = ap_const_lv1_1) and (ap_const_logic_1 = ap_CS_fsm_state2)))) then 
            m_axi_gmem0_RREADY <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_RREADY;
        else 
            m_axi_gmem0_RREADY <= ap_const_logic_0;
        end if; 
    end process;

    m_axi_gmem0_WDATA <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_WDATA;
    m_axi_gmem0_WID <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_WID;
    m_axi_gmem0_WLAST <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_WLAST;
    m_axi_gmem0_WSTRB <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_WSTRB;
    m_axi_gmem0_WUSER <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_WUSER;

    m_axi_gmem0_WVALID_assign_proc : process(ap_CS_fsm_state1, avg_pool_en, grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_WVALID, ap_CS_fsm_state2)
    begin
        if ((((avg_pool_en = ap_const_lv1_1) and (ap_const_logic_1 = ap_CS_fsm_state1)) or ((avg_pool_en = ap_const_lv1_1) and (ap_const_logic_1 = ap_CS_fsm_state2)))) then 
            m_axi_gmem0_WVALID <= grp_avg_pool_Pipeline_avg_pool_loop1_fu_68_m_axi_gmem0_WVALID;
        else 
            m_axi_gmem0_WVALID <= ap_const_logic_0;
        end if; 
    end process;

end behav;