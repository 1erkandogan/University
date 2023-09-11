# Copyright (c) 2016 Technische Universitaet Dresden, Germany
# Chair for VLSI-Design, Diagnostic and Architecture
# Author: Martin Zabel
# All rights reserved.

CWD=$(shell pwd)

SIM ?= icarus
TOPLEVEL_LANG ?=verilog


VERILOG_SOURCES +=$(CWD)/../../Register_file.v
VERILOG_SOURCES +=$(CWD)/../../Mux_16to1.v
VERILOG_SOURCES +=$(CWD)/../../Register_sync_rw.v
VERILOG_SOURCES +=$(CWD)/../../Decoder_4to16.v

TOPLEVEL = Register_file
MODULE := Register_file_test

# include cocotb's make rules to take care of the simulator setup
include $(shell cocotb-config --makefiles)/Makefile.sim
