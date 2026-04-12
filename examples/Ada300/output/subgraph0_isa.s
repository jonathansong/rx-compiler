	.attribute	4, 16
	.attribute	5, "rv64i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"LLVMDialectModule"
	.text
	.globl	subgraph0                       # -- Begin function subgraph0
	.p2align	2
	.type	subgraph0,@function
subgraph0:                              # @subgraph0
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -1024
	.cfi_def_cfa_offset 1024
	sd	ra, 1016(sp)                    # 8-byte Folded Spill
	sd	s0, 1008(sp)                    # 8-byte Folded Spill
	sd	s1, 1000(sp)                    # 8-byte Folded Spill
	sd	s2, 992(sp)                     # 8-byte Folded Spill
	sd	s3, 984(sp)                     # 8-byte Folded Spill
	sd	s4, 976(sp)                     # 8-byte Folded Spill
	sd	s5, 968(sp)                     # 8-byte Folded Spill
	sd	s6, 960(sp)                     # 8-byte Folded Spill
	sd	s7, 952(sp)                     # 8-byte Folded Spill
	sd	s8, 944(sp)                     # 8-byte Folded Spill
	sd	s9, 936(sp)                     # 8-byte Folded Spill
	sd	s10, 928(sp)                    # 8-byte Folded Spill
	sd	s11, 920(sp)                    # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	.cfi_offset s1, -24
	.cfi_offset s2, -32
	.cfi_offset s3, -40
	.cfi_offset s4, -48
	.cfi_offset s5, -56
	.cfi_offset s6, -64
	.cfi_offset s7, -72
	.cfi_offset s8, -80
	.cfi_offset s9, -88
	.cfi_offset s10, -96
	.cfi_offset s11, -104
	addi	s0, sp, 1024
	.cfi_def_cfa s0, 0
	andi	sp, sp, -512
	mv	s1, sp
	mv	s5, a7
	mv	s10, a6
	mv	s8, a3
	mv	s11, a2
	ld	a1, 184(s0)
	sd	a1, 504(s1)                     # 8-byte Folded Spill
	ld	a1, 168(s0)
	sd	a1, 448(s1)                     # 8-byte Folded Spill
	ld	a1, 160(s0)
	sd	a1, 456(s1)                     # 8-byte Folded Spill
	ld	a1, 144(s0)
	sd	a1, 496(s1)                     # 8-byte Folded Spill
	ld	a1, 136(s0)
	sd	a1, 488(s1)                     # 8-byte Folded Spill
	ld	a1, 112(s0)
	sd	a1, 416(s1)                     # 8-byte Folded Spill
	ld	a1, 104(s0)
	sd	a1, 424(s1)                     # 8-byte Folded Spill
	ld	a1, 88(s0)
	sd	a1, 376(s1)                     # 8-byte Folded Spill
	ld	a1, 80(s0)
	sd	a1, 360(s1)                     # 8-byte Folded Spill
	ld	a1, 72(s0)
	sd	a1, 352(s1)                     # 8-byte Folded Spill
	ld	a1, 64(s0)
	sd	a1, 432(s1)                     # 8-byte Folded Spill
	ld	a1, 56(s0)
	sd	a1, 392(s1)                     # 8-byte Folded Spill
	ld	a1, 48(s0)
	sd	a1, 384(s1)                     # 8-byte Folded Spill
	ld	a1, 40(s0)
	sd	a1, 368(s1)                     # 8-byte Folded Spill
	ld	s9, 32(s0)
	ld	a1, 16(s0)
	sd	a1, 400(s1)                     # 8-byte Folded Spill
	ld	a1, 8(s0)
	sd	a1, 408(s1)                     # 8-byte Folded Spill
	sd	a0, 472(s1)                     # 8-byte Folded Spill
	lui	a0, 8
	addi	a0, a0, 64
	call	malloc
	li	s7, 0
	li	s4, 0
	sd	a0, 464(s1)                     # 8-byte Folded Spill
	addi	a0, a0, 63
	slli	s6, s8, 2
	andi	s8, a0, -64
	add	s6, s11, s6
	li	s3, 127
	j	.LBB0_2
.LBB0_1:                                #   in Loop: Header=BB0_2 Depth=1
	addi	s4, s4, 1
	ld	s7, 480(s1)                     # 8-byte Folded Reload
	addi	s7, s7, 512
.LBB0_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_4 Depth 2
	li	a0, 63
	blt	a0, s4, .LBB0_5
# %bb.3:                                #   in Loop: Header=BB0_2 Depth=1
	li	s11, 0
	sd	s7, 480(s1)                     # 8-byte Folded Spill
	bltz	s3, .LBB0_1
.LBB0_4:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	mv	a0, s11
	mv	a1, s10
	call	__muldi3
	mv	s2, a0
	mv	a0, s4
	mv	a1, s5
	call	__muldi3
	add	a0, s2, a0
	slli	a0, a0, 2
	add	a0, s6, a0
	flw	fa5, 0(a0)
	add	a0, s8, s7
	addi	s11, s11, 1
	fsw	fa5, 0(a0)
	addi	s7, s7, 4
	bge	s3, s11, .LBB0_4
	j	.LBB0_1
.LBB0_5:
	li	a0, 576
	call	malloc
	sd	a0, 440(s1)                     # 8-byte Folded Spill
	addi	a0, a0, 63
	andi	s11, a0, -64
	li	s2, 1
.Lpcrel_hi0:
	auipc	a0, %pcrel_hi(.L__constant_1x128xf32)
	addi	a1, a0, %pcrel_lo(.Lpcrel_hi0)
	li	a2, 512
	mv	a0, s11
	call	memcpy
	li	a0, 256
	call	malloc
	mv	s10, a0
	lui	a0, 8
	call	malloc
	mv	s4, a0
	mv	s5, sp
	mv	a0, sp
	addi	a2, a0, -64
	mv	sp, a2
	ld	a1, 352(s1)                     # 8-byte Folded Reload
	sd	a1, -32(a0)
	ld	a1, 360(s1)                     # 8-byte Folded Reload
	sd	a1, -24(a0)
	ld	a1, 376(s1)                     # 8-byte Folded Reload
	sd	a1, -16(a0)
	ld	a1, 368(s1)                     # 8-byte Folded Reload
	sd	a1, -64(a0)
	ld	a1, 384(s1)                     # 8-byte Folded Reload
	sd	a1, -56(a0)
	ld	a1, 392(s1)                     # 8-byte Folded Reload
	sd	a1, -48(a0)
	ld	a1, 432(s1)                     # 8-byte Folded Reload
	sd	a1, -40(a0)
	mv	a0, sp
	addi	a3, a0, -64
	mv	sp, a3
	lui	a1, 5124
	addi	a1, a1, 1
	vsetivli	zero, 4, e64, m2, ta, ma
	vmv.s.x	v10, a1
	addi	a1, a0, -40
	vsext.vf8	v8, v10
	vse64.v	v8, (a1)
	sd	s10, -64(a0)
	sd	s10, -56(a0)
	sd	zero, -48(a0)
	mv	a0, sp
	addi	a1, a0, -16
	mv	sp, a1
	li	a4, 2
	sd	a4, -16(a0)
	sd	a2, -8(a0)
	mv	a0, sp
	addi	a2, a0, -16
	mv	sp, a2
	sd	a4, -16(a0)
	sd	a3, -8(a0)
	li	a0, 4
	call	memrefCopy
	mv	sp, s5
	li	a0, 64
	li	a1, 128
	call	__muldi3
	slli	a2, a0, 2
	mv	a0, s4
	mv	a1, s8
	call	memcpy
	lui	a0, 1032
	addi	a0, a0, 1
	ada300.set_gmm_cfg	a0, zero
	ada300.set_gmm_type	zero, zero
	ada300.set_gmm_iter	s2, s2
	ada300.gmma_mm	s11, s10, s4
	ada300.tcsync
	mv	a0, s10
	call	free
	mv	a0, s4
	call	free
	slli	s4, s9, 7
	li	a0, 576
	call	malloc
	li	s7, 0
	li	s5, 0
	sd	a0, 432(s1)                     # 8-byte Folded Spill
	addi	a0, a0, 63
	ld	s3, 400(s1)                     # 8-byte Folded Reload
	slli	s3, s3, 2
	andi	s10, a0, -64
	ld	a0, 408(s1)                     # 8-byte Folded Reload
	add	s3, a0, s3
	li	s6, 127
	j	.LBB0_7
.LBB0_6:                                #   in Loop: Header=BB0_7 Depth=1
	addi	s5, s5, 1
	ld	s7, 480(s1)                     # 8-byte Folded Reload
	addi	s7, s7, 512
.LBB0_7:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_9 Depth 2
	bgtz	s5, .LBB0_10
# %bb.8:                                #   in Loop: Header=BB0_7 Depth=1
	li	s8, 0
	sd	s7, 480(s1)                     # 8-byte Folded Spill
	bltz	s6, .LBB0_6
.LBB0_9:                                #   Parent Loop BB0_7 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	mv	a0, s5
	mv	a1, s4
	call	__muldi3
	mv	s2, a0
	mv	a0, s8
	mv	a1, s9
	call	__muldi3
	add	a0, s2, a0
	add	a1, s11, s7
	slli	a0, a0, 2
	flw	fa5, 0(a1)
	add	a0, s3, a0
	flw	fa4, 0(a0)
	add	a0, s10, s7
	addi	s8, s8, 1
	fadd.s	fa5, fa4, fa5
	fsw	fa5, 0(a0)
	addi	s7, s7, 4
	bge	s6, s8, .LBB0_9
	j	.LBB0_6
.LBB0_10:
	li	a3, 32
	addi	a0, s10, 128
	addi	a1, s10, 256
	addi	a2, s10, 384
	vsetvli	zero, a3, e32, m8, ta, ma
	vle32.v	v16, (s10)
	vle32.v	v8, (a0)
	vle32.v	v24, (a1)
	vle32.v	v0, (a2)
	addi	a3, sp, -512
	andi	a3, a3, -512
	mv	sp, a3
	addi	a4, a3, 384
	addi	a5, a3, 256
	vse32.v	v0, (a4)
	addi	a4, a3, 128
	vse32.v	v16, (a3)
	vse32.v	v24, (a5)
	li	a5, 16
	vse32.v	v8, (a4)
	ada300.vfpwnl	a3, zero, a5
	addi	a4, a3, 256
	vle32.v	v16, (a4)
	addi	a4, a3, 128
	vle32.v	v8, (a3)
	addi	a3, a3, 384
	vle32.v	v0, (a3)
	vle32.v	v24, (a4)
	vse32.v	v8, (s10)
	vse32.v	v16, (a1)
	vse32.v	v0, (a2)
	vse32.v	v24, (a0)
	addi	a3, sp, -512
	andi	a3, a3, -512
	mv	sp, a3
	addi	a4, a3, 384
	vse32.v	v0, (a4)
	addi	a4, a3, 256
	vse32.v	v16, (a4)
	addi	a4, a3, 128
	vse32.v	v8, (a3)
	vse32.v	v24, (a4)
	li	a4, 2
	ada300.vfpwnl	a3, a4, a5
	addi	a4, a3, 128
	vle32.v	v8, (a3)
	addi	a5, a3, 384
	addi	a3, a3, 256
	vle32.v	v16, (a3)
	vle32.v	v24, (a5)
	vle32.v	v0, (a4)
	vse32.v	v8, (s10)
	vse32.v	v16, (a1)
	vse32.v	v24, (a2)
	vse32.v	v0, (a0)
	lui	a0, 8
	addi	a0, a0, 64
	call	malloc
	li	s8, 0
	li	s4, 0
	sd	a0, 480(s1)                     # 8-byte Folded Spill
	addi	a0, a0, 63
	li	s3, 127
	ld	s6, 416(s1)                     # 8-byte Folded Reload
	slli	s6, s6, 2
	andi	s11, a0, -64
	ld	a0, 424(s1)                     # 8-byte Folded Reload
	add	s6, a0, s6
	li	s9, 63
	j	.LBB0_12
.LBB0_11:                               #   in Loop: Header=BB0_12 Depth=1
	addi	s4, s4, 1
	addi	s8, s8, 256
.LBB0_12:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_14 Depth 2
	blt	s3, s4, .LBB0_15
# %bb.13:                               #   in Loop: Header=BB0_12 Depth=1
	li	s5, 0
	mv	s7, s8
	bltz	s9, .LBB0_11
.LBB0_14:                               #   Parent Loop BB0_12 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	mv	a0, s5
	ld	a1, 488(s1)                     # 8-byte Folded Reload
	call	__muldi3
	mv	s2, a0
	mv	a0, s4
	ld	a1, 496(s1)                     # 8-byte Folded Reload
	call	__muldi3
	add	a0, s2, a0
	slli	a0, a0, 2
	add	a0, s6, a0
	flw	fa5, 0(a0)
	add	a0, s11, s7
	addi	s5, s5, 1
	fsw	fa5, 0(a0)
	addi	s7, s7, 4
	bge	s9, s5, .LBB0_14
	j	.LBB0_11
.LBB0_15:
	li	a0, 320
	call	malloc
	sd	a0, 496(s1)                     # 8-byte Folded Spill
	addi	a0, a0, 63
	li	s3, 1
	andi	s7, a0, -64
.Lpcrel_hi1:
	auipc	a0, %pcrel_hi(.L__constant_1x64xf32)
	addi	a1, a0, %pcrel_lo(.Lpcrel_hi1)
	li	a2, 256
	mv	a0, s7
	call	memcpy
	li	a0, 512
	call	malloc
	mv	s2, a0
	lui	a0, 8
	call	malloc
	mv	s4, a0
	li	a0, 1
	li	a1, 128
	call	__muldi3
	slli	a2, a0, 2
	mv	a0, s2
	mv	a1, s10
	call	memcpy
	li	a0, 128
	li	a1, 64
	call	__muldi3
	slli	a2, a0, 2
	mv	a0, s4
	mv	a1, s11
	call	memcpy
	lui	a0, 2052
	addi	a0, a0, 1
	ada300.set_gmm_cfg	a0, zero
	ada300.set_gmm_type	zero, zero
	ada300.set_gmm_iter	s3, s3
	ada300.gmma_mm	s7, s2, s4
	ada300.tcsync
	mv	a0, s2
	call	free
	mv	a0, s4
	call	free
	ld	s4, 504(s1)                     # 8-byte Folded Reload
	slli	s4, s4, 6
	li	a0, 320
	call	malloc
	li	s10, 0
	li	s11, 0
	sd	a0, 488(s1)                     # 8-byte Folded Spill
	addi	a0, a0, 63
	ld	s6, 448(s1)                     # 8-byte Folded Reload
	slli	s6, s6, 2
	andi	s8, a0, -64
	ld	a0, 456(s1)                     # 8-byte Folded Reload
	add	s6, a0, s6
	li	s3, 63
	j	.LBB0_17
.LBB0_16:                               #   in Loop: Header=BB0_17 Depth=1
	addi	s11, s11, 1
	addi	s10, s10, 256
.LBB0_17:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_19 Depth 2
	bgtz	s11, .LBB0_20
# %bb.18:                               #   in Loop: Header=BB0_17 Depth=1
	li	s5, 0
	mv	s9, s10
	bltz	s3, .LBB0_16
.LBB0_19:                               #   Parent Loop BB0_17 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	mv	a0, s11
	mv	a1, s4
	call	__muldi3
	mv	s2, a0
	mv	a0, s5
	ld	a1, 504(s1)                     # 8-byte Folded Reload
	call	__muldi3
	add	a0, s2, a0
	add	a1, s7, s9
	slli	a0, a0, 2
	flw	fa5, 0(a1)
	add	a0, s6, a0
	flw	fa4, 0(a0)
	add	a0, s8, s9
	addi	s5, s5, 1
	fadd.s	fa5, fa4, fa5
	fsw	fa5, 0(a0)
	addi	s9, s9, 4
	bge	s3, s5, .LBB0_19
	j	.LBB0_16
.LBB0_20:
	ld	a0, 464(s1)                     # 8-byte Folded Reload
	call	free
	ld	a0, 440(s1)                     # 8-byte Folded Reload
	call	free
	ld	a0, 432(s1)                     # 8-byte Folded Reload
	call	free
	ld	a0, 480(s1)                     # 8-byte Folded Reload
	call	free
	ld	a0, 496(s1)                     # 8-byte Folded Reload
	call	free
	ld	a0, 472(s1)                     # 8-byte Folded Reload
	li	a1, 64
	sd	a1, 32(a0)
	sd	a1, 40(a0)
	li	a1, 1
	sd	a1, 48(a0)
	ld	a2, 488(s1)                     # 8-byte Folded Reload
	sd	a2, 0(a0)
	sd	s8, 8(a0)
	sd	zero, 16(a0)
	sd	a1, 24(a0)
	addi	sp, s0, -1024
	.cfi_def_cfa sp, 1024
	ld	ra, 1016(sp)                    # 8-byte Folded Reload
	ld	s0, 1008(sp)                    # 8-byte Folded Reload
	ld	s1, 1000(sp)                    # 8-byte Folded Reload
	ld	s2, 992(sp)                     # 8-byte Folded Reload
	ld	s3, 984(sp)                     # 8-byte Folded Reload
	ld	s4, 976(sp)                     # 8-byte Folded Reload
	ld	s5, 968(sp)                     # 8-byte Folded Reload
	ld	s6, 960(sp)                     # 8-byte Folded Reload
	ld	s7, 952(sp)                     # 8-byte Folded Reload
	ld	s8, 944(sp)                     # 8-byte Folded Reload
	ld	s9, 936(sp)                     # 8-byte Folded Reload
	ld	s10, 928(sp)                    # 8-byte Folded Reload
	ld	s11, 920(sp)                    # 8-byte Folded Reload
	.cfi_restore ra
	.cfi_restore s0
	.cfi_restore s1
	.cfi_restore s2
	.cfi_restore s3
	.cfi_restore s4
	.cfi_restore s5
	.cfi_restore s6
	.cfi_restore s7
	.cfi_restore s8
	.cfi_restore s9
	.cfi_restore s10
	.cfi_restore s11
	addi	sp, sp, 1024
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end0:
	.size	subgraph0, .Lfunc_end0-subgraph0
	.cfi_endproc
                                        # -- End function
	.type	.L__constant_1x64xf32,@object   # @__constant_1x64xf32
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
.L__constant_1x64xf32:
	.zero	256
	.size	.L__constant_1x64xf32, 256

	.type	.L__constant_1x128xf32,@object  # @__constant_1x128xf32
	.p2align	6, 0x0
.L__constant_1x128xf32:
	.zero	512
	.size	.L__constant_1x128xf32, 512

	.section	".note.GNU-stack","",@progbits
