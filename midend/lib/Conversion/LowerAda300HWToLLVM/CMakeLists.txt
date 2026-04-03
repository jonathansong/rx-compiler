add_mlir_library(LowerAda300HWToLLVMPass
  LowerAda300HWToLLVMPass.cpp

  DEPENDS
  MLIRAda300HWOpsIncGen
  BuddyAda300HLAttrsIncGen

  LINK_LIBS PUBLIC
  BuddyAda300HL
  BuddyAda300HW
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
  MLIRMemRefToLLVM
  MLIRTransformUtils
  LLVMSupport
)
