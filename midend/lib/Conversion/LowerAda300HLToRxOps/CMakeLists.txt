add_mlir_library(LowerAda300HLToRxOpsPass
  LowerAda300HLToRxOpsPass.cpp

  DEPENDS
  MLIRAda300HLOpsIncGen
  BuddyAda300HLAttrsIncGen

  LINK_LIBS PUBLIC
  BuddyAda300HL
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRVectorDialect
  MLIRTransformUtils
)
