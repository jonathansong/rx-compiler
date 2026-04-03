add_mlir_library(LowerAda300HLToAda300HWPass
  LowerAda300HLToAda300HWPass.cpp

  DEPENDS
  MLIRAda300HLOpsIncGen
  MLIRAda300HWOpsIncGen
  BuddyAda300HLAttrsIncGen

  LINK_LIBS PUBLIC
  BuddyAda300HL
  BuddyAda300HW
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRTransformUtils
)
