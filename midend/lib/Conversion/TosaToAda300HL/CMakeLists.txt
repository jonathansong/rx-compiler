add_mlir_library(TosaToAda300HLPass
  TosaToAda300HLPass.cpp

  DEPENDS
  MLIRAda300HLOpsIncGen
  BuddyAda300HLAttrsIncGen

  LINK_LIBS PUBLIC
  BuddyAda300HL
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRTosaDialect
  MLIRVectorDialect
  MLIRTransformUtils
)
