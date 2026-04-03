add_mlir_library(LinalgToAda300HLPass
  LinalgToAda300HLPass.cpp

  DEPENDS
  MLIRAda300HLOpsIncGen
  BuddyAda300HLAttrsIncGen

  LINK_LIBS PUBLIC
  BuddyAda300HL
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRTransformUtils
)
