add_mlir_library(MathToAda300HLPass
  MathToAda300HLPass.cpp

  DEPENDS
  MLIRAda300HLOpsIncGen
  BuddyAda300HLAttrsIncGen

  LINK_LIBS PUBLIC
  BuddyAda300HL
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRMathDialect
  MLIRTransformUtils
)
