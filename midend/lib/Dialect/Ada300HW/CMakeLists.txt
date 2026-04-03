add_mlir_dialect_library(BuddyAda300HW
  Ada300HWDialect.cpp
  Ada300HWOps.cpp

  DEPENDS
  MLIRAda300HWOpsIncGen
  MLIRAda300HLOpsIncGen
  BuddyAda300HLAttrsIncGen

  LINK_LIBS PUBLIC
  BuddyAda300HL
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRMemRefDialect
)
