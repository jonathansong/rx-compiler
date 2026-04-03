add_mlir_dialect_library(BuddyAda300HL
  Ada300HLDialect.cpp
  Ada300HLAttrs.cpp
  Ada300HLTypes.cpp
  Ada300HLOps.cpp

  DEPENDS
  MLIRAda300HLOpsIncGen
  BuddyAda300HLAttrsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  MLIRMemRefDialect
)
