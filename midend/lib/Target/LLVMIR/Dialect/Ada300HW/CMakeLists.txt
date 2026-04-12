add_mlir_translation_library(BuddyAda300HWToLLVMIRTranslation
  Ada300HWToLLVMIRTranslation.cpp

  DEPENDS
  buddy_intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  BuddyAda300HW
)
