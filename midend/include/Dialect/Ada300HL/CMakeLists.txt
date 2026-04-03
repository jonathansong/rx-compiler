# Generate op declarations/definitions, type declarations/definitions,
# and dialect declarations/definitions for Ada300HL.
add_mlir_dialect(Ada300HLOps ada300hl)

# Additionally generate attribute and enum tablegen targets, which are not
# covered by add_mlir_dialect.
set(LLVM_TARGET_DEFINITIONS Ada300HLOps.td)
mlir_tablegen(Ada300HLOpsAttrs.h.inc   -gen-attrdef-decls -attrdefs-dialect=ada300hl)
mlir_tablegen(Ada300HLOpsAttrs.cpp.inc -gen-attrdef-defs  -attrdefs-dialect=ada300hl)
mlir_tablegen(Ada300HLOpsEnums.h.inc   -gen-enum-decls)
mlir_tablegen(Ada300HLOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(BuddyAda300HLAttrsIncGen)
