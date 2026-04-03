# Generate op declarations/definitions, type stubs, and dialect
# declarations/definitions for Ada300HW.
# Ada300HWOps.td transitively includes Ada300HL/Ada300HLDialect.td, so the
# Ada300HL tablegen targets must be generated before Ada300HW is compiled.
add_mlir_dialect(Ada300HWOps ada300hw)

# Ensure Ada300HW tablegen runs after Ada300HL headers are generated.
add_dependencies(MLIRAda300HWOpsIncGen
  MLIRAda300HLOpsIncGen
  BuddyAda300HLAttrsIncGen
)
