(struct_specifier
  name: (type_identifier) @struct.name) @struct.def

(enum_specifier
  name: (type_identifier) @enum.name) @enum.def

(function_definition) @function.def

(preproc_include) @import.stmt
(call_expression) @call.expr
