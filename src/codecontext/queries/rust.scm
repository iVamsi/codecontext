(struct_item
  name: (type_identifier) @struct.name) @struct.def

(enum_item
  name: (type_identifier) @enum.name) @enum.def

(trait_item
  name: (type_identifier) @trait.name) @trait.def

(function_item
  name: (identifier) @function.name) @function.def

(use_declaration) @import.stmt
(call_expression) @call.expr
