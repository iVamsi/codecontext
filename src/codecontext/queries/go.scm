(type_spec
  name: (type_identifier) @type.name) @type.def

(function_declaration
  name: (identifier) @function.name) @function.def

(method_declaration
  name: (field_identifier) @method.name) @method.def

(import_declaration) @import.stmt
(call_expression) @call.expr
