(class_declaration
  name: (identifier) @class.name) @class.def

(function_declaration
  name: (identifier) @function.name) @function.def

(method_definition
  name: (_) @method.name) @method.def

(variable_declarator
  name: (identifier) @binding.name) @binding.def

(import_statement) @import.stmt
(export_statement) @export.stmt

(call_expression) @call.expr
(jsx_opening_element) @jsx.open
(jsx_self_closing_element) @jsx.self
