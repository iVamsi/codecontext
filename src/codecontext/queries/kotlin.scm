(class_declaration
  name: (identifier) @class.name) @class.def

(object_declaration
  name: (identifier) @object.name) @object.def

(function_declaration
  name: (identifier) @function.name) @function.def

(import) @import.stmt
(call_expression) @call.expr
