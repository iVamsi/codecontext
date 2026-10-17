(class_definition
  name: (identifier) @class.name) @class.def

(function_definition
  name: (identifier) @function.name) @function.def

(import_statement) @import.stmt
(import_from_statement) @import.from

(call) @call.expr
