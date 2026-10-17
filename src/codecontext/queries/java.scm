(class_declaration
  name: (identifier) @class.name) @class.def

(interface_declaration
  name: (identifier) @interface.name) @interface.def

(enum_declaration
  name: (identifier) @enum.name) @enum.def

(method_declaration
  name: (identifier) @method.name) @method.def

(import_declaration) @import.stmt
(method_invocation) @call.expr
