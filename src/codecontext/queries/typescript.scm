(class_declaration
  name: (type_identifier) @class.name) @class.def

(function_declaration
  name: (identifier) @function.name) @function.def

(function_signature
  name: (identifier) @sig.name) @sig.def

(method_definition
  name: (_) @method.name) @method.def

(method_signature
  name: (_) @msig.name) @msig.def

(interface_declaration
  name: (type_identifier) @interface.name) @interface.def

(type_alias_declaration
  name: (type_identifier) @type.name) @type.def

(enum_declaration
  name: (identifier) @enum.name) @enum.def

(internal_module
  name: (_) @namespace.name) @namespace.def

(variable_declarator
  name: (identifier) @binding.name) @binding.def

(import_statement) @import.stmt
(export_statement) @export.stmt

(call_expression) @call.expr
