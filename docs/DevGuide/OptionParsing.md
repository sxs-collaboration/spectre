\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Option Parsing {#dev_guide_option_parsing}

SpECTRE can read YAML configuration files at runtime to set parameters
and choose between classes implementing an interface.  %Options are
parsed during code initialization and can be used to construct objects
placed in the Parallel::ConstGlobalCache and options passed to the
parallel components.  The types necessary to mark objects for parsing
are declared in `Options/Options.hpp`.

## General option format

An option is defined by an "option tag", represented by a `struct`.  At minimum,
the struct must declare the type of the object to be parsed and provide a brief
description of the meaning.  The name of the option in the input file
defaults to the name of the struct (excluding any template parameters
and scope information), but can be overridden by providing a static
`name()` function.  Several other pieces of information, such as
defaults, limits and grouping, may be provided if desired.  This information is
all included in the generated help output.

Examples:
\snippet Test_Options.cpp options_example_scalar_struct
\snippet Test_Options.cpp options_example_vector_struct

The option type can be any type understood natively by yaml-cpp
(fundamentals, `std::string`, and `std::map`, `std::vector`,
`std::list`, `std::array`, and `std::pair` of parsable types) and
types SpECTRE adds support for.  SpECTRE adds `std::unordered_map`
(but only with ordered keys), and various classes marked as
constructible in their declarations.

An option tag can be placed in a group by adding a `group` type alias to the
struct. The alias should refer to a type that, like option tags, defines a help
string and may override a static `name()` function.

Example:
\snippet Test_Options.cpp options_example_group

## Constructible classes

A class that defines `static constexpr OptionString help` and a
typelist of option structs `options` can be created by the option
parser.  When the class is requested, the option parser will parse
each of the options in the `options` list, and then supply them to the
constructor of the class.  (See [Custom parsing](#custom-parsing)
below for more general creation mechanisms.)

Unlike option descriptions, which should be brief, the class help
string has no length limits and should give a description of the class
and any necessary discussion of its options beyond what can be
described in their individual help strings.

Creatable classes must be default constructible and move assignable.

The `OptionContext` is an optional argument to the constructor that should be
used when the constructor checks for validity of the input. If the input is
invalid, `PARSE_ERROR` is used to propagate the error message back through the
options ensuring that the error message will have a full backtrace so it is easy
for the user to diagnose. Finally, after the `OptionContext` the constructor may
optionally take the Metavariables struct, which is effectively the compile time
input file, and the constructor can use the Metavariables for whatever it wants,
including additional option parsing.

Example:
\snippet Test_CustomTypeConstruction.cpp class_creation_example

## Factory

The factory interface creates an object of type
`std::unique_ptr<Base>` containing a pointer to some class derived
from `Base`.  The base class must define a type alias listing the
derived classes that can be created.
\code{cpp}
 using creatable_classes = tmpl::list<Derived1, ...>;
\endcode
The requested derived class is created in the same way as an
explicitly constructible class.

## <a name="custom-parsing"></a>Custom parsing

Occasionally, the requirements imposed by the default creation
mechanism are too stringent.  In these cases, the construction
algorithm can be overridden by providing a specialization of the
struct
\code{cpp}
template <typename T>
struct create_from_yaml {
  template <typename Metavariables>
  static T create(const Option& options);
};
\endcode
The create function can perform any operations required to construct
the object.

Example of using a specialization to parse an enum:
\snippet Test_CustomTypeConstruction.cpp enum_creation_example

Note that in the case where the `create` function does *not* need to use the
`Metavariables` it is recommended that a general implementation forward to an
explicit instantiation with `void` as the `Metavariables` type. The reason for
using `void` specialization is to reduce compile time. Since we only need one
full implementation of the function independent of what type `Metavariables` is,
we should only parse and compile it once. By having a specialization on `void`
(or some other non-metavariables type like `NoSuchType`) we can handle the
metavariables-independent case efficiently. As a concrete example, the general
definition and forward declaration of the `void` specialization in the header
file would be:

\snippet Test_CustomTypeConstruction.cpp enum_void_creation_header_example

while in the `cpp` file the definition of the `void` specialization is:

\snippet Test_CustomTypeConstruction.cpp enum_void_creation_cpp_example
