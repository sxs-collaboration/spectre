\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Option Parsing {#dev_guide_option_parsing}

\tableofcontents

SpECTRE can read YAML configuration files at runtime to set parameters
and choose between classes implementing an interface.  %Options are
parsed during code initialization and can be used to construct objects
placed in the Parallel::GlobalCache and options passed to the
parallel components.  The types necessary to mark objects for parsing
are declared in `Options/Options.hpp`.

## Metadata and options

YAML input files begin with a metadata section, terminated by `---`, and
followed by the executable options:

```yaml
# Metadata here
Description: |
  Briefly describe the configuration and link to papers for details.
---
# Options start here
```

The metadata section may also be empty:

```yaml
---
---
# Options start here
```

You only need the leading `---` marker if the metadata section is empty. This is
YAML's "document start marker" (see the [YAML spec](https://yaml.org/spec/1.2)).
Any metadata fields at the beginning of the file also imply the start of a
document, so you don't need the first `---` marker.

Metadata provide information for tools, whereas options provide information to
the executable. See tools like `CheckOutputFiles` for details on the metadata
fields that they use. Metadata can also provide information on how to run the
input file, such as the name and version of the executable, and a description
that may refer to published papers for details on the configuration.
Options are defined by the executable and detailed below.

## General option format

An option is defined by an "option tag", represented by a `struct`.  At minimum,
the struct must declare the type of the object to be parsed and provide a brief
description of the meaning.  The name of the option in the input file
defaults to the name of the struct (excluding any template parameters
and scope information), but can be overridden by providing a static
`name()` function.  Several other pieces of information, such as
suggestions, limits and grouping, may be provided if desired.  This
information is all included in the generated help output.

If an option has a suggested value, the value is specified in the
input file as usual, but a warning will be issued if the specified
value does not match the suggestion.

Examples:
\snippet Options/Test_Options.cpp options_example_scalar_struct
\snippet Options/Test_Options.cpp options_example_vector_struct

The option type can be any type understood natively by yaml-cpp
(fundamentals, `std::string`, and `std::map`, `std::vector`,
`std::list`, `std::array`, and `std::pair` of parsable types) and
types SpECTRE adds support for.  SpECTRE adds `std::unordered_map`
(but only with ordered keys), `std::variant` (with alternatives tested
in order), and various classes marked as constructible in their
declarations.

An option tag can be placed in a group by adding a `group` type alias to the
struct. The alias should refer to a type that, like option tags, defines a help
string and may override a static `name()` function.

Example:
\snippet Options/Test_Options.cpp options_example_group

## Constructible classes

A class that defines `static constexpr Options::String help` and a
typelist of option structs `options` can be created by the option
parser.  When the class is requested, the option parser will parse
each of the options in the `options` list, and then supply them to the
constructor of the class.  A class can use Options::Alternatives to
support more than one possible set of options for its creation.  (See
[Custom parsing](#custom-parsing) below for more general creation
mechanisms.)

Unlike option descriptions, which should be brief, the class help
string has no length limits and should give a description of the class
and any necessary discussion of its options beyond what can be
described in their individual help strings.

Creatable classes must be default constructible and move assignable.

The `Options::Context` is an optional argument to the constructor that should be
used when the constructor checks for validity of the input. If the input is
invalid, `PARSE_ERROR` is used to propagate the error message back through the
options ensuring that the error message will have a full backtrace so it is easy
for the user to diagnose.

Example:
\snippet Options/Test_CustomTypeConstruction.cpp class_creation_example

Classes may use the Metavariables struct, which is effectively the compile time
input file, in their parsing by templating the `options` type alias or by taking
the Metavariables as a final argument to the constructor (after the
`Options::Context`).

Example:
\snippet Options/Test_CustomTypeConstruction.cpp class_creation_example_with_metavariables

## Factory

The factory interface creates an object of type
`std::unique_ptr<Base>` containing a pointer to some class derived
from `Base`.  The list of creatable derived classes is specified in
the `factory_creation` struct in the metavariables, which must contain
a `factory_classes` type alias that is a `tmpl::map` from base classes
to lists of derived classes:
\snippet Options/Test_Factory.cpp factory_creation

When a `std::unique_ptr<Base>` is requested, the factory will expect a
single YAML argument specifying the name of the class (as given by a
static `name()` function or, lacking that, the actual class name).  If
the derived class takes no arguments, the name can be given as a YAML
string, otherwise it must be given as a single key-value pair, with
the key the name of the class.  The value portion of this pair is then
used to create the requested derived class in the same way as an
explicitly constructible class.  Examples:
\snippet Options/Test_Factory.cpp factory_without_arguments
\snippet Options/Test_Factory.cpp factory_with_arguments

\anchor custom-parsing
## Custom parsing

Occasionally, the requirements imposed by the default creation
mechanism are too stringent.  In these cases, the construction
algorithm can be overridden by providing a specialization of the
struct
\code{cpp}
template <typename T>
struct Options::create_from_yaml {
  template <typename Metavariables>
  static T create(const Options::Option& options);
};
\endcode
The create function can perform any operations required to construct
the object.

Example of using a specialization to parse an enum:
\snippet Options/Test_CustomTypeConstruction.cpp enum_creation_example

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

\snippet Options/Test_CustomTypeConstruction.cpp enum_void_creation_header_example

while in the `cpp` file the definition of the `void` specialization is:

\snippet Options/Test_CustomTypeConstruction.cpp enum_void_creation_cpp_example
