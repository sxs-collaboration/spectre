\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Protocols {#protocols}

\tableofcontents

# Overview of protocols {#protocols_overview}

Protocols are a concept we use in SpECTRE to define metaprogramming interfaces.
A variation of this concept is built into many languages, so this is a quote
from the [Swift documentation](https://docs.swift.org/swift-book/LanguageGuide/Protocols.html):

> A protocol defines a blueprint of methods, properties, and other requirements
> that suit a particular task or piece of functionality. The protocol can then
> be adopted by a class, structure, or enumeration to provide an actual
> implementation of those requirements. Any type that satisfies the requirements
> of a protocol is said to conform to that protocol.

You should define a protocol when you need a template parameter to conform to an
interface. Here is an example of a protocol that is adapted from the [Swift
documentation](https://docs.swift.org/swift-book/LanguageGuide/Protocols.html):

\snippet Utilities/Test_ProtocolHelpers.cpp named_protocol

The protocol defines an interface that any type that adopts it must implement.
For example, the following class conforms to the protocol we just defined:

\snippet Utilities/Test_ProtocolHelpers.cpp named_conformance

The class indicates it conforms to the protocol by (publicly) inheriting from
`tt::ConformsTo<TheProtocol>`.

Once you have defined a protocol, you can check if a class conforms to it using
the `tt::assert_conforms_to` or `tt::conforms_to` metafunctions:

\snippet Utilities/Test_ProtocolHelpers.cpp conforms_to

Note that checking for protocol conformance is cheap, so you may freely use
protocol conformance checks in your code.

This is how you can write code that relies on the interface defined by the
protocol:

\snippet Utilities/Test_ProtocolHelpers.cpp using_named_protocol

Checking for protocol conformance here makes it clear that we are expecting
a template parameter that exposes the particular interface we have defined in
the protocol. Therefore, the author of the protocol and of the code that uses it
has explicitly defined (and documented!) the interface they expect. And the
developer who consumes the protocol by writing classes that conform to it knows
exactly what needs to be implemented.

Note that the `tt::conforms_to` metafunction is SFINAE-friendly, so you can use
it like this:

\snippet Utilities/Test_ProtocolHelpers.cpp protocol_sfinae

The `tt::conforms_to` metafunction only checks if the class _indicates_ it
conforms to the protocol. Where SFINAE-friendliness is not necessary prefer the
`tt::assert_conforms_to` metafunction that triggers static asserts with
diagnostic messages to understand why the class does not conform to the
protocol.

We typically define protocols in a file named `Protocols.hpp` and within a
`protocols` namespace, similar to how we write \ref DataBoxTagsGroup "tags" in a
`Tags.hpp` file and within a `Tags` namespace. The file should be placed in the
directory associated with the code that depends on classes conforming to the
protocols. For example, the protocol `Named` in the example above would be
placed in directory that also has the `greet` function.

# Protocol users: Conforming to a protocol {#protocols_conforming}

To indicate a class conforms to a protocol it (publicly) inherits from
`tt::ConformsTo<TheProtocol>`. The class must fulfill all requirements defined
by the protocol. The requirements are listed in the protocol's documentation.

Any class that indicates it conforms to a protocol must have a unit test to
check that it actually does. You can use the `tt::assert_conforms_to`
metafunction for the test:

\snippet Utilities/Test_ProtocolHelpers.cpp test_protocol_conformance

# Protocol authors: Writing a protocol {#protocols_author}

To author a new protocol you implement a class that provides a `test`
metafunction and detailed documentation. The `test` metafunction takes a single
template parameter (typically named `ConformingType`) and checks that it
conforms to the requirements laid out in the protocol's documentation. Its
purpose is to provide diagnostic messages as compiler errors to understand why a
type fails to conform to the protocol. You can use `static_assert`s or trigger
standard compiler errors where appropriate. See the protocols defined above for
examples.

Occasionally, you might be tempted to add template parameters to a protocol. In
those situations, add requirements to the protocol instead and retrieve the
parameters from the conforming class. The reason for this guideline is that
conforming classes will always inherit from `tt::ConformsTo<Protocol>`.
Therefore, any template parameters of the protocol must also be template
parameters of their conforming classes, which means the protocol can just
require and retrieve them. For example, we could be tempted to follow this
antipattern:

\snippet Utilities/Test_ProtocolHelpers.cpp named_antipattern

However, instead of adding template parameters to the protocol we should add a
requirement to it:

\snippet Utilities/Test_ProtocolHelpers.cpp named_with_type

Classes would need to specify the template parameters for any
protocols they conform to anyway, if the protocols had any. So they might as
well expose them:

\snippet Utilities/Test_ProtocolHelpers.cpp person_with_name_type

This pattern also allows us to check for protocol conformance first and then add
further checks about the types if we wanted to:

\snippet Utilities/Test_ProtocolHelpers.cpp example_check_name_type

# Protocol authors: Testing a protocol {#protocols_testing}

Protocol authors should provide a unit test for their protocol that includes an
example implementation of a class that conforms to it. The protocol author
should add this example to the documentation of the protocol through a Doxygen
snippet. This gives users a convenient way to see how the author intends their
interface to be implemented.

# Protocols and C++20 "Constraints and concepts" {#protocols_and_constraints}

A feature related to protocols is in C++20 and goes under the name of
[constraints and concepts](https://en.cppreference.com/w/cpp/language/constraints).
Every protocol defines a _concept_, but it defers checking its requirements to
the unit tests to save compile time. In other words, protocols provide a way to
_indicate_ that a class fulfills a set of requirements, whereas C++20
constraints provide a way to _check_ that a class fulfills a set of
requirements. Therefore, the two features complement each other. Once C++20
becomes available in SpECTRE we can either gradually convert our protocols to
concepts and use them as constraints directly if we find the impact on compile
time negligible, or we can add a concept that checks protocol conformance the
same way that `tt::conforms_to_v` currently does (i.e. by checking inheritance).
