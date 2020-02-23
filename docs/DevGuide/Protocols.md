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
interface. Protocols are implemented as unary type traits. Here is an example of
a protocol that is adapted from the
[Swift documentation](https://docs.swift.org/swift-book/LanguageGuide/Protocols.html):

\snippet Utilities/Test_ProtocolHelpers.cpp named_protocol

The protocol defines an interface that any type that adopts it must implement.
For example, the following class conforms to the protocol we just defined:

\snippet Utilities/Test_ProtocolHelpers.cpp named_conformance

The class indicates it conforms to the protocol by (publicly) inheriting from
`tt::ConformsTo<TheProtocol>`.

Once you have defined a protocol, you can check if a class conforms to it using
the `tt::conforms_to` metafunction:

\snippet Utilities/Test_ProtocolHelpers.cpp conforms_to

Note that checking for protocol conformance is cheap, because the
`tt::conforms_to` metafunction only checks if the class _indicates_ it conforms
to the protocol via the above inheritance. The rigorous test whether the class
actually fullfills all of the protocol's requirements is deferred to its unit
tests (see \ref protocols_testing_conformance). Therefore you may freely use
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

Note that the `tt::conforms_to` metafunction is SFINAE-friendly, so you can also
use it like this:

\snippet Utilities/Test_ProtocolHelpers.cpp protocol_sfinae

We typically define protocols in a file named `Protocols.hpp` and within a
`protocols` namespace, similar to how we write \ref DataBoxTagsGroup "tags" in a
`Tags.hpp` file and within a `Tags` namespace. The file should be placed in the
directory associated with the code that depends on classes conforming to the
protocols. For example, the protocol `Named` in the example above would be
placed in directory that also has the `greet` function.

# Protocol users: Testing protocol conformance {#protocols_testing_conformance}

Any class that indicates it conforms to the protocol must test that it actually
does using the `test_protocol_conformance` metafunction from
`tests/Unit/ProtocolTestHelpers.hpp`:

\snippet Utilities/Test_ProtocolHelpers.cpp test_protocol_conformance

# Protocol authors: Protocols must be unary type traits {#protocols_author}

When you author a new protocol, keep in mind that protocols must be unary type
traits. This means they take a single template parameter (typically named
`ConformingType`) and inherit from `std::true_type` or `std::false_type`
depending on whether the `ConformingType` fullfills the protocol's requirements.
Make sure to implement the protocol in a SFINAE-friendly way. You may find the
macros in `Utilities/TypeTraits.hpp` useful. For example, we use
`CREATE_IS_CALLABLE` in the protocols above for testing the existence and return
type of a member function.

Occasionally, you might be tempted to add additional template parameters to the
protocol. In those situations, make the additional parameters part of your
protocol instead. The reason for this guideline is that protocols
will always be used as unary type traits when inheriting from
`tt::ConformsTo<Protocol>`. Therefore, any template parameters of the protocol
must also be template parameters of their conforming classes, which means the
protocol can just check them.

For example, we could be tempted to follow this antipattern:

\snippet Utilities/Test_ProtocolHelpers.cpp named_antipattern

However, instead of making the protocol a non-unary template we should add a
requirement to it:

\snippet Utilities/Test_ProtocolHelpers.cpp named_with_type

Classes would need to specify the additional template parameters for any
protocols they conform to anyway, if the protocols had any. So they might as
well expose them:

\snippet Utilities/Test_ProtocolHelpers.cpp person_with_name_type

This pattern also allows us to check for protocol conformance first and then add
further checks about the types if we wanted to:

\snippet Utilities/Test_ProtocolHelpers.cpp example_check_name_type

# Protocol authors: Testing a protocol {#protocols_testing}

We are currently testing protocol conformance as part of our unit tests, so
that the global `tt::conforms_to` convenience metafunction only needs to check
if a type inherits off the protocol, but doesn't need to check the protocol's
(possibly fairly expensive) implementation. This is primarily to keep compile
times low, and may be reconsidered when transitioning to C++ "concepts". Full
protocol conformance is tested in the `test_protocol_conformance` metafunction
mentioned above.

To make sure their protocol functions correctly, protocol authors must test
its implementation in a unit test (e.g. in a `Test_Protocols.hpp`):

\snippet Utilities/Test_ProtocolHelpers.cpp testing_a_protocol

They should make sure to test the implementation with classes that conform to
the protocol, and others that don't. This means the test will always include an
example implementation of a class that conforms to the protocol, and the
protocol author should add it to the documentation of the protocol through a
Doxygen snippet. This gives users a convenient way to see how the author intends
their interface to be implemented.

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
