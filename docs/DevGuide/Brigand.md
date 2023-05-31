\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Metaprogramming with Brigand {#brigand}

\tableofcontents

\note
This document covers Brigand as of commit
[66b3d9276ed95425ac919ac1841286d088b5f4b1](https://github.com/edouarda/brigand/commit/66b3d9276ed95425ac919ac1841286d088b5f4b1)
in January 2022.

\tableofcontents{HTML:2}

[comment]: # (The \pars improve the spacing in the generated document when)
[comment]: # (many \snippets are involved.)

\par
In SpECTRE, most complex TMP is done using the [Brigand metaprogramming
library](https://github.com/edouarda/brigand).  Brigand is a collection of
templated classes, type aliases, and functions, primarily intended to help with
the manipulation and use of lists of types.  This document is organized to
roughly parallel the structure of the C++ standard, rather than following
Brigand's classifications.

\par
Brigand provides all of its functionality in the `brigand` namespace, but in
SpECTRE we have aliased this namespace to `tmpl`, and the latter should be
preferred.

\par
All functionality described here is provided by SpECTRE's Brigand wrapper
header:
\snippet Test_TMPLDocumentation.cpp include

\par
Examples in this document use the following declarations and definitions:
\snippet Test_TMPLDocumentation.cpp example_declarations


\section Metafunctions

\par
A metafunction is an analog of a familiar C++ function that is coded in the C++
type system.  It turns out that, using metafunction programming, it is possible
to perform arbitrary computations at compile time.

\par
There are multiple ways to encode a calculation in the type system.  When using
Brigand, the relevant ones are eager and lazy metafunctions.


\subsection lazy Eager and lazy metafunctions

\par
Metafunctions commonly appear in two forms: eager and lazy.  Lazy metafunctions
are templated structs (or templated aliases to structs) with a `type` member
alias that indicates the result:
\snippet Test_TMPLDocumentation.cpp metafunctions:lazy
The type traits in the standard library, such as std::is_same, are lazy
metafunctions.

\par
Eager metafunctions are aliases to their result types.  As a trivial case,
struct templates can be viewed as eager metafunctions returning themselves.  An
eager version of the previous example could be implemented as:
\snippet Test_TMPLDocumentation.cpp metafunctions:eager
The standard library provides eager versions of some of its metafunctions
(generally those that modify a type, rather than predicates) using an `_t`
suffix.  When both versions are provided, it is often convenient (and less
error prone!) to define the eager version in terms of the lazy version:
\snippet Test_TMPLDocumentation.cpp metafunctions:eager_from_lazy
And the two definitions agree:
\snippet Test_TMPLDocumentation.cpp metafunctions:agreement

\note
The standard library also provides versions of many of its type traits with an
`_v` suffix.  These evaluate to compile-time *values*, rather than types.  They
can be useful for metaprogramming, but are not the types of metafunctions being
discussed here.

\par
Eager metafunctions are usually more convenient to use, so what is the point of
additionally creating lazy ones?  The answer is that lazy metafunctions can be
used as compile-time functors.  As a simple example, we can write an (eager)
metafunction that calls an arbitrary lazy metafunction twice
\snippet Test_TMPLDocumentation.cpp metafunctions:call_lazy_metafunction
and get the expected output:
\snippet Test_TMPLDocumentation.cpp metafunctions:call_lazy_metafunction_assert
But it fails if you try to call an arbitrary *eager* metafunction
twice in the same way, because the function is evaluated too early,
resulting in the `List1` metafunction being acted upon instead of
`eager_add_list`:
\snippet Test_TMPLDocumentation.cpp metafunctions:call_eager_metafunction
\snippet Test_TMPLDocumentation.cpp metafunctions:call_eager_metafunction_assert
(In this simple case we could have used a template template parameter to pass
the eager metafunction in a form more similar to a runtime lambda, but the
possibilities for generic manipulation of parameter lists containing template
template parameters are limited, so their use must be minimized in complex
metaprogramming.)

\note
In practice, lazy metafunctions are often implemented as empty structs
inheriting from other lazy metafunctions.  The entire inheritance chain then
inherits a `type` alias from the ultimate base class.

\par
Most of the standard Brigand functions are eager, but many have lazy versions
in the nested `tmpl::lazy` namespace.  These are indicated by calls to the
`HAS_LAZY_VERSION` macro in the examples below.


\subsection metalambdas Brigand metalambdas

\par
This use of lazy metafunctions is too limited for general use, however, because
it requires the definition of a new templated struct for every new function.
Brigand uses a more general notation, known as metalambdas.  A metalambda is a
(possibly nested set of) lazy metafunctions with some template arguments
replaced by the placeholders `tmpl::_1`, `tmpl::_2`, etc.  These are the first,
second, etc., arguments of the metalambda, and will be replaced by the actual
arguments when the lambda is used.  The lazy nature of the metafunctions
prevents them from prematurely evaluating to results based on the literal
placeholder types.  The \ref apply "tmpl::apply" function can be used to
evaluate a metalambda with specified arguments, and many other Brigand
functions take metalambdas that are evaluated internally.


\subsection metalambda_structure Evaluation of metalambdas

\note
None of the terminology introduced in this section is standard.

\par
When evaluating a metalambda, the values of any \ref args "arguments"
encountered are taken from the evaluation's argument stack.  The argument stack
is a stack (in the CS sense) of collections of zero or more arguments.  The
values of any \ref args "arguments" that are evaluated are taken from the
collection at the head of the stack.  The remaining collections are arguments
captures in closures, and will not be present in code not using \ref defer
"tmpl::defer".

\par
The \ref apply "tmpl::apply" metafunction is the Brigand metafunction for
evaluating metalambdas.  It can be called explicitly, and is called internally
by many other functions.  It takes a metalambda and arguments to be passed to
that metalambda.  The argument stack for the evaluation has one entry: the
passed arguments.

\par
The argument stack can gain additional entries through the creation of
metaclosures using \ref defer "tmpl::defer".  When \ref defer "tmpl::defer" is
evaluated, it produces a \ref metalambda_metaclosure "metaclosure" containing a
copy of the current argument stack, acting as the lambda captures.  When that
metaclosure is evaluated, the previously active stack is replaced by the stored
stack with the head of the old stack pushed onto it.

\par
This makes \ref args "arguments" in a \ref metalambda_metaclosure "metaclosure"
refer to the arguments "passed" to it.  (No explicit call syntax is used, but
the arguments are inherited from the calling context.)  The captured arguments
are accessible using \ref parent "tmpl::parent", which pops off the last entry
in the argument stack.

\par
There are eight forms that a metalambda can take: an argument, a lazy
expression, a bind expression, a pin expression, a defer expression, a parent
expression, a constant, or a metaclosure.

\subsubsection args Argument

\par
An argument is one of the structs `tmpl::_1`, `tmpl::_2`, or `tmpl::args<n>`
for `unsigned int` n.  The additional aliases `tmpl::_3`, `tmpl::_4`, ...,
`tmpl::_9` are provided to `tmpl::args<2>`, `tmpl::args<3>`, ...,
`tmpl::args<8>`.  (The first two arguments have dedicated types in addition to
the general `tmpl::args<0>` and `tmpl::args<1>`.  My best guess is for
performance reasons.)
\snippet Test_TMPLDocumentation.cpp tmpl::args
When evaluated, they give the first (`tmpl::_1`), second (`tmpl::_2`), or
zero-indexed Nth (`tmpl::args<N>`) entry from the collection of arguments at
the top of the argument stack.
\snippet Test_TMPLDocumentation.cpp tmpl::args:eval
Additionally, `tmpl::_state` and `tmpl::_element` are aliased to `tmpl::_1` and
`tmpl::_2`, primarily for use with \ref fold "tmpl::fold".

\par
When evaluating a metalambdas, the metalambda must be passed enough arguments
to define all argument placeholders in its body.  When evaluating using \ref
apply "tmpl::apply", arguments are passed as template parameters after the
metalambda.  Other Brigand functions that evaluate metalambdas pass them a
specified number of arguments (usually 1 or 2).  Failure to pass enough
arguments may error or produce unintuitive results.

\subsubsection metalambda_lazy Lazy expression

\par
A lazy expression is a fully-specialized struct template with only type
template parameters that is not a specialization of \ref pin "tmpl::pin", \ref
defer "tmpl::defer", or \ref parent "tmpl::parent" and is not a \ref
metalambda_metaclosure "metaclosure".  When evaluated, each of its template
parameters is evaluated as a metalambda and replaced by the result, and then
the struct's `type` type alias is the result of the full lazy-expression.
\snippet Test_TMPLDocumentation.cpp metalambda_lazy

\subsubsection bind Bind expression

\par
A bind expression is a specialization of `tmpl::bind`.  It wraps an eager
metafunction and its arguments.  When evaluated, the arguments are each
evaluated as metalambdas, and then the results are passed to the eager
metafunction.
\snippet Test_TMPLDocumentation.cpp tmpl::bind

\note
The `tmpl::bind` metafunction does not convert an eager metafunction to a lazy
one.  It is handled specially in the evaluation code.

\subsubsection pin Pin expression

\par
A pin expression is a specialization of `tmpl::pin`.  Evaluating a pin
expression gives the (unevaluated) argument to `tmpl::pin`.  This can be used
to force a type to be treated as a \ref metalambda_constant "constant", even if
it would normally be treated as a different type of metalambda (usually a \ref
metalambda_lazy "lazy expression").
\snippet Test_TMPLDocumentation.cpp tmpl::pin

\par
Pin expressions are often used to protect template arguments to eager
metafunctions:
\snippet Test_TMPLDocumentation.cpp tmpl::pin:protect_eager

\subsubsection defer Defer expression

\par
A defer expression is a specialization of `tmpl::defer`.  It does not evaluate
its argument, but results in a \ref metalambda_metaclosure "metaclosure"
containing the passed metalambda and the current argument stack.

\par
Example:
\snippet Test_TMPLDocumentation.cpp tmpl::defer
The evaluation here proceeds as follows:

1. The innermost eager metafunction is the second \ref apply "tmpl::apply".  It
   creates an argument stack with one collection containing the single argument
   `Type1` and proceeds to evaluate `tmpl::defer<tmpl::_1>`.

2. Evaluating the defer expression creates a \ref metalambda_metaclosure
   "metaclosure" with the contents `tmpl::_1` and the argument stack from the
   first apply.  This is the result of the inner \ref apply "tmpl::apply".

3. Next the outer \ref apply "tmpl::apply" is evaluated.  It creates an
   argument stack with one collection containing the single argument`Type2`,
   and proceeds to evaluate the \ref metalambda_metaclosure "metaclosure"
   created above.

[comment]: # (Keep the numbering here in sync with the `parent` example.)

4. Evaluating the \ref metalambda_metaclosure "metaclosure" (see that section
   below) evaluates the contained `tmpl::_1` with a two-element argument stack:
   head to tail [(`Type2`), (`Type1`)].

5. The first (and only) argument (`tmpl::_1`) in the head of the argument stack
   is `Type2`, which is the result of the \ref metalambda_metaclosure
   "metaclosure" and the entire expression.


\par
The primary purposes for `tmpl::defer` are constructing metalambdas to pass to
other metafunctions and preventing "speculative" evaluation of a portion of a
metalambda that is not valid for some arguments.  See the examples below, in
particular \ref make_subtracter, \ref multiplication_table, \ref maybe_first,
and \ref column_with_zeros.

\subsubsection parent Parent expression

\par
A parent expression is a specialization of `tmpl::parent`.  It evaluates its
argument (treated as a metalambda) after popping the top entry off the argument
stack.  This provides access to the captured arguments in a \ref
metalambda_metaclosure "metaclosure".

\par
Example:
\snippet Test_TMPLDocumentation.cpp tmpl::parent
The creation of the \ref metalambda_metaclosure "metaclosure" here is similar
to the example for \ref defer "tmpl::defer", except that the contained
metalambda is `tmpl::parent<tmpl::_1>` instead of a plain `tmpl::_1`.  The
evaluation proceeds as:

1. As in \ref defer "tmpl::defer" example.

2. As in \ref defer "tmpl::defer" example.

3. As in \ref defer "tmpl::defer" example.

4. The \ref metalambda_metaclosure "metaclosure" evaluates the contained
   `tmpl::parent<tmpl::_1>` with the argument stack [(`Type2`), (`Type1`)].

5. Evaluating the `tmpl::parent` pops the stack, and so evaluates `tmpl::_1`
   with the stack [(`Type1`)].

6. The first (and only) argument (`tmpl::_1`) in the stack's head is now
   `Type1`, which is the result of the \ref metalambda_metaclosure
   "metaclosure" and the entire expression.

\warning
Do not call `tmpl::parent` when the argument stack is empty, i.e., do not
attempt to access more sets of captured arguments than have been captured.  If
you want to prevent evaluation of an expression, use \ref pin "tmpl::pin".

\subsubsection metalambda_constant Constant

\par
A constant metalambda is any type that is not a struct template with only type
template parameters, a specialization of \ref bind "tmpl::bind", or a
metaclosure.  A constant metalambda evaluates to itself.
\snippet Test_TMPLDocumentation.cpp metalambda_constant

\subsubsection metalambda_metaclosure Metaclosure

\par
A metaclosure is an opaque type produced by \ref defer "tmpl::defer",
containing a metalambda and an argument stack.  When a metaclosure is
evaluated, it evaluates the packaged metalambda with an argument stack
constructed by pushing the head of the current argument stack onto the argument
stack stored in the metaclosure.  See \ref defer and \ref parent for examples.


\subsection Examples


\subsubsection evens

\par
Finds all numbers in a list that are even.
\snippet Test_TMPLDocumentation.cpp metafunctions:evens
\snippet Test_TMPLDocumentation.cpp metafunctions:evens:asserts

\par
The \ref filter "tmpl::filter" metafunction takes a metalambda as its second
argument.  The \ref integral_constant "tmpl::integral_constant"s have non-type
template parameters, so they are treated as constant expressions.  The \ref
math_comparison "tmpl::equal_to" and \ref math_arithmetic "tmpl::modulo"
metafunctions are lazy, despite not being in the `tmpl::lazy` namespace.


\subsubsection maybe_first

\par
Returns the first element of a list, or \ref no_such_type_ "tmpl::no_such_type_"
if the list is empty.
\snippet Test_TMPLDocumentation.cpp metafunctions:maybe_first
\snippet Test_TMPLDocumentation.cpp metafunctions:maybe_first:asserts

\par
In this example, the inner \ref apply "tmpl::apply" call evaluates the \ref if_
"tmpl::if_" statement, returning either a \ref metalambda_metaclosure
"metaclosure" or \ref no_such_type_ "tmpl::no_such_type_".  The outer \ref
apply "tmpl::apply" either evaluates the metaclosure, calling \ref front
"tmpl::front", or evaluates \ref no_such_type_ "tmpl::no_such_type_", which is
a constant and gives itself.

\par
The reason for creating a metaclosure is that all the arguments to \ref if_
"tmpl::if_" are always evaluated (it is an ordinary metafunction with no
special treatment during evaluation).  This is a problem, because, in a naive
attempt at this metafunction, if `L` is an empty list
`tmpl::front<tmpl::list<>>` would be evaluated for the first branch.  To avoid
this we use \ref defer "tmpl::defer" to wrap the call to \ref front
"tmpl::front" in a metaclosure, which we evaluate only if necessary.

\par
Note that this metaclosure does not capture anything.  `L` is substituted
according to normal C++ rules before any Brigand evaluation.  This means that
the contents of the `tmpl::defer` are, for the first call above,
`tmpl::bind<tmpl::front, tmpl::pin<tmpl::list<Type1>>>`.  Without the \ref pin
"tmpl::pin", the list would be interpreted as a lazy metafunction, resulting in
an error because it does not have a `type` type alias.

\par
The `L` in `tmpl::size<L>` does not need to be protected by a \ref pin
"tmpl::pin" because \ref size "tmpl::size" is an eager metafunction, so that
expression has been converted to a \ref integral_constant
"tmpl::integral_constant" before the metalambda evaluation starts.


\subsubsection factorial

\par
Calculates the factorial using a simple metalambda passed to a \ref fold
"tmpl::fold".
\snippet Test_TMPLDocumentation.cpp metafunctions:factorial
\snippet Test_TMPLDocumentation.cpp metafunctions:factorial:asserts

\par
A nearly literal rewrite of this into runtime C++ is
\snippet Test_TMPLDocumentation.cpp metafunctions:factorial_cpp

\par
The equivalent of a range-based for loop is easier to express in functional
programming (as a fold) than a standard counting for loop.


\subsubsection make_subtracter

\par
Demonstrates the use of captures in metalambdas.
\snippet Test_TMPLDocumentation.cpp metafunctions:make_subtracter
\snippet Test_TMPLDocumentation.cpp metafunctions:make_subtracter:asserts

\par
This metafunction returns a \ref metalambda_metaclosure "metaclosure" that
subtracts a given number from it's argument.  That metaclosure uses both the
argument passed to it (`tmpl::_1`, which is 5 in the example) and the value
captured at it's creation (`tmpl::parent<tmpl::_1>`, which is 3 in the
example).

\par
(This `make_subtracter` could be implemented more simply as
\snippet Test_TMPLDocumentation.cpp metafunctions:make_subtracter_simple
but that doesn't demonstrate metaclosures.)


\subsubsection multiplication_table

\par
Constructs a multiplication table.
\snippet Test_TMPLDocumentation.cpp metafunctions:multiplication_table
\snippet Test_TMPLDocumentation.cpp metafunctions:multiplication_table:asserts

\par
This demonstrates the use of \ref defer "tmpl::defer" to pass a closure as an
argument to a metafunction (\ref transform "tmpl::lazy::transform"), while
capturing an argument from the outer context (the metalambda evaluated for the
outer \ref transform "tmpl::transform").  This is the use most similar to
common uses of the C++ lambda.

\par
The outer (eager) \ref transform "tmpl::transform" evaluates its second
argument as a metalambda.  This first evaluates the arguments to the inner \ref
transform "tmpl::lazy::transform".  The first argument is a \ref list
"tmpl::list" of \ref integral_constant "tmpl::integral_constant"s (because the
\ref range "tmpl::range" is eager and has already been evaluated).  This must
be protected by a \ref pin "tmpl::pin" because it looks like a lazy
metafunction.  The second argument gives a metaclosure, capturing the value
from the outer \ref transform "tmpl::transform" (available as
`tmpl::parent<tmpl::_1>`).  The \ref list "tmpl::list" (without the \ref pin
"tmpl::pin", which has already been evaluated) and metaclosure are then passed
to the inner \ref transform "tmpl::lazy::transform".


\subsubsection column_with_zeros

\par
Extracts a column from a row-major matrix, extending any short rows with zeros.
\snippet Test_TMPLDocumentation.cpp metafunctions:column_with_zeros
\snippet Test_TMPLDocumentation.cpp metafunctions:column_with_zeros:asserts

\par
This example shows another use of \ref defer "tmpl::defer" to avoid evaluating
an invalid expression, similar to \ref maybe_first.  The use of an \ref args
"argument" in the deferred branch makes this case more complicated: a \ref
parent "tmpl::parent" expression is used to access arguments from where the
\ref defer "tmpl::defer" occurs to avoid having to pass the argument explicitly
using the \ref apply "tmpl::apply" call.

\par
This is the "apply-defer-parent" pattern for lazy evaluation.  A \ref parent
"tmpl::parent" is placed immediately inside a \ref defer "tmpl::defer" with a
(not immediately) surrounding \ref apply "tmpl::apply".  The \ref apply
"tmpl::apply" and \ref defer "tmpl::defer" collectively add an (empty) element
to the head of the argument stack, which is then popped off to restore the
original value.  This causes the interior metalambda to have the same result it
would have had without the \ref defer "tmpl::defer".


\subsubsection factorial_recursion

\par
Again calculates the factorial, but using a recursive algorithm.
\snippet Test_TMPLDocumentation.cpp metafunctions:factorial_recursion
\snippet Test_TMPLDocumentation.cpp metafunctions:factorial_recursion:asserts

\par
This is a direct translation of the common definition \f$f(N) = N f(N-1)\f$ for
nonzero \f$N\f$, and \f$f(0) = 1\f$.  The metalambda is passed a copy of itself
as the first argument and the value to take the factorial of as the second.

\par
This again uses the "apply-defer-parent" pattern to prevent "speculative"
evaluation of conditional branches.  In this example, speculative evaluation of
the branch is invalid because it would recurse infinitely.


\subsubsection primes

\par
Generates a list of prime numbers less than `N` using the sieve of
Eratosthenes.  This example defines three helper metafunctions.  Two, `zero`
and `replace_at`, are defined only for clarity's sake and could be inlined.
The third, `range_from_types`, is not easily inlinable, and works around
Brigand's lack of sequence generating functions without non-type template
parameters.
\snippet Test_TMPLDocumentation.cpp metafunctions:primes
\snippet Test_TMPLDocumentation.cpp metafunctions:primes:asserts

\par
This is roughly equivalent to the following C++ code with loops converted fo
fold expressions over ranges.
\snippet Test_TMPLDocumentation.cpp metafunctions:primes_cpp


\section metafunction_guidelines Guidelines for writing metafunctions

\par
This section covers a few general guidelines for writing metafunctions in
SpECTRE.  Some of the Brigand functions mentioned below have specific advice as
well.

1. For general metafunctions, write both lazy and eager versions.  Follow the
   STL convention of `foo` being lazy, `foo_t` being eager, and `foo_v` being a
   constexpr value (if applicable).  This does not apply to internal-use or
   special-purpose metafunctions.
   \snippet Test_TMPLDocumentation.cpp guidelines:lazy_eager

2. Don't perform unnecessary return-type conversions.  We often recommend using
   STL types over equivalent Brigand types, but if the implementation naturally
   produces a Brigand type do not do extra work to convert it.


\section function_docs Brigand types and functions

\par
In this section, CamelCase identifiers indicate [type template
parameters](https://en.cppreference.com/w/cpp/language/template_parameters#Type_template_parameter)
or, occasionally, [template template
parameters](https://en.cppreference.com/w/cpp/language/template_parameters#Template_template_parameter).
Identifiers in all lowercase indicate [non-type template
parameters](https://en.cppreference.com/w/cpp/language/template_parameters#Non-type_template_parameter).
Identifiers in [brackets] are optional, and the default value will be
identified in the prose description.  Identifiers with ellipses... represent
zero or more parameters.

\par
The *head* of a fully specialized class template is the class template itself
(e.g., the head of `tmpl::list<T, U, V>` is `tmpl::list`).  When taken as a
metafunction parameter, these are template template parameters and are usually
called `Head` below.

\par
An identifier called `Sequence` must be a full specialization of a class
template with no non-type template parameters.  Many functions do not make
sense on sequences with a fixed length, and these require the head of their
sequence arguments to take a variable number of template arguments.  In
practical applications, sequence arguments will usually be specializations of
\ref list "tmpl::list".

\par
Parameters called `Predicate` must be unary metalambdas returning \ref
integral_constant "tmpl::integral_constant"s of `bool` or compatible classes.

\par
Parameters called `Comparator` must be binary metalambdas returning \ref
integral_constant "tmpl::integral_constant"s of `bool` or compatible classes.
They must establish a [strict weak
ordering](https://en.wikipedia.org/wiki/Weak_ordering#Strict_weak_orderings) on
the types they will be applied to in the same manner as runtime comparators
from the STL.

\par
Metafunctions documented here are eager unless otherwise noted.  In many cases,
Brigand provides lazy versions of its metafunctions under the same name in the
`tmpl::lazy` namespace.  These cases are indicated by the presence of the
`HAS_LAZY_VERSION` macro in the usage example.


\subsection Containers

\par
Brigand provides container classes with the sole purpose of wrapping other
things.


\subsubsection integral_constant integral_constant<T, value>

\par
A compile-time value `value` of type `T`.  Very similar to
std::integral_constant, except that the `constexpr` specifiers on the member
functions have been omitted.
\snippet Test_TMPLDocumentation.cpp tmpl::integral_constant

\par
Brigand supplies type aliases for constants of some specific types:
\snippet Test_TMPLDocumentation.cpp tmpl::integral_constant::abbreviations

\par
Most metafunctions that accept integral_constants will accept any type with a
`value` static member variable.

\par
Because of the `type` type alias, integral_constants behave like lazy
metafunctions returning themselves.  Most lazy metafunctions producing an
integral_constant will actually inherit from their result, so `value` will be
directly available without needing to go through the `type` alias.

\remark
Prefer std::integral_constant, except for the convenience wrapper
`tmpl::size_t` or when necessary for type equality comparison.


\subsubsection list list<T...>

\par
An empty struct templated on a parameter pack, with no additional
functionality.
\snippet Test_TMPLDocumentation.cpp tmpl::list

\par
Most metafunctions that operate on lists will work on any struct template.


\subsubsection map map<Pair...>

\par
A collection of key-value \ref pair "tmpl::pair"s with unique keys.  See the
section on \ref map_operations "operations on maps" for details.
\snippet Test_TMPLDocumentation.cpp tmpl::map

\par
The actual type of a map is unspecified, but it has the same template
parameters as a call to `map` that would produce it.

\warning
Equivalent maps may have different types, depending on the order their keys are
stored in internally.


\subsubsection pair pair<T1, T2>

\par
A pair of types, with easy access to each type in the pair.
\snippet Test_TMPLDocumentation.cpp tmpl::pair


\subsubsection set set<T...>

\par
An unordered collection of distinct types.  Trying to create a `set` with
duplicate entries is an error (but \ref set_insert "tmpl::insert" ignores
duplicate entries).  See the section on \ref set_operations
"operations on sets" for details.
\snippet Test_TMPLDocumentation.cpp tmpl::set

\par
The actual type of a set is unspecified, but it has the same template
parameters as a call to `set` that would produce it.

\warning
Equivalent sets may have different types, depending on the order their elements
are stored in internally.


\subsubsection type_ type_<T>

\par
A struct containing a `type` alias to `T`.
\snippet Test_TMPLDocumentation.cpp tmpl::type_

\par
When extracting the type, programmers are encouraged to use \ref type_from
"tmpl::type_from" to make it clear that the \c \::type that would otherwise
appear is not an evaluation of a lazy metafunction.  See \ref always
"tmpl::always" or \ref identity "tmpl::identity" for similar functionality that
is intended for use as a metafunction.


\subsection Constants

\par
Brigand defines a few concrete types and type aliases.


\subsubsection empty_base

\par
An empty struct used by \ref inherit "tmpl::inherit" and \ref inherit_linearly
"tmpl::inherit_linearly".  Primarily for internal use.
\snippet Test_TMPLDocumentation.cpp tmpl::empty_base


\subsubsection empty_sequence

\par
An empty \ref list "tmpl::list".
\snippet Test_TMPLDocumentation.cpp tmpl::empty_sequence

\remark
Prefer just writing `tmpl::list<>`.


\subsubsection false_type

\par
A \ref integral_constant "tmpl::integral_constant" representing `false`.
Similar to std::false_type.
\snippet Test_TMPLDocumentation.cpp tmpl::false_type

\remark
Prefer std::false_type.


\subsubsection no_such_type_

\par
An empty struct returned as the failure case for various searching operations.
\snippet Test_TMPLDocumentation.cpp tmpl::no_such_type_


\subsubsection true_type

\par
A \ref integral_constant "tmpl::integral_constant" representing `true`.
Similar to std::true_type.
\snippet Test_TMPLDocumentation.cpp tmpl::true_type

\remark
Prefer std::true_type.


\subsection list_constructor Constructor-like functions for lists

\par
These functions produce \ref list "tmpl::list"s from non-list values.  They are
often similar to constructors in the STL.


\subsubsection filled_list filled_list<Entry, n, [Head]>

\par
Creates a list containing `n` (passed as an `unsigned int`) of `Entry`.  The
head of the list defaults to \ref list "tmpl::list".
\snippet Test_TMPLDocumentation.cpp tmpl::filled_list


\subsubsection integral_list integral_list<T, n...>

\par
Shorthand for a \ref list "tmpl::list" of \ref integral_constant
"tmpl::integral_constant"s of the type `T` with values `n...`.
\snippet Test_TMPLDocumentation.cpp tmpl::integral_list

\remark
Prefer std::integer_sequence when used for pack expansion.  Prefer
`tmpl::integral_list` when the contents need to be manipulated for more
complicated metaprogramming.


\subsubsection make_sequence make_sequence<Start, n, [Next], [Head]>

\par
Produces a list with first element `Start` and length `n` (provided as an
`unsigned int`).  The remaining elements are obtained by repeated applications
of the \ref metalambdas "metalambda" `Next`, defaulting to \ref next
"tmpl::next".  The head of the sequence can be specified, and defaults to \ref
list "tmpl::list".
\snippet Test_TMPLDocumentation.cpp tmpl::make_sequence

\see \ref range "tmpl::range", \ref repeat "tmpl::repeat"


\subsubsection range range<T, start, stop>

\par
Produces a \ref list "tmpl::list" of \ref integral_constant
"tmpl::integral_constant"s of type `T` representing adjacent ascending integers
from `start` to `stop`, including the starting value and excluding the ending
value.
\snippet Test_TMPLDocumentation.cpp tmpl::range

\see \ref reverse_range "tmpl::reverse_range"


\subsubsection reverse_range reverse_range<T, start, stop>

\par
Produces a \ref list "tmpl::list" of \ref integral_constant
"tmpl::integral_constant"s of type `T` representing adjacent descending
integers from `start` to `stop`, including the starting value and excluding the
ending value.
\snippet Test_TMPLDocumentation.cpp tmpl::reverse_range

\see \ref range "tmpl::range"


\subsection list_query Functions for querying lists

\par
These tend to be similar to const member functions in the STL and the
non-modifying sequence operations in `<algorithm>`.  They are most frequently
used with \ref list "tmpl::list", but similar classes will also work.


\subsubsection all all<Sequence, [Predicate]>

\par
Checks if `Predicate` is true for all elements of `Sequence`.  The default
predicate checks that the element's `value` is not equal to zero.
\snippet Test_TMPLDocumentation.cpp tmpl::all

\note
The predicate must return the same true value for each element for `all` to
return true.
\snippet Test_TMPLDocumentation.cpp tmpl::all:inhomogeneous

\see \ref any "tmpl::any", \ref none "tmpl::none"


\subsubsection any any<Sequence, [Predicate]>

\par
Checks if `Predicate` is true for at least one element of `Sequence`.  The
default predicate checks that the element's `value` is not equal to zero.
\snippet Test_TMPLDocumentation.cpp tmpl::any

\remark
The \ref any "tmpl::any" and \ref none "tmpl::none" metafunctions perform the
same tasks as \ref found "tmpl::found" and \ref not_found "tmpl::not_found",
but use different algorithms.  In general, \ref any "tmpl::any" and \ref none
"tmpl::none" are much faster, but \ref found "tmpl::found" and \ref not_found
"tmpl::not_found" short-circuit, so they may be preferable with short lists and
expensive predicates.

\note
The predicate must return the same false value for each element for `any` to
return false.
\snippet Test_TMPLDocumentation.cpp tmpl::any:inhomogeneous

\see \ref all "tmpl::all", \ref found "tmpl::found", \ref none "tmpl::none"


\subsubsection at at<Sequence, Index>

\par
Retrieves a given element of `Sequence`, similar to `operator[]` of the STL
containers.  The `Index` is supplied as a \ref integral_constant
"tmpl::integral_constant" or similar type.
\snippet Test_TMPLDocumentation.cpp tmpl::at

\par
This operator is \ref map_at "overloaded for maps".

\see \ref at_c "tmpl::at_c"


\subsubsection at_c at_c<Sequence, n>

\par
Retrieves a given element of `Sequence`, similar to `operator[]` of the STL
containers.  The index `n` is supplied as an `unsigned int`.
\snippet Test_TMPLDocumentation.cpp tmpl::at_c

\see \ref at "tmpl::at"


\subsubsection back back<Sequence>

\par
Retrieves the last element of `Sequence`.
\snippet Test_TMPLDocumentation.cpp tmpl::back


\subsubsection count_if count_if<Sequence, Predicate>

\par
Returns the number of elements of `Sequence` satisfying `Predicate`.
\snippet Test_TMPLDocumentation.cpp tmpl::count_if


\subsubsection fold fold<Sequence, State, Functor>

\par
Performs a [left
fold](https://en.wikipedia.org/wiki/Fold_(higher-order_function)), i.e., given
a list `Sequence`, initial state `State`, and \ref metalambdas "metalambda"
`Functor`, updates the state by calling `Functor` on the state and the first
element of `Sequence`, repeats with the second, and so on, returning the final
state.
\snippet Test_TMPLDocumentation.cpp tmpl::fold

\par
Brigand provides `tmpl::_state` and `tmpl::_element` aliases to the appropriate
\ref args "arguments" for use in folds.

\see \ref reverse_fold "tmpl::reverse_fold"


\subsubsection found found<Sequence, [Predicate]>

\par
Returns, as a \ref integral_constant "tmpl::integral_constant" of `bool`,
whether `Predicate` matches any element of `Sequence`.  The default predicate
checks that the element's `value` is not equal to zero.
\snippet Test_TMPLDocumentation.cpp tmpl::found

\remark
This function performs the same operation as \ref any "tmpl::any".  See \ref
any "tmpl::any" for discussion.

\see \ref any "tmpl::any", \ref find "tmpl::find", \ref not_found
"tmpl::not_found"


\subsubsection front front<Sequence>

\par
Retrieves the first element of `Sequence`.
\snippet Test_TMPLDocumentation.cpp tmpl::front


\subsubsection index_if<Sequence, Predicate, [NotFound]>

\par
Finds the index as a `size_t` \ref integral_constant "tmpl::integral_constant"
of the first type in `Sequence` satisfying `Predicate`.  Returns `NotFound`,
defaulting to \ref no_such_type_ "tmpl::no_such_type_" if no elements match.
\snippet Test_TMPLDocumentation.cpp tmpl::index_if


\subsubsection index_of index_of<Sequence, T>

\par
Finds the index as a `size_t` \ref integral_constant "tmpl::integral_constant"
of the first occurrence of `T` in `Sequence`.  Returns \ref no_such_type_
"tmpl::no_such_type_" if the type is not found.
\snippet Test_TMPLDocumentation.cpp tmpl::index_of


\subsubsection list_contains list_contains<Sequence, T>

\par
Checks whether `T` is contained in `Sequence`, returning a \ref
integral_constant "tmpl::integral_constant" of `bool`.
\snippet Test_TMPLDocumentation.cpp tmpl::list_contains

\note
This is not a Brigand metafunction.  It is implemented in SpECTRE.


\subsubsection none none<Sequence, [Predicate]>

\par
Checks if `Predicate` is false for all elements of `Sequence`.  The default
predicate checks that the element's `value` is not equal to zero.
\snippet Test_TMPLDocumentation.cpp tmpl::none

\remark
This function performs the same operation as \ref not_found "tmpl::not_found".
See \ref any "tmpl::any" for discussion.

\note
The predicate must return the same false value for each element for `none` to
return true.
\snippet Test_TMPLDocumentation.cpp tmpl::none:inhomogeneous

\see \ref all "tmpl::all", \ref any "tmpl::any", \ref not_found
"tmpl::not_found"


\subsubsection not_found not_found<Sequence, [Predicate]>

\par
Returns, as a \ref integral_constant "tmpl::integral_constant" of `bool`,
whether `Predicate` matches no elements of `Sequence`.  The default predicate
checks that the element's `value` is not equal to zero.
\snippet Test_TMPLDocumentation.cpp tmpl::not_found

\remark
This function performs the same operation as \ref none "tmpl::none".  See \ref
any "tmpl::any" for discussion.

\see \ref find "tmpl::find", \ref found "tmpl::found", \ref none "tmpl::none"


\subsubsection size size<Sequence>

\par
Returns the number of elements in `Sequence` as a \ref integral_constant
"tmpl::integral_constant" of type `unsigned int`.
\snippet Test_TMPLDocumentation.cpp tmpl::size

\see \ref count "tmpl::count"


\subsection list_to_list Functions producing lists from other lists

\par
These tend to be similar to non-const member functions in the STL and the
mutating sequence operations in `<algorithm>`, but due to the nature of
metaprogramming all return a new list rather than modifying an argument.  They
are most frequently used with \ref list "tmpl::list", but similar classes will
also work.


\subsubsection append append<Sequence...>

\par
Concatenates all of its arguments, keeping the head of the first (or \ref list
"tmpl::list" if passed no arguments).
\snippet Test_TMPLDocumentation.cpp tmpl::append

\see \ref join "tmpl::join"


\subsubsection clear clear<Sequence>

\par
Produces a list with the same head as `Sequence` but no elements.
\snippet Test_TMPLDocumentation.cpp tmpl::clear

\remark
If the head is known, prefer writing it explicitly.  If the head is irrelevant,
write an empty \ref list "tmpl::list".


\subsubsection erase erase<Sequence, Index>

\par
Produces a copy of `Sequence` with the element at index `Index` (passed as a
\ref integral_constant "tmpl::integral_constant" or similar type) removed.
\snippet Test_TMPLDocumentation.cpp tmpl::erase

\par
This operator is overloaded \ref map_erase "for maps" and \ref set_erase
"for sets".

\see \ref erase_c


\subsubsection erase_c erase_c<Sequence, n>

\par
Produces a copy of `Sequence` with the element at index `n` (passed as an
`unsigned int`) removed.
\snippet Test_TMPLDocumentation.cpp tmpl::erase_c


\subsubsection filter filter<Sequence, Predicate>

\par
Removes all types not matching `Predicate` from `Sequence`.
\snippet Test_TMPLDocumentation.cpp tmpl::filter

\see \ref remove_if "tmpl::remove_if"


\subsubsection find find<Sequence, [Predicate]>

\par
Returns a list containing the first element of `Sequence` for which `Predicate`
returns true and all subsequent elements.  The default predicate checks that
the element's `value` is not equal to zero.  Returns an empty list if the
predicate returns false for all elements.
\snippet Test_TMPLDocumentation.cpp tmpl::find

\see \ref found "tmpl::found", \ref not_found "tmpl::not_found", \ref
reverse_find "tmpl::reverse_find"


\subsubsection flatten flatten<Sequence>

\par
Recursively inlines the contents of elements of `Sequence` that are sequences
with the same head.
\snippet Test_TMPLDocumentation.cpp tmpl::flatten

\see \ref join "tmpl::join"


\subsubsection join join<Sequence>

\par
Combines lists in the same manner as \ref append "tmpl::append", but takes a
list of lists instead of multiple arguments.
\snippet Test_TMPLDocumentation.cpp tmpl::join


\subsubsection list_difference list_difference<Sequence1, Sequence2>

\par
Remove all elements that occur in `Sequence2` from `Sequence1`.
\snippet Test_TMPLDocumentation.cpp tmpl::list_difference

\note
This is not a Brigand metafunction.  It is implemented in SpECTRE.


\subsubsection merge merge<Sequence1, Sequence2, [Comparator]>

\par
Given two sorted lists, returns a sorted list containing the elements of both.
A comparator metalambda can be provided, defaulting to \ref math_comparison
"tmpl::less".
\snippet Test_TMPLDocumentation.cpp tmpl::merge

\note
If there are equivalent elements, those from the second list are placed
earlier.
\snippet Test_TMPLDocumentation.cpp tmpl::merge:equiv

\see std::merge


\subsubsection partition partition<Sequence, Predicate>

\par
Returns a \ref pair "tmpl::pair" containing a list of the elements of
`Sequence` for which the `Predicate` returns true and a list of the elements of
`Sequence` for which the `Predicate` returns false.
\snippet Test_TMPLDocumentation.cpp tmpl::partition

\see \ref filter "tmpl::filter", \ref remove_if "tmpl::remove_if"


\subsubsection pop_back pop_back<Sequence, [Count]>

\par
Remove `Count` elements from the end of `Sequence`.  The number of elements to
remove is supplied as a \ref integral_constant "tmpl::integral_constant" and
defaults to 1.
\snippet Test_TMPLDocumentation.cpp tmpl::pop_back


\subsubsection pop_front pop_front<Sequence, [Count]>

\par
Remove `Count` elements from the beginning of `Sequence`.  The number of
elements to remove is supplied as a \ref integral_constant
"tmpl::integral_constant" and defaults to 1.
\snippet Test_TMPLDocumentation.cpp tmpl::pop_front


\subsubsection push_back push_back<Sequence, T...>

\par
Appends types `T...` to `Sequence`.
\snippet Test_TMPLDocumentation.cpp tmpl::push_back


\subsubsection push_front push_front<Sequence, T...>

\par
Prepends types `T...` to `Sequence`.  The order of the prepended items is
retained: they are pushed as a unit, not one-by-one.
\snippet Test_TMPLDocumentation.cpp tmpl::push_front


\subsubsection remove remove<Sequence, T>

\par
Removes all occurrences of `T` from `Sequence`.
\snippet Test_TMPLDocumentation.cpp tmpl::remove


\subsubsection remove_duplicates remove_duplicates<Sequence>

\par
Remove duplicates from `Sequence`.  The first occurrence of each type is kept.
\snippet Test_TMPLDocumentation.cpp tmpl::remove_duplicates

\note
This is not a Brigand metafunction.  It is implemented in SpECTRE.


\subsubsection remove_if remove_if<Sequence, Predicate>

\par
Removes all types matching `Predicate` from `Sequence`.
\snippet Test_TMPLDocumentation.cpp tmpl::remove_if

\see \ref filter "tmpl::filter"


\subsubsection replace replace<Sequence, Old, New>

\par
Replaces all occurrences of `Old` in `Sequence` with `New`.
\snippet Test_TMPLDocumentation.cpp tmpl::replace


\subsubsection replace_if replace_if<Sequence, Predicate, T>

\par
Replaces all types in `Sequence` matching `Predicate` with `T`.
\snippet Test_TMPLDocumentation.cpp tmpl::replace_if


\subsubsection reverse reverse<Sequence>

\par
Reverses the order of types in `Sequence`.
\snippet Test_TMPLDocumentation.cpp tmpl::reverse


\subsubsection reverse_find reverse_find<Sequence, [Predicate]>

\par
Returns a list containing the last element of `Sequence` for which `Predicate`
returns true and all preceding elements.  The default predicate checks that the
element's `value` is not equal to zero.  Returns an empty list if the predicate
returns false for all elements.
\snippet Test_TMPLDocumentation.cpp tmpl::reverse_find

\see \ref find "tmpl::find"


\subsubsection reverse_fold reverse_fold<Sequence, State, Functor>

\par
Performs a [right
fold](https://en.wikipedia.org/wiki/Fold_(higher-order_function)), i.e., given
a list `Sequence`, initial state `State`, and \ref metalambdas "metalambda"
`Functor`, updates the state by calling `Functor` on the state and the last
element of `Sequence`, repeats with the second to last, and so on, returning
the final state.
\snippet Test_TMPLDocumentation.cpp tmpl::reverse_fold

\par
Brigand provides `tmpl::_state` and `tmpl::_element` aliases to the appropriate
\ref args "arguments" for use in folds.

\see \ref fold "tmpl::fold"


\subsubsection sort sort<Sequence, [Comparator]>

\par
Sorts `Sequence` according to `Comparator`, which defaults to \ref
math_comparison "tmpl::less".
\snippet Test_TMPLDocumentation.cpp tmpl::sort

\note
The sort is not stable.
\snippet Test_TMPLDocumentation.cpp tmpl::sort:equiv


\subsubsection split split<Sequence, Delimiter>

\par
Splits `Sequence` into parts separated by `Delimiter`, discarding empty parts.
\snippet Test_TMPLDocumentation.cpp tmpl::split


\subsubsection split_at split_at<Sequence, Index>

\par
Returns a list of two of lists, the first containing the first `Index`
(supplied as a \ref integral_constant "tmpl::integral_constant") elements or
`Sequence`, and the second containing the remaining elements.
\snippet Test_TMPLDocumentation.cpp tmpl::split_at


\subsubsection transform transform<Sequence, Sequences..., Functor>

\par
Calls a `Functor` on each element of `Sequence`, collecting the results in a
new list.  If additional `Sequences...` are supplied, elements from those lists
are passed as additional arguments to `Functor`.
\snippet Test_TMPLDocumentation.cpp tmpl::transform


\subsection map_operations Operations on maps

\par
Brigand's \ref map "tmpl::map" type can be manipulated by several
metafunctions.

\par
Examples in this section use this map as an example:
\snippet Test_TMPLDocumentation.cpp example_map


\subsubsection map_at at<Map, Key>

\par
Returns the value associated with `Key` in `Map`.  Returns \ref
no_such_type_ "tmpl::no_such_type_" if `Key` is not in the map.
\snippet Test_TMPLDocumentation.cpp tmpl::at:map

\par
This operator is \ref at "overloaded for lists".  When called on a \ref map
"tmpl::map", this is the same as \ref lookup "tmpl::lookup".


\subsubsection map_erase erase<Map, Key>

\par
Produces a copy of `Map` with the element with the key `Key` removed.  If `Key`
is not in the map, returns `Map` unchanged.
\snippet Test_TMPLDocumentation.cpp tmpl::erase:map

\par
This operator is overloaded \ref erase "for lists" and \ref set_erase
"for sets".


\subsubsection map_has_key has_key<Map, Key>

\par
Returns a \ref integral_constant "tmpl::integral_constant" of `bool`
indicating whether `Map` contains the key `Key`.
\snippet Test_TMPLDocumentation.cpp tmpl::has_key:map

\par
This operator is \ref set_has_key "overloaded for sets".


\subsubsection map_insert insert<Map, Pair>

\par
Returns `Map` with `Pair` added.  If the key of `Pair` is already in the map,
the map is returned unchanged.
\snippet Test_TMPLDocumentation.cpp tmpl::insert:map

\par
This operator is \ref set_insert "overloaded for sets".


\subsubsection keys_as_sequence keys_as_sequence<Map, [Head]>

\par
Returns the keys from `Map` as a sequence with head `Head`, defaulting to \ref
set "tmpl::set".
\snippet Test_TMPLDocumentation.cpp tmpl::keys_as_sequence

\par
If the key-value pairs are required, they can be extracted directly from the
template arguments of the \ref map "tmpl::map".

\see \ref values_as_sequence "tmpl::values_as_sequence"


\subsubsection lookup lookup<Map, Key>

\par
Returns the value associated with `Key` in `Map`.  Returns \ref no_such_type_
"tmpl::no_such_type_" if `Key` is not in the map.
\snippet Test_TMPLDocumentation.cpp tmpl::lookup

\see \ref map_at "tmpl::at"


\subsubsection lookup_at lookup_at<Map, Key>

\par
Returns the value associated with `Key` in `Map`, wrapped in a \ref type_
"tmpl::type_".  Returns `type_<no_such_type_>` if `Key` is not in the map.
This function has no eager version, but is still in the `tmpl::lazy` namespace.
\snippet Test_TMPLDocumentation.cpp tmpl::lookup_at

\see \ref lookup "tmpl::lookup"


\subsubsection values_as_sequence values_as_sequence<Map, [Head]>

\par
Returns the values from `Map` as a sequence with head `Head`, defaulting to
\ref list "tmpl::list".
\snippet Test_TMPLDocumentation.cpp tmpl::values_as_sequence

\par
If the key-value pairs are required, they can be extracted directly from the
template arguments of the \ref map "tmpl::map".

\see \ref keys_as_sequence "tmpl::keys_as_sequence"


\subsection set_operations Operations on sets

\par
Brigand's \ref set "tmpl::set" type can be manipulated by several
metafunctions.


\subsubsection contains contains<Set, T>

\par
Returns a \ref integral_constant "tmpl::integral_constant" of `bool`
indicating whether `Set` contains `T`.
\snippet Test_TMPLDocumentation.cpp tmpl::contains


\subsubsection set_erase erase<Set, T>

\par
Produces a copy of `Set` with the element `T` removed.  If the element is not
in the set, returns the set unchanged.
\snippet Test_TMPLDocumentation.cpp tmpl::erase:set

\par
This operator is overloaded \ref erase "for lists" and \ref map_erase
"for maps".


\subsubsection set_has_key has_key<Set, T>

\par
Returns a \ref integral_constant "tmpl::integral_constant" of `bool`
indicating whether `Set` contains `T`.
\snippet Test_TMPLDocumentation.cpp tmpl::has_key:set

\par
This operator is \ref map_has_key "overloaded for maps".


\subsubsection set_insert insert<Set, T>

\par
Returns a copy of `Set` containing an additional element `T`.  If `T` is
already in the set, the set is returned unchanged.
\snippet Test_TMPLDocumentation.cpp tmpl::insert:set

\par
This operator is \ref map_insert "overloaded for maps".


\subsection math Mathematical functions

\par
These perform the same operations at their language counterparts, but on \ref
integral_constant "tmpl::integral_constant"s (or anything else with a `value`
static member type of type `value_type`).  The results inherit from \ref
integral_constant "tmpl::integral_constant"s of types noted below.

\par
These are all lazy metafunctions.


\subsubsection math_arithmetic Arithmetic operators

\par
These operations return classes inheriting from \ref integral_constant
"tmpl::integral_constant"s of the same type as the result of the language
operator on their arguments.  The integral promotion and conversion rules are
applied.  (Contrast the \ref math_bitwise "bitwise operators".)
\snippet Test_TMPLDocumentation.cpp math_arithmetic

\par
The standard library runtime functors have the same names for std::plus,
std::minus, std::divides, and std::negate, but the other two are
std::multiplies and std::modulus.


\subsubsection math_bitwise Bitwise operators

\par
These operations return classes inheriting from \ref integral_constant
"tmpl::integral_constant"s of the same type as their first argument's `value`.
This is *not* generally the same type as the language operator, even when the
types of the values of both arguments are the same.  (The integer promotion and
conversion rules are not applied.)
\snippet Test_TMPLDocumentation.cpp math_bitwise

\par
The standard library runtime functors are called std::bit_not, std::bit_and,
std::bit_or, and std::bit_xor.


\subsubsection math_comparison Comparison operators

\par
These operations return classes inheriting from \ref integral_constant
"tmpl::integral_constant"s of `bool`.
\snippet Test_TMPLDocumentation.cpp math_comparison

\par
The standard library runtime functors have the same names, such as
std::equal_to.


\subsubsection math_logical Logical operators

\par
These operations return classes inheriting from \ref integral_constant
"tmpl::integral_constant"s of `bool`.  They should only be used on types
wrapping `bool`s.  The `and_` and `or_` structs can take any number of
arguments.
\snippet Test_TMPLDocumentation.cpp math_logical

\par
The standard library runtime functors are called std::logical_and,
std::logical_or, and std::logical_not.  The xor operation is equivalent to \ref
math_comparison "tmpl::not_equal_to".


\subsubsection identity identity<T>

\par
The identity function.  Unlike most math functions, this returns the same type
as its argument, even if that is not a \ref integral_constant
"tmpl::integral_constant".
\snippet Test_TMPLDocumentation.cpp tmpl::identity

\see \ref always "tmpl::always"


\subsubsection max max<T1, T2>

\par
Computes the larger of `T1` and `T2`, returning a \ref integral_constant
"tmpl::integral_constant" of the common type of its arguments.
\snippet Test_TMPLDocumentation.cpp tmpl::max


\subsubsection min min<T1, T2>

\par
Computes the smaller of `T1` and `T2`, returning a \ref integral_constant
"tmpl::integral_constant" of the common type of its arguments.
\snippet Test_TMPLDocumentation.cpp tmpl::min


\subsubsection next next<T>

\par
Computes `T` plus one, returning a \ref integral_constant
"tmpl::integral_constant" of the same type as its argument.
\snippet Test_TMPLDocumentation.cpp tmpl::next


\subsubsection prev prev<T>

\par
Computes `T` minus one, returning a \ref integral_constant
"tmpl::integral_constant" of the same type as its argument.
\snippet Test_TMPLDocumentation.cpp tmpl::prev


\subsection misc Miscellaneous functions

\par
Functions that don't fit into any of the other sections.


\subsubsection always always<T>

\par
A lazy identity function.
\snippet Test_TMPLDocumentation.cpp tmpl::always

\see \ref identity "tmpl::identity"


\subsubsection apply apply<Lambda, [Arguments...]>

\par
Calls a \ref metalambdas "metalambda" `Lambda` with arguments `Arguments...`.
\snippet Test_TMPLDocumentation.cpp tmpl::apply


\subsubsection count count<T...>

\par
Returns the number of template parameters provided as a \ref integral_constant
"tmpl::integral_constant" of `unsigned int`.
\snippet Test_TMPLDocumentation.cpp tmpl::count


\subsubsection conditional_t conditional_t<b, TrueResult, FalseResult>

\par
Returns `TrueResult` if the `bool` `b` is true, otherwise `FalseResult`.  An
optimized version of std::conditional_t.
\snippet Test_TMPLDocumentation.cpp tmpl::conditional_t

\note
This is not a Brigand metafunction.  It is implemented in SpECTRE.


\subsubsection eval_if eval_if<Condition, TrueFunction, FalseFunction>

\par
A lazy metafunction that, if `Condition` has a true `value`, evaluates and
returns the result of the lazy metafunction (*not* metalambda) `TrueFunction`,
otherwise, evaluates and returns the result of the lazy metafunction
`FalseFunction`.
\snippet Test_TMPLDocumentation.cpp tmpl::eval_if

\par
This performs lazy evaluation of conditional branches outside of a metalambda.


\subsubsection eval_if_c eval_if_c<b, TrueFunction, FalseFunction>

\par
The same as \ref eval_if "tmpl::eval_if", but takes its first argument as a
`bool` instead of a type.
\snippet Test_TMPLDocumentation.cpp tmpl::eval_if_c


\subsubsection has_type has_type<Ignored, [T]>

\par
A lazy metafunction that returns `T` (defaulting to `void`), ignoring its first
argument.
\snippet Test_TMPLDocumentation.cpp tmpl::has_type

\par
This can be used to expand a parameter pack to repetitions of the same type.
\snippet Test_TMPLDocumentation.cpp tmpl::has_type:pack_expansion
\snippet Test_TMPLDocumentation.cpp tmpl::has_type:pack_expansion:asserts


\subsubsection if_ if_<Condition, TrueResult, FalseResult>

\par
A lazy metafunction that returns `TrueResult` if the `value` static member
value of `Condition` is true, and otherwise `FalseResult`.
\snippet Test_TMPLDocumentation.cpp tmpl::if_

\warning
The second and third arguments are both evaluated, independent of which is
returned.  Use \ref defer "tmpl::defer" or \ref eval_if "tmpl::eval_if" if this
is undesirable.


\subsubsection if_c if_c<Condition, TrueResult, FalseResult>

\par
The same as std::conditional.
\snippet Test_TMPLDocumentation.cpp tmpl::if_c


\subsubsection inherit inherit<T...>

\par
A lazy metafunction that produces a type with all of its template arguments as
base classes.  All the arguments must be unique.
\snippet Test_TMPLDocumentation.cpp tmpl::inherit

\remark
This task can be performed more simply than the algorithm used by Brigand by
directly using pack expansions:
\snippet Test_TMPLDocumentation.cpp tmpl::inherit:pack:definitions
\snippet Test_TMPLDocumentation.cpp tmpl::inherit:pack:asserts

\note
The \ref empty_base "tmpl::empty_base" type is used internally as a sentinel.
The result may or may not inherit from \ref empty_base "tmpl::empty_base",
independently of whether it is supplied as an argument.


\subsubsection inherit_linearly inherit_linearly<Sequence, NodePattern, [Root]>

\par
Transforms `Sequence` into a linked list.  The `NodePattern` must be a class
template (*not* a lazy metafunction) instantiated with metalambdas.  The
function performs a [left
fold](https://en.wikipedia.org/wiki/Fold_(higher-order_function)), with the
`Root` (defaulting to \ref empty_base "tmpl::empty_base") as the initial state
and the transform function evaluating the arguments to the node pattern.
\snippet Test_TMPLDocumentation.cpp tmpl::inherit_linearly

\remark
This function handles its function-like argument differently from any other
function in Brigand.  Prefer \ref fold "tmpl::fold", which can perform the same
task and has a more standard interface.


\subsubsection is_set is_set<T...>

\par
Tests if all of its arguments are distinct, producing a \ref integral_constant
"tmpl::integral_constant" of `bool`.
\snippet Test_TMPLDocumentation.cpp tmpl::is_set

\note
This is unrelated to the Brigand \ref set "tmpl::set" type.


\subsubsection real_ real_<RealType, IntType, value>

\par
Represents a floating point number of type `RealType` at compile time via its
internal memory representation.  The value is stored as a \ref
integral_constant "tmpl::integral_constant" of type `IntType` with value
`value` (which must be the same size as `RealType`) and can be extracted at
runtime using the conversion operator.  Brigand provides the aliases
`single_<value>` and `double_<value>` with `RealType` and `IntType` set to
appropriate values.
\snippet Test_TMPLDocumentation.cpp tmpl::real_

\par
There are no compile-time mathematical functions provided for floating point
types.  They are opaque (or sometimes treated as integers) until runtime.

\remark
Consider whether you really need to represent floating point values at compile
time.

\see std::ratio


\subsubsection repeat repeat<Function, Count, Initial>

\par
Calls a unary eager metafunction `Function` on `Initial`, then on the result of
that, then on the result of that, and so on, up to `Count` calls.
\snippet Test_TMPLDocumentation.cpp tmpl::repeat

\note
This function has a lazy version, but it cannot be used in a metalambda because
the template template parameter prevents manipulation of the parameter list.
\snippet Test_TMPLDocumentation.cpp tmpl::repeat:lazy

\see \ref make_sequence "tmpl::make_sequence"


\subsubsection sizeof_ sizeof_<T>

\par
A lazy metafunction that computes `sizeof` its argument as an `unsigned int`
\ref integral_constant "tmpl::integral_constant".
\snippet Test_TMPLDocumentation.cpp tmpl::sizeof_


\subsubsection substitute substitute<Pattern, ArgumentList>

\par
Substitutes values from `ArgumentList` for appearances of \ref args
"tmpl::args" (but *not* `tmpl::_1` or `tmpl::_2`) appearing in `Pattern`.
\snippet Test_TMPLDocumentation.cpp tmpl::substitute


\subsubsection type_from type_from<T>

\par
Extracts the `type` from `T`.
\snippet Test_TMPLDocumentation.cpp tmpl::type_from

\remark
This function will work on any class with a `type` type alias, but, when used
outside of a metafunction, it should only be used with \ref type_ "tmpl::type_"
for clarity.


\subsubsection wrap wrap<Sequence, Head>

\par
Replaces the head of `Sequence` with `Head`.
\snippet Test_TMPLDocumentation.cpp tmpl::wrap

\note
This function has a lazy version, but it cannot be used in a metalambda because
the template template parameter prevents manipulation of the parameter list.
\snippet Test_TMPLDocumentation.cpp tmpl::wrap:lazy


\subsection runtime Runtime functionality

\par
Brigand provides a few C++ functions that execute at runtime.

\par
The examples in this section use the following definition:
\snippet Test_TMPLDocumentation.cpp runtime_declarations


\subsubsection for_each_args for_each_args(functor, arguments...)

\par
Calls `functor` on each of `arguments...`, in order.  Returns `functor`.
\snippet Test_TMPLDocumentation.cpp tmpl::for_each_args:defs
\snippet Test_TMPLDocumentation.cpp tmpl::for_each_args

\par

\note
This uses a std::reference_wrapper internally, but I don't see a reason for
that.  If it were removed then this function could be constexpr starting in
C++14.


\subsubsection for_each for_each<Sequence>(functor)

\par
Calls `functor` on \ref type_ "tmpl::type_" objects wrapping each type in
`Sequence`, in order.  Returns `functor`.
\snippet Test_TMPLDocumentation.cpp tmpl::for_each:defs
\snippet Test_TMPLDocumentation.cpp tmpl::for_each

\see \ref type_from "tmpl::type_from"


\subsubsection select select<Condition>(true_result, false_result)

\par
Returns `true_result` if `Condition`'s `value` member is true,
and `false_result` if it is false.
\snippet Test_TMPLDocumentation.cpp tmpl::select


\subsection external External integration

\par
Brigand provides metafunctions for interfacing with some types from the
standard library and Boost.  They usually come in pairs, with `as_X` taking a
list and `X_wrapper` taking a parameter pack.  This makes `X_wrapper`
equivalent to the wrapped class.

\remark
Avoid the `*_wrapper` functions in favor of using the class directly.


\subsubsection boost_integration Boost

\par
Brigand provides functions to produce the `boost::fusion` types `deque`,
`list`, `set`, and `vector`, as well as `boost::variant`.
\snippet Test_TMPLDocumentation.cpp boost_integration

\note
These functions are unavailable if `BRIGAND_NO_BOOST_SUPPORT` is defined, as is
the case in SpECTRE.


\subsubsection stl_integration STL

\par
Brigand provides functions to produce the STL types std::pair and std::tuple.
In addition to the usual functions, Brigand provides `pair_wrapper_`, which is
a lazy form of `pair_wrapper`.  The pair functions all assert that they have
received two types.
\snippet Test_TMPLDocumentation.cpp stl_integration


\subsubsection make_integral integral_constant

\par
Brigand provides two functions for converting from std::integral_constant (or a
similar class with a `value_type` and a `value`) to \ref integral_constant
"tmpl::integral_constant".  The lazy metafunction `make_integral` performs this
conversion.  The `as_integral_list` eager metafunction performs this operation
on all elements of a list.
\snippet Test_TMPLDocumentation.cpp tmpl::make_integral

\warning
The standard library std::integer_sequence is not a list of types, and so
cannot be used as input to `as_integral_list`.


\subsubsection as_list list

\par
Brigand provides two metafunctions for converting types to Brigand sequences.
The more general function, `as_sequence`, is equivalent to \ref wrap
"tmpl::wrap".  The specialized version, `tmpl::as_list`, produces a \ref list
"tmpl::list".
\snippet Test_TMPLDocumentation.cpp tmpl::as_list

\remark
Using `as_list` is often not necessary because most metafunctions operate on
arbitrary template classes.


\subsubsection as_set set

\par
Brigand provides the standard two metafunctions for converting types to Brigand
\ref set "tmpl::set"s.
\snippet Test_TMPLDocumentation.cpp tmpl::as_set


\section oddities Bugs/Oddities

* \ref push_front and \ref pop_front have lazy versions, but \ref push_back,
  and \ref pop_back do not.

* \ref reverse_range validates its arguments, but \ref range does not.
  (Probably because the former is called incorrectly more often.)

* Brigand inconsistently uses `unsigned int` and `size_t` for size-related
  things.  (Most blatantly, the result of \ref sizeof_ is represented as an
  `unsigned int`.)

* Brigand has a file containing operator overloads for \ref integral_constant
  ""s, but it is not included by the main brigand header.  They work poorly,
  mostly because it inexplicably puts them all in namespace std where the
  compiler can't find them.


\section TODO



```
      2 tmpl::get_source
      2 tmpl::get_destination
      2 tmpl::edge
```

```
./adapted:
fusion.hpp        - Done
integral_list.hpp - Done
list.hpp          - Done
pair.hpp          - Done
tuple.hpp         - Done
variant.hpp       - Done

./algorithms:
all.hpp           - Done
any.hpp           - Done
count.hpp         - Done
find.hpp          - Done
flatten.hpp       - Done
fold.hpp          - Done
for_each.hpp      - Done
for_each_args.hpp - Done
index_of.hpp      - Done
is_set.hpp        - Done
merge.hpp         - Done
none.hpp          - Done
partition.hpp     - Done
remove.hpp        - Done
replace.hpp       - Done
reverse.hpp       - Done
select.hpp        - Done
sort.hpp          - Done
split.hpp         - Done
split_at.hpp      - Done
transform.hpp     - Done
wrap.hpp          - Done

./functions:
eval_if.hpp  - Done
if.hpp       - Done

./functions/arithmetic:
complement.hpp          - Done
divides.hpp             - Done
identity.hpp            - Done
max.hpp                 - Done
min.hpp                 - Done
minus.hpp               - Done
modulo.hpp              - Done
negate.hpp              - Done
next.hpp                - Done
plus.hpp                - Done
prev.hpp                - Done
times.hpp               - Done

./functions/bitwise:
bitand.hpp           - Done
bitor.hpp            - Done
bitxor.hpp           - Done
shift_left.hpp       - Done
shift_right.hpp      - Done

./functions/comparison:
equal_to.hpp            - Done
greater.hpp             - Done
greater_equal.hpp       - Done
less.hpp                - Done
less_equal.hpp          - Done
not_equal_to.hpp        - Done

./functions/lambda:
apply.hpp           - Done
bind.hpp            - Done
substitute.hpp      - Done

./functions/logical:
and.hpp              - Done
not.hpp              - Done
or.hpp               - Done
xor.hpp              - Done

./functions/misc:
always.hpp        - Done
repeat.hpp        - Done
sizeof.hpp        - Done

./sequences:
append.hpp             - Done
at.hpp                 - Done
back.hpp               - Done
clear.hpp              - Done
contains.hpp           - Done
erase.hpp              - Done
filled_list.hpp        - Done
front.hpp              - Done
has_key.hpp            - Done
insert.hpp             - Done
keys_as_sequence.hpp   - Done
list.hpp               - Done
make_sequence.hpp      - Done
map.hpp                - Done
pair.hpp               - Done
range.hpp              - Done
set.hpp                - Done
size.hpp               - Done
values_as_sequence.hpp - Done

./types:
args.hpp              - Done
bool.hpp              - Done
empty_base.hpp        - Done
has_type.hpp          - Done
inherit.hpp           - Done
inherit_linearly.hpp  - Done
integer.hpp           - Done
integral_constant.hpp - Done
no_such_type.hpp      - Done
operators.hpp         - Broken and unused
real.hpp              - Done
type.hpp              - Done
voidp.hpp             - Not included by main header.  Special case of has_type.
```
