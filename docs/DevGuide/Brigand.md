\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Metaprogramming with Brigand {#brigand}

\note
This document covers Brigand as of commit
[85baf9e685eb0c942764b7224fa1ce034bb3beba](https://github.com/edouarda/brigand/commit/85baf9e685eb0c942764b7224fa1ce034bb3beba)
in Summer 2017.  There have been only minor changes since then.

\tableofcontents

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
Metafunctions commonly appear in two forms: eager and lazy.  Lazy functions are
templated structs (or templated aliases to structs) with a `type` member
function that indicates the result:
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
used as compile-time functors.  As a simple example, we can write a function
that calls an arbitrary lazy metafunction several times
\snippet Test_TMPLDocumentation.cpp metafunctions:lazy_call_metafunction
and get the expected output:
\snippet Test_TMPLDocumentation.cpp metafunctions:lazy_call_metafunction_assert
A similar attempt with an eager metafunction fails, because the function is
evaluated too early, acting as a function composition, rather than a lambda:
\snippet Test_TMPLDocumentation.cpp metafunctions:eager_call_metafunction
\snippet Test_TMPLDocumentation.cpp metafunctions:eager_call_metafunction_assert
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
placeholder types.  The \ref apply function can be used to evaluate a
metalambda with specified arguments, and many other Brigand functions take
metalambdas that are evaluated internally.


\subsection metalambda_structure Evaluation of metalambdas

\note
None of the terminology introduced in this section is standard.

\par
A metalambda is always evaluated in some *context* describing the possible
argument substitutions to be performed.  The context is a list of arguments
and, possibly, a parent context.  When evaluating a metalambda, for example by
calling \ref apply, the initial context contains the passed set of arguments
with no parent context.

\par
There are eight forms that a metalambda can take: an argument, a lazy
expression, a bind expression, a pin expression, a defer expression, a parent
expression, a constant, or a metaclosure.

\subsubsection args Argument

\par
An argument is one of the structs `tmpl::_1`, `tmpl::_2`, or `tmpl::args<N>`
for `unsigned int` N.  The additional aliases `tmpl::_3`, `tmpl::_4`, ...,
`tmpl::_9` are provided to `tmpl::args<2>`, `tmpl::args<3>`, ...,
`tmpl::args<8>`.
\snippet Test_TMPLDocumentation.cpp tmpl::args
When evaluated, they give the first (`tmpl::_1`), second (`tmpl::_2`), or
zero-indexed Nth (`tmpl::args<N>`) element of the context's argument list.
\snippet Test_TMPLDocumentation.cpp tmpl::args:eval
Additionally, `tmpl::_state` and `tmpl::_element` are aliased to `tmpl::_1` and
`tmpl::_2`, primarily for use with \ref fold.

\par
Metalambdas must be passed enough arguments to define all argument placeholders
in their bodies.  Failure to pass enough arguments may error or produce
unintuitive results.

\subsubsection metalambda_lazy Lazy expression

\par
A lazy expression is a fully-specialized struct template with only type
template parameters that is not a specialization of \ref pin "pin", \ref defer
"defer", or \ref parent "parent" and is not a \ref metalambda_metaclosure
"metaclosure".  When evaluated, each of its template parameters is evaluated as
a metafunction and replaced by the result, and then the struct's `type` type
alias is the result of the full lazy-expression.
\snippet Test_TMPLDocumentation.cpp metalambda_lazy

\subsubsection bind Bind expression

\par
A bind expression is a specialization of `tmpl::bind`.  It wraps an eager
metafunction and its arguments.  When evaluated, the arguments are each
evaluated as metafunctions, and then the results are passed to the eager
metafunction.
\snippet Test_TMPLDocumentation.cpp tmpl::bind

\note
The `tmpl::bind` metafunction does not convert an eager metafunction to a lazy
one.  It is handled specially in the evaluation code.

\subsubsection pin Pin expression

\par
A pin expression is a specialization of `tmpl::pin`.  Evaluating a pin
expression gives the argument to `tmpl::pin`.  This can be used to force a type
to be treated as a \ref metalambda_constant "constant", even if it would
normally be treated as a different type of metalambda (usually a \ref
metalambda_lazy "lazy expression").
\snippet Test_TMPLDocumentation.cpp tmpl::pin

\subsubsection defer Defer expression

\par
A defer expression is a specialization of `tmpl::defer`.  It does not evaluate
its argument, but results in a \ref metalambda_metaclosure "metaclosure"
containing the passed metalambda and the current evaluation context.
\snippet Test_TMPLDocumentation.cpp tmpl::defer

\par
The primary purposes for `tmpl::defer` are constructing metalambdas to pass to
other metafunctions and preventing "speculative" evaluation of a portion of a
metalambda that is not valid for some arguments.  See the examples below, in
particular \ref multiplication_table, \ref maybe_first, and \ref
column_with_zeros.

\warning
The metalambda contained in a `tmpl::defer` must be a \ref metalambda_lazy
"lazy expression" or a \ref bind "bind expression".  This is presumably a bug.
If another type is needed, it can be wrapped in \ref always.

\subsubsection parent Parent expression

\par
A parent expression is a specialization of `tmpl::parent`.  It evaluates its
argument, replacing the current context with its parent.  This provides access
to the captured arguments in a metaclosure.
\snippet Test_TMPLDocumentation.cpp tmpl::parent

\warning
Do not call `tmpl::parent` outside of a metaclosure context.  This results in
an empty evaluation context, causing unintuitive changes to the evaluation
rules.  (Most, but not all, expressions are left unevaluated in such a
context.)  Use \ref pin "pin" to prevent evaluation.

\par

\warning
There is a bug that prevents `tmpl::parent` from working in a metaclosure being
evaluated in a metaclosure context.  In some cases this can be worked around by
evaluating the metaclosure in the parent of the metaclosure context.
\snippet Test_TMPLDocumentation.cpp tmpl::parent:bug

\subsubsection metalambda_constant Constant

\par
A constant metalambda is any type that is not a struct template with only type
template parameters, a specialization of \ref bind "bind", or a metaclosure.  A
constant metalambda evaluates to itself.
\snippet Test_TMPLDocumentation.cpp metalambda_constant

\subsubsection metalambda_metaclosure Metaclosure

\par
A metaclosure is an opaque type produced by \ref defer "defer", containing a
metalambda and an evaluation context.  When a metaclosure is evaluated, it
evaluates the packaged metalambda in the current evaluation context with the
parent context replaced by the packaged context.  See \ref defer and \ref
parent for examples.


\subsection Examples


\subsubsection evens

\par
Finds all numbers in a list that are even.
\snippet Test_TMPLDocumentation.cpp metafunctions:evens
\snippet Test_TMPLDocumentation.cpp metafunctions:evens:asserts


\subsubsection maybe_first

\par
Returns the first element of a list, or \ref no_such_type_ if the list is
empty.
\snippet Test_TMPLDocumentation.cpp metafunctions:maybe_first
\snippet Test_TMPLDocumentation.cpp metafunctions:maybe_first:asserts

\par
This example demonstrates the use of \ref defer "defer" to lazily evaluate a
branch of the \ref if_, preventing an attempt to evaluate
`tmpl::front<tmpl::list<>>`.


\subsubsection factorial

\par
Calculates the factorial using a simple metalambda passed to a \ref fold.
\snippet Test_TMPLDocumentation.cpp metafunctions:factorial
\snippet Test_TMPLDocumentation.cpp metafunctions:factorial:asserts


\subsubsection multiplication_table

\par
Constructs a multiplication table.
\snippet Test_TMPLDocumentation.cpp metafunctions:multiplication_table
\snippet Test_TMPLDocumentation.cpp metafunctions:multiplication_table:asserts

\par
This demonstrates the use of \ref defer "defer" to pass a closure as an
argument to a metafunction (\ref transform "tmpl::lazy::transform"), while
capturing an argument from the outer context (the metalambda evaluated for the
outer `tmpl::transform`).


\subsubsection column_with_zeros

\par
Extracts a column from a row-major matrix, extending any short rows with zeros.
\snippet Test_TMPLDocumentation.cpp metafunctions:column_with_zeros
\snippet Test_TMPLDocumentation.cpp metafunctions:column_with_zeros:asserts

\par
This example shows another use of \ref defer "defer" to avoid evaluating an
invalid expression, similar to \ref maybe_first.  The use of an \ref args
"argument" in the deferred branch makes this case more complicated: a \ref
parent "parent" expression is used to access the context where the \ref defer
"defer" occurs to avoid having to pass the argument explicitly using the \ref
apply call.

\par
This is the "apply-defer-parent" pattern for lazy evaluation.  A \ref parent
"parent" is placed immediately inside a \ref defer "defer" (with an \ref always
to work around a Brigand bug) with a (not immediately) surrounding \ref apply.
The \ref parent "parent" causes its contents to be executed (when called by
\ref apply "apply") in the context where the \ref defer "defer" was evaluated,
so the deferral is unobservable by the contents.


\subsubsection factorial_recursion

\par
Again calculates the factorial, but using a recursive algorithm.  After some
setup code to start the recursion, the recursive metalambda is called with
itself as the first argument (as a plain lambda, not a closure).  The other two
arguments are an accumulator and the number of remaining iterations.
\snippet Test_TMPLDocumentation.cpp metafunctions:factorial_recursion
\snippet Test_TMPLDocumentation.cpp metafunctions:factorial_recursion:asserts

\par
This again uses the "apply-defer-parent" pattern to prevent "speculative"
evaluation of conditional branches.  In this example, speculative evaluation of
the branch is invalid because it would recurse infinitely.


\subsubsection primes

\par
Generates a list of prime numbers using the sieve of Eratosthenes.  This
example defines three helper metafunctions.  Two, `zero` and `replace_at`, are
defined only for clarity's sake and could be inlined.  The third,
`range_from_types`, is not easily inlinable, and works around Brigand's lack of
sequence generating functions without non-type template parameters.
\snippet Test_TMPLDocumentation.cpp metafunctions:primes
\snippet Test_TMPLDocumentation.cpp metafunctions:primes:asserts


\section function_docs Brigand functions

\subsection Containers

\par
Brigand provides container classes with the sole purpose of wrapping other
things.


\subsubsection integral_constant

\par
Very similar to std::integral_constant, except that the `constexpr` specifiers
on the member functions have been omitted.
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


\subsubsection list

\par
An empty struct templated on a parameter pack, with no additional
functionality.
\snippet Test_TMPLDocumentation.cpp tmpl::list

\par
Most metafunctions that operate on lists will work on any struct template.


\subsubsection map

\par
A collection of key-value \ref pair ""s with unique keys.  See the section on
\ref map_operations "operations on maps" for details.
\snippet Test_TMPLDocumentation.cpp tmpl::map

\par
The actual type of a map is implementation-defined, but it has the same
template parameters as a call to `map` that would produce it.

\warning
Equivalent maps may have different types, depending on the order their keys are
stored in internally.


\subsubsection pair

\par
A pair of types, with easy access to each type in the pair.
\snippet Test_TMPLDocumentation.cpp tmpl::pair


\subsubsection set

\par
An unordered collection of distinct types.  Trying to create a `set`
with duplicate entries is an error (but \ref set_insert "insert" ignores
duplicate entries).
\snippet Test_TMPLDocumentation.cpp tmpl::set

\par
The actual type of a set is implementation-defined, but it has the same
template parameters as a call to `set` that would produce it.

\warning
Equivalent sets may have different types, depending on the order their elements
are stored in internally.


\subsubsection type_

\par
A struct templated on a single type `T` containing an alias `type` to `T`.
\snippet Test_TMPLDocumentation.cpp tmpl::type_

\par
When extracting the type, programmers are encouraged to use \ref type_from to
make it clear that the `::type` that would otherwise appear is not an
evaluation of a lazy metafunction.  See \ref always or \ref identity for
similar functionality that is intended for use as a metafunction.


\subsection Constants

\par
Brigand defines a few concrete types and type aliases.


\subsubsection empty_base

\par
An empty struct used by \ref inherit and \ref inherit_linearly.  Primarily for
internal use.
\snippet Test_TMPLDocumentation.cpp tmpl::empty_base


\subsubsection empty_sequence

\par
An empty \ref list.
\snippet Test_TMPLDocumentation.cpp tmpl::empty_sequence

\remark
Prefer just writing an empty list.


\subsubsection false_type

\par
An \ref integral_constant representing `false`.  Similar to std::false_type.
\snippet Test_TMPLDocumentation.cpp tmpl::false_type

\remark
Prefer std::false_type.


\subsubsection no_such_type_

\par
An empty struct returned as the failure case for various searching operations.
\snippet Test_TMPLDocumentation.cpp tmpl::no_such_type_


\subsubsection true_type

\par
An \ref integral_constant representing `true`.  Similar to std::true_type.
\snippet Test_TMPLDocumentation.cpp tmpl::true_type

\remark
Prefer std::true_type.


\subsection list_constructor Constructor-like functions for lists

\par
These functions produce \ref list ""s from non-list values.  They are often
similar to constructors in the STL.


\subsubsection filled_list

\par
Creates a list containing a given number (passed as an `unsigned int`) of the
same type.  The head of the list defaults to \ref list.
\snippet Test_TMPLDocumentation.cpp tmpl::filled_list


\subsubsection integral_list

\par
Shorthand for a \ref list of \ref integral_constant ""s of the same type
\snippet Test_TMPLDocumentation.cpp tmpl::integral_list

\remark
Prefer std::integer_sequence when used for pack expansion.  Prefer this when
the contents need to be manipulated for more complicated metaprogramming.


\subsubsection make_sequence

\par
Produces a list with a given first element and length (provided as an `unsigned
int`).  The remaining elements are obtained by repeated applications of a
\ref metalambdas "metalambda", defaulting to \ref next.  The head of the
sequence can be specified, and defaults to \ref list.
\snippet Test_TMPLDocumentation.cpp tmpl::make_sequence

\see \ref range, \ref repeat


\subsubsection range

\par
Produces a \ref list of \ref integral_constant ""s representing adjacent
ascending integers, including the starting value and excluding the ending
value.
\snippet Test_TMPLDocumentation.cpp tmpl::range

\see \ref reverse_range


\subsubsection reverse_range

\par
Produces a \ref list of \ref integral_constant ""s representing adjacent
descending integers, including the starting value and excluding the ending
value.
\snippet Test_TMPLDocumentation.cpp tmpl::reverse_range

\see \ref range


\subsection list_query Functions for querying lists

\par
These tend to be similar to const member functions in the STL and the
non-modifying sequence operations in `<algorithm>`.  They are most frequently
used with \ref list, but similar classes will also work.


\subsubsection all

\par
Checks if a predicate is true for all elements of a list.  The default
predicate checks that the element's `value` is not equal to zero.
\snippet Test_TMPLDocumentation.cpp tmpl::all

\note
The predicate must return the same true value for each element for `all` to
return true.
\snippet Test_TMPLDocumentation.cpp tmpl::all:inhomogeneous

\see \ref any, \ref none


\subsubsection any

\par
Checks if a predicate is true for at least one element of a list.  The default
predicate checks that the element's `value` is not equal to zero.
\snippet Test_TMPLDocumentation.cpp tmpl::any

\remark
The \ref any and \ref none metafunctions perform the same tasks as \ref found
and \ref not_found, but use different algorithms.  In general, \ref any and
\ref none are much faster, but \ref found and \ref not_found short-circuit, so
they may be preferable with short lists and expensive predicates.

\note
The predicate must return the same false value for each element for `any` to
return false.
\snippet Test_TMPLDocumentation.cpp tmpl::any:inhomogeneous

\see \ref all, \ref found, \ref none


\subsubsection at

\par
Retrieves a given element of a list, similar to `operator[]` of the STL
containers.  The index number is supplied as an \ref integral_constant or
similar type.
\snippet Test_TMPLDocumentation.cpp tmpl::at

\par
This operator is \ref map_at "overloaded for maps".

\see \ref at_c


\subsubsection at_c

\par
Retrieves a given element of a list, similar to `operator[]` of the STL
containers.  The index number is supplied as an `unsigned int`.
\snippet Test_TMPLDocumentation.cpp tmpl::at_c

\see \ref at


\subsubsection back

\par
Retrieves the last element of a list.
\snippet Test_TMPLDocumentation.cpp tmpl::back


\subsubsection count_if

\par
Returns the number of elements of a list satisfying a predicate.
\snippet Test_TMPLDocumentation.cpp tmpl::count_if

\note
If the predicate is neither a \ref bind "bind expression" with a single
argument of `_1` nor a \ref metalambda_lazy "lazy expression" with a single
argument of `_1`, then a bug causes the `type` of the result of the predicate
to be used instead of the result itself.  As long as the predicate returns an
\ref integral_constant or a std::integral_constant this does not matter, as
`type` is a no-op for those classes.
\snippet Test_TMPLDocumentation.cpp tmpl::count_if:bug:definitions
\snippet Test_TMPLDocumentation.cpp tmpl::count_if:bug:asserts


\subsubsection fold

\par
Performs a
[left fold](https://en.wikipedia.org/wiki/Fold_(higher-order_function)), i.e.,
given a list, initial state, and \ref metalambdas "metalambda", updates the
state by calling the metalambda on the state and the first element, repeats
with the second, and so on, returning the final state.
\snippet Test_TMPLDocumentation.cpp tmpl::fold

\par
Brigand provides `tmpl::_state` and `tmpl::_element` aliases to the appropriate
\ref args "arguments" for use in folds.

\see \ref reverse_fold


\subsubsection found

\par
Returns, as an \ref integral_constant of `bool`, whether a predicate matches
any element of a list.  The default predicate checks that the element's `value`
is not equal to zero.
\snippet Test_TMPLDocumentation.cpp tmpl::found

\remark
This function performs the same operation as \ref any.  See \ref any for
discussion.

\see \ref any, \ref find, \ref not_found


\subsubsection front

\par
Retrieves the first element of a list.
\snippet Test_TMPLDocumentation.cpp tmpl::front


\subsubsection index_if

\par
Finds the index as a `size_t` \ref integral_constant of the first type in a
list satisfying a \ref metalambdas "predicate".  Returns a supplied type,
defaulting to \ref no_such_type_ if no elements match.
\snippet Test_TMPLDocumentation.cpp tmpl::index_if


\subsubsection index_of

\par
Finds the index as a `size_t` \ref integral_constant of the first occurrence of
a type in a list.  Returns \ref no_such_type_ if the type is not found.
\snippet Test_TMPLDocumentation.cpp tmpl::index_of


\subsubsection list_contains

\par
Checks whether a particular type is contained in a list, returning an
\ref integral_constant of `bool`.
\snippet Test_TMPLDocumentation.cpp tmpl::list_contains

\note
This is not a Brigand metafunction.  It is implemented in SpECTRE.


\subsubsection none

\par
Checks if a predicate is false for all elements of a list.  The default
predicate checks that the element's `value` is not equal to zero.
\snippet Test_TMPLDocumentation.cpp tmpl::none

\remark
This function performs the same operation as \ref not_found.  See \ref any for
discussion.

\note
The predicate must return the same false value for each element for `none` to
return true.
\snippet Test_TMPLDocumentation.cpp tmpl::none:inhomogeneous

\see \ref all, \ref any, \ref not_found


\subsubsection not_found

\par
Returns, as an \ref integral_constant of `bool`, whether a predicate matches
no elements of a list.  The default predicate checks that the element's `value`
is not equal to zero.
\snippet Test_TMPLDocumentation.cpp tmpl::not_found

\remark
This function performs the same operation as \ref none.  See \ref any for
discussion.

\see \ref find, \ref found, \ref none


\subsubsection size

\par
Returns the size of a list as an \ref integral_constant of type `unsigned int`.
\snippet Test_TMPLDocumentation.cpp tmpl::size


\subsection list_to_list Functions producing lists from other lists

\par
These tend to be similar to non-const member functions in the STL and the
mutating sequence operations in `<algorithm>`, but due to the nature of
metaprogramming all return a new list rather than modifying an argument.  They
are most frequently used with \ref list, but similar classes will also work.


\subsubsection append

\par
Appends the contents of several lists to the contents of a list.
\snippet Test_TMPLDocumentation.cpp tmpl::append

\par
For a version taking its arguments as a list of lists, see \ref join.

\warning
A flaw in the implementation makes use of this metafunction very error-prone.
Prefer \ref push_back or \ref push_front when possible.
\snippet Test_TMPLDocumentation.cpp tmpl::append::bug


\subsubsection clear

\par
Produces list with the same head but no elements.
\snippet Test_TMPLDocumentation.cpp tmpl::clear

\remark
If the head is known, prefer writing it explicitly.  If the head is irrelevant,
write an empty \ref list.


\subsubsection erase

\par
Produces a copy of a list with the element at the given index (passed as an
\ref integral_constant or similar type) removed.
\snippet Test_TMPLDocumentation.cpp tmpl::erase

\par
This operator is overloaded \ref map_erase "for maps" and \ref set_erase
"for sets".

\see \ref erase_c


\subsubsection erase_c

\par
Produces a copy of a list with the element at the given index (passed as an
`unsigned int`) removed.
\snippet Test_TMPLDocumentation.cpp tmpl::erase_c


\subsubsection filter

\par
Removes all types not matching a \ref metalambdas "predicate" from a list.
\snippet Test_TMPLDocumentation.cpp tmpl::filter

\see \ref remove_if


\subsubsection find

\par
Given a list and a \ref metalambdas "predicate", returns a list containing the
first element for which the predicate returns true and all subsequent elements.
The default predicate checks that the element's `value` is not equal to zero.
Returns an empty list if the predicate returns false for all elements.
\snippet Test_TMPLDocumentation.cpp tmpl::find

\see \ref found, \ref not_found, \ref reverse_find


\subsubsection flatten

\par
Given a list, recursively inlines the contents of elements that are lists with
the same head.
\snippet Test_TMPLDocumentation.cpp tmpl::flatten


\subsubsection join

\par
Appends to a list in the same manner as \ref append, but takes a list of lists
instead of multiple arguments.
\snippet Test_TMPLDocumentation.cpp tmpl::join

\warning
A flaw in the implementation makes use of this metafunction very error-prone.
Prefer \ref push_back or \ref push_front when possible.
\snippet Test_TMPLDocumentation.cpp tmpl::join::bug

\par

\warning
This metafunction has a lazy version, but its behavior does not match the eager
version, as it determines the head of the resulting list differently.
\snippet Test_TMPLDocumentation.cpp tmpl::join::bug-lazy


\subsubsection list_difference

\par
Remove all elements that occur in the second list from the first list.
\snippet Test_TMPLDocumentation.cpp tmpl::list_difference

\note
This is not a Brigand metafunction.  It is implemented in SpECTRE.


\subsubsection merge

\par
Given two sorted lists, returns a sorted list containing the elements of both.
A comparator metalambda can be provided, defaulting to \ref math_comparison
"less".
\snippet Test_TMPLDocumentation.cpp tmpl::merge

\note
If there are equivalent elements, those from the second list are placed
earlier.
\snippet Test_TMPLDocumentation.cpp tmpl::merge:equiv

\see std::merge


\subsubsection partition

\par
Given a list and a \ref metalambdas "predicate", returns a \ref pair containing
a list of the elements for which the predicate returns true and a list of the
elements for which the predicate returns false.
\snippet Test_TMPLDocumentation.cpp tmpl::partition

\see \ref filter, \ref remove_if


\subsubsection pop_back

\par
Remove elements from the end of a list.  The number of elements to remove is
supplied as an \ref integral_constant and defaults to 1.
\snippet Test_TMPLDocumentation.cpp tmpl::pop_back


\subsubsection pop_front

\par
Remove elements from the beginning of a list.  The number of elements to remove
is supplied as an \ref integral_constant and defaults to 1.
\snippet Test_TMPLDocumentation.cpp tmpl::pop_front


\subsubsection push_back

\par
Appends types to a list.
\snippet Test_TMPLDocumentation.cpp tmpl::push_back


\subsubsection push_front

\par
Prepends types to a list.  The order of the prepended items is retained: they
are pushed as a unit, not one-by-one.
\snippet Test_TMPLDocumentation.cpp tmpl::push_front


\subsubsection remove

\par
Removes all occurrences of a given type from a list.
\snippet Test_TMPLDocumentation.cpp tmpl::remove


\subsubsection remove_duplicates

\par
Remove duplicates from a list.  The first occurrence of each type is kept.
\snippet Test_TMPLDocumentation.cpp tmpl::remove_duplicates

\note
This is not a Brigand metafunction.  It is implemented in SpECTRE.


\subsubsection remove_if

\par
Removes all types matching a \ref metalambdas "predicate" from a list.
\snippet Test_TMPLDocumentation.cpp tmpl::remove_if

\see \ref filter


\subsubsection replace

\par
Replaces all occurrences of one type in a list with another.
\snippet Test_TMPLDocumentation.cpp tmpl::replace

\warning
This metafunction has a lazy version, but it is broken because it is
implemented as a non-trivial type alias.  Use \ref bind "bind" on the eager
version instead.
\snippet Test_TMPLDocumentation.cpp tmpl::replace:bug


\subsubsection replace_if

\par
Replaces all types matching a \ref metalambdas "predicate" in a list with a
given type.
\snippet Test_TMPLDocumentation.cpp tmpl::replace_if


\subsubsection reverse

\par
Reverses the order of types in a list.
\snippet Test_TMPLDocumentation.cpp tmpl::reverse


\subsubsection reverse_find

\par
Given a list and a \ref metalambdas "predicate", returns a list containing the
last element for which the predicate returns true and all preceding elements.
The default predicate checks that the element's `value` is not equal to zero.
Returns an empty list if the predicate returns false for all elements.
\snippet Test_TMPLDocumentation.cpp tmpl::reverse_find

\see \ref find


\subsubsection reverse_fold

\par
Performs a
[right fold](https://en.wikipedia.org/wiki/Fold_(higher-order_function)), i.e.,
given a list, initial state, and \ref metalambdas "metalambda", updates the
state by calling the metalambda on the state and the last element, repeats with
the second to last, and so on, returning the final state.
\snippet Test_TMPLDocumentation.cpp tmpl::reverse_fold

\par
Brigand provides `tmpl::_state` and `tmpl::_element` aliases to the appropriate
\ref args "arguments" for use in folds.

\see \ref fold


\subsubsection sort

\par
Sorts a list according to a comparator, which defaults to \ref math_comparison
"less".
\snippet Test_TMPLDocumentation.cpp tmpl::sort

\note
The sort is not stable.
\snippet Test_TMPLDocumentation.cpp tmpl::sort:equiv


\subsubsection split

\par
Splits a list into parts separated by a specified delimiter, discarding empty
parts.
\snippet Test_TMPLDocumentation.cpp tmpl::split


\subsubsection split_at

\par
Given a list and an integer \f$N\f$ (supplied as an \ref integral_constant),
returns a list of two of lists, the first containing the first \f$N\f$
elements, and the second containing the remaining elements.
\snippet Test_TMPLDocumentation.cpp tmpl::split_at


\subsubsection transform

\par
Given a list, calls a \ref metalambdas "metalambda" on each type in the list,
collecting the results in a new list.  If additional lists are supplied,
elements from those lists are passed as additional arguments to the metalambda.
\snippet Test_TMPLDocumentation.cpp tmpl::transform


\subsection map_operations Operations on maps

\par
Brigand's \ref map type can be manipulated by several metafunctions.

\par
Examples in this section use this map as an example:
\snippet Test_TMPLDocumentation.cpp example_map


\subsubsection map_at at

\par
Returns the value associated with a key in a \ref map.  Returns \ref
no_such_type_ if the key is not in the map.
\snippet Test_TMPLDocumentation.cpp tmpl::at:map

\par
This operator is \ref at "overloaded for lists".  When called on a \ref map,
this is the same as \ref lookup.


\subsubsection map_erase erase

\par
Produces a copy of a map with the element with the given key removed.  If the
key is not in the map, returns the map unchanged.
\snippet Test_TMPLDocumentation.cpp tmpl::erase:map

\par
This operator is overloaded \ref erase "for lists" and \ref set_erase
"for sets".


\subsubsection map_has_key has_key

\par
Returns an \ref integral_constant of `bool` indicating whether a \ref map has a
given key.
\snippet Test_TMPLDocumentation.cpp tmpl::has_key:map

\par
This operator is \ref set_has_key "overloaded for sets".


\subsubsection map_insert insert

\par
Returns a new map containing an additional key-value pair.  If the key is
already in the map, the map is returned unchanged.
\snippet Test_TMPLDocumentation.cpp tmpl::insert:map

\par
This operator is \ref set_insert "overloaded for sets".

\warning
A bug allows invalid maps with duplicate keys to be constructed using
`insert`.  The insertion check improperly compares the values, as well as the
keys, to decide whether to alter the map.  In cases where this may occur, the
check may be made manually using \ref map_has_key "has_key".
\snippet Test_TMPLDocumentation.cpp tmpl::insert:map:bug


\subsubsection keys_as_sequence

\par
Returns the keys from a map as a sequence of a specified type, defaulting to
\ref set.
\snippet Test_TMPLDocumentation.cpp tmpl::keys_as_sequence

\par
If the key-value pairs are required, they can be extracted directly from the
template arguments of the \ref map.

\see \ref values_as_sequence


\subsubsection lookup

\par
Returns the value associated with a key in a \ref map.  Returns \ref
no_such_type_ if the key is not in the map.
\snippet Test_TMPLDocumentation.cpp tmpl::lookup

\see \ref map_at "at"


\subsubsection lookup_at

\par
Returns the value associated with a key in a \ref map, wrapped in a \ref type_.
Returns `type_<no_such_type_>` if the key is not in the map.  This function has
no eager version, but is still in the `tmpl::lazy` namespace.
\snippet Test_TMPLDocumentation.cpp tmpl::lookup_at

\see \ref lookup


\subsubsection values_as_sequence

\par
Returns the values from a map as a sequence of a specified type, defaulting to
\ref list.
\snippet Test_TMPLDocumentation.cpp tmpl::values_as_sequence

\par
If the key-value pairs are required, they can be extracted directly from the
template arguments of the \ref map.

\see \ref keys_as_sequence


\subsection set_operations Operations on sets

\par
Brigand's \ref set type can be manipulated by several metafunctions.


\subsubsection contains

\par
Returns an \ref integral_constant of `bool` indicating whether a set contains
a particular element.
\snippet Test_TMPLDocumentation.cpp tmpl::contains


\subsubsection set_erase erase

\par
Produces a copy of a set with the given element removed.  If the element is not
in the set, returns the set unchanged.
\snippet Test_TMPLDocumentation.cpp tmpl::erase:set

\par
This operator is overloaded \ref erase "for lists" and \ref map_erase
"for maps".


\subsubsection set_has_key has_key

\par
Returns an \ref integral_constant of `bool` indicating whether a \ref set
contains a given element.
\snippet Test_TMPLDocumentation.cpp tmpl::has_key:set

\par
This operator is \ref map_has_key "overloaded for maps".


\subsubsection set_insert insert

\par
Returns a new set containing an additional element.  If the element is already
in the set, the set is returned unchanged.
\snippet Test_TMPLDocumentation.cpp tmpl::insert:set

\par
This operator is \ref map_insert "overloaded for maps".


\subsection math Mathematical functions

\par
These perform the same operations at their language counterparts, but on \ref
integral_constant ""s (or anything else with a `value` static member type of
type `value_type`).  The results inherit from \ref integral_constant ""s of
types noted below.

\par
These are all lazy metafunctions.


\subsubsection math_arithmetic Arithmetic operators

\par
These operations return classes inheriting from \ref integral_constant ""s of
the same type as the result of the language operator on their arguments.  The
integral promotion and conversion rules are applied.  (Contrast the \ref
math_bitwise "bitwise operators".)
\snippet Test_TMPLDocumentation.cpp math_arithmetic

\par
The standard library runtime functors have the same names for std::plus,
std::minus, std::divides, and std::negate, but the other two are
std::multiplies and std::modulus.


\subsubsection math_bitwise Bitwise operators

\par
These operations return classes inheriting from \ref integral_constant ""s of
the same type as their first argument's `value`.  This is *not* generally the
same type as the language operator, even when the types of the values of both
arguments are the same.  (The integer promotion and conversion rules are not
applied.)
\snippet Test_TMPLDocumentation.cpp math_bitwise

\par
The standard library runtime functors are called std::bit_not, std::bit_and,
std::bit_or, and std::bit_xor.


\subsubsection math_comparison Comparison operators

\par
These operations return classes inheriting from \ref integral_constant ""s of
`bool`.
\snippet Test_TMPLDocumentation.cpp math_comparison

\par
The standard library runtime functors have the same names, such as
std::equal_to.


\subsubsection math_logical Logical operators

\par
These operations return classes inheriting from \ref integral_constant ""s of
`bool`.  They should only be used on types wrapping `bool`s.
\snippet Test_TMPLDocumentation.cpp math_logical

\par
The standard library runtime functors are called std::logical_and,
std::logical_or, and std::logical_not.  The xor operation is equivalent to \ref
math_comparison "not_equal_to".


\subsubsection identity

\par
The identity function.  Unlike most math functions, this returns the same type
as its argument, even if that is not an \ref integral_constant.
\snippet Test_TMPLDocumentation.cpp tmpl::identity

\see \ref always


\subsubsection max

\par
Computes the larger of two values, returning an \ref integral_constant of the
common type of its arguments.
\snippet Test_TMPLDocumentation.cpp tmpl::max


\subsubsection min

\par
Computes the smaller of two values, returning an \ref integral_constant of the
common type of its arguments.
\snippet Test_TMPLDocumentation.cpp tmpl::min


\subsubsection next

\par
Computes the passed value plus one, returning an \ref integral_constant of the
same type as its argument.
\snippet Test_TMPLDocumentation.cpp tmpl::next


\subsubsection prev

\par
Computes the passed value minus one, returning an \ref integral_constant of the
same type as its argument.
\snippet Test_TMPLDocumentation.cpp tmpl::prev


\subsection misc Miscellaneous functions

\par
Functions that don't fit into any of the other sections.


\subsubsection always

\par
A lazy identity function.
\snippet Test_TMPLDocumentation.cpp tmpl::always

\see \ref identity


\subsubsection apply

\par
Calls a \ref metalambdas "metalambda" with given arguments.
\snippet Test_TMPLDocumentation.cpp tmpl::apply


\subsubsection count

\par
Returns the number of template parameters provided as an \ref integral_constant
of `unsigned int`.
\snippet Test_TMPLDocumentation.cpp tmpl::count


\subsubsection conditional_t

\par
Returns the second argument if the first is true, otherwise the third.  An
optimized version of std::conditional_t.
\snippet Test_TMPLDocumentation.cpp tmpl::conditional_t

\note
This is not a Brigand metafunction.  It is implemented in SpECTRE.


\subsubsection eval_if

\par
A lazy metafunction that, if the conditional (first) argument has a true
`value`, evaluates and returns the result of the first lazy metafunction (*not*
metalambda), otherwise, evaluates and returns the result of the second lazy
metafunction.
\snippet Test_TMPLDocumentation.cpp tmpl::eval_if

\par
This is performs lazy evaluation of conditional branches outside of a
metalambda.


\subsubsection eval_if_c

\par
The same as \ref eval_if, but takes its first argument as a `bool` instead of a
type.
\snippet Test_TMPLDocumentation.cpp tmpl::eval_if_c


\subsubsection has_type

\par
A lazy metafunction that returns its second argument (defaulting to `void`),
ignoring its first argument.
\snippet Test_TMPLDocumentation.cpp tmpl::has_type

\par
This can be used to expand a parameter pack to repetitions of the same type.
\snippet Test_TMPLDocumentation.cpp tmpl::has_type:pack_expansion
\snippet Test_TMPLDocumentation.cpp tmpl::has_type:pack_expansion:asserts


\subsubsection if_

\par
A lazy metafunction that returns the second argument if the `value` static
member value of the first is true, and otherwise the third.
\snippet Test_TMPLDocumentation.cpp tmpl::if_

\warning
The second and third arguments are both evaluated, independent of which is
returned.  Use \ref defer "defer" or \ref eval_if if this is undesirable.


\subsubsection if_c

\par
The same as std::conditional.
\snippet Test_TMPLDocumentation.cpp tmpl::if_c


\subsubsection inherit

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
The \ref empty_base type is used internally as a sentinel.  The result may or
may not inherit from \ref empty_base, independently of whether it is supplied
as an argument.


\subsubsection inherit_linearly

\par
Transforms a list into a linked list.  This metafunction takes three arguments:
the list of types, the node structure, and the root, which defaults to \ref
empty_base.  The node structure must be a class template (*not* a lazy
metafunction) instantiated with metalambdas.  The function performs a [left
fold](https://en.wikipedia.org/wiki/Fold_(higher-order_function)), with the
root as the initial state and the transform function evaluating the arguments
to the node structure.
\snippet Test_TMPLDocumentation.cpp tmpl::inherit_linearly

\remark
This function handles its function-like argument differently from any other
function in Brigand.  Prefer \ref fold, which can perform the same task and has
a more standard interface.


\subsubsection is_set

\par
Tests if all of its arguments are distinct, producing a \ref integral_constant
"bool_".
\snippet Test_TMPLDocumentation.cpp tmpl::is_set

\note
This is unrelated to the Brigand \ref set type.


\subsubsection real_

\par
Represents a floating point number at compile time via its internal memory
representation.  The value is stored as an \ref integral_constant of the same
size as the floating point type, and can be extracted at runtime using the
conversion operator.  Brigand provides the aliases `single_` and `double_` for
the built-in floating point types.
\snippet Test_TMPLDocumentation.cpp tmpl::real_

\par
There are no compile-time mathematical functions provided for floating point
types.  They are opaque (or sometimes treated as integers) until runtime.

\remark
Consider whether you really need to represent floating point values at compile
time.

\see std::ratio


\subsubsection repeat

\par
Calls a unary eager metafunction on a given type, then on the result of that,
then on the result of that, and so on, up to a specified (as an \ref
integral_constant) number of calls.
\snippet Test_TMPLDocumentation.cpp tmpl::repeat

\note
This function has a lazy version, but it cannot be used in a metalambda because
the template template parameter prevents manipulation of the parameter list.
\snippet Test_TMPLDocumentation.cpp tmpl::repeat:lazy

\see \ref make_sequence


\subsubsection sizeof_

\par
A lazy metafunction that computes `sizeof` its argument as an `unsigned int`
\ref integral_constant.
\snippet Test_TMPLDocumentation.cpp tmpl::sizeof_


\subsubsection substitute

\par
Substitutes values for \ref args "args" (but *not* `_1` or `_2`) appearing in a
type.
\snippet Test_TMPLDocumentation.cpp tmpl::substitute


\subsubsection type_from

\par
Extracts the `type` from a struct.
\snippet Test_TMPLDocumentation.cpp tmpl::type_from

\remark
This function will work on any class with a `type` type alias, but, when used
outside of a metafunction, it should only be used with \ref type_ for clarity.


\subsubsection wrap

\par
Replaces the head of a sequence.
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


\subsubsection for_each_args

\par
Calls the first argument (a functor) on each of the remaining arguments, in
order.  Returns the functor.
\snippet Test_TMPLDocumentation.cpp tmpl::for_each_args:defs
\snippet Test_TMPLDocumentation.cpp tmpl::for_each_args

\note
The functor must be copyable.  This is a bug.

\par

\note
This uses a std::reference_wrapper internally, but I don't see a reason for
that.  If it were removed then this function could be constexpr (before C++20).


\subsubsection for_each

\par
Calls a functor on \ref type_ objects wrapping each type in a list, in order.
Returns the functor.
\snippet Test_TMPLDocumentation.cpp tmpl::for_each:defs
\snippet Test_TMPLDocumentation.cpp tmpl::for_each

\note
The functor must be copyable.  This is a bug.

\par

\note
An object of the list template parameter type is constructed, so the list must
be a complete type.

\see \ref type_from


\subsubsection select

\par
Returns its first argument if the template argument's `value` is true,
and its second argument if it is false.
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
similar class with a `value_type` and a `value`) to \ref integral_constant.
The lazy metafunction `make_integral` performs this conversion.  The
`as_integral_list` eager metafunction performs this operation on all elements
of a list.
\snippet Test_TMPLDocumentation.cpp tmpl::make_integral

\warning
The standard library std::integer_sequence is not a list of types, and so
cannot be used as input to `as_integral_list`.


\subsubsection as_list list

\par
Brigand provides two metafunctions for converting types to Brigand sequences.
The more general function, `as_sequence`, is equivalent to \ref wrap.  The
specialized version, `as_list`, produces a \ref list.
\snippet Test_TMPLDocumentation.cpp tmpl::as_list

\remark
Using `as_list` is often not necessary because most metafunctions operate on
arbitrary template classes.


\subsubsection as_set set

\par
Brigand provides the standard two metafunctions for converting types to Brigand
\ref set ""s.
\snippet Test_TMPLDocumentation.cpp tmpl::as_set


\section oddities Bugs/Oddities

* \ref join has eager and lazy versions that don't agree.

* \ref push_front and \ref pop_front have lazy versions, but \ref push_back,
  and \ref pop_back do not.

* \ref reverse_range validates its arguments, but \ref range does not.
  (Probably because the former is called incorrectly more often.)

* \ref repeat and \ref wrap have unusable (in metalambdas) lazy versions
  (because they have a template template parameter).

* \ref replace has a completely broken lazy version (because it is a
  non-trivial type alias instead of a struct).

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
