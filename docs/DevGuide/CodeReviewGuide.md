\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Code Review Guide {#code_review_guide}

Code must follow the
<a href="https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md">C++ Core Guidelines</a>
and the [Google style guide](https://google.github.io/styleguide/cppguide.html).
If the Google style guide disagrees with the Core Guidelines, follow the Core
Guidelines.

Here we summarize what we view as the more important portions of the guides.

Stylistic Items:

* Adhere to [Google style](https://google.github.io/styleguide/cppguide.html).
  [Can use `clang-format -style=google`]
  (http://clang.llvm.org/docs/ClangFormat.html).
* CamelCase: class names template parameters, file names, and directory names.
* snake_case: function, variable, metafunction and metavariable names.
* SCREAMING_SNAKE_CASE: macros.
* Functions or classes only used internally in a library should be in a
  namespace named `LibraryOrFile_detail`. For example, `databox_detail`,
  `Tensor_detail` or `ConstantExpressions_detail`.
* Name unused function parameters `/*parameter_name*/` or `/*meta*/` for TMP
  cases
* Type aliases that wrap type traits have a trailing `_t` following the STL
* Private member variables have a [trailing underscore]
  (https://google.github.io/styleguide/cppguide.html#Variable_Names).
* Do not use
  [Hungarian notation](https://en.wikipedia.org/wiki/Hungarian_notation),
  e.g. `double* pd_blah` is bad
* Header order:
  1. (If a test:) `tests/Unit/TestingFramework.hpp`, followed by one blank line
  2. (If a cpp file with a corresponding hpp file:) hpp corresponding to cpp,
     followed by one blank line
  3. STL and externals (in alphabetical order)
  4. Blank line
  5. SpECTRE includes (in alphabetical order)
* Template definitions in header files are separated from the declaration of
  the class by the following line, which contains exactly 64 equal signs

``` cpp
// ================================================================
```

* File lists in CMake are alphabetical.
* No blank lines surrounding Doxygen group comments
  (<code>// \@{</code> and <code>// \@}</code>).
* Use the [alternative tokens]
  (http://en.cppreference.com/w/cpp/language/operator_alternative)
  `or`, `and`, and `not` instead of `||`, `&&`, and `!`.
* Use C-style Doxygen comments (`/*! ... */`) when using multi-line math,
  otherwise C-style and C++ style comments are accepted.
* When addressing requests on a PR, the commit message must start with
  `fixup` followed by a descriptive message.

Code Quality Items:

* All code passes Clang and CppCheck static analyzers. For help
  with these tools see \ref static_analysis_tools "here".
* Almost always `auto`, except with expression templates, i.e. `DataVector`
* All loops and if statements use braces.
* Order of declaration in classes is `public` before `private` and member
  functions before data. See
  [Google](https://google.github.io/styleguide/cppguide.html#Declaration_Order).
* Prefer return by value over pass-by-mutable-reference except when mutable
  reference provides superior performance (in practice if you need a mutable
  reference use `const gsl::not_null<Type*>` instead of `Type&`). The mutable
  references must be the first arguments passed to the function.
* All commits for performance changes provide quantitative evidence and the
  tests used to obtain said evidence.
* Never include `<iostream>`, use `Parallel::printf` inside
  `Parallel/Printf.hpp` instead, which is safe to use in parallel.
* When using charm++ nodelocks include `<converse.h>` instead of `<lrtslock.h>`.
* Do not add anything to [the `std` namespace]
  (http://en.cppreference.com/w/cpp/language/extending_std).
* Virtual functions are explicitly overridden using the `override` keyword.
* `pragma once` is to be used for header guards
* Prefer range-based for loops
* Use `size_t` for positive integers rather than `int`, specifically when
  looping over containers. This is in compliance with what the STL uses.
* Error messages should be helpful. An example of a bad error message is "Size
  mismatch". Instead this message could read "The number of grid points in the
  matrix 'F' is not the same as the number of grid points in the determinant.",
  along with the runtime values of the mentioned quantities if applicable.
* Mark immutable objects as `const`
* Make classes serializable by writing a `pup` function
* If a class stores an object passed into a constructor the object should
  be taken by-value and `std::move`d into the member variable.
* Definitions of function and class templates should be in `.cpp` files with
  explicit instantiations whenever possible. The macro
  `GENERATE_EXPLICIT_INSTANTIATIONS` is useful for generating many
  explicit instantiations.
* Explicit instantiations of functions marked as noexcept should be marked
  as noexcept as well.
* Functions that do not throw should be marked `noexcept`. If you're unsure
  and the function does not use an `OptionContext` or is generating python
  bindings, mark it `noexcept`.
* Variable names in macros must avoid name collisions, e.g. inside the
  `PARSE_ERROR` macro you would write
  `double variable_name_avoid_name_collisions_PARSE_ERROR = 0.0;`
* Avoid macros if possible. Prefer `constexpr` functions, constexpr
  variables, and/or template metaprogramming
* Explicitly specify `double`s, e.g. `sqrt(2.)` or `sqrt(2.0)` instead of
  `sqrt(2)`.
* When the index of a `Tensor` is known at compile time, use
  `get<a, b>(tensor)` instead of the member `get` function.
* All necessary header files must be included. In header files, prefer
  forward declarations if class definitions aren't necessary.
* Explicitly make numbers floating point, e.g. `2.` or `2.0` over `2`.
