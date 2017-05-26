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

* Adhere to [Google style](http://clang.llvm.org/docs/ClangFormat.html).
Can use `clang-format -style=google`.
* CamelCase: class names, type names, namespaces, template
parameters, file names, and directory names.
* snake_case: (meta)function and (meta)variables names.
* SCREAMING_SNAKE_CASE: macros.
* Member variables of classes (not structs!) have a trailing underscore. See
[Google](https://google.github.io/styleguide/cppguide.html#Variable_Names).
* Types NOT specified in the name (e.g. `double* pd_blah` is bad)
* Header order: hpp corresponding to cpp, blank line, STL and externals,
blank line, SpECTRE includes
* Header order is alphabetical
* Template definitions in header files are separated from the declaration of
the class by the following line, which contains exactly 70 equal signs

``` cpp
// ======================================================================
```

* No blank lines surrounding Doxygen group comments (`//\@{` and `//\@}`).
* Use [alternative tokens](http://en.cppreference.com/w/cpp/language/operator_alternative)
whenever possible. E.g. `or` and `and` keywords instead of `||` and `&&`,
respectively.
* Use C-style Doxygen comments except for documenting the file `/*! ... */`

Code Quality Items:

* All code passes Clang and CppCheck static analyzers. For help
with these tools see [[here]].
* Almost always `auto`
* All loops and if statements use braces.
* Order of declaration in classes is `public` before `private` and member
functions before data. See
[Google](https://google.github.io/styleguide/cppguide.html#Declaration_Order).
* Use a functional style unless pass-by-mutable-reference provides superior
performance.
* All commits for performance changes provide quantitative evidence and the
tests used to obtain said evidence.
* Never include `<iostream>`
* Do not add anything to
[the `std` namespace](http://en.cppreference.com/w/cpp/language/extending_std).
* Virtual functions are explicitly overridden using the `override` keyword.
* `pragma once` is to be used for header guards
* Use `size_t` for positive integers rather than `int`, specifically when
looping over containers. This is in compliance with what the STL uses.
* Error messages are helpful. An example of a poor error message is
"Size mismatch" which should read "The number of grid points in the matrix 'F'
is not the same as the number of grid points in the determinant."
