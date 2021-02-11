\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond

# Build Profiling and Optimization {#build_profiling_and_optimization}

\tableofcontents

# Why is our build so expensive?

SpECTRE makes heavy use of compile-time logic, which is responsible for a lot of
the nice type-checking, performance, and flexibility the code has to offer.
For instance, our central data structure, the \ref DataBoxGroup "DataBox", uses
a type list of several "tags" to determine its contents, as well as to
automatically propagate dependencies in compute items.

This use of compile-time logic, however, has the trade-off of making our builds
take longer and use more memory than a similar implementation in runtime logic
would.
There is good reason to believe that some of these costs are payed back at
runtime, because many of our compile-time switches permit the final runtime code
to be more efficient or avoid unnecessary computation.
There is certainly room for optimization, though, either in finding better
compile-time implementations of the algorithms, eliminating expensive template
instantiations, or moving inefficient parts to runtime.
This guide gives a quick outline of some of the methods that can be used to
profile the build and possible pitfalls

# Understanding template expenses

The cost of compile-time template logic is a bit non-intuitive if you are used
to thinking only of runtime performance.
The main reason is that the typical unit of 'cost' in a compile-time operation
is the number of instantiated types and functions.
Re-using a type that has been instantiated elsewhere (in the same translation
unit) typically has a very low compile-time cost, where instantiating a type
with new template parameters will incur its own cost, plus any new types that it
requires in e.g. type aliases or member functions.

Consider a Fibonacci calculation at compile-time:
```
template <size_t N>
struct Fibonacci {
  static constexpr size_t value =
      Fibonacci<N - 1>::value + Fibonacci<N - 2>::value;
};

template <>
struct Fibonacci<1> {
  static constexpr size_t value = 1;
};

template <>
struct Fibonacci<0> {
  static constexpr size_t value = 1;
};
```
If we were to write the same logic at runtime, the algorithm would be hopelessly
inefficient; the recursive calls would cause each call to Fibonacci to make two
calls to the same function, resulting in an exponential time algorithm!
However, the C++ language will only instantiate unique types, so only N types
will be created, giving a linear in compile-time operation.


Compile-time lists and list operations frequently appear in SpECTRE, and should
be thought of differently from runtime list operations.

In compile-time lists, we have no access to true constant-time lookup, speedy
algorithms that rely on sorted structures, or more sophisticated data structures
(balanced trees, hash tables, etc.).
The limitations of compile-time list techniques can cause list operations to be
more costly than we could achieve with runtime data structures.

Consider a basic version of the compile-time list accessor:
```
template <typename List, size_t Index>
struct list_at;

template <typename ListItem0, typename... ListItems>
struct list_at<tmpl::list<ListItem0, ListItems...>, 0> {
  using type = ListItem0;
};

template <typename ListItem0, typename... ListItems, size_t Index>
struct list_at<tmpl::list<ListItem0, ListItems...>, Index> {
  using type = typename list_at<tmpl::list<ListItems...>, Index - 1>::type;
};
```
Now, to access the Nth item in the list, we need to instantiate \f$O(N)\f$
types.
The above implementation is significantly more costly than we would find in
practice in template metaprogramming libraries -- in particular, our chosen
TMPL backend, [Brigand](https://github.com/edouarda/brigand), manages the task
in \f$O(\log(N))\f$ (at least in type instantiation count).

Most of the list operations in SpECTRE cannot take advantage of any particular
ordering or hashing of the list, so must resort to naive list operations --
so, searching a list (`tmpl::list_contains` or `tmpl::index_of`) is
\f$O(N)\f$ cost, `tmpl::remove_duplicates` is \f$O(N^2)\f$, and
`tmpl::list_difference` is similarly \f$O(N^2)\f$.
So, complicated type logic scales pretty badly with long lists, and improvements
can sometimes be made by reducing a list's size or avoiding the more costly list
operations when a list is known to be long.

# Profiling the build

In the current version of SpECTRE, the most expensive builds are the final
translation units associated with the executables (particularly the most
complicated executables, like Generalized Harmonic and Valencia), which should
be unsurprising from the above discussion: it is in these compilation steps that
we instantiate the Parallel components, and in turn, all of the \ref
DataBoxGroup "DataBox" types that will be used during the evolution.

Similarly, a number of tests have now shown that in the current version of
SpECTRE (as of early 2021), by far the most expensive part of the build is
\ref DataBoxGroup "DataBox" operations and instantiations, and the best build
performance gains are available by either reducing the number of
\ref DataBoxGroup "DataBox"es that are instantiated, reducing the number of tags
(particularly compute tags) stored in the \ref DataBoxGroup "DataBox", or
optimizing the implementations of the \ref DataBoxGroup "DataBox" and its
utilities.
So, generally speaking, profiling should start by focusing on the
\ref DataBoxGroup "DataBox", and move to other utilities if it becomes clear
that there are other parts of the code that are contributing significantly to
the compilation time or memory usage.


## Specialized tests and feature exploration

This is simultaneously the most reliable and most labor-intensive strategy for
understanding build costs.
The procedure is to identify a feature you'd like to profile, create a
specialized test for that feature, and put that test in the
`tests/Unit/RunSingleTest/CMakeLists.txt`.
Then, you can easily include or exclude uses of functions or classes that you
want to profile, and compare the relative total cost of building
`RunSingleTest`.

There are a number of tools for profiling the cost of an individual process,
but for compilation, the detailed tools like `perf` or hpctoolkit are unlikely
to give useful information about what parts of our code are slow to build.
Instead, it's best to just carefully measure the full build of the target
in question, and rapidly iterate to include or exclude potentially expensive
parts to understand the build costs.

There are a lot of tools that can give you the global resource usage
information, including the `/proc/$PID/status` file from kernel information,
`top`, or tools from the `sysstat` package.
The `time` utility is particularly user-friendly, available on ubuntu (and
therefore in the SpECTRE build container) via `apt-get install time`.
Then, it can be invoked by `/usr/bin/time -v your_command` (note that simply
`time` will route to a different alias in the default environment on ubuntu in
the container, so the full path `/usr/bin/time` is required).
After completion, it will print a readable report of the time and memory usage.

One important feature to be aware of in profiling the build by this method is
the implicit memoization of many compile-time features.
For instance, if feature `A` and `B` both instantiate a class `C` that is
expensive to build, you'll see a difference in the build cost when _either_ `A`
or `B` are included, but the cost won't be additive - the second feature will
just 'reuse' the instantiation from the first.
To optimize this type of situation, either `C` must be improved to be less
costly to instantiate, or its use must be eliminated from _both_ `A` and `B` --
removing `C` from only one of the classes that use it won't help the build much
at all.

## Templight++

Detailed profiling of a C++ build is a surprisingly hard task, and there are few
useful tools for getting a good idea for what parts of a compilation are
expensive. One tool that can perform some profiling of the build is
[templight++](https://github.com/mikael-s-persson/templight).
_It is important to note that this tool often produces build profiles that are
misleading or incomplete!_ It is included in this guide under the philosophy
that a flawed tool can be better than no tool at all in some circumstances,
but the templight++ profiles should be taken primarily as a loose guide for
features to investigate with more careful follow-up investigations like the
above suggestion of specialized tests and feature exploration.

The `templight++` build and usage instructions work nicely with the current
SpECTRE build system, and the cmake trick suggested by the `templight++`
documentation
```bash
export CC="/path/to/llvm/build/bin/templight -Xtemplight -profiler\
 -Xtemplight -memory"
export CXX="/path/to/llvm/build/bin/templight++ -Xtemplight -profiler\
 -Xtemplight -memory"
```
works well in SpECTRE. Build profiling with `templight++` are incredibly slow,
and seem to produce increasingly misleading data for larger builds, so it is
recommended to avoid using the tool for our most expensive evolution
executables. Experience indicates that you will likely wait for hours and be
disappointed by deeply flawed results.


After building a target, you will find along side each `.o` file in the
`build/src` tree an additional file that ends with `.trace.pbf`.
These are the `templight++` trace files, and (like many performance tool
outputs), require post-processing to recover human-readable data.
The companion package
[templight-tools](https://github.com/mikael-s-persson/templight-tools)
can be built to obtain the `templight-convert` utility that converts
the templight traces to more managable formats.
It is recommended to install and use
[KCacheGrind](https://kcachegrind.github.io/html/Home.html)
(which does, unfortunately require some KDE libraries, but doesn't require you
to use the full KDE window system) to visualize the output -- the larger graphs
produced by templight are inefficient to render in the graphviz format.

## Clang AST syntax generation

There is a nice and poorly documented feature of the `clang++` compiler that it
can produce a rough approximation of the collection of C++ template
instantiations produced by a particular executable.
Adding `-Xclang -ast-print -fsyntax-only` to the `CXX_FLAGS` will cause this
information to be printed to stdout, which should probably be redirected to file
because it will be an enormous output.
Importantly, to the best of our knowledge, this tool has not yet been used to
successfully profile any SpECTRE build, but with sufficient post-processing the
C++-syntax version of the AST might be useful to determine the number and nature
of instantiations produced by a particular piece of code, and might offer some
proxy for build performance.

For instance, if we put the above `Fibonacci` struct in a source file with:
```cpp
int main(int argc, char** argv) {
  std::cout << Fibonacci<6>::value << "\n";
}
```
and we compile it with
```
clang++-10 -Xclang -ast-print -fsyntax-only -o fib ./fib.cpp >> fib_out
```
we obtain in `fib_out` (after thousands of lines of STL-generated code):
```cpp
template <size_t N> struct Fibonacci {
  static constexpr size_t value =
      Fibonacci<N - 1>::value + Fibonacci<N - 2>::value;
};
template <> struct Fibonacci<6> {
  static constexpr size_t value =
      Fibonacci<6UL - 1>::value + Fibonacci<6UL - 2>::value;
};
template <> struct Fibonacci<5> {
  static constexpr size_t value =
      Fibonacci<5UL - 1>::value + Fibonacci<5UL - 2>::value;
};
template <> struct Fibonacci<4> {
  static constexpr size_t value =
      Fibonacci<4UL - 1>::value + Fibonacci<4UL - 2>::value;
};
template <> struct Fibonacci<3> {
  static constexpr size_t value =
      Fibonacci<3UL - 1>::value + Fibonacci<3UL - 2>::value;
};
template <> struct Fibonacci<2> {
  static constexpr size_t value =
      Fibonacci<2UL - 1>::value + Fibonacci<2UL - 2>::value;
};
template <> struct Fibonacci<1> { static constexpr size_t value = 1; };
template <> struct Fibonacci<0> { static constexpr size_t value = 1; };
int main(int argc, char **argv) { std::cout << Fibonacci<6>::value << "\n"; }
```
Which is actually pretty illuminating about what the compiler decided to do in
this simple case.
Unfortunately, the AST produced by `clang++` in more complicated cases produces
extremely large outputs, so realistic cases are likely too large to be usefully
human-readable.
It may be possible, though, to script post-processing tools to sift through the
collections of template instantiations for particular classes to understand
specific cases of template logic.

