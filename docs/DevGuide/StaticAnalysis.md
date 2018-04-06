\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Static Analysis Tools {#static_analysis_tools}

SpECTRE code is frequently run through the
[Clang Tidy](http://clang.llvm.org/extra/clang-tidy/) static analyzer.
Since analyzing a single source file can take over half a minute it
is generally not advisable to run clang-tidy over the entire code base.
If CMake isn't finding clang-tidy, make sure clang-tidy is installed and that
you have chosen clang as your compiler. To analyze a single source file
run, for example
`make clang-tidy FILE=/path/to/source/src/DataStructures/DataVector.cpp`.
To analyze the entire code base run `make clang-tidy-all`. To analyze all
changed C++ source files in the commits from `FIRST_HASH` to `HEAD`, run `make
clang-tidy-hash HASH=FIRST_HASH`.
