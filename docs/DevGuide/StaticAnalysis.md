\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Static Analysis Tools {#static_analysis_tools}

SpECTRE code is frequently run through the static analyzers
[Cppcheck](http://cppcheck.sourceforge.net/) and
[clang-tidy](http://clang.llvm.org/extra/clang-tidy/).
All Travis builds run these over the pull
requests to catch as many errors as possible before they enter
main code. To run Cppcheck locally run `make cppcheck`. Be patient
as this can take several minutes.

[Clang Tidy](http://clang.llvm.org/extra/clang-tidy/)
takes significantly longer to run than Cppcheck and so
can be either run on a single source file or on the entire code base.
Since analyzing a single source file can take over half a minute it
is generally not advisable to run clang-tidy over the entire code base.
If CMake isn't finding clang-tidy, make sure clang-tidy is installed and that
you have chosen clang as your compiler. To analyze a single source file
run, for example
`make clang-tidy FILE=/path/to/source/src/DataStructures/DataVector.cpp`.
To analyze the entire code base run `make clang-tidy-all`.
