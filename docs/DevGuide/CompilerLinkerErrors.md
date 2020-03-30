\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond

# Understanding Compiler and Linker Errors {#compiler_and_linker_errors}

# Linker Errors {#understanding_linker_errors}

There are a few common mistakes that can lead to linker problems, specifically
problems where there is an `undefined reference`. These include:
- forgetting to add a `.cpp` file to the list of sources of a library in a
  `CMakeLists.txt` file
- missing an explicit instantiation of a function or class template in a `cpp`
  file
- not including a `tpp` file inside a `cpp` file
- the template specialization or function overload has been explicitly disabled
  via SFINAE (usually through the use of a `Requires`)
- not linking a library (explained below)

Possibly the most difficult part of fixing linking errors is understanding what
they are trying to tell you. Let's take the following example
```
error: undefined reference to 'Tensor<DataVector,
brigand::list<brigand::integral_constant<int, 1> >,
brigand::list<Tensor_detail::TensorIndexType<3ul, (UpLo)0, Frame::Inertial,
(IndexType)0> > >
random_unit_normal<DataVector>(gsl::not_null<std::mersenne_twister_engine<
unsigned long, 32ul, 624ul, 397ul, 31ul, 2567483615ul, 11ul, 4294967295ul, 7ul,
2636928640ul, 15ul, 4022730752ul, 18ul, 1812433253ul>*>, Tensor<DataVector,
brigand::list<brigand::integral_constant<int, 1>,
brigand::integral_constant<int, 1> >,
brigand::list<Tensor_detail::TensorIndexType<3ul, (UpLo)1, Frame::Inertial,
(IndexType)0>, Tensor_detail::TensorIndexType<3ul, (UpLo)1, Frame::Inertial,
(IndexType)0> > > const&)' in
lib/libTest_GeneralizedHarmonic.a(Test_UpwindFlux.cpp.o):
Test_UpwindFlux.cpp:function (anonymous namespace)::test_upwind_flux_random()
```
We can start by splitting out information about different parts of the
error. First,
```
error: undefined reference to
```
tells us that we forgot to add a link dependency for a library or
executable. The next relevant part of information is which library (executable)
and file the missing function/class was in. This is at the end of the error
message (unfortunately, where in the error message can depend on your linker,
these examples used `ld.lld` v9):
```
lib/libTest_GeneralizedHarmonic.a(Test_UpwindFlux.cpp.o):
Test_UpwindFlux.cpp:function (anonymous namespace)::test_upwind_flux_random()
```
What this means is that the missing link dependency is used in the library
`Test_GeneralizedHarmonic`, the file `Test_UpwindFlux.cpp`, and the function
`test_upwind_flux_random()`.

We have now determined what the linker error is (a missing link dependency), and
in which library, file, and function the missing link dependency is used. We now
need to understand what the missing link dependency is. Since SpECTRE uses a lot
of templates, the missing reference (link dependency) can be quite
long. In this case it is:
```
Tensor<DataVector,
brigand::list<brigand::integral_constant<int, 1> >,
brigand::list<Tensor_detail::TensorIndexType<3ul, (UpLo)0, Frame::Inertial,
(IndexType)0> > >
random_unit_normal<DataVector>(gsl::not_null<std::mersenne_twister_engine<
unsigned long, 32ul, 624ul, 397ul, 31ul, 2567483615ul, 11ul, 4294967295ul, 7ul,
2636928640ul, 15ul, 4022730752ul, 18ul, 1812433253ul>*>, Tensor<DataVector,
brigand::list<brigand::integral_constant<int, 1>,
brigand::integral_constant<int, 1> >,
brigand::list<Tensor_detail::TensorIndexType<3ul, (UpLo)1, Frame::Inertial,
(IndexType)0>, Tensor_detail::TensorIndexType<3ul, (UpLo)1, Frame::Inertial,
(IndexType)0> > > const&)
```
In order to make this easier to read, it is recommended to run the code through
ClangFormat. This can be done by copying the linker output into an empty `cpp`
file and running `clang-format -i EMPTY_FILE_WITH_LINKER_OUTPUT`. Keep in mind
that ClangFormat will only work if you have only copied the part of the linker
output that resembles valid C++. Doing so in this case gives:
\code{.cpp}
Tensor<DataVector, brigand::list<brigand::integral_constant<int, 1> >,
       brigand::list<Tensor_detail::TensorIndexType<
           3ul, (UpLo)0, Frame::Inertial, (IndexType)0> > >
random_unit_normal<DataVector>(
    gsl::not_null<std::mersenne_twister_engine<
        unsigned long, 32ul, 624ul, 397ul, 31ul, 2567483615ul, 11ul,
        4294967295ul, 7ul, 2636928640ul, 15ul, 4022730752ul, 18ul,
        1812433253ul>*>,
    Tensor<DataVector,
           brigand::list<brigand::integral_constant<int, 1>,
                         brigand::integral_constant<int, 1> >,
           brigand::list<
               Tensor_detail::TensorIndexType<3ul, (UpLo)1, Frame::Inertial,
                                              (IndexType)0>,
               Tensor_detail::TensorIndexType<3ul, (UpLo)1, Frame::Inertial,
                                              (IndexType)0> > > const&);
\endcode
We see that it is the function `random_unit_normal` that isn't found. Now we can
search in the code base where than function is defined. Doing a
`git grep "random_unit_normal"` points to
`tests/Unit/Helpers/DataStructures/RandomUnitNormal.?pp`. Looking
at `tests/Unit/Helpers/DataStructures/RandomUnitNormal.hpp` we see that the
function `random_unit_normal` is declared there, and looking in the
corresponding `cpp` we see `random_unit_normal` is instantiated in the source
file. Opening up `tests/Unit/Helpers/DataStructures/CMakeLists.txt` we see that
the library name is `DataStructuresHelpers`, and `RandomUnitNormal.cpp` is in
the list of sources for the library. Thus, linking
`Test_GeneralizedHarmonic` against `DataStructuresHelpers` will resolve our
error. To link against a library, you must add it to the
`target_link_libraries`, or the last argument passed to `add_test_library`.

If `random_unit_normal` had been defined in the header file, then the error
would've indicated that we did not include the header (or `tpp`) file into
`Test_UpwindFlux.cpp`, and so the compiler could not generate an
instantiation.

In summary:
- Identify target with undefined reference, i.e. the file included in a library
  or executable.
- Identify missing source definition (usually a function or static variable).
- Find source declaration and definition in repository.
- If source definition is in a `cpp` file, make sure it is in the list of
  sources in the `CMakeLists.txt` in the same directory and that the
  corresponding library is linked against by the target.

  If the undefined reference is a template, make sure the required instantiation
  exists.
- If the undefined reference's definition is in a `tpp` file, make sure the
  `tpp` file is included in the target file.
- If the undefined reference's source definition is in an `hpp` file, make sure
  the specific instantiation is possible (e.g. not forbidden by a `Requires`)
