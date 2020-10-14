\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond

# Implementing SpECTRE vectors {#implementing_vectors}

\tableofcontents

# Overview of SpECTRE Vectors {#general_structure}

In SpECTRE, sets of contiguous or related data are stored in specializations of
vector data types. The canonical implementation of this is the `DataVector`,
which is used for storage of a contiguous sequence of doubles which support a
wide variety of mathematical operations and represent data on a grid used during
an evolution or elliptic solve. However, we support the ability to easily
generate similar vector types which can hold data of a different type
(e.g. `std::complex<double>`), or support a different set of mathematical
operations. SpECTRE vector classes are derived from the class template
`VectorImpl`. The remainder of this brief guide gives a description of the tools
for defining additional vector types.

For reference, all functions described here can also be found in brief in the
Doxygen documentation for VectorImpl.hpp, and a simple reference implementation
can be found in DataVector.hpp and DataVector.cpp.

# The class definition {#class_definition}

SpECTRE vector types inherit from vector types implemented in the
high-performance arithmetic library
[Blaze](https://bitbucket.org/blaze-lib/blaze). Using inheritance, SpECTRE
vectors gracefully make use of the math functions defined for the Blaze types,
but can be customized for the specific needs in SpECTRE computations.

The pair of template parameters for `VectorImpl` are the type of the stored data
(e.g. `double` for `DataVector`), and the result type for mathematical
operations. The result type is used by Blaze to ensure that only compatible
vector types are used together in mathematical expressions. For example, a
vector representing `double` data on a grid (`DataVector`) cannot be added to a
vector representing spectral coefficients (`ModalVector`). This avoids subtle
bugs that arise when vector types are unintentionally mixed.  In nearly all
cases the result type will be the vector type that is being defined, so, for
instance, `DataVector` is a derived class of `VectorImpl<double,
DataVector>`. This template pattern is known as the
["Curiously Recurring Template Pattern"]
(https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern) (CRTP).

The class template `VectorImpl` defines various constructors, assignment
operators, and iterator generation members. Most of these are inherited from
Blaze types, but in addition, the methods `set_data_ref`, and `pup` are defined
for use in SpECTRE. All except for the assignment operators and constructors
will be implicitly inherited from `VectorImpl`. The assignment and constructors
may be inherited calling the following alias code in the vector class
definition:

```
using VectorImpl<T,VectorType>::operator=;
using VectorImpl<T,VectorType>::VectorImpl;
```

Only the mathematical operations supported on the base Blaze types are supported
by default. Those operations are determined by the storage type `T` and by the
Blaze library. See [blaze-wiki/Vector_Operations]
(https://bitbucket.org/blaze-lib/blaze/wiki/Vector%20Operations).

Other math operations may be defined either in the class definition or
outside. For ease in defining some of these math operations, `PointerVector.hpp`
defines the macro `MAKE_EXPRESSION_MATH_ASSIGN(OP, TYPE)`, which can be used to
define math assignment operations such as `+=` provided the underlying operation
(e.g. `+`) is defined on the Blaze type.

# Allowed operator specification {#blaze_definitions}

Blaze keeps track of the return type of unary and binary operations using "type
trait" structs. These specializations for vector types should be placed in the
header file associated with the `VectorImpl` specialization. For `DataVector`,
the specializations are defined in `DataStructures/DataVector.hpp`. The presence
or absence of template specializations of these structs also determines the set
of allowed operations between the vector type and other types. For example, if
adding a `double` to a `DataVector` should be allowed and the result should be
treated as a `DataVector` for subsequent operations, then the struct
`blaze::AddTrait<DataVector, double>` needs to be defined as follows:

```
namespace blaze {
// the `template <>` head tells the compiler that
// `AddTrait<DataVector, double>` is a class template specialization
template <>
struct AddTrait<DataVector, double> {
    // the `Type` alias tells blaze that the result should be treated like a
    // `DataVector` for any further operations
    using Type = DataVector;
};
}  // namespace blaze
```

Note that this only adds support for `DataVector + double`, not `double +
DataVector`. To get the latter the following AddTrait specialization must be
defined

```
namespace blaze {
// the `template <>` head tells the compiler that
// `AddTrait<double, DataVector>` is a class template specialization
template <>
struct AddTrait<double, DataVector> {
    // the `Type` alias tells blaze that the result should be treated like a
    // `DataVector` for any further operations
    using Type = DataVector;
};
}  // namespace blaze
```

Four helper macros are defined to assist with generating the many
specializations that binary operations may require. Both of these macros must be
put inside the blaze namespace for them to work correctly.

The first of these helper macros is
`BLAZE_TRAIT_SPECIALIZE_BINARY_TRAIT(VECTOR_TYPE, BLAZE_MATH_TRAIT)`, which will
define all of the pairwise operations (`BLAZE_MATH_TRAIT`) for the vector type
(`VECTOR_TYPE`) with itself and for the vector type with its `value_type`. This
reduces the three specializations similar to the above code blocks to a single
line call,

```
namespace blaze {
BLAZE_TRAIT_SPECIALIZE_BINARY_TRAIT(DataVector, AddTrait)
}  // namespace blaze
```

The second helper macro is provided to easily define all of the arithmetic
operations that will typically be supported for a vector type with its value
type. The macro is
`VECTOR_BLAZE_TRAIT_SPECIALIZE_ARITHMETIC_TRAITS(VECTOR_TYPE)`, and defines all
of:
- `IsVector<VECTOR_TYPE>` to `std::true_type`
- `TransposeFlag<VECTOR_TYPE>`, which informs Blaze of the interpretation of the
  data as a "column" or "row" vector
- `AddTrait` for the `VECTOR_TYPE` and its value type (3 Blaze struct
  specializations)
- `SubTrait` for the `VECTOR_TYPE` and its value type (3 Blaze struct
  specializations)
- `MultTrait` for the `VECTOR_TYPE` and its value type (3 Blaze struct
  specializations)
- `DivTrait` for the `VECTOR_TYPE` and its value type (3 Blaze struct
  specializations)

This macro is similarly intended to be used in the `blaze` namespace and can
substantially simplify these specializations for new vector types. For instance,
the call for `DataVector` is:

```
namespace blaze {
VECTOR_BLAZE_TRAIT_SPECIALIZE_ARITHMETIC_TRAITS(DataVector)
}  // namespace blaze
```

The third helper macro is provided to define a combination of Blaze traits for
symmetric operations of a vector type with a second type (which may or may not
be a vector type). The macro is
`BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(VECTOR, COMPATIBLE, TRAIT)`, and
defines the appropriate trait for the two combinations `<VECTOR, COMPATIBLE>`
and `<COMPATIBLE, VECTOR>`, and defines the result type to be `VECTOR`. For
instance, to support the multiplication of a `ComplexDataVector` with a
`DataVector` and have the result be a `ComplexDataVector`, the following macro
call should be included in the `blaze` namespace:

```
namespace blaze {
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexDataVector, DataVector,
                                               MultTrait);
}  // namespace blaze
```

Finally, the fourth helper macro is provided to define all of the blaze traits
which are considered either unary or binary maps. This comprises most named
unary functions (like `sin()` or `sqrt()`) and named binary functions (like
`hypot()` and `atan2()`). The macro
`VECTOR_BLAZE_TRAIT_SPECIALIZE_ALL_MAP_TRAITS(VECTOR_TYPE)` broadly specializes
all blaze-defined maps in which the given `VECTOR_TYPE` as the sole argument
(for unary maps) or both arguments (for binary maps). This macro is also
intended to be used in the blaze namespace. The call for `DataVector` is:

```
namespace blaze {
VECTOR_BLAZE_TRAIT_SPECIALIZE_ALL_MAP_TRAITS(DataVector)
}  // namespace blaze
```

# Supporting operations for `std::array`s of vectors {#array_vector_definitions}

In addition to operations between SpECTRE vectors, it is useful to gracefully
handle operations between `std::arrays` of vectors element-wise. There are
general macros defined for handling operations between array specializations:
`DEFINE_STD_ARRAY_BINOP` and `DEFINE_STD_ARRAY_INPLACE_BINOP` from
`Utilities/StdArrayHelpers.hpp`.

In addition, there is a macro for rapidly generating addition and subtraction
between arrays of vectors and arrays of their data types. The macro
`MAKE_STD_ARRAY_VECTOR_BINOPS(VECTOR_TYPE)` will define:
- the element-wise `+` and `-` with `std::array<VECTOR_TYPE, N>` and
  `std::array<VECTOR_TYPE, N>`
- the element-wise `+` and `-` of either ordering of
  `std::array<VECTOR_TYPE, N>` with `std::array<VECTOR_TYPE::value_type, N>`
- the `+=` and `-=` of `std::array<VECTOR_TYPE, N>` with a
  `std::array<VECTOR_TYPE, N>`
- the `+=` and `-=` of `std::array<VECTOR_TYPE, N>` with a
  `std::array<VECTOR_TYPE::value_type, N>`.

# Equivalence operators {#Vector_type_equivalence}

Equivalence operators are supported by the Blaze type inheritance. The
equivalence operator `==` evaluates to true on a pair of vectors if they are the
same size and contain the same values, regardless of ownership.

# MakeWithValueImpl {#Vector_MakeWithValueImpl}

SpECTRE offers the convenience function `make_with_value` for various types. The
typical behavior for a SpECTRE vector type is to create a new vector type of the
same type and length initialized with the value provided as the second argument
in all entries. This behavior may be created by placing the macro
`MAKE_WITH_VALUE_IMPL_DEFINITION_FOR(VECTOR_TYPE)` in the .hpp file. Any other
specializations of `MakeWithValueImpl` will need to be written manually.

# Interoperability with other data types {#Vector_tensor_and_variables}

When additional vector types are added, small changes are necessary if they are
to be used as the base container type either for `Tensor`s or for `Variables` (a
`Variables` contains `Tensor`s), which contain some vector type.

In `Tensor.hpp`, there is a `static_assert` which white-lists the possible types
that can be used as the storage type in `Tensor`s. Any new vectors must be added
to that white-list if they are to be used within `Tensor`s.

`Variables` is templated on the storage type of the stored `Tensor`s. However,
any new data type should be appropriately tested. New vector types should be
tested by invoking new versions of existing testing functions templated on the
new vector type, rather than `DataVector`.

# Writing tests {#Vector_tests}

In addition to the utilities for generating new vector types, there are a number
of convenience functions and utilities for easily generating the tests necessary
to verify that the vectors function appropriately. These utilities are in
`VectorImplTestHelper.hpp`, and documented individually in the
TestingFrameworkGroup. Presented here are the salient details for rapidly
assembling basic tests for vectors.

## Utility check functions
Each of these functions is intended to encapsulate a single frequently used unit
test and is templated (in order) on the vector type and the value type to be
generated. The default behavior is to uniformly sample values between -100 and
100, but alternative bounds may be passed in via the function arguments.

### `TestHelpers::VectorImpl::vector_test_construct_and_assign()`
 This function tests a battery of construction and assignment operators for the
 vector type.

### `TestHelpers::VectorImpl::vector_test_serialize()`
This function tests that vector types can be serialized and deserialized,
retaining their data.

### `TestHelpers::VectorImpl::vector_test_ref()`
This function tests the `set_data_ref` method of sharing data between vectors,
and that the appropriate owning flags and move operations are handled correctly.

### `TestHelpers::VectorImpl::vector_test_math_after_move()`
Tests several combinations of math operations and ownership before and after use
of `std::move`.

### `TestHelpers::VectorImpl::vector_ref_test_size_error()`
This function intentionally generates an error when assigning values from one
vector to a differently sized, non-owning vector (made non-owning by use of
`set_data_ref`). The assertion test which calls this function should search for
the string "Must copy into same size". Three forms of the test are provided,
which are switched between using a value from the enum `RefSizeErrorTestKind` in
the first function argument:
- `RefSizeErrorTestKind::Copy`: tests that the size error is appropriately
  generated when copying to a non-owning vector of the wrong size.
- `RefSizeErrorTestKind::ExpressionAssign`: tests that the size error is
  appropriately generated when assigning the result of a mathematical expression
  to a non-owning vector of the wrong size.
- `RefSizeErrorTestKind::Move`: tests that the size error is appropriately
  generated when a vector is `std::move`d into a non-owning vector of the wrong
  size

## `TestHelpers::VectorImpl::test_functions_with_vector_arguments()`

This is a general function for testing the mathematical operation of vector
types with other vector types and/or their base types, with or without various
reference wrappers. This may be used to efficiently test the full set of
permitted math operations on a vector. See the documentation of
`test_functions_with_vector_arguments()` for full usage details.

An example simple use case for the math test utility:
\snippet Test_DataVector.cpp test_functions_with_vector_arguments_example

More use cases of this functionality can be found in `Test_DataVector.cpp`.

# Vector storage nuts and bolts {#Vector_storage}

Internally, all vector classes inherit from the templated `VectorImpl`, which
inherits from `PointerVector`, which inherits from a `blaze::DenseVector`. Most
of the mathematical operations are supported through the Blaze inheritance,
which ensures that the math operations execute the optimized forms in Blaze.

SpECTRE vectors can be either "owning" or "non-owning". If a vector is owning,
it allocates and controls the data it has access to, and is responsible for
eventually freeing that data when the vector goes out of scope. If the vector is
non-owning, it acts as a (possibly complete) "view" of otherwise allocated
memory. Non-owning vectors do not manage memory, nor can they change size. The
two cases of data ownership cause the underlying data to be handled fairly
differently, so we will discuss each in turn.

In both cases of ownership, the base `PointerVector` contains a pointer to the
allocated contiguous block of memory via a raw pointer `v_` (accessible via
`PointerVector.data()` and the extent of that memory `size_` (accessible via
`PointerVector.size()`). The raw pointer in `PointerVector` then gives access to
the memory block to the base Blaze types, which perform the work of the actual
math operations. `PointerVector` is closely patterned off of Blaze internal
functionality (`CustomVector`), and we stress that direct alteration of
`PointerVector` should be avoided unless completely necessary. The discussion in
this section is intended to provide a better understanding of how the interface
is used for the SpECTRE vector types so that future customization of derived
classes of `VectorImpl` is as easy as possible.

When a SpECTRE vector is constructed as owning, or becomes owning, it allocates
its own block of memory of appropriate size, and stores a pointer to that memory
in a `std::unique_ptr` named `owned_data_`. The `std::unique_ptr` ensures that
the SpECTRE vector needs to perform no further direct memory management, and
that the memory will be appropriately managed whenever the `std::unique_ptr
owned_data_` member is deleted or moved. The base `PointerVector` must also be
told about the pointer, which is always accomplished by calling the protected
function `VectorImpl.reset_pointer_vector(const size_t set_size)`, which sets
the `PointerVector` internal pointer to the pointer obtained by
`std::unique_pointer.get()`.

When a SpECTRE vector is constructed as non-owning by the `VectorImpl(ValueType*
start, size_t set_size)` constructor, or becomes non-owning by the
`set_data_ref` function, the internal `std::unique_ptr` named `owned_data_` no
longer points to the data represented by the vector and can be thought of as
"inactive" for the purposes of computation and memory management. This behavior
is desirable, because otherwise the `std::unique_ptr` would attempt to free
memory that is presumed to be also used elsewhere, causing difficult to diagnose
memory errors. The non-owning SpECTRE vector updates the base `PointerVector`
pointer directly by calling `PointerVector.reset` from the derived class (on
itself).
