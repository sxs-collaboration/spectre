\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Implementing new Vector types {#implementing_vectors}

\tableofcontents

# Overview of SpECTRE Vectors {#general_structure}

In SpECTRE, sets of contiguous or related data are stored in specializations of
vector data types. The canonical implementation of this is the `DataVector`,
which is used for storage of a contiguous sequence of doubles which support a
wide variety of mathematical operations. However, we support the ability to
easily generate `DataVector`s for alternative types or which support alternative
sets of mathematical operations. All such data types are derived classes from
the class template `VectorImpl`. The remainder of this brief guide gives a
description of the tools for defining additional Vector types.

For reference, all functions described here can also be found in brief in the
Doxygen documentation for VectorImpl.hpp, and a simple reference implementation
can be found in DataVector.hpp and DataVector.cpp.

# The class definition {#class_definition}

SpECTRE vector types inherit from vector types implemented in the
high-performance arithmetic libarary
[Blaze](https://bitbucket.org/blaze-lib/blaze). Using this inheritance
technique, SpECTRE vectors gracefully make use of the math functions defined for
the Blaze types, but can be customized for the specific needs in SpECTRE
computations.

The pair of template parameters for `VectorImpl` are the type of the stored data
(`double` for `DataVector`), and the result type for mathematical operations,
which is necessary to supply at compile time to pass on to the underlying blaze
vector types. In nearly all cases this will be the type that is being declared
itself, so, for instance, `DataVector` is the derived class of
`VectorImpl<double,DataVector>`.

The class template `VectorImpl` defines various constructors, assignment
operators, and iterator generation members. Most of these are inherited from
blaze types, but in addition, the methods `set_data_ref`, and `pup` are defined
for use in SpECTRE. All except for the assignment and constructors will inherit
gracefully to the derived class. The assignment and constructors may be
obtained via

```
using VectorImpl<T,VectorType>::operator=
using VectorImpl<T,VectorType>::VectorImpl
```

Only the mathematical operations supported on the base blaze types are supported
by default. Those operations are determined by the storage type `T` and by the
blaze library. See [blaze-wiki/Vector_Operations]
(https://bitbucket.org/blaze-lib/blaze/wiki/Vector%20Operations).

Other math operations may be defined either in the class definition or
outside. For ease in defining some of these math operations, the
`PointerVector.hpp` defines the macro `MAKE_EXPRESSION_MATH_ASSIGN(OP, TYPE)`,
which can be used to define math assignment operations such as `+=` provided the
underlying operation (e.g. `+`) is defined on the blaze type.

# Blaze operation typing {#blaze_definitions}

In order for the math operations which pass through blaze to return the new
desired vector type, instead of the blaze vector base type, we need to
communicate that return type to the various templated blaze objects which
account for those operations.

To define these, it is necessary to define a template specialization in the
blaze namespace for each of the type combinations for the blaze struct
associated with each operation you'd like to support. For example, to support
addition returning the `DataVector` type when a `DataVector` is added to a
`double`, we would declare:

```
namespace blaze {
template <> //for template specialization
struct AddTrait<DataVector, double> {
    using Type = DataVector;
};
}  // namespace blaze
```
Note that if you want to support adding a `double` to a `DataVector`
(reversing the arguments), you must define a separate blaze struct
specialization. To assist in the many specializations this would require,
two helper macros are defined, both intended to be put in the blaze namespace.

The first is `BLAZE_TRAIT_SPEC_BINTRAIT(VECTORTYPE, BLAZE_MATH_TRAIT)`, which
will define all of the pairwise operations (`BLAZE_MATH_TRAIT`) for the vector
type (`VECTORTYPE`) with itself and for the vector type with its
`value_type`. This reduces the three specializations like the above to a single
line call,

```
namespace blaze {
BLAZE_TRAIT_SPEC_BINTRAIT(DataVector,AddTrait)
}  // namespace blaze
```

Finally, a second macro is defined to easily define all of the arithmetic
operations that will typically be supported for a vector type with its value
type. This macro is defined as `VECTOR_BLAZE_TRAIT_SPEC(VECTORTYPE)`, and
defines all of:
- `IsVector<VECTORTYPE>` to compile-time true
- `TransposeFlag<VECTORTYPE>`
- `AddTrait` for the `VECTORTYPE` and its value type (3 blaze struct
  specializations)
- `SubTrait` for the `VECTORTYPE` and its value type (3 blaze struct
  specializations)
- `MultTrait` for the `VECTORTYPE` and its value type (3 blaze struct
  specializations)
- `DivTrait` for the `VECTORTYPE` and its value type (3 blaze struct
  specializations)
- `UnaryMapTrait` for the `VECTORTYPE` for any unary map
- `BinaryMapTrait` for the `VECTORTYPE` with another `VECTORTYPE`

This macro is similarly intended to be used in the blaze namespace and can
substantially simplify these specializations for new vector types.

```
namespace blaze {
VECTOR_BLAZE_TRAIT_SPEC(DataVector)
}  // namespace blaze
```

# Arrays of vector operations {#array_VECTORTYPE_definitions}

In addition to operations between SpECTRE vectors, it is useful to gracefully
handle operations between arrays of vectors element-wise. There are general
macros defined for handling operations between array specializations:
`DEFINE_ARRAY_BINOP` and `DEFINE_ARRAY_INPLACE_BINOP` from
`Utilities/StdArrayHelpers.hpp`.

In addition, there is a macro for rapidly generating addition and subtraction
between arrays of vectors and arrays of their data types. The macro
`MAKE_ARRAY_VECTOR_BINOPS(VECTORTYPE)` will define the element-wise `+` and `-`
with `VECTORTYPE` and `VECTORTYPE`, as well as either ordering of VECTORTYPE
with `VECTORTYPE::value_type`, and the `+=` and `-=` of `VECTORTYPE` with a
`VECTORTYPE` or with its `VECTORTYPE::value_type`.

# Equivalence operators {#Vector_type_equivalence}

Equivalence operators are one of the few functions which do not automatically
type infer to the base type, and therefore must be separately defined for each
new vector type. As these equivalence operators are typically defined simply as
the element-wise equivalence, this may be implemented easily via the macro
`IMPLEMENT_VECTOR_EQUIV(VECTORTYPE)`. This should ideally be placed in an
accompanying .cpp source file. Additionally, an accompanying
`DECLARE_VECTOR_EQUIV(VECTORTYPE)` macro is provided for putting all of the
appropriate forward-declarations of the equivalence operators in the .hpp.

# MakeWithValueImpl {#Vector_MakeWithValueImpl}

SpECTRE offers the convenience function `MakeWithValueImpl::apply` for various
types. The typical behavior for a SpECTRE vector type is to create a new vector
type of the same type and length initialized with the value provided as the
second argument in all entries. This behavior may be created with simply
`MAKE_VECTOR_MAKEWITHVALUES(VECTORTYPE)`. Any other behavior will need to be
created manually.

# Functionality with other data types {#Vector_tensor_and_variables}

When additional vector types are added, small changes are necessary if they are
to be used as the base container type eiher for `Tensor`s or for `Variables`
(`Variables` contain `Tensor`s, which contain some vector type.

In `Tensor.hpp`, there is a `static_assert` which white-lists the possible types
that can be used as the storage type in a `Tensor`s. Any new vectors must be
added to that white-list if they are to be used within `Tensor`s.

`Variables` templates on the storage type of the stored `Tensor`s. However, any
new data type should be appropriately tested. New vector types should be tested
by invoking new versions of existing testing functions templated on the new
vector type, rather than `DataVector`.

# Writing tests {#Vector_tests}

In addition to the utilities for generating new vector types, there are a number
of convenience functions and utilities for easily generating the tests necessary
to verify that the vectors function appropriately. These utilities are in
`VectorImplTestHelper.hpp`, and documented individually in the
TestingFrameworkGroup. Presented here are the salient details for rapidly
assembling basic tests for vectors.

## check_vectors
A utility function for checking the equality of two vector-like containers or
arrays thereof, which overloads appropriately to primitive types. More exotic
equality tests will likely require additional specialization.

## utility check functions

Each of these functions is intended to encapsulate a single frequently used unit
test and is templated (in order) on the vector type and the value type to be
generated. The default behavior is to uniformly sample values between -100 and
100, but alternative bounds may be passed in via the function arguments.

### `TestVectorImpl::vector_test_construct_and_assign()`
 This function tests a battery of construction and assignment operators for the
 vector type.

### `TestVectorImpl::vector_test_serialize()`
This function tests that vector types can be serialized and deserialized,
retaining their data.

### `TestVectorImpl::vector_test_ref()`
This function tests the `set_data_ref` method of sharing data between vectors,
and that the appropriate owning flags and move operations are handled correctly.

### `TestVectorImpl::vector_test_math_after_move()`
Tests several combinations of math operations and ownership before and after
movement with std::move.

### `TestVectorImpl::vector_ref_test_size_error()`
This function intentionally generates an error when assigning values from one
vector to a differently sized, non-owning vector (made not owning by use of
`set_data_ref`). The assertion test that this is called in should search for the
string "Must copy into same size"

### `TestVectorImpl::vector_ref_test_move_size_error()`
Intentionally generates an error as above, but instead checks moving to the
wrong size. The calling assertion test should also search for the string "Must
copy into same size"

## `TestVectorImpl::VectorTestFunctors`

This struct is a generic template object for testing the mathematical operation
of vector types with other vector types and/or their base types, with or without
various reference wrappers. This may be used to efficiently test the full set of
permitted math operations on a vector. See VectorTestFunctors documentation for
full usage details. An example use of this functionality can be found in `Test_DataVector.cpp`.

# Vector storage nuts and bolts {#Vector_storage}

Internally, all vector classes inherit from the templated `VectorImpl`, which
inherits from the `PointerVector`, which inherits from a
`blaze::DenseVector`. Most of the mathematical operations are supported through
the blaze inheritance, which ensures that the math operations execute the
optimized forms in blaze.

The data itself is either stored in the `std::vector` `owned_data_` private
member variable of the `VectorImpl` if the `VectorImpl` is currently owning its
data, or is stored in otherwise allocated memory. In either case, the access to
the data is set by the raw pointer `v_` in `PointerVector`, accessible via the
`data()` member function. In the case where the `VectorImpl` is owning and its
`owned_data_` `std::vector` is populated, the `v_` in `PointerVector` points
into the underlying array from `data()` member function from the
`std::vector`. Note that the access member functions for the vector types are
not defined in `VectorImpl`, but inherited from `PointerVector`. So, the data
access operations act on the raw pointer, rather than calling any functions on
the std::vector in `VectorImpl`, even when `owning_` is true.

The core blaze operations act on the contents of the vectors through the methods
defined by `PointerVector` to access the data via the member functions of `PointerVector` which act on or return the raw pointer.
