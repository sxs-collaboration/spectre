\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Writing Tensor Equations with TensorExpressions {#writing_tensorexpressions}

\tableofcontents

SpECTRE's `TensorExpression`s interface allows you to write tensor equations in
SpECTRE in C++ with syntax that resembles tensor index notation. To use it,
simply add this include to the top of your file:
```
#include "DataStructures/Tensor/Tensor.hpp"
```
The following guide assumes a basic understanding of the `Tensor` class and
\ref tnsr "tnsr" type aliases. **RHS** refers to the right hand side expression
that we wish to compute and **LHS** refers to the resulting left hand side
tensor that stores the result of computing the RHS expression.

# Syntax {#te_syntax}
`TensorExpression`s are arithmetic expressions of `Tensor`s that can be
evaluated using `tenex::evaluate`. Terms used in the expression may be `Tensor`s
or numbers (see [supported types](#te_data_type_support)).

As a simple example of how `TensorExpression`s are used, if you would like to
raise the index of some `Tensor` `R` with some inverse spacetime metric `Tensor`
`g`, i.e. \f$R^c{}_b = R_{ab} g^{ac}\f$, you can compute this with
`TensorExpression`s by doing:

\snippet Expressions/Test_Examples.cpp te_example_evaluate_lhs_return

where `R_up`, `R`, and `g` are rank 2 spacetime `Tensor`s and the `ti::*`
variables are `TensorIndex`s representing generic tensor indices. Here is a
breakdown of the different parts of this line:
- the RHS expression to compute is the argument to
\ref tenex::evaluate "evaluate": `R(ti::a, ti::b) * g(ti::A, ti::C)`
- the result LHS `Tensor` is `R_up`
- the LHS `Tensor`'s indices are the template arguments to
\ref tenex::evaluate "evaluate": `ti::C, ti::b`

The LHS \ref ::Symmetry "Symmetry" will be deduced from the RHS tensors'
symmetries and order of operations.

Alternatively, if you already have a LHS `Tensor` variable, you can pass it into
the following \ref tenex::evaluate "evaluate" overload, where the LHS `Tensor`
provided will be assigned to the result of the RHS expression:

\snippet Expressions/Test_Examples.cpp te_example_evaluate_lhs_arg

Note that to use this \ref tenex::evaluate "evaluate" overload, the LHS `Tensor`
does not need to be previously sized unless the data type is a Blaze vector type
(e.g. `DataVector`) *and* the RHS expression contains no `Tensor` terms
(see [example](#te_assigning_to_a_number) where sizing is necessary). This
overload is useful in a couple cases:
- [Specifying the LHS symmetry](#te_specify_lhs_symmetry): One advantage of this
overload is that it uses the \ref ::Symmetry "Symmetry" of the provided LHS
tensor instead of deducing it from the RHS expression. This enables you to
specify the LHS symmetry in cases where the previous
\ref tenex::evaluate "evaluate" overload does not deduce the one you want. While
the LHS index order in this example (`ti::C, ti::b`) could theoretically be
deduced from the index types of `R_up`, we still require specifying them because
this isn't the case for all equations and we would like to have a unified
interface. See [this example](#te_specify_lhs_symmetry), which demonstrates a
case where you might want to specify the LHS symmetry and where the LHS index
order would not be deducible.
- **[Using spatial and time indices on LHS spacetime indices](#te_spatial_time_index_lhs)**

## Tensor indices {#te_tensor_indices}

`TensorIndex`s represent generic tensor indices and are supplied as
comma-separated lists in two places:
- in parentheses for each tensor in the RHS expression
- in the template parameters of \ref tenex::evaluate "evaluate" to specify the
order of the LHS result tensor's indices.

Each `TensorIndex` takes the form `ti::*` where `*` is a letter that encodes
index properties:
- Uppercase letters denote upper indices and lowercase letters denote lower
indices
- Letters `A/a - H/h` indicate spacetime indices, `I/i - N/n` indicate spatial
indices, and `T/t` indicates a concrete time index. This is what is currently
defined, but more spatial and spacetime indices (letters) can easily be added if
needed. Note that there is no precedence or difference between the indices of
some type, e.g. `ti::a`, `ti::b`, ... `ti::h` are equivalent

The properties of each `TensorIndex` and the `Tensor`'s indices (typelist of
\ref SpacetimeIndex "TensorIndexType"s) must be compatible:
- valences (being upper or lower indices) must match
- if a `Tensor`'s index is spacetime, you can use a spacetime `TensorIndex`,
spatial `TensorIndex`, or concrete time `TensorIndex`
- if a `Tensor`'s index is spatial, you must use a spatial `TensorIndex`

To demonstrate correct and incorrect usage, let's say we have tensors
\f$R_{ab}\f$ (two spacetime indices, e.g. type \ref tnsr "tnsr::ab") and
\f$S_{ij}\f$ (two spatial indices, e.g. type \ref tnsr "tnsr::ij"):

```
R(ti::c, ti::d) // OK
R(ti::c, ti::k) // OK, can use spatial TensorIndex on spacetime index
R(ti::c, ti::t) // OK, can use time TensorIndex on spacetime index
R(ti::c, ti::D) // ERROR, ti::D is upper but the 2nd index is lower
S(ti::j, ti::k) // OK
S(ti::a, ti::k) // ERROR, can't use spacetime TensorIndex on a spatial index
S(ti::j, ti::t) // ERROR, can't use time TensorIndex on a spatial index
```

# Examples {#te_examples}
## Basic operations {#te_basic_operations}

In the following examples:
- `R` is type \ref tnsr "tnsr::ab<DataVector, 3>"
- `S` is type \ref tnsr "tnsr::ab<DataVector, 3>"
- `T` is type \ref Scalar "Scalar<DataVector>"
- `U` is type \ref tnsr "tnsr::Ab<DataVector, 3>"
- `V` is type \ref tnsr "tnsr::aBC<DataVector, 3>"
- `G` is type \ref tnsr "tnsr::a<DataVector, 3>"
- `H` is type \ref tnsr "tnsr::A<DataVector, 3>"

### Addition and subtraction {#te_addition_and_subtraction}
\f$L_{ab} = R_{ab} + S_{ba}\f$

\snippet Expressions/Test_Examples.cpp te_example_addition

\f$L = 1 - T\f$

\snippet Expressions/Test_Examples.cpp te_example_subtraction

### Contraction of a single tensor {#te_contraction}
\f$L = U^{a}{}_{a}\f$

\snippet Expressions/Test_Examples.cpp te_example_contraction_to_scalar

\f$L^b = V_{a}{}^{ba}\f$

\snippet Expressions/Test_Examples.cpp te_example_contraction_to_tensor

### Inner and outer products {#te_products}
\f$L = G_a H^{a}\f$

\snippet Expressions/Test_Examples.cpp te_example_inner_product

\f$L_{cb} = T G_a G_c U^{a}{}_{b}\f$

\snippet Expressions/Test_Examples.cpp te_example_inner_and_outer_product

### Division {#te_division}
\f$L_a = \frac{G_a}{2}\f$

\snippet Expressions/Test_Examples.cpp te_example_division_by_number

\f$L_{ba} = \frac{R_{ab}}{T}\f$

\snippet Expressions/Test_Examples.cpp te_example_division_by_tensor

\f$L = \frac{5}{U^{a}{}_{a} + 1}\f$

\snippet Expressions/Test_Examples.cpp te_example_division_by_tensor_expression

### Square root {#te_square_root}
\f$L = \sqrt{T}\f$

\snippet Expressions/Test_Examples.cpp te_example_square_root_tensor

\f$L = \sqrt{G_a H^a}\f$

\snippet Expressions/Test_Examples.cpp te_example_square_root_inner_product

## More features {#te_more_features}

### Specifying the LHS symmetry {#te_specify_lhs_symmetry}
When using the \ref tenex::evaluate "evaluate" overload that returns the LHS
`Tensor`, the \ref ::Symmetry "Symmetry" of the LHS `Tensor` will be deduced
from the RHS expression. However, in some cases the deduced LHS symmetry may
not be what you want. To specify it yourself, you can pass your LHS `Tensor`
(that has the desired \ref ::Symmetry "Symmetry") to the
\ref tenex::evaluate "evaluate" overload that takes the LHS `Tensor` as the
first argument.

For example, if we have \f$L_{ab} = R_a R_b\f$, the indices of \f$L\f$ are
symmetric. However, when we do:

\snippet Expressions/Test_Examples.cpp te_example_deduced_lhs_symmetry_fail

the type of `L` will be \ref tnsr "tnsr::ab" because it is not known at
compile time that the two vectors in the product are the same. To override the
deduced symmetry and make it a symmetric result, we can create a
\ref tnsr "tnsr::aa" and pass it into the other overload:

\snippet Expressions/Test_Examples.cpp te_example_deduced_lhs_symmetry_force

### Assigning to a number {#te_assigning_to_a_number}
You can assign a number (e.g. `double`) to a `Tensor` of any rank to fill all
components with that value, e.g. \f$L_{ab} = -1\f$. How you do that is slightly
different depending on the underlying data type of your `Tensor`:

- When your `Tensor`'s data type is a number type (e.g. `double`,
`std::complex<double>`):

\snippet Expressions/Test_Examples.cpp te_example_assign_number_to_tensor_of_numbers

- When your `Tensor`'s data type is a Blaze vector type (e.g. `DataVector`,
`ComplexDataVector`), the `Tensor` must first be sized before calling
\ref tenex::evaluate "evaluate" because there is no sizing information (from a
`Tensor` component) in the RHS expression:

\snippet Expressions/Test_Examples.cpp te_example_assign_number_to_tensor_of_vectors

See [supported number types](#te_data_type_support) for the data types that the
RHS number can be.

### Using spatial and time indices on RHS spacetime indices {#te_spatial_time_index_rhs}
If a `Tensor` has spacetime indices, you can use generic spatial indices and
concrete time indices to refer to a subset of the components, as we see in
literature.

Lapse \f$\alpha\f$ computed from the spacetime metric \f$g_{ab}\f$ and shift
\f$\beta^i\f$:

\f$\alpha = \sqrt{\beta^i g_{it} - g_{tt}}\f$

\snippet Expressions/Test_Examples.cpp te_example_rhs_spatial_and_time_indices

### Using spatial and time indices on LHS spacetime indices {#te_spatial_time_index_lhs}
Related to the previous example, you can also use generic spatial indices and
concrete time indices for the spacetime indices of the LHS `Tensor` to assign
subsets of the LHS `Tensor`'s components.

Spacetime metric \f$g_{ab}\f$ computed from the lapse \f$\alpha\f$, shift
\f$\beta^i\f$, and spatial metric \f$\gamma_{ij}\f$:

\f$g_{tt} = -\alpha^2 + \beta^m \beta^n \gamma_{mn}\f$

\f$g_{ti} = \gamma_{mi} \beta^m\f$

\f$g_{ij} = \gamma_{ij}\f$

\snippet Expressions/Test_Examples.cpp te_example_lhs_spatial_and_time_indices

\parblock
\note The above example is for demonstration purposes only. In practice, you
would want to avoid repeating computing the reused quantity
\f$\beta^i \gamma_{ij}\f$ by e.g. storing the result of the reused quantity in
another variable.
\endparblock

### Using the LHS Tensor in the RHS expression {#te_using_lhs_tensor_in_rhs}
You can use the LHS `Tensor` in the RHS expression to emulate operations like
`+=`, `*=`, etc. For example, say you would like to emulate the following:
```
// pseudocode
L_ab = R_ab
L_ab += 2.0 * S_ba
```
You can emulate the `+=` operation by calling \ref tenex::update "update"
instead of \ref tenex::evaluate "evaluate":
```
auto L = tenex::evaluate<ti::a, ti::b>(R(ti::a, ti::b));
// use the LHS tensor in the RHS
tenex::update<ti::a, ti::b>(
    make_not_null(&L), L(ti::a, ti::b) + 2.0 * S(ti::b, ti::a));
```
One limitation is that when using the LHS tensor in the RHS expression, the
LHS tensor's index order must be the same on the LHS and RHS. This means that
the order of the `TensorIndex` template parameters of
\ref tenex::update "update" must match the order of the `TensorIndex` arguments
in the parentheses that come after the LHS `Tensor` in the RHS expression. For
example, the following is not allowed and will yield a runtime error:
```
// ERROR: index order for L on LHS and RHS is not the same
tenex::update<ti::a, ti::b>(
    make_not_null(&L), L(ti::b, ti::a) + 2.0 * S(ti::b, ti::a));
```

\parblock
\note It is not advised to use very large RHS expressions with
\ref tenex::update "update" because runtime performance does not scale well as
the number of operations gets very large. This is because
\ref tenex::evaluate "evaluate" breaks up large expressions into smaller ones,
but \ref tenex::update "update" cannot. One way around this is to break up the
expression and use more than one call to \ref tenex::update "update".
\endparblock

# Compile time math checks {#te_compile_time_math_checks}
For all operations, mathematical legality is checked at compile time. The
compiler will catch what is not sound to write on paper, which includes things
like no repeated indices, can't divide by a tensor with rank > 0, and that
spatial dimensions, frames, valences, index types (spatial or spacetime), and
ranks of tensors match where they should.

Here are some examples of illegal math that the compiler will catch:

```
tnsr::ab<double, 3, Frame::Inertial> R{};
tnsr::ab<double, 3, Frame::Inertial> S{};
tnsr::ab<double, 3, Frame::Grid> T{};
tnsr::AB<double, 2, Frame::Inertial> G{};
// ERROR: LHS and RHS indices don't match
auto result1 = tenex::evaluate<ti::a, ti::c>(R(ti::a, ti::b) + S(ti::a, ti::b));
// ERROR: Can't add Tensors with different indices
auto result2 = tenex::evaluate<ti::a, ti::b>(R(ti::a, ti::b) + S(ti::a, ti:c));
// ERROR: Repeated index in the RHS
auto result3 =
    tenex::evaluate<ti::a, ti::b, ti::c>(R(ti::a, ti::b) * S(ti::a, ti::c));
// ERROR: Can't add Tensors with different Frame types
auto result4 = tenex::evaluate<ti::a, ti::b>(R(ti::a, ti::b) + T(ti::a, ti::b));
// ERROR: Can't contract indices with different number of spatial dimensions
auto result5 = tenex::evaluate(R(ti::a, ti::b) * G(ti::A, ti::B));
// ERROR: Can't divide by a rank > 0 Tensor
auto result6 = tenex::evaluate<ti::a, ti::b>(R(ti::a, ti::b) / S(ti::a, ti::b));
```

# Support for data types and operations {#te_data_type_and_op_support}

## Data types {#te_data_type_support}
The RHS expression may contain a mixture of number terms and `Tensor` terms,
e.g. `0.5 * T(ti::a)`.

Currently supported data types for number terms:
- `double`
- `std::complex<double>`

Currently supported underlying data types for `Tensor` terms:
- `double`
- `std::complex<double>`
- `DataVector`
- `ComplexDataVector`

Support for more types can be added.

## Operations {#te_operation_support}
It's possible for terms in the RHS expression to have different data types. The
mixture of `double`s and `Tensor<double>`s is shown in the
[subtraction example](#te_addition_and_subtraction) and the
[division examples](#te_division).

It's also possible for terms to be a mixture of both real-valued and
complex-valued numbers or `Tensor`s. For example, we can compute
\f$z^i = x^i + i y*i\f$, where \f$x\f$ and \f$y\f$ are real-valued `Tensor`s,
`i` is a `std::complex<double>`, and \f$z\f$ is a complex-valued `Tensor`.

\snippet Expressions/Test_Examples.cpp te_example_complex_vector

In the above example, a `std::complex<double>` is multiplied by a
`Tensor<DataVector>`, which can be thought to have an "intermediate" type
`Tensor<ComplexDataVector>`, and then a `Tensor<DataVector>` is added to that
intermediate `Tensor<ComplexDataVector>` to yield the resulting type,
`Tensor<ComplexDataVector>`.

The following table shows the data type that results from performing a binary
operation (`+`, `-`, `*`, `/`) between two terms of given data types:

<table>
  <caption id="multi_row">
      Data types resulting from binary operations between supported
      TensorExpression operand types
  </caption>

  <tr>
    <th></th>
    <th><code>double</code></th>
    <th><code>std::complex&lt;double&gt;</code></th>
    <th><code>Tensor&lt;double&gt;</code></th>
    <th><code>Tensor&lt;std::complex&lt;double&gt;&gt;</code></th>
    <th><code>Tensor&lt;DataVector&gt;</code></th>
    <th><code>Tensor&lt;ComplexDataVector&gt;</code></th>
  </tr>

  <tr>
    <th><code>double</code></th>
    <td><code>double</code></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>

  <tr>
    <th><code>std::complex&lt;double&gt;</code></th>
    <td><code>std::complex&lt;double&gt;</code></td>
    <td><code>std::complex&lt;double&gt;</code></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>

  <tr>
    <th><code>Tensor&lt;double&gt;</code></th>
    <td><code>Tensor&lt;double&gt;</code></td>
    <td><code>Tensor&lt;std::complex&lt;double&gt;&gt;</code></td>
    <td><code>Tensor&lt;double&gt;</code></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>

  <tr>
    <th><code>Tensor&lt;std::complex&lt;double&gt;&gt;</code></th>
    <td><code>Tensor&lt;std::complex&lt;double&gt;&gt;</code></td>
    <td><code>Tensor&lt;std::complex&lt;double&gt;&gt;</code></td>
    <td><code>Tensor&lt;std::complex&lt;double&gt;&gt;</code></td>
    <td><code>Tensor&lt;std::complex&lt;double&gt;&gt;</code></td>
    <td>-</td>
    <td>-</td>
  </tr>

  <tr>
    <th><code>Tensor&lt;DataVector&gt;</code></th>
    <td><code>Tensor&lt;DataVector&gt;</code></td>
    <td><code>Tensor&lt;ComplexDataVector&gt;</code><strong>*</strong></td>
    <td>Not supported</td>
    <td>Not supported</td>
    <td><code>Tensor&lt;DataVector&gt;</code></td>
    <td>-</td>
  </tr>

  <tr>
    <th><code>Tensor&lt;ComplexDataVector&gt;</code></th>
    <td><code>Tensor&lt;ComplexDataVector&gt;</code></td>
    <td><code>Tensor&lt;ComplexDataVector&gt;</code></td>
    <td>Not supported</td>
    <td>Not supported</td>
    <td><code>Tensor&lt;ComplexDataVector&gt;</code></td>
    <td><code>Tensor&lt;ComplexDataVector&gt;</code></td>
  </tr>
</table>

\parblock
\note The only binary operation that is supported between `std::complex<double>`
and `Tensor<DataVector>` is multiplication. This is because Blaze does not
support addition, subtraction, nor division between `std::complex<double>` and
`DataVector`.
\endparblock