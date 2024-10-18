\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Writing tensor equations with TensorExpressions {#writing_tensorexpressions}

\tableofcontents

SpECTRE's `TensorExpression`s interface allows you to write tensor equations in
SpECTRE in C++ with syntax that resembles tensor index notation. To use it,
simply add this include to the top of your file:
```
#include "DataStructures/Tensor/Tensor.hpp"
```
The following guide assumes a basic understanding of the `Tensor` class and
\ref tnsr "tnsr" type aliases.

# Syntax {#te_syntax}
`TensorExpression`s are arithmetic expressions of `Tensor`s that can be
evaluated using `tenex::evaluate`. Terms used in the expression may be `Tensor`s
or numbers (see [supported types](#te_data_type_support)).

As a simple example of how `TensorExpression`s are used, if you would like to
raise the index of some `Tensor` `R` with some spacetime metric `Tensor` `g`,
i.e. \f$R^c{}_b = R_{ab} g^{ac}\f$, you can compute this with
`TensorExpression`s by doing:
```
auto R_up = tenex::evaluate<ti::C, ti::b>(R(ti::a, ti::b) * g(ti::A, ti::C));
```

where `R_up`, `R`, and `g` are rank 2 spacetime `Tensor`s and the `ti::*`
variables are `TensorIndex`s. Here, the argument to
\ref tenex::evaluate "evaluate" is the RHS tensor expression to compute, `R_up`
is the result LHS `Tensor`, and the template arguments to
\ref tenex::evaluate "evaluate" are the LHS `Tensor`'s indices. The LHS symmetry
will be deduced from the RHS tensors' symmetries and order of operations.

Alternatively, if you already have a LHS `Tensor` variable, you can pass it into
the following \ref tenex::evaluate "evaluate" overload, where the LHS `Tensor`
provided will be assigned to the result of the RHS expression:

```
tenex::evaluate<ti::C, ti::b>(
    make_not_null(&R_up), R(ti::a, ti::b) * g(ti::A, ti::C));
```

One advantage of this overload is that it uses the symmetry and index types
(spatial/spacetime) of the provided LHS tensor instead of deducing them from the
RHS expression, so this enables you to specify the LHS index structure in cases
where the previous \ref tenex::evaluate "evaluate" overload does not deduce the
one you want. Note that the LHS tensor does not need to be previously sized
unless the data type is a Blaze vector type (e.g. `DataVector`) *and* the RHS
expression contains no `Tensor` terms (see [example](#te_assigning_to_a_number)
where sizing is necessary). This overload is also used to
[assign subsets of tensor components](#te_assigning_subsets_of_components).

## Tensor indices (TensorIndexs) {#te_tensor_indices}

`TensorIndex`s represent generic tensor indices and are supplied as
comma-separated lists in two places: (1) in parentheses for each tensor in the
RHS expression and (2) in the template parameters of
\ref tenex::evaluate "evaluate" to specify the order of the LHS result tensor's
indices.

Each `TensorIndex` takes the form `ti::*` where `*` is a letter that encodes
index properties:
- Uppercase letters denote upper indices and lowercase letters denote lower
indices
- Letters `A/a - H/h` indicate spacetime indices, `I/i - N/n` indicate spatial
indices, and `T/t` indicates a concrete time index. This is what is currently
defined, but more spatial and spacetime indices (letters) can easily be added if
needed

The properties of each `TensorIndex` and the `Tensor`'s indices (typelist of
\ref SpacetimeIndex "TensorIndexType"s) must be compatible:
- valences must match
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
```
auto L = tenex::evaluate<ti::a, ti::b>(R(ti::a, ti::b) + S(ti::b, ti::a));
```
\f$L = 1 - T\f$
```
auto L = tenex::evaluate(1.0 - T());
```

### Contraction of a single tensor {#te_contraction}
\f$L = U^{a}{}_{a}\f$
```
auto L = tenex::evaluate(U(ti::A, ti::a));
```
\f$L^b = V_{a}{}^{ba}\f$
```
auto L = tenex::evaluate<ti::B>(V(ti::a, ti::B, ti::A));
```

### Inner and outer products {#te_products}
\f$L = G_a H^{a}\f$
```
auto L = tenex::evaluate(G(ti::a) * H(ti::A));
```
\f$L_{cb} = T G_a G_c U^{a}{}_{b}\f$
```
auto L =
    tenex::evaluate<ti::c, ti::b>(T() * G(ti::a) * G(ti::c) * U(ti::A, ti::b));
```

### Division {#te_division}
\f$L_a = \frac{G_a}{2}\f$
```
auto L = tenex::evaluate<ti::a>(G(ti::a) / 2.0);
```
\f$L_{ba} = \frac{R_{ab}}{T}\f$
```
auto L = tenex::evaluate<ti::b, ti::a>(R(ti::a, ti::b) / T());
```
\f$L = \frac{5}{U^{a}{}_{a} + 1}\f$
```
auto L = tenex::evaluate(5.0 / (U(ti::A, ti::a) + 1.0));
```

### Square root {#te_square_root}
\f$L = \sqrt{T}\f$
```
auto L = tenex::evaluate(sqrt(T()));
```
\f$L = \sqrt{G_a H^a}\f$
```
auto L = tenex::evaluate(sqrt(G(ti::a) * H(ti::A)));
```

## More features {#te_more_features}

### Assigning to a number {#te_assigning_to_a_number}
You can assign a number (e.g. `double`) to a `Tensor` of any rank to fill all
components with that value:

\f$L_{ab} = -1\f$
```
tnsr::ab<double, 3> L{};
tenex::evaluate<ti::a, ti::b>(make_not_null(&L), -1.0);
```
If the data type of your LHS `Tensor` is a Blaze vector type
(e.g. `DataVector`), the `Tensor` must be sized before calling
\ref tenex::evaluate "evaluate" because there is no sizing information (from a
`Tensor` component) in the RHS expression:
```
// construct LHS tensor with size 5 DataVector
tnsr::ab<DataVector, 3> L{DataVector(0.0, 5)};
tenex::evaluate<ti::a, ti::b>(make_not_null(&L), -1.0);
```

See [supported number types](#te_data_type_support).

### Using spatial and time indices on spacetime indices {#te_spatial_time_index}
If a `Tensor` has spacetime indices, you can use generic spatial indices and
concrete time indices to refer to a subset of the components, as we see in
literature.

Lapse \f$\alpha\f$ computed from the spacetime metric \f$g_{ab}\f$ and spatial
metric \f$\gamma_{ij}\f$:

\f$\alpha = \sqrt{\gamma^{ij} g_{jt} g_{it} - g_{tt}}\f$
```
// spatial_metric is type tnsr::ii<DataVector, 3> and spacetime_metric is type
// tnsr::aa<DataVector, 3>
auto lapse = tenex::evaluate(
      sqrt(spatial_metric(ti::I, ti::J) * spacetime_metric(ti::j, ti::t) *
               spacetime_metric(ti::i, ti::t) -
           spacetime_metric(ti::t, ti::t)));
```

### Assigning subsets of tensor components {#te_assigning_subsets_of_components}
Related to the previous example, you can also use generic spatial indices and
concrete time indices for the spacetime indices of the LHS `Tensor` to assign
subsets of the LHS `Tensor`'s components.

Spacetime metric \f$g_{ab}\f$ computed from the lapse \f$\alpha\f$, shift
\f$\beta^I\f$, and spatial metric \f$\gamma_{ij}\f$:

\f{align}{
  g_{tt} &= - \alpha^2 + \beta^m \beta^n \gamma_{mn} \\
  g_{ti} &= \gamma_{mi} \beta^m  \\
  g_{ij} &= \gamma_{ij}
\f}
```
// spatial_metric is type tnsr::ii<DataVector, 3>, shift is type
// tnsr::I<DataVector, 3>, and lapse is type Scalar<DataVector>

tnsr::aa<DataVector, 3> spacetime_metric{};
tenex::evaluate<ti::t, ti::t>(
    make_not_null(&spacetime_metric),
    -lapse() * lapse() + shift(ti::M) * shift(ti::N) *
                spatial_metric(ti::m, ti::n));
tenex::evaluate<ti::t, ti::i>(
    make_not_null(&spacetime_metric),
    spatial_metric(ti::m, ti::i) * shift(ti::M));
tenex::evaluate<ti::i, ti::j>(
    make_not_null(&spacetime_metric), spatial_metric(ti::i, ti::j));
```

### Using the LHS Tensor in the RHS expression {#te_using_lhs_tensor_in_rhs}
You can use the LHS `Tensor` in the RHS expression to emulate operations like
`+=`, `*=`, etc. For example, say you would like to emulate the following:
```
// pseudocode
L_ab = R_ab
L_ab += + 2.0 * S_ba
```
You can do the operation in the 2nd line above by calling
\ref tenex::update "update" instead of \ref tenex::evaluate "evaluate":
```
auto L = tenex::evaluate<ti::a, ti::b>(R(ti::a, ti::b));
// use the LHS tensor in the RHS
tenex::update<ti::a, ti::b>(
    make_not_null(&L), L(ti::a, ti::b) + 2.0 * S(ti::b, ti::a));
```
One limitation is that when using the LHS tensor in the RHS expression, the
index order used for the LHS tensor must be the same in the LHS and RHS, e.g.
the following is not allowed and will yield a runtime error:
```
// ERROR: index order for L on LHS and RHS is not the same
tenex::update<ti::a, ti::b>(
    make_not_null(&L), L(ti::b, ti::a) + 2.0 * S(ti::b, ti::a));
```

**Note:** It is not advised to use very large RHS expressions with
\ref tenex::update "update" because runtime performance does not scale well as
the number of operations gets very large. This is because
\ref tenex::evaluate "evaluate" breaks up large expressions into smaller ones,
but \ref tenex::update "update" cannot. One way around this is to break up the
expression and use more than one call to \ref tenex::update "update".

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
auto result1 = tenex::evaluate<ti::a, ti::c>(R(ti::a, ti::b) + S(ti:a::ti::b));
// ERROR: Can't add Tensors with different indices
auto result2 = tenex::evaluate<ti::a, ti::b>(R(ti::a, ti::b) + S(ti:a::ti:c));
// ERROR: Repeated index in the RHS
auto result3 =
    tenex::evaluate<ti::a, ti::b, ti::c>(R(ti::a, ti::b) * S(ti:a::ti::c));
// ERROR: Can't add Tensors with different Frame types
auto result4 = tenex::evaluate<ti::a, ti::b>(R(ti::a, ti::b) + T(ti:a::ti::b));
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
It's possible for terms in the expression to have different data types. The
following table shows the data type that results from performing a binary
operation (`+`, `-`, `*`, `/`) between two terms of given data types:

<table>
  <caption id="multi_row">
      Data types resulting from binary operations between supported
      TensorExpression operand types
  </caption>

  <tr>
    <th></th>
    <th colspan="8">RHS operand type</th>
  </tr>

  <tr>
    <th rowspan="8">LHS operand type</th>
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
    <td><code>std::complex&lt;double&gt;</code></td>
    <td><code>Tensor&lt;double&gt;</code></td>
    <td><code>Tensor&lt;std::complex&lt;double&gt;&gt;</code></td>
    <td><code>Tensor&lt;DataVector&gt;</code></td>
    <td><code>Tensor&lt;ComplexDataVector&gt;</code></td>
  </tr>

  <tr>
    <th><code>std::complex&lt;double&gt;</code></th>
    <td>-</td>
    <td><code>std::complex&lt;double&gt;</code></td>
    <td><code>Tensor&lt;std::complex&lt;double&gt;&gt;</code></td>
    <td><code>Tensor&lt;std::complex&lt;double&gt;&gt;</code></td>
    <td><code>Tensor&lt;ComplexDataVector&gt;</code>*</td>
    <td><code>Tensor&lt;ComplexDataVector&gt;</code></td>
  </tr>

  <tr>
    <th><code>Tensor&lt;double&gt;</code></th>
    <td>-</td>
    <td>-</td>
    <td><code>Tensor&lt;double&gt;</code></td>
    <td><code>Tensor&lt;std::complex&lt;double&gt;&gt;</code></td>
    <td>Not supported</td>
    <td>Not supported</td>
  </tr>

  <tr>
    <th><code>Tensor&lt;std::complex&lt;double&gt;&gt;</code></th>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td><code>Tensor&lt;std::complex&lt;double&gt;&gt;</code></td>
    <td>Not supported</td>
    <td>Not supported</td>
  </tr>

  <tr>
    <th><code>Tensor&lt;DataVector&gt;</code></th>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td><code>Tensor&lt;DataVector&gt;</code></td>
    <td><code>Tensor&lt;ComplexDataVector&gt;</code></td>
  </tr>

  <tr>
    <th><code>Tensor&lt;ComplexDataVector&gt;</code></th>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td><code>Tensor&lt;ComplexDataVector&gt;</code></td>
  </tr>
</table>

For example, if `R` is a `Tensor<DataVector, ...>` and `S` is a
`Tensor<ComplexDataVector, ...>`, `L` will be a
`Tensor<ComplexDataVector, ...>`:
```
auto L = tenex::evaluate<ti::a>(R(ti::a) - S(ti::a));
```

<strong>\* Note:</strong> The only binary operation that is supported between
`std::complex<double>` and `Tensor<DataVector>` is multiplication. This is
because Blaze does not support addition, subtraction, nor division between
`std::complex<double>` and `DataVector`.
