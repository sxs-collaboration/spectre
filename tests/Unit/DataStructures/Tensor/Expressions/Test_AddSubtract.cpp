// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <climits>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <typename... Ts>
void assign_unique_values_to_tensor(
    const gsl::not_null<Tensor<double, Ts...>*> tensor) noexcept {
  std::iota(tensor->begin(), tensor->end(), 0.0);
}

template <typename... Ts>
void assign_unique_values_to_tensor(
    const gsl::not_null<Tensor<DataVector, Ts...>*> tensor) noexcept {
  double value = 0.0;
  for (auto index_it = tensor->begin(); index_it != tensor->end(); index_it++) {
    for (auto vector_it = index_it->begin(); vector_it != index_it->end();
         vector_it++) {
      *vector_it = value;
      value += 1.0;
    }
  }
}

// \brief Test the sum and difference of a `double` and tensor expression is
// correctly evaluated
//
// \details
// The cases tested are:
// - \f$L = R + S\f$
// - \f$L = R - S\f$
// - \f$L = G^{i}{}_{i} + R\f$
// - \f$L = G^{i}{}_{i} - R\f$
// - \f$L = R + S + T\f$
// - \f$L = R - G^{i}{}_{i} + T\f$
//
// where \f$R\f$ and \f$T\f$ are `double`s and \f$S\f$, \f$G\f$, and \f$L\f$
// are Tensors with data type `double` or DataVector.
//
// \tparam DataType the type of data being stored in the tensor expression
// operand of the sums and differences
template <typename DataType>
void test_addsub_double(const DataType& used_for_size) noexcept {
  Tensor<DataType> S{{{used_for_size}}};
  if (std::is_same_v<DataType, double>) {
    // Replace tensor's value from `used_for_size` to a proper test value
    S.get() = 2.4;
  } else {
    assign_unique_values_to_tensor(make_not_null(&S));
  }

  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      G(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&G));

  DataType G_trace = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t i = 0; i < 3; i++) {
    G_trace += G.get(i, i);
  }

  // \f$L = R + S\f$
  const Tensor<DataType> R_plus_S = TensorExpressions::evaluate(5.6 + S());
  // \f$L = R - S\f$
  const Tensor<DataType> R_minus_S = TensorExpressions::evaluate(1.1 - S());
  // \f$L = G^{i}{}_{i} + R\f$
  const Tensor<DataType> G_plus_R =
      TensorExpressions::evaluate(G(ti_I, ti_i) + 8.2);
  // \f$L = G^{i}{}_{i} - R\f$
  const Tensor<DataType> G_minus_R =
      TensorExpressions::evaluate(G(ti_I, ti_i) - 3.5);
  // \f$L = R + S + T\f$
  const Tensor<DataType> R_plus_S_plus_T =
      TensorExpressions::evaluate(0.7 + S() + 9.8);
  // \f$L = R - G^{i}{}_{i} + T\f$
  const Tensor<DataType> R_minus_G_plus_T =
      TensorExpressions::evaluate(5.9 - G(ti_I, ti_i) + 4.7);

  CHECK(R_plus_S.get() == 5.6 + S.get());
  CHECK(R_minus_S.get() == 1.1 - S.get());
  CHECK(G_plus_R.get() == G_trace + 8.2);
  CHECK(G_minus_R.get() == G_trace - 3.5);
  CHECK(R_plus_S_plus_T.get() == 0.7 + S.get() + 9.8);
  CHECK(R_minus_G_plus_T.get() == 5.9 - G_trace + 4.7);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.AddSubtract",
                  "[DataStructures][Unit]") {
  test_addsub_double(std::numeric_limits<double>::signaling_NaN());
  test_addsub_double(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()));

  // Test adding scalars
  const Tensor<double> scalar_1{{{2.1}}};
  const Tensor<double> scalar_2{{{-0.8}}};
  Tensor<double> lhs_scalar =
      TensorExpressions::evaluate(scalar_1() + scalar_2());
  CHECK(lhs_scalar.get() == 1.3);

  Tensor<double, Symmetry<1, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      All{};
  std::iota(All.begin(), All.end(), 0.0);
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Hll{};
  std::iota(Hll.begin(), Hll.end(), 0.0);
  // [use_tensor_index]
  const Tensor<double, Symmetry<2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Gll = TensorExpressions::evaluate<ti_a, ti_b>(All(ti_a, ti_b) +
                                                    Hll(ti_a, ti_b));
  const Tensor<double, Symmetry<2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Gll2 = TensorExpressions::evaluate<ti_a, ti_b>(All(ti_a, ti_b) +
                                                     Hll(ti_b, ti_a));
  const Tensor<double, Symmetry<2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Gll3 = TensorExpressions::evaluate<ti_a, ti_b>(
          All(ti_a, ti_b) + Hll(ti_b, ti_a) + All(ti_b, ti_a) -
          Hll(ti_b, ti_a));
  // [use_tensor_index]
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      CHECK(Gll.get(i, j) == All.get(i, j) + Hll.get(i, j));
      CHECK(Gll2.get(i, j) == All.get(i, j) + Hll.get(j, i));
      CHECK(Gll3.get(i, j) == 2.0 * All.get(i, j));
    }
  }
  // Test 3 indices add subtract
  Tensor<double, Symmetry<1, 1, 2>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Alll{};
  std::iota(Alll.begin(), Alll.end(), 0.0);
  Tensor<double, Symmetry<1, 2, 3>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Hlll{};
  std::iota(Hlll.begin(), Hlll.end(), 0.0);
  Tensor<double, Symmetry<2, 1, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Rlll{};
  std::iota(Rlll.begin(), Rlll.end(), 0.0);
  Tensor<double, Symmetry<1, 2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Slll{};
  std::iota(Slll.begin(), Slll.end(), 0.0);

  const Tensor<double, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Glll = TensorExpressions::evaluate<ti_a, ti_b, ti_c>(
          Alll(ti_a, ti_b, ti_c) + Hlll(ti_a, ti_b, ti_c));
  const Tensor<double, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Glll2 = TensorExpressions::evaluate<ti_a, ti_b, ti_c>(
          Alll(ti_a, ti_b, ti_c) + Hlll(ti_b, ti_a, ti_c));
  const Tensor<double, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Glll3 = TensorExpressions::evaluate<ti_a, ti_b, ti_c>(
          Alll(ti_a, ti_b, ti_c) + Hlll(ti_b, ti_a, ti_c) +
          Alll(ti_b, ti_a, ti_c) - Hlll(ti_b, ti_a, ti_c));
  // testing LHS symmetry is nonsymmetric when RHS operands do not have
  // symmetries in common
  const Tensor<double, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Glll4 = TensorExpressions::evaluate<ti_a, ti_b, ti_c>(
          Alll(ti_b, ti_c, ti_a) + Rlll(ti_c, ti_a, ti_b));
  // testing LHS symmetry preserves shared RHS symmetry when RHS operands have
  // symmetries in common
  const Tensor<double, Symmetry<2, 1, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Glll5 = TensorExpressions::evaluate<ti_a, ti_b, ti_c>(
          Alll(ti_b, ti_c, ti_a) - Rlll(ti_a, ti_c, ti_b));

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        CHECK(Glll.get(i, j, k) == Alll.get(i, j, k) + Hlll.get(i, j, k));
        CHECK(Glll2.get(i, j, k) == Alll.get(i, j, k) + Hlll.get(j, i, k));
        CHECK(Glll3.get(i, j, k) == 2.0 * Alll.get(i, j, k));
        CHECK(Glll4.get(i, j, k) == Alll.get(j, k, i) + Rlll.get(k, i, j));
        CHECK(Glll5.get(i, j, k) == Alll.get(j, k, i) - Rlll.get(i, k, j));
      }
    }
  }

  // testing with expressions having spatial indices for spacetime indices
  const Tensor<double, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                          SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Glll6 = TensorExpressions::evaluate<ti_a, ti_j, ti_k>(
          Rlll(ti_a, ti_j, ti_k) + Slll(ti_a, ti_j, ti_k));
  Tensor<double, Symmetry<3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Glll7{};
  TensorExpressions::evaluate<ti_j, ti_a, ti_k>(
      make_not_null(&Glll7), Slll(ti_j, ti_k, ti_a) - Rlll(ti_k, ti_a, ti_j));

  for (int a = 0; a < 4; ++a) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        CHECK(Glll6.get(a, j, k) ==
              Rlll.get(a, j + 1, k + 1) + Slll.get(a, j, k + 1));
        CHECK(Glll7.get(j, a, k + 1) ==
              Slll.get(j + 1, k, a) - Rlll.get(k + 1, a, j + 1));
      }
    }
  }
}
