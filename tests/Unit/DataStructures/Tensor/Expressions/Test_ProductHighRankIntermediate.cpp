// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <typename... Ts>
void assign_unique_values_to_tensor(
    gsl::not_null<Tensor<double, Ts...>*> tensor) noexcept {
  std::iota(tensor->begin(), tensor->end(), 0.0);
}

template <typename... Ts>
void assign_unique_values_to_tensor(
    gsl::not_null<Tensor<DataVector, Ts...>*> tensor) noexcept {
  double value = 0.0;
  for (auto index_it = tensor->begin(); index_it != tensor->end(); index_it++) {
    for (auto vector_it = index_it->begin(); vector_it != index_it->end();
         vector_it++) {
      *vector_it = value;
      value += 1.0;
    }
  }
}

// \brief Test the product of three rank 4 tensors involving both inner and
// outer products of indices is correctly evaluated
//
// \details
// The product case tested is:
// - \f$L^{c}{}_{dkl} = R_{ijb}{}^{a} * (S_{da}{}^{BC} * T^{j}{}_{kl}{}^{i})\f$
//
// This is intended as a stress test for TensorContract and OuterProduct.
//
// \tparam DataType the type of data being stored in the product operands
template <typename DataType>
void test_high_rank_intermediate(const DataType& used_for_size) noexcept {
  using frame = Frame::Inertial;
  using A_index = SpacetimeIndex<3, UpLo::Up, frame>;
  using a_index = SpacetimeIndex<3, UpLo::Lo, frame>;
  using B_index = SpacetimeIndex<3, UpLo::Up, frame>;
  using b_index = SpacetimeIndex<3, UpLo::Lo, frame>;
  using C_index = B_index;
  using d_index = SpacetimeIndex<3, UpLo::Lo, frame>;
  using I_index = SpatialIndex<3, UpLo::Up, frame>;
  using i_index = SpatialIndex<3, UpLo::Lo, frame>;
  using J_index = SpatialIndex<3, UpLo::Up, frame>;
  using j_index = i_index;
  using k_index = SpatialIndex<3, UpLo::Lo, frame>;
  using l_index = SpatialIndex<3, UpLo::Lo, frame>;

  Tensor<DataType, Symmetry<3, 3, 2, 1>,
         index_list<i_index, j_index, b_index, A_index>>
      R(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&R));
  Tensor<DataType, Symmetry<3, 2, 1, 1>,
         index_list<d_index, a_index, B_index, C_index>>
      S(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&S));
  Tensor<DataType, Symmetry<4, 3, 2, 1>,
         index_list<J_index, k_index, l_index, I_index>>
      T(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&T));

  // \f$L^{c}{}_{dkl} = R_{ijb}{}^{a} * (S_{da}{}^{BC} * T^{j}{}_{kl}{}^{i})\f$
  const Tensor<DataType, Symmetry<4, 3, 2, 1>,
               index_list<C_index, d_index, k_index, l_index>>
      actual_result = TensorExpressions::evaluate<ti_C, ti_d, ti_k, ti_l>(
          R(ti_i, ti_j, ti_b, ti_A) *
          (S(ti_d, ti_a, ti_B, ti_C) * T(ti_J, ti_k, ti_l, ti_I)));

  for (size_t c = 0; c < C_index::dim; c++) {
    for (size_t d = 0; d < d_index::dim; d++) {
      for (size_t k = 0; k < k_index::dim; k++) {
        for (size_t l = 0; l < l_index::dim; l++) {
          DataType expected_product_component =
              make_with_value<DataType>(used_for_size, 0.0);
          for (size_t i = 0; i < i_index::dim; i++) {
            for (size_t j = 0; j < j_index::dim; j++) {
              for (size_t b = 0; b < b_index::dim; b++) {
                for (size_t a = 0; a < a_index::dim; a++) {
                  expected_product_component +=
                      R.get(i, j, b, a) * S.get(d, a, b, c) * T.get(j, k, l, i);
                }
              }
            }
          }
          CHECK_ITERABLE_APPROX(actual_result.get(c, d, k, l),
                                expected_product_component);
        }
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression.ProductHighRankIntermediate",
    "[DataStructures][Unit]") {
  test_high_rank_intermediate(std::numeric_limits<double>::signaling_NaN());
  test_high_rank_intermediate(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
