// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/Product.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

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

// Computes \f$L_{a} = R_{ab}* S^{b} + G_{a} - H_{ba}{}^{b} * T\f$
template <typename R_t, typename S_t, typename G_t, typename H_t, typename T_t,
          typename DataType>
G_t compute_expected_result(const R_t& R, const S_t& S, const G_t& G,
                            const H_t& H, const T_t& T,
                            const DataType& used_for_size) noexcept {
  using result_tensor_type = G_t;
  result_tensor_type expected_result{};
  const size_t dim = tmpl::front<typename R_t::index_list>::dim;
  for (size_t a = 0; a < dim; a++) {
    DataType expected_Rab_SB_product =
        make_with_value<DataType>(used_for_size, 0.0);
    DataType expected_HbaB_contracted_value =
        make_with_value<DataType>(used_for_size, 0.0);
    for (size_t b = 0; b < dim; b++) {
      expected_Rab_SB_product += R.get(a, b) * S.get(b);
      expected_HbaB_contracted_value += H.get(b, a, b);
    }
    expected_result.get(a) = expected_Rab_SB_product + G.get(a) -
                             (expected_HbaB_contracted_value * T.get());
  }

  return expected_result;
}

template <typename DataType>
void test_mixed_operations(const DataType& used_for_size) noexcept {
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      R(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&R));

  Tensor<DataType, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      S(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&S));

  Tensor<DataType, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      G(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&G));

  Tensor<DataType, Symmetry<3, 2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      H(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&H));

  Tensor<DataType> T{{{used_for_size}}};
  if constexpr (std::is_same_v<DataType, double>) {
    // Replace tensor's value from `used_for_size` with a proper test value
    T.get() = -2.2;
  } else {
    assign_unique_values_to_tensor(make_not_null(&T));
  }

  using result_tensor_type = decltype(G);
  result_tensor_type expected_result_tensor =
      compute_expected_result(R, S, G, H, T, used_for_size);
  // \f$L_{a} = R_{ab}* S^{b} + G_{a} - H_{ba}{}^{b} * T\f$
  result_tensor_type actual_result_tensor = TensorExpressions::evaluate<ti_a>(
      R(ti_a, ti_b) * S(ti_B) + G(ti_a) - H(ti_b, ti_a, ti_B) * T());

  for (size_t a = 0; a < 4; a++) {
    CHECK_ITERABLE_APPROX(actual_result_tensor.get(a),
                          expected_result_tensor.get(a));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.MixedOperations",
                  "[DataStructures][Unit]") {
  test_mixed_operations(std::numeric_limits<double>::signaling_NaN());
  test_mixed_operations(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
