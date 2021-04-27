// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
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

// \brief Test the square root of a rank 0 tensor expression is correctly
// evaluated
//
// \details
// The cases tested are:
// - \f$L = \sqrt{R}\f$
// - \f$L = \sqrt{S_{k}{}^{k}}\f$
// - \f$L = \sqrt{S_{k}{}^{k} * T}\f$
//
// where \f$R\f$ and \f$L\f$ are rank 0 Tensors, \f$S\f$ is a rank 2 Tensor,
// and \f$T\f$ is a `double`.
//
// \tparam DataType the type of data being stored in the tensor operands of the
// expressions tested
template <typename DataType>
void test_sqrt(const DataType& used_for_size) noexcept {
  Tensor<DataType> R{{{used_for_size}}};
  if (std::is_same_v<DataType, double>) {
    // Replace tensor's value from `used_for_size` to a proper test value
    R.get() = 5.7;
  } else {
    assign_unique_values_to_tensor(make_not_null(&R));
  }

  // \f$L = \sqrt{R}\f$
  Tensor<DataType> sqrt_R = TensorExpressions::evaluate(sqrt(R()));
  CHECK(sqrt_R.get() == sqrt(R.get()));

  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Up, Frame::Inertial>>>
      S(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&S));

  DataType S_trace = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t i = 0; i < 3; i++) {
    S_trace += S.get(i, i);
  }

  // \f$L = \sqrt{S_{k}{}^{k}}\f$
  const Tensor<DataType> sqrt_S =
      TensorExpressions::evaluate(sqrt(S(ti_k, ti_K)));
  // \f$L = \sqrt{S_{k}{}^{k} * T}\f$
  const Tensor<DataType> sqrt_S_T =
      TensorExpressions::evaluate(sqrt(S(ti_k, ti_K) * 3.6));
  CHECK(sqrt_S.get() == sqrt(S_trace));
  CHECK(sqrt_S_T.get() == sqrt(S_trace * 3.6));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.SquareRoot",
                  "[DataStructures][Unit]") {
  test_sqrt(std::numeric_limits<double>::signaling_NaN());
  test_sqrt(DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
