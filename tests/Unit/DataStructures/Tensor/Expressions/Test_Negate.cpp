// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <climits>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"

namespace {
// \brief Test the unary `-` correctly negates a tensor expression
//
// \details
// The cases tested are:
// - \f$L_{ji}{}^{k} = -R^{k}_{ji}\f$
// - \f$L^{i}{}_{kj} = -(R^{i}_{kj} + S^{i}_{kj})\f$
//
// \tparam DataType the type of data being stored in the tensors
template <typename Generator, typename DataType>
void test_negate(const gsl::not_null<Generator*> generator,
                 const DataType& used_for_size) {
  std::uniform_real_distribution<> distribution(-2.0, 2.0);
  constexpr size_t dim = 3;

  const auto R =
      make_with_random_values<tnsr::Ijj<DataType, dim, Frame::Inertial>>(
          generator, make_not_null(&distribution), used_for_size);
  const auto S =
      make_with_random_values<tnsr::Ijj<DataType, dim, Frame::Inertial>>(
          generator, make_not_null(&distribution), used_for_size);

  // \f$L_{ji}{}^{k} = -R^{k}_{ji}\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  const Tensor<DataType, Symmetry<2, 2, 1>,
               index_list<SpatialIndex<dim, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<dim, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<dim, UpLo::Up, Frame::Inertial>>>
      result1 =
          TensorExpressions::evaluate<ti_j, ti_i, ti_K>(-R(ti_K, ti_j, ti_i));
  // \f$L^{i}{}_{kj} = -(R^{i}_{kj} + S^{i}_{kj})\f$
  const tnsr::Ijj<DataType, dim> result2 =
      TensorExpressions::evaluate<ti_I, ti_k, ti_j>(
          -(R(ti_I, ti_k, ti_j) + S(ti_I, ti_k, ti_j)));

  for (size_t i = 0; i < dim; i++) {
    for (size_t j = 0; j < dim; j++) {
      for (size_t k = 0; k < dim; k++) {
        CHECK(result1.get(j, i, k) == -R.get(k, j, i));
        CHECK(result2.get(i, k, j) == -R.get(i, k, j) - S.get(i, k, j));
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Negate",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  test_negate(make_not_null(&generator),
              std::numeric_limits<double>::signaling_NaN());
  test_negate(make_not_null(&generator),
              DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
