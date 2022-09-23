// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <climits>
#include <complex>
#include <cstddef>
#include <random>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"

namespace {
// Checks that the number of ops in the expressions match what is expected
void test_tensor_ops_properties() {
  const Scalar<double> G{5.0};
  const double H = 5.0;
  const tnsr::ii<double, 3> R{};
  const tnsr::ij<double, 3> S{};

  const auto neg_G = -G();
  const auto neg_R = -R(ti::i, ti::j);
  // Below, -H should be `NumberAsExpression(-H)`, so the * should be the only
  // op for this expression
  const auto neg_H_times_G = -H * G();
  const auto neg_S_minus_R_times_G = -(S(ti::j, ti::i) - R(ti::i, ti::j) * G());

  CHECK(neg_G.num_ops_subtree == 1);
  CHECK(neg_R.num_ops_subtree == 1);
  CHECK(neg_H_times_G.num_ops_subtree == 1);
  CHECK(neg_S_minus_R_times_G.num_ops_subtree == 3);
}

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
      result1 = tenex::evaluate<ti::j, ti::i, ti::K>(-R(ti::K, ti::j, ti::i));
  // \f$L^{i}{}_{kj} = -(R^{i}_{kj} + S^{i}_{kj})\f$
  const tnsr::Ijj<DataType, dim> result2 = tenex::evaluate<ti::I, ti::k, ti::j>(
      -(R(ti::I, ti::k, ti::j) + S(ti::I, ti::k, ti::j)));

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

  test_tensor_ops_properties();
  test_negate(make_not_null(&generator),
              std::numeric_limits<double>::signaling_NaN());
  test_negate(
      make_not_null(&generator),
      std::complex<double>(std::numeric_limits<double>::signaling_NaN(),
                           std::numeric_limits<double>::signaling_NaN()));
  test_negate(make_not_null(&generator),
              DataVector(5, std::numeric_limits<double>::signaling_NaN()));
  test_negate(
      make_not_null(&generator),
      ComplexDataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
