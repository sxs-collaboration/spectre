// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <climits>
#include <complex>
#include <cstddef>
#include <random>
#include <type_traits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
// Checks that the number of ops in the expressions match what is expected
void test_tensor_ops_properties() {
  const Scalar<double> G{5.0};
  const double H = 5.0;

  const auto sqrt_G = sqrt(G());
  const auto sqrt_HGG = sqrt(H * G() * G());
  // Expected: 2 sqrt + 2 negations + 2 multiplies = 6 total ops
  const auto sqrt_sqrt_GHG = sqrt(sqrt(-G() * H * -G()));

  CHECK(sqrt_G.num_ops_subtree == 1);
  CHECK(sqrt_HGG.num_ops_subtree == 3);
  CHECK(sqrt_sqrt_GHG.num_ops_subtree == 6);
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
template <typename Generator, typename DataType>
void test_sqrt(const gsl::not_null<Generator*> generator,
               const DataType& used_for_size) {
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  const auto R = make_with_random_values<Scalar<DataType>>(
      generator, distribution, used_for_size);

  // \f$L = \sqrt{R}\f$
  Tensor<DataType> sqrt_R = tenex::evaluate(sqrt(R()));
  CHECK(sqrt_R.get() == sqrt(R.get()));

  const auto S =
      make_with_random_values<tnsr::iJ<DataType, 3, Frame::Inertial>>(
          generator, distribution, used_for_size);

  DataType S_trace = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t i = 0; i < 3; i++) {
    S_trace += S.get(i, i);
  }

  // \f$L = \sqrt{S_{k}{}^{k}}\f$
  const Tensor<DataType> sqrt_S = tenex::evaluate(sqrt(S(ti::k, ti::K)));
  // \f$L = \sqrt{S_{k}{}^{k} * T}\f$
  const Tensor<DataType> sqrt_S_T =
      tenex::evaluate(sqrt(S(ti::k, ti::K) * 3.6));
  CHECK_ITERABLE_APPROX(sqrt_S.get(), sqrt(S_trace));
  CHECK_ITERABLE_APPROX(sqrt_S_T.get(), sqrt(S_trace * 3.6));

  const auto G = make_with_random_values<tnsr::I<DataType, 3, Frame::Grid>>(
      generator, distribution, used_for_size);
  const auto H = make_with_random_values<tnsr::a<DataType, 3, Frame::Grid>>(
      generator, distribution, used_for_size);

  DataType GH_product = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t i = 0; i < 3; i++) {
    GH_product += G.get(i) * H.get(i + 1);
  }

  // Test expression that uses generic spatial index for a spacetime index
  // \f$L = \sqrt{G^{j} H_{j}\f$
  const Tensor<DataType> sqrt_GH_product =
      tenex::evaluate(sqrt(G(ti::J) * H(ti::j)));
  CHECK_ITERABLE_APPROX(sqrt_GH_product.get(), sqrt(GH_product));

  const auto T = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>>(
      generator, distribution, used_for_size);

  // Test expression that uses concrete time index for a spacetime index
  // \f$L = \sqrt{T_{t, t}\f$
  const Scalar<DataType> sqrt_T_time = tenex::evaluate(sqrt(T(ti::t, ti::t)));
  CHECK(sqrt_T_time.get() == sqrt(T.get(0, 0)));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.SquareRoot",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  test_tensor_ops_properties();
  test_sqrt(make_not_null(&generator),
            std::numeric_limits<double>::signaling_NaN());
  test_sqrt(make_not_null(&generator),
            std::complex<double>(std::numeric_limits<double>::signaling_NaN(),
                                 std::numeric_limits<double>::signaling_NaN()));
  test_sqrt(make_not_null(&generator),
            DataVector(5, std::numeric_limits<double>::signaling_NaN()));
  test_sqrt(make_not_null(&generator),
            ComplexDataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
