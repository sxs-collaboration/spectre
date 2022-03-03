// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/PolynomialInterpolation.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <size_t Degree>
void test() {
  CAPTURE(Degree);
  const DataVector coords{0.1, 0.4, 0.5, 0.9, 1.2, 1.6, 1.8, 2.1};
  const double target_x = 0.2;
  DataVector y{coords.size(), 0.0};
  double expected_y = 0.0;
  for (size_t i = 0; i <= Degree; ++i) {
    y += pow(coords, i);
    expected_y += pow(target_x, i);
  }
  double error_in_y = 0.0;
  double target_y = 0.0;
  intrp::polynomial_interpolation<Degree>(
      make_not_null(&target_y), make_not_null(&error_in_y), target_x,
      gsl::make_span(y.data(), Degree + 1),
      gsl::make_span(coords.data(), Degree + 1));
  CHECK_ITERABLE_APPROX(target_y, expected_y);
  // Nils Deppe: The error estimate is quite bad I thinks because we do not have
  // any decrease in power as we go to higher degree polynomials and so the
  // difference between the Degree and Degree-1 polynomial is generally fairly
  // large.
  CHECK(std::abs(error_in_y) < 0.5);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Interpolation.PolynomialInterpolation",
                  "[Unit][NumericalAlgorithms]") {
  test<1>();
  test<2>();
  test<3>();
  test<4>();
  test<5>();
  test<6>();
  test<7>();
}
