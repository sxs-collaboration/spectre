// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "NumericalAlgorithms/Interpolation/LagrangePolynomial.hpp"
#include "Utilities/Gsl.hpp"

SPECTRE_TEST_CASE("Unit.Numerical.Interpolation.LagrangePolynomial",
                  "[Unit][NumericalAlgorithms]") {
  std::array<double, 4> control{{-1., 0., 1.5, 6.}};

  for (size_t i = 0; i < control.size(); ++i) {
    for (size_t j = 0; j < control.size(); ++j) {
      CHECK(lagrange_polynomial(j, gsl::at(control, i),
                                control.begin(), control.end())
            == approx(i == j ? 1. : 0.));
      CHECK(lagrange_polynomial(control.begin() + static_cast<ptrdiff_t>(j),
                                gsl::at(control, i),
                                control.begin(),
                                control.end()) ==
            approx(i == j ? 1. : 0.));
    }
  }

  // Check interpolating a cubic
  const auto func =
      [](double x) { return 1.2 + x * (2.3 + x * (3.4 + x * 4.5)); };
  const double test_point = 2.7;
  double interpolated = 0.;
  for (auto it = control.begin(); it != control.end(); ++it) {
    interpolated +=
        func(*it) * lagrange_polynomial(it, test_point, control.begin(),
                                        control.end());
  }
  CHECK(interpolated == approx(func(test_point)));
}
