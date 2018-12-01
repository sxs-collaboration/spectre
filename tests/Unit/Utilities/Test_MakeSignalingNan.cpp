// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <complex>

#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/MakeSignalingNan.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.MakeSignalingNaN", "[Unit][Utilities]") {
  disable_floating_point_exceptions();
  auto double_snan = make_signaling_NaN<double>();
  CHECK(std::isnan(double_snan));
  auto float_snan = make_signaling_NaN<float>();
  CHECK(std::isnan(float_snan));
  std::complex<double> complex_snan =
      make_signaling_NaN<std::complex<double>>();
  CHECK(std::isnan(complex_snan.real()));
  CHECK(std::isnan(complex_snan.imag()));
  enable_floating_point_exceptions();
}
