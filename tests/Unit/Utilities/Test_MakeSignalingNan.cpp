// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <complex>

#include "Utilities/IsNan.hpp"
#include "Utilities/MakeSignalingNan.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.MakeSignalingNaN", "[Unit][Utilities]") {
  const auto double_snan = make_signaling_NaN<double>();
  CHECK(is_nan(double_snan));
  const auto float_snan = make_signaling_NaN<float>();
  CHECK(is_nan(float_snan));
  const auto complex_snan = make_signaling_NaN<std::complex<double>>();
  CHECK(is_nan(complex_snan.real()));
  CHECK(is_nan(complex_snan.imag()));
}
