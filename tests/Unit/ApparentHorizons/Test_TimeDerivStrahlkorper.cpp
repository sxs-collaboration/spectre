// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <deque>
#include <utility>

#include "ApparentHorizons/TimeDerivStrahlkorper.hpp"
#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Utilities/Gsl.hpp"

namespace {
void test_time_deriv_strahlkorper() {
  const size_t l_max = 2;
  Strahlkorper<Frame::Inertial> strahlkorper{l_max, l_max, 1.0,
                                             std::array{0.0, 0.0, 0.0}};
  SpherepackIterator iter{l_max, l_max};
  // Set a random coefficient non-zero
  strahlkorper.coefficients()[iter.set(2, 1)()] = 1.3;

  for (size_t num_times = 2; num_times <= 4; num_times++) {
    CAPTURE(num_times);
    std::deque<std::pair<double, Strahlkorper<Frame::Inertial>>>
        previous_strahlkorpers{};

    // Set all strahlkorpers to be the same
    for (size_t i = 0; i < num_times; i++) {
      // If num_times = 3, set one of the times == NaN to test that we get back
      // zero
      previous_strahlkorpers.emplace_front(std::make_pair(
          (num_times == 3 and i == 0) ? std::numeric_limits<double>::quiet_NaN()
                                      : static_cast<double>(i),
          strahlkorper));
    }

    auto time_deriv = strahlkorper;

    ah::time_deriv_of_strahlkorper(make_not_null(&time_deriv),
                                   previous_strahlkorpers);

    const DataVector& time_deriv_strahlkorper_coefs = time_deriv.coefficients();

    // Since we made all the Strahlkorpers the same (or there is a NaN time),
    // the time deriv should be zero.
    CHECK_ITERABLE_APPROX(
        time_deriv_strahlkorper_coefs,
        (DataVector{time_deriv_strahlkorper_coefs.size(), 0.0}));

    // Check that the deriv works for 2 times (easy to calculate by hand)
    if (num_times == 2) {
      // Set a single coefficient to a random value for each previous
      // strahlkorper
      previous_strahlkorpers.front().second.coefficients()[iter.set(2, -1)()] =
          2.0;
      previous_strahlkorpers.back().second.coefficients()[iter()] = 2.5;

      ah::time_deriv_of_strahlkorper(make_not_null(&time_deriv),
                                     previous_strahlkorpers);

      const DataVector& coefs = time_deriv.coefficients();

      DataVector expected_coefs{coefs.size(), 0.0};
      expected_coefs[iter()] = -0.5;

      CHECK_ITERABLE_APPROX(coefs, expected_coefs);
    }
  }
}

SPECTRE_TEST_CASE("Unit.ApparentHorizons.TimeDerivStrahlkorper", "[Unit]") {
  test_time_deriv_strahlkorper();
}

}  // namespace
