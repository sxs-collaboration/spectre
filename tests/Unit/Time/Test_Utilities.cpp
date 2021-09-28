// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <initializer_list>

#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/Utilities.hpp"

namespace {
void check_slab(const double start, const double end) {
  const double duration = end - start;
  {
    const Slab slab = Slab::with_duration_from_start(start, duration);
    CHECK(abs(slab.end().value() - end) < slab_rounding_error(slab.start()));
    CHECK(abs(slab.end().value() - end) < slab_rounding_error(slab.end()));
  }
  {
    const Slab slab = Slab::with_duration_to_end(end, duration);
    CHECK(abs(slab.start().value() - start) <
          slab_rounding_error(slab.start()));
    CHECK(abs(slab.start().value() - start) < slab_rounding_error(slab.end()));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.slab_rounding_error", "[Unit][Time]") {
  const std::initializer_list<double> test_times{
      -1.0e5 - 1.0,  -1.0e5,  -1.0e5 + 1.0, -2.0,   -1.0 - 1.0e-5, -1.0,
      -1.0 + 1.0e-5, -1.0e-5, 0.0,          1.0e-5, 1.0 - 1.0e-5,  1.0,
      1.0 + 1.0e-5,  2.0,     1.0e5 - 1.0,  1.0e5,  1.0e5 + 1.0};

  for (auto start = test_times.begin(); start != test_times.end(); ++start) {
    for (auto end = start + 1; end != test_times.end(); ++end) {
      check_slab(*start, *end);
    }
  }
}
