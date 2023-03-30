// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <deque>

#include "NumericalAlgorithms/FiniteDifference/NonUniform1D.hpp"
#include "Utilities/Gsl.hpp"

namespace {

template <size_t StencilSize>
void test_impl() {
  std::deque<double> times{};
  times.emplace_front(4.0);
  times.emplace_back(3.0);
  if constexpr (StencilSize > 2) {
    times.emplace_back(2.5);
    if constexpr (StencilSize > 3) {
      times.emplace_back(0.5);
    }
  }

  const auto weights = fd::non_uniform_1d_weights<StencilSize>(times);

  std::array<double, StencilSize> zeroth_weights{};
  gsl::at(zeroth_weights, 0) = 1.0;
  // All stencil sizes have this as the zeroth values
  CHECK(gsl::at(weights, 0) == zeroth_weights);

  if constexpr (StencilSize == 2) {
    CHECK_ITERABLE_APPROX(gsl::at(weights, 1), (std::array{1.0, -1.0}));
  } else if constexpr (StencilSize == 3) {
    CHECK_ITERABLE_APPROX(gsl::at(weights, 1),
                          (std::array{5.0 / 3.0, -3.0, 4.0 / 3.0}));
    CHECK_ITERABLE_APPROX(gsl::at(weights, 2),
                          (std::array{4.0 / 3.0, -4.0, 8.0 / 3.0}));
  } else if constexpr (StencilSize == 4) {
    CHECK_ITERABLE_APPROX(
        gsl::at(weights, 1),
        (std::array{41.0 / 21.0, -21.0 / 5.0, 7.0 / 3.0, -3.0 / 35.0}));
    CHECK_ITERABLE_APPROX(gsl::at(weights, 2),
                          (std::array{16.0 / 7.0, -8.0, 6.0, -2.0 / 7.0}));
    CHECK_ITERABLE_APPROX(
        gsl::at(weights, 3),
        (std::array{8.0 / 7.0, -24.0 / 5.0, 4.0, -12.0 / 35.0}));
  }
}

template <size_t StencilSize>
void test_errors() {
#ifdef SPECTRE_DEBUG
  std::deque<double> times{};
  times.emplace_front(1.0);

  CHECK_THROWS_WITH(fd::non_uniform_1d_weights<StencilSize>(times),
                    Catch::Contains("The size of the times passed in"));

  times.emplace_back(2.0);
  if constexpr (StencilSize > 2) {
    times.emplace_back(0.5);
    if constexpr (StencilSize > 3) {
      times.emplace_back(0.3);
    }
  }

  CHECK_THROWS_WITH(fd::non_uniform_1d_weights<StencilSize>(times),
                    Catch::Contains("Times must be monotonically decreasing"));
#endif
}

template <size_t StencilSize>
void test() {
  test_impl<StencilSize>();
  test_errors<StencilSize>();
}
}  // namespace

SPECTRE_TEST_CASE("Unit.FiniteDifference.NonUniform1D",
                  "[Unit][NumericalAlgorithms]") {
  test<2>();
  test<3>();
  test<4>();
}
