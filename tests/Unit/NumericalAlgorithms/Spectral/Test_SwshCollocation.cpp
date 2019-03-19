// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <sharp_cxx.h>

#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/Spectral/ComplexDataView.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace Spectral {
namespace Swsh {
namespace {

// IWYU pragma: no_include <sharp_geomhelpers.h>
// IWYU pragma: no_include <sharp_lowlevel.h>

template <ComplexRepresentation Representation>
void test_spherical_harmonic_collocation() noexcept {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> sdist{8, 64};
  const size_t l_max = sdist(gen);

  CAPTURE(l_max);
  const Collocation<Representation>& precomputed_collocation_value =
      precomputed_collocation<Representation>(l_max);

  const Collocation<Representation>& another_precomputed_collocation =
      precomputed_collocation<Representation>(l_max);

  // checks that the same pointer is in both
  CHECK(precomputed_collocation_value.get_sharp_geom_info() ==
        another_precomputed_collocation.get_sharp_geom_info());

  const Collocation<Representation> computed_collocation{l_max};

  CHECK(precomputed_collocation_value.l_max() == l_max);
  CHECK(computed_collocation.l_max() == l_max);

  const int expected_stride = detail::ComplexDataView<Representation>::stride();

  sharp_geom_info* manual_sgi;
  sharp_make_gauss_geom_info(
      l_max + 1, 2 * l_max + 1, 0.0, expected_stride,
      static_cast<unsigned long>(expected_stride) * (2 * l_max + 1),
      &manual_sgi);

  // clang-tidy doesn't like the pointer manipulation needed to work with the
  // libsharp types.
  CHECK(precomputed_collocation_value  // NOLINT
            .get_sharp_geom_info()
            ->pair[0]
            .r1.stride == expected_stride);  // NOLINT
  CHECK(precomputed_collocation_value        // NOLINT
            .get_sharp_geom_info()
            ->pair[0]
            .r2.stride == expected_stride);  // NOLINT
  CHECK(computed_collocation                 // NOLINT
            .get_sharp_geom_info()
            ->pair[0]
            .r1.stride == expected_stride);  // NOLINT
  CHECK(computed_collocation                 // NOLINT
            .get_sharp_geom_info()
            ->pair[0]
            .r2.stride == expected_stride);  // NOLINT

  size_t offset_counter = 0;

  sharp_geom_info* computed_sgi = computed_collocation.get_sharp_geom_info();

  // check iterator equivalence. The for loop below is also a check of the
  // iterator functionality.
  CHECK(precomputed_collocation_value.begin() ==
        precomputed_collocation_value.cbegin());
  CHECK(precomputed_collocation_value.end() ==
        precomputed_collocation_value.cend());
  CHECK(precomputed_collocation_value.begin() !=
        precomputed_collocation_value.end());
  for (const auto& collocation_point : precomputed_collocation_value) {
    CHECK(collocation_point.offset == offset_counter);
    CHECK(collocation_point.theta ==
          computed_collocation.theta(offset_counter));
    CHECK(collocation_point.phi == computed_collocation.phi(offset_counter));
    CAPTURE(offset_counter);
    // check theta values:
    if (offset_counter < (2 * l_max + 1) * (l_max / 2 + 1)) {
      CAPTURE(offset_counter / (2 * l_max + 1));
      CHECK(approx(computed_collocation.theta(offset_counter)) ==
            computed_sgi  // NOLINT
                ->pair[offset_counter / (2 * l_max + 1)]
                .r1.theta);  // NOLINT
      CHECK(approx(computed_collocation.theta(offset_counter)) ==
            manual_sgi  // NOLINT
                ->pair[offset_counter / (2 * l_max + 1)]
                .r1.theta);  // NOLINT
    } else {
      CAPTURE(l_max - (offset_counter / (2 * l_max + 1)));
      CHECK(approx(computed_collocation.theta(offset_counter)) ==
            computed_sgi  // NOLINT
                ->pair[l_max - offset_counter / (2 * l_max + 1)]
                .r2.theta);  // NOLINT
      CHECK(approx(computed_collocation.theta(offset_counter)) ==
            manual_sgi  // NOLINT
                ->pair[l_max - offset_counter / (2 * l_max + 1)]
                .r2.theta);  // NOLINT
    }

    // check phi values:
    CHECK(approx(computed_collocation.phi(offset_counter)) ==
          2.0 * M_PI *
              ((offset_counter % (2 * l_max + 1)) / (2.0 * l_max + 1.0)));
    CHECK(approx(computed_collocation.phi(offset_counter)) ==
          2.0 * M_PI *
              ((offset_counter % (2 * l_max + 1)) / (2.0 * l_max + 1.0)));
    offset_counter++;
  }
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Spectral.SwshCollocation",
                  "[Unit][NumericalAlgorithms]") {
  {
    INFO("Collocation based on contiguous complex data (Interleaved)");
    test_spherical_harmonic_collocation<ComplexRepresentation::Interleaved>();
  }
  {
    INFO(
        "Collocation based on a pair of contiguous blocks representing complex "
        "data (RealsThenImags)");
    test_spherical_harmonic_collocation<
        ComplexRepresentation::RealsThenImags>();
  }
}

// [[OutputRegex, is not below the maximum l_max]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.Spectral.SwshCollocation.PrecomputationOverrun",
    "[Unit][NumericalAlgorithms]") {
  ERROR_TEST();
  precomputed_collocation<ComplexRepresentation::RealsThenImags>(
      collocation_maximum_l_max + 1);
  ERROR("Failed to trigger ERROR in an error test");
}
}  // namespace
}  // namespace Swsh
}  // namespace Spectral
