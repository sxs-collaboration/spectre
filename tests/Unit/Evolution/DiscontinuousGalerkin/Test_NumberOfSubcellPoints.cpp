// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <string>

#include "Evolution/DiscontinuousGalerkin/NumberOfSubcellPoints.hpp"
#include "Evolution/DiscontinuousGalerkin/OptionTags.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"

namespace {
void test_number_of_subcell_points() {
  using evolution::dg::number_of_subcell_points;
  constexpr size_t max_num_fd_points =
      Spectral::maximum_number_of_points<Spectral::Basis::FiniteDifference,
                                         Spectral::Quadrature::CellCentered>;
  for (size_t n = 1; n < 22; ++n) {
    CHECK(number_of_subcell_points(
              n, evolution::dg::auto_number_of_subcell_points_coefficients) ==
          2 * n - 1);
    // The number of FD points may equal the number of DG points
    CHECK(number_of_subcell_points(n, {{0, 1, 0}}) == n);
    if (n * n - 2 * n + 3 <= max_num_fd_points) {
      CHECK(number_of_subcell_points(n, {{1, -2, 3}}) == n * n - 2 * n + 3);
    }
    if (3 * n + 2 <= max_num_fd_points) {
      CHECK(number_of_subcell_points(n, {{0, 3, 2}}) == 3 * n + 2);
    }
    // A negative quadratic coefficient is fine as long as the result is valid
    if (n <= 6) {
      CHECK(number_of_subcell_points(n, {{-1, 9, -2}}) == 9 * n - n * n - 2);
    }
  }
  CHECK(number_of_subcell_points(1, {{0, 0, 1}}) == 1);
  CHECK(number_of_subcell_points(6, {{1, 0, 6}}) == max_num_fd_points);

  // Fewer FD points than DG points
  CHECK_THROWS_WITH(
      number_of_subcell_points(5, {{0, 1, -1}}),
      Catch::Matchers::ContainsSubstring(
          "The number of FD grid points (4 = 0 * 5^2 + 1 * 5 + -1) must be at "
          "least the number of DG grid points (5)"));
  CHECK_THROWS_WITH(number_of_subcell_points(4, {{0, 0, 0}}),
                    Catch::Matchers::ContainsSubstring(
                        "must be at least the number of DG grid points (4)"));
  CHECK_THROWS_WITH(
      number_of_subcell_points(10, {{-1, 9, 0}}),
      Catch::Matchers::ContainsSubstring(
          "The number of FD grid points (-10 = -1 * 10^2 + 9 * 10 + 0) must be "
          "at least the number of DG grid points (10)"));
  // More FD points than supported
  CHECK_THROWS_WITH(
      number_of_subcell_points(7, {{1, 0, 0}}),
      Catch::Matchers::ContainsSubstring(
          "The number of FD grid points (49 = 1 * 7^2 + 0 * 7 + 0) for 7 DG "
          "grid points must be in [1, " +
          std::to_string(max_num_fd_points) + "]"));
  CHECK_THROWS_WITH(number_of_subcell_points(6, {{1, 0, 7}}),
                    Catch::Matchers::ContainsSubstring(
                        "for 6 DG grid points must be in [1, " +
                        std::to_string(max_num_fd_points) + "]"));
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      number_of_subcell_points(
          0, evolution::dg::auto_number_of_subcell_points_coefficients),
      Catch::Matchers::ContainsSubstring(
          "The number of DG grid points must be positive."));
#endif  // SPECTRE_DEBUG
}

void test_tag() {
  using Tag = evolution::dg::Tags::NumberOfSubcellPointsCoefficients;
  TestHelpers::db::test_simple_tag<Tag>("NumberOfSubcellPointsCoefficients");
  CHECK(Tag::create_from_options(std::nullopt) ==
        evolution::dg::auto_number_of_subcell_points_coefficients);
  CHECK(Tag::create_from_options(std::array{1, -2, 3}) == std::array{1, -2, 3});
  CHECK(TestHelpers::test_option_tag<
            evolution::dg::OptionTags::NumberOfSubcellPoints>("Auto") ==
        std::nullopt);
  CHECK(TestHelpers::test_option_tag<
            evolution::dg::OptionTags::NumberOfSubcellPoints>("[1, -2, 3]") ==
        std::array{1, -2, 3});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.NumberOfSubcellPoints",
                  "[Evolution][Unit]") {
  test_number_of_subcell_points();
  test_tag();
}
