// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <vector>

#include "NumericalAlgorithms/SphericalHarmonics/WignerThreeJ.hpp"

namespace {
void test_wigner_three_j() {
  // For these, tests, Mark computed a few Wigner 3J symbols
  // analytically using an online tool.
  WignerThreeJ coefa(3, 2, 4, -3);
  CHECK(coefa.l1_min() == 1);
  CHECK(coefa.l1_max() == 7);
  CHECK(coefa(0) == 0.0);  // Triangle condition violated, return 0 no error.
  CHECK(coefa(1) == approx(-0.5 * sqrt(1.0 / 3.0)));
  CHECK(coefa(2) == approx(-(1.0 / 6.0) * sqrt(1.0 / 5.0)));
  CHECK(coefa(3) == approx((1.0 / 3.0) * sqrt(1.0 / 11.0)));
  CHECK(coefa(4) == approx(sqrt(1.0 / 33.0)));
  CHECK(coefa(5) == approx((1.0 / 6.0) * sqrt(11.0 / 13.0)));
  CHECK(coefa(6) == approx((17.0 / 6.0) * sqrt(1.0 / 1001.0)));
  CHECK(coefa(7) == approx(4.0 * sqrt(1.0 / 15015.0)));
  CHECK(coefa(8) == 0.0);  // Triangle condition violated, return 0 no error.

  WignerThreeJ coefb(1, -1, 2, -1);
  CHECK(coefb.l1_min() == 2);
  CHECK(coefb.l1_max() == 3);
  CHECK(coefb(1) == 0.0);  // Triangle condition violated, return 0 no error.
  CHECK(coefb(2) == approx(sqrt(1.0 / 15.0)));
  CHECK(coefb(3) == approx(-sqrt(2.0 / 21.0)));
  CHECK(coefb(4) == 0.0);  // Triangle condition violated, return 0 no error.

  // Intentionally pass |m| > l to get error.
  CHECK_THROWS_WITH(WignerThreeJ(3, 4, 4, -3),
                    Catch::Matchers::ContainsSubstring("m2 is 4 but l2 is 3"));
  CHECK_THROWS_WITH(WignerThreeJ(3, 3, 4, -5),
                    Catch::Matchers::ContainsSubstring("m3 is -5 but l3 is 4"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.SphericalHarmonics.WignerThreeJ",
                  "[NumericalAlgorithms][Unit]") {
  test_wigner_three_j();
}
