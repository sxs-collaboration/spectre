// Distributed under the MIT License.
// See LICENSE.txt for details.
#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <tuple>

#include "Utilities/Gsl.hpp"
#include "tests/Unit/DataStructures/Test_VectorImplHelper.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.VectorImplHelpers.ApplyTupleCombinations",
                  "[Utilities][Unit]") {
  // for more generic applications, with lambdas:
  /// [tuple_combos_lambda]
  const int zero = 0;
  const double one = 1.2;  // for general lambda, 1.2 casts to 1
  const std::complex<double> two{2.5, 5.0};  // for lambda, real and casts to 2
  std::array<std::array<bool, 3>, 3> checklist {{{{false, false, false}},
                                                 {{false, false, false}},
                                                 {{false, false, false}}}};
  const auto lambda_tuple = std::make_tuple(zero, one, two);
  TestHelpers::VectorImpl::apply_tuple_combinations<2>(
      lambda_tuple, [&checklist](auto x, auto y) noexcept {
        gsl::at(gsl::at(checklist, static_cast<size_t>(std::real(x))),
                static_cast<size_t>(std::real(y))) = true;
        CHECK(std::imag(x) + std::real(y) >= 0);
      });
  CHECK(checklist == std::array<std::array<bool, 3>, 3>{
          {{{true, true, true}}, {{true, true, true}}, {{true, true, true}}}});
  /// [tuple_combos_lambda]
}
