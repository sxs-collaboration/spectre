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

SPECTRE_TEST_CASE("Unit.DataStructures.VectorImplHelpers.TupleHelpers",
                  "[Utilities][Unit]") {
  auto test_tup = std::make_tuple(5.5, std::complex<int>{2, 3}, 'c');
  CHECK(TestHelpers::VectorImpl::remove_nth<0>(test_tup) ==
        std::make_tuple(std::complex<int>{2, 3}, 'c'));
  CHECK(TestHelpers::VectorImpl::remove_nth<1>(test_tup) ==
        std::make_tuple(5.5, 'c'));
  CHECK(TestHelpers::VectorImpl::remove_nth<2>(test_tup) ==
        std::make_tuple(5.5, std::complex<int>{2, 3}));
  const auto addr_test_tup =
      TestHelpers::VectorImpl::addressof(make_not_null(&test_tup));
  CHECK(addr_test_tup == std::make_tuple(&std::get<0>(test_tup),
                                         &std::get<1>(test_tup),
                                         &std::get<2>(test_tup)));
}
