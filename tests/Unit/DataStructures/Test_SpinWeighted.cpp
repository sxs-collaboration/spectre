// Distributed under the MIT License.
// See LICENSE.txt for details

#include "tests/Unit/TestingFramework.hpp"

#include <complex>
#include <random>

#include "DataStructures/SpinWeighted.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace {
template <typename T>
void test_spinweights() {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<T> dist{static_cast<T>(-100.0),
                                    static_cast<T>(100.0)};

  const auto v1 = make_with_random_values<SpinWeighted<T, 0>>(
      make_not_null(&gen), make_not_null(&dist));
  const auto v2 = make_with_random_values<SpinWeighted<T, 1>>(
      make_not_null(&gen), make_not_null(&dist));
  const auto v3 = make_with_random_values<SpinWeighted<T, -2>>(
      make_not_null(&gen), make_not_null(&dist));

  const auto v4 = make_with_random_values<SpinWeighted<std::complex<T>, 0>>(
      make_not_null(&gen), make_not_null(&dist));
  const auto v5 = make_with_random_values<SpinWeighted<std::complex<T>, 1>>(
      make_not_null(&gen), make_not_null(&dist));

  // check compile-time spin values
  static_assert(decltype(v1)::spin == 0,
                "assert failed for the spin of a spin-weight 0");
  static_assert(decltype(v2)::spin == 1,
                "assert failed for the spin of a spin-weight 1");
  static_assert(decltype(v4 / v3)::spin == 2,
                "assert failed for the spin of a spin-weight ratio.");
  static_assert(decltype(v5 * v2)::spin == 2,
                "assert failed for the spin of a spin-weight product.");

  // check that valid spin combinations work
  CHECK(v1 + v1 == SpinWeighted<T, 0>{v1.data + v1.data});
  CHECK(v2 - v5 == SpinWeighted<std::complex<T>, 1>{v2.data - v5.data});
  CHECK(v2 * v3 == SpinWeighted<T, -1>{v2.data * v3.data});
  CHECK(v5 / v3 == SpinWeighted<std::complex<T>, 3>{v5.data / v3.data});
}

using SpinWeightedTypes = tmpl::list<double, int, long>;

SPECTRE_TEST_CASE("Unit.DataStructures.SpinWeight", "[DataStructures][Unit]") {
  tmpl::for_each<SpinWeightedTypes>([](auto x) noexcept {
    test_spinweights<typename decltype(x)::type>();
  });
}
}  // namespace
