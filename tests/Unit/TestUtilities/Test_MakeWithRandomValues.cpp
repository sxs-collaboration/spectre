// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

namespace {
template <size_t Dim>
struct Var1 {
  using type = tnsr::i<DataVector, Dim, Frame::Grid>;
};

struct Var2 {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
using one_var = tmpl::list<Var1<Dim>>;

template <size_t Dim>
using two_vars = tmpl::list<Var1<Dim>, Var2>;

void check_random_values(
    const gsl::not_null<std::unordered_set<double>*> values,
    const gsl::not_null<size_t*> counter, const double v,
    const double lower_bound, const double upper_bound) noexcept {
  CHECK(v >= lower_bound);
  CHECK(v <= upper_bound);
  CHECK(values->insert(v).second);
  ++(*counter);
}

// clang-tidy is wrong, this is a function definition
template <typename T>
void check_random_values(
    const gsl::not_null<std::unordered_set<double>*> values,        // NOLINT
    const gsl::not_null<size_t*> counter, const T& c,               // NOLINT
    const double lower_bound, const double upper_bound) noexcept {  // NOLINT
  for (const auto& v : c) {
    check_random_values(values, counter, v, lower_bound, upper_bound);
  }
}

template <typename... Tags>
void check_random_values(
    const gsl::not_null<std::unordered_set<double>*> values,
    const gsl::not_null<size_t*> counter,
    const Variables<tmpl::list<Tags...>>& v, const double lower_bound,
    const double upper_bound) noexcept {
  expand_pack((check_random_values<decltype(get<Tags>(v))>(
                   values, counter, get<Tags>(v), lower_bound, upper_bound),
               cpp17::void_type{})...);
}

template <typename T, typename U>
void test_make_with_random_values(const U& used_for_size) noexcept {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-10.0, 10.0);
  // check that the data structure is filled with unique random values
  // by adding them to a set and checking that the size of the set
  // is equal to the number of doubles inserted
  std::unordered_set<double> values;
  size_t counter = 0;
  const auto d = make_with_random_values<T>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  check_random_values(make_not_null(&values), make_not_null(&counter), d, -10.0,
                      10.0);
  CHECK(values.size() == counter);
}
}  // namespace

SPECTRE_TEST_CASE("Test.TestHelpers.MakeWithRandomValues", "[Unit]") {
  const double d = std::numeric_limits<double>::signaling_NaN();
  test_make_with_random_values<double>(d);
  test_make_with_random_values<Scalar<double>>(d);
  test_make_with_random_values<tnsr::A<double, 3>>(d);
  test_make_with_random_values<tnsr::Abb<double, 3>>(d);
  test_make_with_random_values<tnsr::aBcc<double, 3>>(d);

  const DataVector dv(5);
  test_make_with_random_values<DataVector>(dv);
  test_make_with_random_values<Scalar<DataVector>>(dv);
  test_make_with_random_values<tnsr::A<DataVector, 3>>(dv);
  test_make_with_random_values<tnsr::Abb<DataVector, 3>>(dv);
  test_make_with_random_values<tnsr::aBcc<DataVector, 3>>(dv);
  test_make_with_random_values<Variables<one_var<3>>>(dv);
  test_make_with_random_values<Variables<two_vars<3>>>(dv);
}
