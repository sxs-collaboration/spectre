// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <cstddef>
#include <random>
#include <unordered_set>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/DataStructures/TestHelpers.hpp"
#include "tests/Unit/TestingFramework.hpp"

namespace {
template <size_t Dim>
struct Var1 {
  using type = tnsr::i<DataVector, Dim, Frame::Grid>;
};

struct Var2 {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
using two_vars = tmpl::list<Var1<Dim>, Var2>;

template <size_t Dim>
using one_var = tmpl::list<Var1<Dim>>;

template <typename R, typename T>
void check_make_with_value(const R& expected, const T& input,
                           const double value) {
  const auto computed = make_with_value<R>(input, value);
  CHECK(expected == computed);
}

void test_make_tagged_tuple() {
  for (size_t n_pts = 1; n_pts < 4; ++n_pts) {
    check_make_with_value(
        tuples::TaggedTuple<Var2>(Scalar<DataVector>(n_pts, -5.7)),
        DataVector(n_pts, 0.0), -5.7);
    check_make_with_value(
        tuples::TaggedTuple<Var2>(Scalar<DataVector>(n_pts, -5.7)),
        tnsr::ab<DataVector, 2, Frame::Inertial>(n_pts, 0.0), -5.7);

    check_make_with_value(tuples::TaggedTuple<Var1<3>, Var2>(
                              tnsr::i<DataVector, 3, Frame::Grid>(n_pts, 3.8),
                              Scalar<DataVector>(n_pts, 3.8)),
                          DataVector(n_pts, 0.0), 3.8);

    check_make_with_value(tuples::TaggedTuple<Var1<3>, Var2>(
                              tnsr::i<DataVector, 3, Frame::Grid>(n_pts, 3.8),
                              Scalar<DataVector>(n_pts, 3.8)),
                          tnsr::ab<DataVector, 2, Frame::Inertial>(n_pts, 0.0),
                          3.8);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.MakeWithValue",
                  "[DataStructures][Unit]") {
  check_make_with_value(8.3, 1.3, 8.3);
  check_make_with_value(Scalar<double>(8.3), 1.3, 8.3);
  check_make_with_value(tnsr::I<double, 3, Frame::Grid>(8.3), 1.3, 8.3);
  check_make_with_value(8.3, tnsr::I<double, 3, Frame::Grid>(1.3), 8.3);
  check_make_with_value(tnsr::Ij<double, 3, Frame::Grid>(8.3),
                        tnsr::aB<double, 1, Frame::Inertial>(1.3), 8.3);

  for (size_t n_pts = 1; n_pts < 4; ++n_pts) {
    // create DataVector from DataVector
    check_make_with_value(DataVector(n_pts, -2.3), DataVector(n_pts, 4.5),
                          -2.3);
    check_make_with_value(Scalar<DataVector>(n_pts, -2.3),
                          DataVector(n_pts, 4.5), -2.3);
    check_make_with_value(tnsr::ab<DataVector, 2, Frame::Inertial>(n_pts, -2.3),
                          DataVector(n_pts, 4.5), -2.3);
    check_make_with_value(DataVector(n_pts, -2.3),
                          Scalar<DataVector>(n_pts, 4.5), -2.3);
    check_make_with_value(DataVector(n_pts, -2.3),
                          tnsr::ab<DataVector, 2, Frame::Inertial>(n_pts, 4.5),
                          -2.3);
    check_make_with_value(
        tnsr::ijj<DataVector, 2, Frame::Inertial>(n_pts, -2.3),
        tnsr::ab<DataVector, 3, Frame::Grid>(n_pts, 4.5), -2.3);
    check_make_with_value(
        tnsr::ijj<DataVector, 2, Frame::Inertial>(n_pts, -2.3),
        Scalar<DataVector>(n_pts, 4.5), -2.3);
    check_make_with_value(Scalar<DataVector>(n_pts, -2.3),
                          tnsr::abc<DataVector, 3, Frame::Inertial>(n_pts, 4.5),
                          -2.3);
    check_make_with_value(Variables<two_vars<3>>(n_pts, -2.3),
                          DataVector(n_pts, 4.5), -2.3);
    check_make_with_value(Variables<one_var<3>>(n_pts, -2.3),
                          DataVector(n_pts, 4.5), -2.3);
    check_make_with_value(Variables<two_vars<3>>(n_pts, -2.3),
                          tnsr::ab<DataVector, 3, Frame::Grid>(n_pts, 4.5),
                          -2.3);
    check_make_with_value(Variables<one_var<3>>(n_pts, -2.3),
                          Scalar<DataVector>(n_pts, 4.5), -2.3);
    check_make_with_value(Variables<one_var<3>>(n_pts, -2.3),
                          Variables<two_vars<3>>(n_pts, 4.5), -2.3);
    check_make_with_value(Variables<two_vars<3>>(n_pts, -2.3),
                          Variables<one_var<3>>(n_pts, 4.5), -2.3);
  }
  test_make_tagged_tuple();
}

namespace {
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
  swallow((check_random_values<decltype(get<Tags>(v))>(
               values, counter, get<Tags>(v), lower_bound, upper_bound),
           cpp17::void_type{})...);
}

template <typename T, typename U>
void test_make_with_random_values(const U& used_for_size) noexcept {
  std::random_device r;
  const auto seed = r();
  std::mt19937 generator(seed);
  INFO("seed = " << seed);
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
