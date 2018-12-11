// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>

#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
// We wish to explicitly test implicit type conversion when adding std::arrays
// of different fundamentals, so we supress -Wsign-conversion.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include "Utilities/StdArrayHelpers.hpp"
#pragma GCC diagnostic pop

SPECTRE_TEST_CASE("Unit.Utilities.StdArrayHelpers.Arithmetic",
                  "[DataStructures][Unit]") {
  const size_t Dim = 3;

  std::array<double, Dim> p1{{2.3, -1.4, 0.2}};
  std::array<double, Dim> p2{{-12.4, 4.5, 2.6}};

  const std::array<double, Dim> expected_plus{{-10.1, 3.1, 2.8}};
  const std::array<double, Dim> expected_minus{{14.7, -5.9, -2.4}};

  const auto plus = p1 + p2;
  const auto minus = p1 - p2;

  for (size_t i = 0; i < Dim; ++i) {
    CHECK(gsl::at(plus, i) == approx(gsl::at(expected_plus, i)));
    CHECK(gsl::at(minus, i) == approx(gsl::at(expected_minus, i)));
  }

  p1 += expected_plus;
  p2 -= expected_minus;

  const std::array<double, Dim> expected_plus_equal{{-7.8, 1.7, 3.}};
  const std::array<double, Dim> expected_minus_equal{{-27.1, 10.4, 5.}};

  for (size_t i = 0; i < Dim; ++i) {
    CHECK(gsl::at(p1, i) == approx(gsl::at(expected_plus_equal, i)));
    CHECK(gsl::at(p2, i) == approx(gsl::at(expected_minus_equal, i)));
  }

  const double scale = -1.8;
  const auto left_scaled_array = scale * p1;
  const std::array<double, Dim> expected_left_scaled_array{
      {14.04, -3.06, -5.4}};
  const auto right_scaled_array = p1 * scale;
  const auto array_divided_by_double = p1 / (1 / scale);

  for (size_t i = 0; i < Dim; ++i) {
    CHECK(gsl::at(left_scaled_array, i) ==
          approx(gsl::at(expected_left_scaled_array, i)));
    CHECK(gsl::at(left_scaled_array, i) ==
          approx(gsl::at(right_scaled_array, i)));
    CHECK(gsl::at(left_scaled_array, i) ==
          approx(gsl::at(array_divided_by_double, i)));
  }

  const auto neg_p1 = -p1;
  const auto expected_neg_p1 = -1. * p1;
  for (size_t i = 0; i < Dim; ++i) {
    CHECK(gsl::at(neg_p1, i) == approx(gsl::at(expected_neg_p1, i)));
  }

  const std::array<double, 2> double_array{{2.2, -1.0}};
  const std::array<int, 2> int_array{{1, 3}};
  const std::array<size_t, 2> size_array{{2, 10}};
  const std::array<double, 2> double_plus_int{{3.2, 2.0}};
  const std::array<double, 2> double_plus_size{{4.2, 9.0}};
  const std::array<size_t, 2> int_plus_size{{3, 13}};
  const std::array<size_t, 2> size_minus_int{{1, 7}};
  CHECK(double_array + int_array == double_plus_int);
  CHECK(double_array + size_array == double_plus_size);
  CHECK(int_array + size_array == int_plus_size);
  CHECK(int_array + double_array == double_plus_int);
  CHECK(size_array + double_array == double_plus_size);
  CHECK(size_array + int_array == int_plus_size);
  CHECK(size_array - -int_array == int_plus_size);
  CHECK(size_array - -double_array == double_plus_size);
  CHECK(size_array - int_array == size_minus_int);
}

SPECTRE_TEST_CASE("Unit.Utilities.StdArrayHelpers.Magnitude",
                  "[DataStructures][Unit]") {
  std::array<double, 1> p1{{-2.5}};
  CHECK(2.5 == magnitude(p1));
  const std::array<double, 2> p2{{3., -4.}};
  CHECK(magnitude(p2) == approx(5.));
  const std::array<double, 3> p3{{-2., 10., 11.}};
  CHECK(magnitude(p3) == approx(15.));
}

SPECTRE_TEST_CASE("Unit.Utilities.StdArrayHelpers.AllButSpecifiedElementOf",
                  "[DataStructures][Unit]") {
  const std::array<size_t, 3> a3{{5, 2, 3}};
  const std::array<size_t, 2> a2{{2, 3}};
  const std::array<size_t, 1> a1{{3}};
  const std::array<size_t, 0> a0{{}};
  CHECK(a2 == all_but_specified_element_of(a3, 0));
  CHECK(a1 == all_but_specified_element_of(a2, 0));
  CHECK(a0 == all_but_specified_element_of(a1, 0));
  const std::array<size_t, 2> b2{{5, 3}};
  const std::array<size_t, 1> b1{{5}};
  auto c2 = all_but_specified_element_of(a3, 1);
  CHECK(b2 == c2);
  auto c1 = all_but_specified_element_of(b2, 1);
  CHECK(b1 == c1);
  CHECK(a0 == all_but_specified_element_of(b1, 0));
}

SPECTRE_TEST_CASE("Unit.Utilities.StdArrayHelpers.InsertElement",
                  "[DataStructures][Unit]") {
  const std::array<int, 0> a0{{}};
  const std::array<int, 1> a1{{3}};
  const std::array<int, 2> a2{{2, 3}};
  const std::array<int, 3> a3{{2, 4, 3}};
  CHECK(insert_element(a0, 0, 3) == a1);
  CHECK(insert_element(a1, 0, 2) == a2);
  CHECK(insert_element(a2, 1, 4) == a3);
}

SPECTRE_TEST_CASE("Unit.Utilities.StdArrayHelpers.Prepend",
                  "[Utilities][Unit]") {
  const std::array<size_t, 3> a3{{5, 2, 3}};
  const std::array<size_t, 2> a2{{2, 3}};
  const std::array<size_t, 1> a1{{3}};
  const std::array<size_t, 0> a0{{}};
  const auto p1 = prepend(a0, 3_st);
  CHECK(p1 == a1);
  const auto p2 = prepend(a1, 2_st);
  CHECK(p2 == a2);
  const auto p3 = prepend(a2, 5_st);
  CHECK(p3 == a3);
}

SPECTRE_TEST_CASE("Unit.Utilities.StdArrayHelpers.map_array",
                  "[Utilities][Unit]") {
  const auto func = [](const int x) noexcept { return 2. * x; };
  CHECK(map_array(std::array<int, 3>{{4, 3, 2}}, func) ==
        std::array<double, 3>{{8, 6, 4}});
  CHECK(map_array(std::array<int, 1>{{4}}, func) == std::array<double, 1>{{8}});
  CHECK(map_array(std::array<int, 0>{}, func) == std::array<double, 0>{});

  // Check function returning a reference
  CHECK(
      map_array(std::array<int, 1>{{4}}, [](const int& x) noexcept->const int& {
        return x;
      }) == std::array<int, 1>{{4}});
}

namespace {
DEFINE_STD_ARRAY_BINOP(int, int, int, operator+, std::plus<>())
DEFINE_STD_ARRAY_BINOP(double, double, double, f, std::multiplies<>())

DEFINE_STD_ARRAY_INPLACE_BINOP(int, int, operator+=, std::plus<>())
DEFINE_STD_ARRAY_INPLACE_BINOP(double, double, g, std::multiplies<>())

SPECTRE_TEST_CASE("Unit.Utilities.StdArrayHelpers.define_std_array_binop",
                  "[Utilities][Unit]") {
  // tests DEFINE_STD_ARRAY_BINOP
  CHECK(std::array<int, 2>{{1, 2}} + std::array<int, 2>{{2, 3}} ==
        std::array<int, 2>{{3, 5}});
  CHECK(
      f(std::array<double, 2>{{1.0, 2.0}}, std::array<double, 2>{{2.0, 3.0}}) ==
      std::array<double, 2>{{2.0, 6.0}});

  // tests DEFINE_STD_ARRAY_INPLACE_BINOP
  std::array<int, 2> a{{1, 2}};
  CHECK((a += std::array<int, 2>{{2, 3}}) == std::array<int, 2>{{3, 5}});
  CHECK(a == std::array<int, 2>{{3, 5}});
  std::array<double, 2> b{{1.0, 2.0}};
  CHECK(g(b, std::array<double, 2>{{2.0, 3.0}}) ==
        std::array<double, 2>{{2.0, 6.0}});
  CHECK(b == std::array<double, 2>{{2.0, 6.0}});
}
}  // namespace
