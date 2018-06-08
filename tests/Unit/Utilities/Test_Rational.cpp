// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <functional>
#include <string>

#include "ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Rational.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.Rational", "[Unit][Utilities]") {
  CHECK(Rational(6, 8).numerator() == 3);
  CHECK(Rational(6, 8).denominator() == 4);
  CHECK(Rational(-6, 8).numerator() == -3);
  CHECK(Rational(-6, 8).denominator() == 4);
  CHECK(Rational(6, -8).numerator() == -3);
  CHECK(Rational(6, -8).denominator() == 4);
  CHECK(Rational(-6, -8).numerator() == 3);
  CHECK(Rational(-6, -8).denominator() == 4);

  CHECK(Rational(0, 2).numerator() == 0);
  CHECK(Rational(0, 2).denominator() == 1);
  CHECK(Rational(0, -2).numerator() == 0);
  CHECK(Rational(0, -2).denominator() == 1);

  CHECK(Rational(3, 4).value() == 0.75);
  CHECK(Rational(-3, 2).value() == -1.5);
  CHECK(Rational(1, 3).value() == approx(1./3.));

  check_cmp(Rational(3, 4), Rational(5, 4));
  check_cmp(Rational(3, 5), Rational(3, 4));

  CHECK(-Rational(3, 4) == Rational(-3, 4));
  CHECK(-Rational(-3, 4) == Rational(3, 4));

  CHECK(Rational(3, 4).inverse() == Rational(4, 3));
  CHECK(Rational(-3, 4).inverse() == Rational(-4, 3));

  CHECK_OP(Rational(3, 4), +, Rational(2, 5), Rational(23, 20));
  CHECK_OP(Rational(3, 4), +, Rational(7, 4), Rational(5, 2));
  CHECK_OP(Rational(3, 4), -, Rational(2, 5), Rational(7, 20));
  CHECK_OP(Rational(3, 4), -, Rational(7, 4), Rational(-1));
  CHECK_OP(Rational(3, 4), *, Rational(5, 2), Rational(15, 8));
  CHECK_OP(Rational(3, 4), *, Rational(2, 3), Rational(1, 2));
  CHECK_OP(Rational(3, 4), /, Rational(2, 5), Rational(15, 8));
  CHECK_OP(Rational(3, 4), /, Rational(3, 2), Rational(1, 2));

  const auto hash = std::hash<Rational>{};
  CHECK(hash(Rational(1, 2)) != hash(Rational(1, 3)));
  CHECK(hash(Rational(1, 2)) != hash(Rational(3, 2)));
  CHECK(hash(Rational(1, 2)) != hash(Rational(-1, 2)));

  CHECK(get_output(Rational(3, 4)) == "3/4");
  CHECK(get_output(Rational(-3, 4)) == "-3/4");

  test_serialization(Rational(3, 4));
}

SPECTRE_TEST_CASE("Unit.Utilities.Rational.internal_overflow",
                  "[Unit][Utilities]") {
  CHECK(Rational(6000000, 8000000) == Rational(3, 4));

  check_cmp(Rational(5, 1 << 16), Rational(1 << 16));

  CHECK_OP(Rational((1 << 21) + 1, 1 << 17), +,
           Rational((1 << 22) - 1, 1 << 18), Rational((1 << 23) + 1, 1 << 18));
  CHECK_OP(Rational((1 << 21) + 1, 1 << 17), -,
           Rational((1 << 22) - 1, 1 << 18), Rational(3, 1 << 18));
  CHECK_OP(Rational((1 << 20) - 1, 1 << 20), *,
           Rational(1 << 17, (1 << 10) + 1), Rational((1 << 10) - 1, 1 << 3));
  CHECK_OP(Rational((1 << 20) - 1, 1 << 20), /,
           Rational((1 << 10) + 1, 1 << 17), Rational((1 << 10) - 1, 1 << 3));
}

// [[OutputRegex, Division by zero]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Utilities.Rational.denominator_zero",
                               "[Unit][Utilities]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Rational(1, 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Division by zero]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Utilities.Rational.invert_zero",
                               "[Unit][Utilities]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Rational(0).inverse();
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Division by zero]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Utilities.Rational.divide_zero",
                               "[Unit][Utilities]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Rational(1) / 0;
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Division by zero]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Utilities.Rational.divide_equal_zero",
                               "[Unit][Utilities]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Rational r(1);
  r /= 0;
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Rational overflow: 1000000000000/1]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Utilities.Rational.overflow_numerator",
                               "[Unit][Utilities]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Rational(1000000) * Rational(1000000);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Rational overflow: 1/1000000000000]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Utilities.Rational.overflow_denominator",
                               "[Unit][Utilities]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Rational(1, 1000000) * Rational(1, 1000000);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
