// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>

#include "Utilities/Functional.hpp"

namespace funcl {
namespace {
void test_divides() {
  CHECK(Divides<>{}(1, -2) == 0);
  CHECK(Divides<>{}(1, -2.0) == -0.5);
}
void test_get_argument() {
  CHECK(GetArgument<1>{}(-2) == -2);
  CHECK(GetArgument<1, 0>{}(-2) == -2);
  CHECK(GetArgument<2, 0>{}(-2, 1) == -2);
  CHECK(GetArgument<2, 1>{}(-2, 8) == 8);
  CHECK(GetArgument<3, 0>{}(-2, 8, -10) == -2);
  CHECK(GetArgument<3, 1>{}(-2, 8, -10) == 8);
  CHECK(GetArgument<3, 2>{}(-2, 8, -10) == -10);
}
void test_identity() {
  CHECK(Identity{}(-2) == -2);
  CHECK(Identity{}(-2.0) == -2.0);
}
void test_minus() {
  CHECK(Minus<>{}(1, -2) == 3);
  CHECK(Minus<>{}(1, -2.0) == 3.0);
}
void test_multiplies() {
  CHECK(Multiplies<>{}(1, -2) == -2);
  CHECK(Multiplies<>{}(1, -2.0) == -2.0);
}
void test_negate() {
  CHECK(Negate<>{}(10) == -10);
  CHECK(Negate<>{}(-10.0) == 10.0);
}
void test_plus() {
  CHECK(Plus<>{}(1, -2) == -1);
  CHECK(Plus<>{}(1, -2.0) == -1.0);
}
void test_sqrt() {
  CHECK(Sqrt<>{}(3.0) == sqrt(3.0));
}
void test_square() {
  CHECK(Square<>{}(3.0) == 9.0);
  CHECK(Square<>{}(-3.0) == 9.0);
}

void test_composition() {
  CHECK(Plus<Negate<>>{}(2.0, 7.0) == -9.0);
  CHECK(Plus<Negate<>, Identity>{}(2.0, 7.0) == 5.0);
  CHECK(Negate<Plus<>>{}(2.0, 7.0) == -9.0);
  CHECK(Multiplies<Plus<Plus<>, Plus<>>, Identity>{}(1, 2, 3, 4, 5) == 50);
  CHECK(Minus<Plus<Identity, Multiplies<>>, Identity>{}(1, 2, 3, 4) == 3);
  CHECK(Negate<Plus<Identity, Negate<>>>{}(2.0, 7.0) == 5.0);
  CHECK(Plus<Identity, Multiplies<>>{}(1, 2, 3) == 7);
}

SPECTRE_TEST_CASE("Unit.Utilities.Functional", "[Unit][Utilities]") {
  test_divides();
  test_get_argument();
  test_identity();
  test_minus();
  test_multiplies();
  test_negate();
  test_plus();
  test_sqrt();
  test_square();

  test_composition();
}
}  // namespace
}  // namespace funcl
