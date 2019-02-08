// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/TypeTraits.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.Math", "[Unit][Utilities]") {
  {
    INFO("Test number_of_digits");
    CHECK(2 == number_of_digits(10));
    CHECK(1 == number_of_digits(0));
    CHECK(1 == number_of_digits(-1));
    CHECK(1 == number_of_digits(9));
    CHECK(2 == number_of_digits(-99));
  }

  {
    INFO("Test evaluate_polynomial");
    const std::vector<double> poly_coeffs{1., 2.5, 0.3, 1.5};
    CHECK_ITERABLE_APPROX(evaluate_polynomial(poly_coeffs, 0.5), 2.5125);
    CHECK_ITERABLE_APPROX(
        evaluate_polynomial(poly_coeffs,
                            DataVector({-0.5, -0.1, 0., 0.8, 1., 12.})),
        DataVector({-0.3625, 0.7515, 1., 3.96, 5.3, 2666.2}));
    const std::vector<DataVector> poly_variable_coeffs{DataVector{1., 0., 2.},
                                                       DataVector{0., 2., 1.}};
    CHECK_ITERABLE_APPROX(
        evaluate_polynomial(poly_variable_coeffs, DataVector({0., 0.5, 1.})),
        DataVector({1., 1., 3.}));
  }

  {
    INFO("Test inverse roots and step_function");
    CHECK(step_function(1.0) == 1.0);
    CHECK(step_function(0.5) == 1.0);
    CHECK(step_function(-10) == 0);
    CHECK(step_function(0.0) == 1.0);
    CHECK(step_function(0) == 1);
    CHECK(invsqrt(4.0) == 0.5);
    CHECK(invsqrt(10) == 1.0 / sqrt(10));
    CHECK(approx(invcbrt(27.0)) == (1 / 3.0));
    CHECK(approx(invcbrt(1.0 / 64.0)) == 4.0);
  }

  {
    INFO("Test sign function");
    CHECK(sgn(2) == 1);
    CHECK(sgn(0) == 0);
    CHECK(sgn(-2) == -1);
    static_assert(cpp17::is_same_v<decltype(sgn(-2)), int>,
                  "Failed testing type of sgn");

    CHECK(sgn(2.14) == 1.0);
    CHECK(sgn(0.0) == 0.0);
    CHECK(sgn(-3.87) == -1.0);
    static_assert(cpp17::is_same_v<decltype(sgn(2.14)), double>,
                  "Failed testing type of sgn");

    CHECK(sgn(static_cast<unsigned>(2)) == 1);
    CHECK(sgn(static_cast<unsigned>(0)) == 0);
    CHECK(sgn(static_cast<unsigned>(-1)) == 1);
    static_assert(
        cpp17::is_same_v<decltype(sgn(static_cast<unsigned>(2))), unsigned>,
        "Failed testing type of sgn");
  }
}
