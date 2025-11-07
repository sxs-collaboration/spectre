// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <random>

#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctions/Zernike.hpp"

namespace Spectral {
namespace {
void test_values() {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> x_distribution(0.0, 1.0);
  const double x = x_distribution(generator);
  CHECK(Zernike<2>::basis_function_value(0, 0, x) == approx(1.0));
  CHECK(Zernike<2>::basis_function_value(1, 1, x) == approx(x));
  CHECK(Zernike<2>::basis_function_value(2, 0, x) ==
        approx(2.0 * square(x) - 1.0));
  CHECK(Zernike<2>::basis_function_value(2, 2, x) == approx(square(x)));
  CHECK(Zernike<2>::basis_function_value(3, 1, x) ==
        approx(3.0 * cube(x) - 2.0 * x));
  CHECK(Zernike<2>::basis_function_value(3, 3, x) == approx(cube(x)));
  CHECK(Zernike<2>::basis_function_value(4, 0, x) ==
        approx(6.0 * pow<4>(x) - 6.0 * square(x) + 1.0));
  CHECK(Zernike<2>::basis_function_value(4, 2, x) ==
        approx(4.0 * pow<4>(x) - 3.0 * square(x)));
  CHECK(Zernike<2>::basis_function_value(4, 4, x) == approx(pow<4>(x)));
  CHECK(Zernike<2>::basis_function_value(5, 1, x) ==
        approx(10.0 * pow<5>(x) - 12.0 * cube(x) + 3.0 * x));
  CHECK(Zernike<2>::basis_function_value(5, 3, x) ==
        approx(5.0 * pow<5>(x) - 4.0 * cube(x)));
  CHECK(Zernike<2>::basis_function_value(5, 5, x) == approx(pow<5>(x)));
  CHECK(Zernike<2>::basis_function_value(6, 0, x) ==
        approx(20.0 * pow<6>(x) - 30.0 * pow<4>(x) + 12.0 * square(x) - 1.0));
  CHECK(Zernike<2>::basis_function_value(6, 2, x) ==
        approx(15.0 * pow<6>(x) - 20.0 * pow<4>(x) + 6.0 * square(x)));
  CHECK(Zernike<2>::basis_function_value(6, 4, x) ==
        approx(6.0 * pow<6>(x) - 5.0 * pow<4>(x)));
  CHECK(Zernike<2>::basis_function_value(6, 6, x) == approx(pow<6>(x)));
  CHECK(Zernike<2>::basis_function_value(7, 1, x) ==
        approx(35.0 * pow<7>(x) - 60.0 * pow<5>(x) + 30.0 * cube(x) - 4.0 * x));
  CHECK(Zernike<2>::basis_function_value(7, 3, x) ==
        approx(21.0 * pow<7>(x) - 30.0 * pow<5>(x) + 10.0 * cube(x)));
  CHECK(Zernike<2>::basis_function_value(7, 5, x) ==
        approx(7.0 * pow<7>(x) - 6.0 * pow<5>(x)));
  CHECK(Zernike<2>::basis_function_value(7, 7, x) == approx(pow<7>(x)));
  const Approx custom_approx = Approx::custom().epsilon(1.0e-11).scale(1.0);
  CHECK(Zernike<2>::basis_function_value(12, 0, x) ==
        custom_approx(924.0 * pow<12>(x) - 2772.0 * pow<10>(x) +
                      3150.0 * pow<8>(x) - 1680.0 * pow<6>(x) +
                      420.0 * pow<4>(x) - 42.0 * square(x) + 1.0));
  CHECK(Zernike<2>::basis_function_value(13, 1, x) ==
        custom_approx(1716.0 * pow<13>(x) - 5544.0 * pow<11>(x) +
                      6930.0 * pow<9>(x) - 4200.0 * pow<7>(x) +
                      1260.0 * pow<5>(x) - 168.0 * cube(x) + 7.0 * x));

  CHECK(Zernike<3>::basis_function_value(0, 0, x) == approx(1.0));
  CHECK(Zernike<3>::basis_function_value(1, 1, x) == approx(x));
  CHECK(Zernike<3>::basis_function_value(2, 0, x) ==
        approx(2.5 * square(x) - 1.5));
  CHECK(Zernike<3>::basis_function_value(2, 2, x) == approx(square(x)));
  CHECK(Zernike<3>::basis_function_value(3, 1, x) ==
        approx(3.5 * cube(x) - 2.5 * x));
  CHECK(Zernike<3>::basis_function_value(3, 3, x) == approx(cube(x)));
  CHECK(Zernike<3>::basis_function_value(4, 0, x) ==
        approx(63.0 / 8.0 * pow<4>(x) - 70.0 / 8.0 * square(x) + 15.0 / 8.0));
  CHECK(Zernike<3>::basis_function_value(4, 2, x) ==
        approx(4.5 * pow<4>(x) - 3.5 * square(x)));
  CHECK(Zernike<3>::basis_function_value(4, 4, x) == approx(pow<4>(x)));
  CHECK(Zernike<3>::basis_function_value(12, 0, x) ==
        custom_approx(
            1300075.0 / 1024.0 * pow<12>(x) - 4056234.0 / 1024.0 * pow<10>(x) +
            4849845.0 / 1024.0 * pow<8>(x) - 2771340.0 / 1024.0 * pow<6>(x) +
            765765.0 / 1024.0 * pow<4>(x) - 90090.0 / 1024.0 * square(x) +
            3003.0 / 1024.0));
  CHECK(Zernike<3>::basis_function_value(13, 1, x) ==
        custom_approx(
            2340135.0 / 1024.0 * pow<13>(x) - 7800450.0 / 1024.0 * pow<11>(x) +
            10140585.0 / 1024.0 * pow<9>(x) - 6466460.0 / 1024.0 * pow<7>(x) +
            2078505.0 / 1024.0 * pow<5>(x) - 306306.0 / 1024.0 * cube(x) +
            15015.0 / 1024.0 * x));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.BasisFunctions.Zernike",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  test_values();
}
}  // namespace Spectral
