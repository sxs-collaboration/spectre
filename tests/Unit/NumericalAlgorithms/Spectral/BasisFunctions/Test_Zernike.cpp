// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <random>

#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctions/Jacobi.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctions/Zernike.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace Spectral {
namespace {
// Copy of normalization of basis from Zernike.cpp's anonymous namespace
template <size_t Dim>
double I(const size_t n, const size_t m) {
  static_assert(Dim == 1 or Dim == 2 or Dim == 3);
  ASSERT(n >= m, "n = " << n << "; m = " << m);
  ASSERT((n + m) % 2 == 0,
         "n and m must have same parity, got n = " << n << " m = " << m);
  constexpr double betaM = Dim == 1 ? 0.0 : (Dim == 2 ? 1.0 : 2.0);
  double result = 0.5 / (static_cast<double>(m) + 0.5 + 0.5 * betaM);
  for (size_t i = m + 2; i <= n; i += 2) {
    const auto ii = static_cast<double>(i);
    result *= (2. * ii + betaM - 3.) / (2. * ii + betaM + 1.);
  }
  return result;
}

// The implemented Zernike basis is normalized, but we relate the basis to
// known (unnnormalized) Jacobi polynomials, so we have to undo this
// normalization to check against Jacobi implementation
template <size_t Dim>
double Zernike_basis_function_value_unnormalized(size_t n, size_t m,
                                                 const double xi) {
  return Zernike<Dim>::basis_function_value(n, m, xi) * sqrt(I<Dim>(n, m));
}

template <size_t Dim, typename T>
T Zernike_basis_function_value_Jacobi(const size_t n, const size_t m,
                                      const T& r) {
  static_assert(Dim == 1 or Dim == 2 or Dim == 3);
  ASSERT(n >= m, "m " << m << " must be at most n " << n);
  ASSERT((n + m) % 2 == 0, "m " << m << " plus n " << n << " must be even");
  const size_t k = (n - m) / 2;
  const auto mm = static_cast<double>(m);
  const double beta = Dim == 1 ? mm - 0.5 : Dim == 2 ? mm : mm + 0.5;
  const T x = 2.0 * square(r) - 1.0;
  T result = pow(r, mm);
  result *= Jacobi::basis_function_value(0.0, beta, k, x);
  return result;
}

void test_values() {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> r_distribution(0.0, 1.0);
  {
    INFO("Testing against analytic");
    const double r = r_distribution(generator);
    const double xi = 2.0 * r - 1.0;
    const Approx custom_approx = Approx::custom().epsilon(1.0e-11).scale(1.0);

    CHECK(Zernike_basis_function_value_unnormalized<1>(0, 0, xi) ==
          approx(1.0));
    CHECK(Zernike_basis_function_value_unnormalized<1>(1, 1, xi) == approx(r));
    CHECK(Zernike_basis_function_value_unnormalized<1>(2, 0, xi) ==
          approx(1.5 * square(r) - 0.5));
    CHECK(Zernike_basis_function_value_unnormalized<1>(2, 2, xi) ==
          approx(square(r)));
    CHECK(Zernike_basis_function_value_unnormalized<1>(3, 1, xi) ==
          approx(2.5 * cube(r) - 1.5 * r));
    CHECK(Zernike_basis_function_value_unnormalized<1>(3, 3, xi) ==
          approx(cube(r)));
    CHECK(Zernike_basis_function_value_unnormalized<1>(4, 0, xi) ==
          approx(35.0 / 8.0 * pow<4>(r) - 15.0 / 4.0 * square(r) + 3.0 / 8.0));
    CHECK(Zernike_basis_function_value_unnormalized<1>(4, 2, xi) ==
          approx(3.5 * pow<4>(r) - 2.5 * square(r)));
    CHECK(Zernike_basis_function_value_unnormalized<1>(4, 4, xi) ==
          approx(pow<4>(r)));
    CHECK(Zernike_basis_function_value_unnormalized<1>(11, 1, xi) ==
          custom_approx(
              88179.0 / 256.0 * pow<11>(r) - 230945.0 / 256.0 * pow<9>(r) +
              109395.0 / 128.0 * pow<7>(r) - 45045.0 / 128.0 * pow<5>(r) +
              15015.0 / 256.0 * cube(r) - 693.0 / 256.0 * r));
    CHECK(Zernike_basis_function_value_unnormalized<1>(12, 0, xi) ==
          custom_approx(
              676039.0 / 1024.0 * pow<12>(r) - 969969.0 / 512.0 * pow<10>(r) +
              2078505.0 / 1024.0 * pow<8>(r) - 255255.0 / 256.0 * pow<6>(r) +
              225225.0 / 1024.0 * pow<4>(r) - 9009.0 / 512.0 * square(r) +
              231.0 / 1024.0));

    CHECK(Zernike_basis_function_value_unnormalized<2>(0, 0, xi) ==
          approx(1.0));
    CHECK(Zernike_basis_function_value_unnormalized<2>(1, 1, xi) == approx(r));
    CHECK(Zernike_basis_function_value_unnormalized<2>(2, 0, xi) ==
          approx(2.0 * square(r) - 1.0));
    CHECK(Zernike_basis_function_value_unnormalized<2>(2, 2, xi) ==
          approx(square(r)));
    CHECK(Zernike_basis_function_value_unnormalized<2>(3, 1, xi) ==
          approx(3.0 * cube(r) - 2.0 * r));
    CHECK(Zernike_basis_function_value_unnormalized<2>(3, 3, xi) ==
          approx(cube(r)));
    CHECK(Zernike_basis_function_value_unnormalized<2>(4, 0, xi) ==
          approx(6.0 * pow<4>(r) - 6.0 * square(r) + 1.0));
    CHECK(Zernike_basis_function_value_unnormalized<2>(4, 2, xi) ==
          approx(4.0 * pow<4>(r) - 3.0 * square(r)));
    CHECK(Zernike_basis_function_value_unnormalized<2>(4, 4, xi) ==
          approx(pow<4>(r)));
    CHECK(Zernike_basis_function_value_unnormalized<2>(5, 1, xi) ==
          approx(10.0 * pow<5>(r) - 12.0 * cube(r) + 3.0 * r));
    CHECK(Zernike_basis_function_value_unnormalized<2>(5, 3, xi) ==
          approx(5.0 * pow<5>(r) - 4.0 * cube(r)));
    CHECK(Zernike_basis_function_value_unnormalized<2>(5, 5, xi) ==
          approx(pow<5>(r)));
    CHECK(Zernike_basis_function_value_unnormalized<2>(6, 0, xi) ==
          approx(20.0 * pow<6>(r) - 30.0 * pow<4>(r) + 12.0 * square(r) - 1.0));
    CHECK(Zernike_basis_function_value_unnormalized<2>(6, 2, xi) ==
          approx(15.0 * pow<6>(r) - 20.0 * pow<4>(r) + 6.0 * square(r)));
    CHECK(Zernike_basis_function_value_unnormalized<2>(6, 4, xi) ==
          approx(6.0 * pow<6>(r) - 5.0 * pow<4>(r)));
    CHECK(Zernike_basis_function_value_unnormalized<2>(6, 6, xi) ==
          approx(pow<6>(r)));
    CHECK(
        Zernike_basis_function_value_unnormalized<2>(7, 1, xi) ==
        approx(35.0 * pow<7>(r) - 60.0 * pow<5>(r) + 30.0 * cube(r) - 4.0 * r));
    CHECK(Zernike_basis_function_value_unnormalized<2>(7, 3, xi) ==
          approx(21.0 * pow<7>(r) - 30.0 * pow<5>(r) + 10.0 * cube(r)));
    CHECK(Zernike_basis_function_value_unnormalized<2>(7, 5, xi) ==
          approx(7.0 * pow<7>(r) - 6.0 * pow<5>(r)));
    CHECK(Zernike_basis_function_value_unnormalized<2>(7, 7, xi) ==
          approx(pow<7>(r)));
    CHECK(Zernike_basis_function_value_unnormalized<2>(12, 0, xi) ==
          custom_approx(924.0 * pow<12>(r) - 2772.0 * pow<10>(r) +
                        3150.0 * pow<8>(r) - 1680.0 * pow<6>(r) +
                        420.0 * pow<4>(r) - 42.0 * square(r) + 1.0));
    CHECK(Zernike_basis_function_value_unnormalized<2>(13, 1, xi) ==
          custom_approx(1716.0 * pow<13>(r) - 5544.0 * pow<11>(r) +
                        6930.0 * pow<9>(r) - 4200.0 * pow<7>(r) +
                        1260.0 * pow<5>(r) - 168.0 * cube(r) + 7.0 * r));

    CHECK(Zernike_basis_function_value_unnormalized<3>(0, 0, xi) ==
          approx(1.0));
    CHECK(Zernike_basis_function_value_unnormalized<3>(1, 1, xi) == approx(r));
    CHECK(Zernike_basis_function_value_unnormalized<3>(2, 0, xi) ==
          approx(2.5 * square(r) - 1.5));
    CHECK(Zernike_basis_function_value_unnormalized<3>(2, 2, xi) ==
          approx(square(r)));
    CHECK(Zernike_basis_function_value_unnormalized<3>(3, 1, xi) ==
          approx(3.5 * cube(r) - 2.5 * r));
    CHECK(Zernike_basis_function_value_unnormalized<3>(3, 3, xi) ==
          approx(cube(r)));
    CHECK(Zernike_basis_function_value_unnormalized<3>(4, 0, xi) ==
          approx(63.0 / 8.0 * pow<4>(r) - 70.0 / 8.0 * square(r) + 15.0 / 8.0));
    CHECK(Zernike_basis_function_value_unnormalized<3>(4, 2, xi) ==
          approx(4.5 * pow<4>(r) - 3.5 * square(r)));
    CHECK(Zernike_basis_function_value_unnormalized<3>(4, 4, xi) ==
          approx(pow<4>(r)));
    CHECK(Zernike_basis_function_value_unnormalized<3>(12, 0, xi) ==
          custom_approx(
              1300075.0 / 1024.0 * pow<12>(r) -
              4056234.0 / 1024.0 * pow<10>(r) + 4849845.0 / 1024.0 * pow<8>(r) -
              2771340.0 / 1024.0 * pow<6>(r) + 765765.0 / 1024.0 * pow<4>(r) -
              90090.0 / 1024.0 * square(r) + 3003.0 / 1024.0));
    CHECK(Zernike_basis_function_value_unnormalized<3>(13, 1, xi) ==
          custom_approx(2340135.0 / 1024.0 * pow<13>(r) -
                        7800450.0 / 1024.0 * pow<11>(r) +
                        10140585.0 / 1024.0 * pow<9>(r) -
                        6466460.0 / 1024.0 * pow<7>(r) +
                        2078505.0 / 1024.0 * pow<5>(r) -
                        306306.0 / 1024.0 * cube(r) + 15015.0 / 1024.0 * r));
  }
  {
    INFO("Testing against scaled Jacobi, all dimensions");
    const double r = r_distribution(generator);
    const double xi = 2.0 * r - 1.0;
    std::array<size_t, 8> all_k{};
    std::iota(all_k.begin(), all_k.end(), 2);
    for (const auto& k : random_sample<3>(all_k, make_not_null(&generator))) {
      for (size_t m = k % 2; m <= k; m += 2) {
        CHECK(Zernike_basis_function_value_unnormalized<1>(k, m, xi) ==
              approx(Zernike_basis_function_value_Jacobi<1>(k, m, r)));
        CHECK(Zernike_basis_function_value_unnormalized<2>(k, m, xi) ==
              approx(Zernike_basis_function_value_Jacobi<2>(k, m, r)));
        CHECK(Zernike_basis_function_value_unnormalized<3>(k, m, xi) ==
              approx(Zernike_basis_function_value_Jacobi<3>(k, m, r)));
      }
    }
  }
}

void test_errors() {
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      Zernike<1>::basis_function_value(2, 4, 1.0),
      Catch::Matchers::ContainsSubstring("m, 4, must be at most n, 2"));
  CHECK_THROWS_WITH(
      Zernike<2>::basis_function_value(4, 3, 1.0),
      Catch::Matchers::ContainsSubstring("n, 4, plus m, 3, must be even."));
#endif
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.BasisFunctions.Zernike",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  test_values();
  test_errors();
}
}  // namespace Spectral
