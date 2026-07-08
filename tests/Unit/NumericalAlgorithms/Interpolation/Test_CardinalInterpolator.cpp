// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <random>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/NumericalAlgorithms/Spectral/BallTestFunctions.hpp"
#include "Helpers/NumericalAlgorithms/Spectral/DiskTestFunctions.hpp"
#include "Helpers/NumericalAlgorithms/SphericalHarmonics/YlmTestFunctions.hpp"
#include "NumericalAlgorithms/Interpolation/CardinalInterpolator.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Math.hpp"

namespace {
// Polynomial of given degree with leading coefficinet \f$a_0\f$, and with
// \f$a_{n+1} = a_n / falloff
class Polynomial {
 public:
  Polynomial(const size_t degree, const double a_0, const double falloff)
      : coefficients_(degree + 1) {
    double n = falloff * a_0;
    std::generate(coefficients_.begin(), coefficients_.end(), [&falloff, &n]() {
      n /= falloff;
      return n;
    });
  }
  Polynomial() = default;
  DataVector operator()(const DataVector& x) const {
    return evaluate_polynomial(coefficients_, x);
  }

 private:
  std::vector<double> coefficients_;
};

template <size_t Dim>
class ProductOfPolynomials {
 public:
  ProductOfPolynomials(const std::array<size_t, Dim>& degree,
                       const std::array<double, Dim>& a_0,
                       const std::array<double, Dim>& falloff) {
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(polynomials_, d) =
          Polynomial{gsl::at(degree, d), gsl::at(a_0, d), gsl::at(falloff, d)};
    }
  }
  DataVector operator()(
      const tnsr::I<DataVector, Dim, Frame::ElementLogical>& x) const {
    DataVector result = polynomials_[0](get<0>(x));
    for (size_t d = 1; d < Dim; ++d) {
      result *= gsl::at(polynomials_, d)(x.get(d));
    }
    return result;
  }

 private:
  std::array<Polynomial, Dim> polynomials_;
};

void test_1d(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);
  for (size_t n_target_points = 1; n_target_points < 101;
       n_target_points += 11) {
    const auto xi_target =
        make_with_random_values<tnsr::I<DataVector, 1, Frame::ElementLogical>>(
            generator, make_not_null(&xi_distribution), n_target_points);
    const tnsr::I<double, 1, Frame::ElementLogical> xi_target_single{
        {{get<0>(xi_target)[0]}}};
    for (const auto basis :
         std::array{Spectral::Basis::Legendre, Spectral::Basis::Chebyshev}) {
      for (const auto quadrature :
           std::array{Spectral::Quadrature::Gauss,
                      Spectral::Quadrature::GaussLobatto}) {
        for (size_t n_xi = 2; n_xi < 21; ++n_xi) {
          const Mesh<1> source_mesh{n_xi, basis, quadrature};
          const Polynomial f{n_xi - 1, 1.5, 2.0};
          const auto xi_source = logical_coordinates(source_mesh);
          const DataVector f_source = f(get<0>(xi_source));
          const DataVector f_expected = f(get<0>(xi_target));
          {
            const intrp::Cardinal<1> interpolator(source_mesh, xi_target);
            const DataVector f_interpolated =
                interpolator.interpolate(f_source);
            CHECK_ITERABLE_APPROX(f_interpolated, f_expected);
          }
          if (n_target_points == 1) {
            const intrp::Cardinal<1> interpolator(source_mesh,
                                                  xi_target_single);
            const DataVector f_interpolated =
                interpolator.interpolate(f_source);
            CHECK(f_interpolated.size() == 1);
            CHECK(f_interpolated[0] == approx(f_expected[0]));
          }
        }
      }
    }
  }
}

void test_2d_cartesian(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);
  const auto bases =
      std::array{Spectral::Basis::Legendre, Spectral::Basis::Chebyshev};
  const auto quadratures = std::array{Spectral::Quadrature::Gauss,
                                      Spectral::Quadrature::GaussLobatto};
  for (size_t n_target_points = 1; n_target_points < 13;
       n_target_points += 11) {
    const auto xi_target =
        make_with_random_values<tnsr::I<DataVector, 2, Frame::ElementLogical>>(
            generator, make_not_null(&xi_distribution), n_target_points);
    const tnsr::I<double, 2, Frame::ElementLogical> xi_target_single{
        {{get<0>(xi_target)[0], get<1>(xi_target)[0]}}};
    for (const auto xi_basis : bases) {
      for (const auto xi_quadrature : quadratures) {
        for (const auto eta_basis : bases) {
          for (const auto eta_quadrature : quadratures) {
            for (size_t n_xi = 2; n_xi < 21; n_xi += 3) {
              for (size_t n_eta = 2; n_eta < 21; n_eta += 3) {
                const Mesh<2> source_mesh{
                    std::array{n_xi, n_eta}, std::array{xi_basis, eta_basis},
                    std::array{xi_quadrature, eta_quadrature}};
                const ProductOfPolynomials<2> f{std::array{n_xi - 1, n_eta - 1},
                                                std::array{1.5, 2.5},
                                                std::array{2.0, 4.0}};
                const auto xi_source = logical_coordinates(source_mesh);
                const DataVector f_source = f(xi_source);
                const DataVector f_expected = f(xi_target);
                {
                  const intrp::Cardinal<2> interpolator(source_mesh, xi_target);
                  const DataVector f_interpolated =
                      interpolator.interpolate(f_source);
                  CHECK_ITERABLE_APPROX(f_interpolated, f_expected);
                }
                if (n_target_points == 1) {
                  const intrp::Cardinal<2> interpolator(source_mesh,
                                                        xi_target_single);
                  const DataVector f_interpolated =
                      interpolator.interpolate(f_source);
                  CHECK(f_interpolated.size() == 1);
                  CHECK(f_interpolated[0] == approx(f_expected[0]));
                }
              }
            }
          }
        }
      }
    }
  }
}

void test_2d_spherical(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);
  std::uniform_real_distribution<> phi_distribution(0.0, 2.0 * M_PI);
  for (size_t n_target_points = 1; n_target_points < 13;
       n_target_points += 11) {
    tnsr::I<DataVector, 2, Frame::ElementLogical> xi_target{n_target_points};
    get<0>(xi_target) = acos(make_with_random_values<DataVector>(
        generator, make_not_null(&xi_distribution), xi_target));
    get<1>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&phi_distribution), xi_target);
    const tnsr::I<double, 2, Frame::ElementLogical> xi_target_single{
        {{get<0>(xi_target)[0], get<1>(xi_target)[0]}}};
    for (size_t n_z = 0; n_z < 4; ++n_z) {
      for (size_t n_y = 0; n_y < 4; ++n_y) {
        for (size_t n_x = 0; n_x < 4; ++n_x) {
          const Mesh<2> source_mesh{
              std::array{n_x + n_y + n_z + 2, 2 * (n_x + n_y) + 3},
              std::array{Spectral::Basis::SphericalHarmonic,
                         Spectral::Basis::SphericalHarmonic},
              std::array{Spectral::Quadrature::Gauss,
                         Spectral::Quadrature::Equiangular}};
          const YlmTestFunctions::ProductOfPolynomials f(n_x, n_y, n_z);
          const auto xi_source = logical_coordinates(source_mesh);
          const DataVector f_source = f(xi_source);
          const DataVector f_expected = f(xi_target);
          {
            const intrp::Cardinal<2> interpolator(source_mesh, xi_target);
            const DataVector f_interpolated =
                interpolator.interpolate(f_source);
            CHECK_ITERABLE_APPROX(f_interpolated, f_expected);
          }
          if (n_target_points == 1) {
            const intrp::Cardinal<2> interpolator(source_mesh,
                                                  xi_target_single);
            const DataVector f_interpolated =
                interpolator.interpolate(f_source);
            CHECK(f_interpolated.size() == 1);
            CHECK(f_interpolated[0] == approx(f_expected[0]));
          }
        }
      }
    }
  }
}

void test_3d_cartesian(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);
  for (size_t n_target_points = 1; n_target_points < 13;
       n_target_points += 11) {
    const auto xi_target =
        make_with_random_values<tnsr::I<DataVector, 3, Frame::ElementLogical>>(
            generator, make_not_null(&xi_distribution), n_target_points);
    const tnsr::I<double, 3, Frame::ElementLogical> xi_target_single{
        {{get<0>(xi_target)[0], get<1>(xi_target)[0], get<2>(xi_target)[0]}}};
    for (size_t n_xi = 2; n_xi < 21; n_xi += 3) {
      for (size_t n_eta = 2; n_eta < 21; n_eta += 3) {
        for (size_t n_zeta = 2; n_zeta < 21; n_zeta += 3) {
          const Mesh<3> source_mesh{std::array{n_xi, n_eta, n_zeta},
                                    Spectral::Basis::Legendre,
                                    Spectral::Quadrature::GaussLobatto};
          const ProductOfPolynomials<3> f{
              std::array{n_xi - 1, n_eta - 1, n_zeta - 1},
              std::array{1.5, 2.5, 3.5}, std::array{2.0, 4.0, 2.5}};
          const auto xi_source = logical_coordinates(source_mesh);
          const DataVector f_source = f(xi_source);
          const DataVector f_expected = f(xi_target);
          {
            const intrp::Cardinal<3> interpolator(source_mesh, xi_target);
            const DataVector f_interpolated =
                interpolator.interpolate(f_source);
            CHECK_ITERABLE_APPROX(f_interpolated, f_expected);
          }
          if (n_target_points == 1) {
            const intrp::Cardinal<3> interpolator(source_mesh,
                                                  xi_target_single);
            const DataVector f_interpolated =
                interpolator.interpolate(f_source);
            CHECK(f_interpolated.size() == 1);
            CHECK(f_interpolated[0] == approx(f_expected[0]));
          }
        }
      }
    }
  }
}

void test_3d_spherical(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);
  std::uniform_real_distribution<> phi_distribution(0.0, 2.0 * M_PI);
  for (size_t n_target_points = 1; n_target_points < 13;
       n_target_points += 11) {
    tnsr::I<DataVector, 3, Frame::ElementLogical> xi_target{n_target_points};
    get<0>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&xi_distribution), xi_target);
    get<1>(xi_target) = acos(make_with_random_values<DataVector>(
        generator, make_not_null(&xi_distribution), xi_target));
    get<2>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&phi_distribution), xi_target);
    const tnsr::I<double, 3, Frame::ElementLogical> xi_target_single{
        {{get<0>(xi_target)[0], get<1>(xi_target)[0], get<2>(xi_target)[0]}}};
    for (size_t n_r = 2; n_r < 4; ++n_r) {
      for (size_t n_z = 0; n_z < 4; ++n_z) {
        for (size_t n_y = 0; n_y < 4; ++n_y) {
          for (size_t n_x = 0; n_x < 4; ++n_x) {
            const Mesh<3> source_mesh{
                std::array{n_r, n_x + n_y + n_z + 2, 2 * (n_x + n_y) + 3},
                std::array{Spectral::Basis::Legendre,
                           Spectral::Basis::SphericalHarmonic,
                           Spectral::Basis::SphericalHarmonic},
                std::array{Spectral::Quadrature::GaussLobatto,
                           Spectral::Quadrature::Gauss,
                           Spectral::Quadrature::Equiangular}};
            const Polynomial f_r{n_r - 1, 1.5, 2.0};
            const YlmTestFunctions::ProductOfPolynomials f_a(n_x, n_y, n_z);
            const auto xi_source = logical_coordinates(source_mesh);
            const DataVector f_source =
                f_r(get<0>(xi_source)) *
                f_a(get<1>(xi_source), get<2>(xi_source));
            const DataVector f_expected =
                f_r(get<0>(xi_target)) *
                f_a(get<1>(xi_target), get<2>(xi_target));
            {
              const intrp::Cardinal<3> interpolator(source_mesh, xi_target);
              const DataVector f_interpolated =
                  interpolator.interpolate(f_source);
              CHECK_ITERABLE_APPROX(f_interpolated, f_expected);
            }
            if (n_target_points == 1) {
              const intrp::Cardinal<3> interpolator(source_mesh,
                                                    xi_target_single);
              const DataVector f_interpolated =
                  interpolator.interpolate(f_source);
              CHECK(f_interpolated.size() == 1);
              CHECK(f_interpolated[0] == approx(f_expected[0]));
            }
          }
        }
      }
    }
  }
}

void test_2d_disk(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);
  std::uniform_real_distribution<> phi_distribution(0.0, 2.0 * M_PI);
  for (size_t n_target_points = 1; n_target_points < 13;
       n_target_points += 11) {
    tnsr::I<DataVector, 2, Frame::ElementLogical> xi_target{n_target_points};
    get<0>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&xi_distribution), xi_target);
    get<1>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&phi_distribution), xi_target);
    const tnsr::I<double, 2, Frame::ElementLogical> xi_target_single{
        {{get<0>(xi_target)[0], get<1>(xi_target)[0]}}};
    for (size_t n_y = 0; n_y < 4; ++n_y) {
      for (size_t n_x = 0; n_x < 4; ++n_x) {
        const Mesh<2> source_mesh{
            n_x + n_y == 0 ? std::array{1_st, 1_st}
                           : std::array{n_x + n_y + 1, 2 * (n_x + n_y) + 1},
            std::array{Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2},
            std::array{Spectral::Quadrature::GaussRadauUpper,
                       Spectral::Quadrature::Equiangular}};
        const DiskTestFunctions::ProductOfPolynomials f{n_x, n_y};
        const auto xi_source = logical_coordinates(source_mesh);
        const DataVector f_source =
            f(0.5 * (get<0>(xi_source) + 1.0), get<1>(xi_source));
        const DataVector f_expected =
            f(0.5 * (get<0>(xi_target) + 1.0), get<1>(xi_target));
        {
          const intrp::Cardinal<2> interpolator(source_mesh, xi_target);
          const DataVector f_interpolated = interpolator.interpolate(f_source);
          CHECK_ITERABLE_APPROX(f_interpolated, f_expected);
        }
        if (n_target_points == 1) {
          const intrp::Cardinal<2> interpolator(source_mesh, xi_target_single);
          const DataVector f_interpolated = interpolator.interpolate(f_source);
          CHECK(f_interpolated.size() == 1);
          CHECK(f_interpolated[0] == approx(f_expected[0]));
        }
      }
    }
  }
}

void test_3d_cylinder(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);
  std::uniform_real_distribution<> phi_distribution(0.0, 2.0 * M_PI);
  for (size_t n_target_points = 1; n_target_points < 13;
       n_target_points += 11) {
    tnsr::I<DataVector, 3, Frame::ElementLogical> xi_target{n_target_points};
    get<0>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&xi_distribution), xi_target);
    get<1>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&phi_distribution), xi_target);
    get<2>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&xi_distribution), xi_target);
    const tnsr::I<double, 3, Frame::ElementLogical> xi_target_single{
        {{get<0>(xi_target)[0], get<1>(xi_target)[0], get<2>(xi_target)[0]}}};
    for (size_t n_z = 2; n_z < 4; ++n_z) {
      for (size_t n_y = 0; n_y < 4; ++n_y) {
        for (size_t n_x = 0; n_x < 4; ++n_x) {
          const Mesh<3> source_mesh{
              n_x + n_y == 0
                  ? std::array{1_st, 1_st, n_z}
                  : std::array{n_x + n_y + 1, 2 * (n_x + n_y) + 1, n_z},
              std::array{Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2,
                         Spectral::Basis::Legendre},
              std::array{Spectral::Quadrature::GaussRadauUpper,
                         Spectral::Quadrature::Equiangular,
                         Spectral::Quadrature::GaussLobatto}};
          const DiskTestFunctions::ProductOfPolynomials f{n_x, n_y};
          const Polynomial f_z{n_z - 1, 1.5, 2.0};
          const auto xi_source = logical_coordinates(source_mesh);
          const DataVector f_source =
              f(0.5 * (get<0>(xi_source) + 1.0), get<1>(xi_source)) *
              f_z(get<2>(xi_source));
          const DataVector f_expected =
              f(0.5 * (get<0>(xi_target) + 1.0), get<1>(xi_target)) *
              f_z(get<2>(xi_target));
          {
            const intrp::Cardinal<3> interpolator(source_mesh, xi_target);
            const DataVector f_interpolated =
                interpolator.interpolate(f_source);
            CHECK_ITERABLE_APPROX(f_interpolated, f_expected);
          }
          if (n_target_points == 1) {
            const intrp::Cardinal<3> interpolator(source_mesh,
                                                  xi_target_single);
            const DataVector f_interpolated =
                interpolator.interpolate(f_source);
            CHECK(f_interpolated.size() == 1);
            CHECK(f_interpolated[0] == approx(f_expected[0]));
          }
        }
      }
    }
  }
}

void test_3d_ball(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);
  std::uniform_real_distribution<> phi_distribution(0.0, 2.0 * M_PI);
  for (const size_t n_target_points : {1_st, 12_st}) {
    tnsr::I<DataVector, 3, Frame::ElementLogical> xi_target{n_target_points};
    get<0>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&xi_distribution), xi_target);
    get<1>(xi_target) = acos(make_with_random_values<DataVector>(
        generator, make_not_null(&xi_distribution), xi_target));
    get<2>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&phi_distribution), xi_target);
    const tnsr::I<double, 3, Frame::ElementLogical> xi_target_single{
        {{get<0>(xi_target)[0], get<1>(xi_target)[0], get<2>(xi_target)[0]}}};
    for (size_t n_z = 2; n_z < 4; ++n_z) {
      CAPTURE(n_z);
      for (size_t n_y = 0; n_y < 4; ++n_y) {
        CAPTURE(n_y);
        for (size_t n_x = 0; n_x < 4; ++n_x) {
          CAPTURE(n_x);
          // SPHEREPACK requires m_max >= 2 (n_phi >= 5)
          const size_t n_phi = 2 * (n_x + n_y) + 3;
          if (n_phi < 5) {
            continue;
          }
          const size_t n_theta = n_x + n_y + n_z + 2;
          const Mesh<3> source_mesh{
              std::array{n_theta / 2 + 1, n_theta, n_phi},
              std::array{Spectral::Basis::ZernikeB3, Spectral::Basis::ZernikeB3,
                         Spectral::Basis::ZernikeB3},
              std::array{Spectral::Quadrature::GaussRadauUpper,
                         Spectral::Quadrature::Gauss,
                         Spectral::Quadrature::Equiangular}};
          const BallTestFunctions::ProductOfPolynomials f{n_x, n_y, n_z};
          const auto xi_source = logical_coordinates(source_mesh);
          const DataVector f_source = f(0.5 * (get<0>(xi_source) + 1.0),
                                        get<1>(xi_source), get<2>(xi_source));
          const DataVector f_expected = f(0.5 * (get<0>(xi_target) + 1.0),
                                          get<1>(xi_target), get<2>(xi_target));
          {
            const intrp::Cardinal<3> interpolator(source_mesh, xi_target);
            const DataVector f_interpolated =
                interpolator.interpolate(f_source);
            CHECK_ITERABLE_APPROX(f_interpolated, f_expected);
            {
              INFO("Testing B3 serialization");
              // The fields that are handled in complex ways by the pup function
              // are excluded from the equality operator
              const DataVector f_pup_interpolated =
                  serialize_and_deserialize(interpolator).interpolate(f_source);
              CHECK_ITERABLE_APPROX(f_pup_interpolated, f_expected);
            }
          }
          if (n_target_points == 1) {
            const intrp::Cardinal<3> interpolator(source_mesh,
                                                  xi_target_single);
            const DataVector f_interpolated =
                interpolator.interpolate(f_source);
            CHECK(f_interpolated.size() == 1);
            CHECK(f_interpolated[0] == approx(f_expected[0]));
          }
        }
      }
    }
  }
}

void test_errors() {
  {
    INFO("Testing SphericalHarmonic with unsupported quadrature");
    CHECK_THROWS_WITH(
        (intrp::Cardinal<2>{Mesh<2>{{3, 3},
                                    {Spectral::Basis::SphericalHarmonic,
                                     Spectral::Basis::SphericalHarmonic},
                                    {Spectral::Quadrature::GaussLobatto,
                                     Spectral::Quadrature::Equiangular}},
                            tnsr::I<DataVector, 2, Frame::ElementLogical>{
                                {{{1.0, 2.0}, {0.5, 1.5}}}}}),
        Catch::Matchers::ContainsSubstring(
            "Quadrature must be Gauss or Equiangular for Basis "
            "SphericalHarmonic"));
  }
  {
    INFO("Testing ZernikeB3 with unsupported quadrature");
    CHECK_THROWS_WITH(
        (intrp::Cardinal<3>{
            Mesh<3>{{3, 5, 9},
                    {Spectral::Basis::ZernikeB3, Spectral::Basis::ZernikeB3,
                     Spectral::Basis::ZernikeB3},
                    {Spectral::Quadrature::GaussLobatto,
                     Spectral::Quadrature::Gauss,
                     Spectral::Quadrature::Equiangular}},
            tnsr::I<DataVector, 3, Frame::ElementLogical>{
                {{{0.5}, {1.0}, {2.0}}}}}),
        Catch::Matchers::ContainsSubstring(
            "Quadrature must be GaussRadauUpper, Gauss, or Equiangular for "
            "Basis ZernikeB3"));
  }
  {
    INFO("Testing ZernikeB2 with unsupported quadrature");
    CHECK_THROWS_WITH(
        (intrp::Cardinal<2>{
            Mesh<2>{{3, 3},
                    {Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2},
                    {Spectral::Quadrature::Gauss,
                     Spectral::Quadrature::Equiangular}},
            tnsr::I<DataVector, 2, Frame::ElementLogical>{
                {{{0.5, 1.0}, {0.0, 1.5}}}}}),
        Catch::Matchers::ContainsSubstring(
            "Quadrature must be GaussRadauUpper or Equiangular for Basis "
            "ZernikeB2"));
  }

#ifdef SPECTRE_DEBUG
  {
    INFO("Testing N_phi odd assertion for ZernikeB2");
    CHECK_THROWS_WITH(
        (intrp::Cardinal<2>{
            Mesh<2>{{3, 4},
                    {Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2},
                    {Spectral::Quadrature::GaussRadauUpper,
                     Spectral::Quadrature::Equiangular}},
            tnsr::I<DataVector, 2, Frame::ElementLogical>{
                {{{0.5, 1.0}, {0.0, 1.5}}}}}),
        Catch::Matchers::ContainsSubstring(
            "Need N_phi to be odd for stability"));
  }
  {
    INFO("Testing minimum resolution assertion for ZernikeB3");
    CHECK_THROWS_WITH(
        (intrp::Cardinal<3>{
            Mesh<3>{{2, 5, 9},
                    {Spectral::Basis::ZernikeB3, Spectral::Basis::ZernikeB3,
                     Spectral::Basis::ZernikeB3},
                    {Spectral::Quadrature::GaussRadauUpper,
                     Spectral::Quadrature::Gauss,
                     Spectral::Quadrature::Equiangular}},
            tnsr::I<DataVector, 3, Frame::ElementLogical>{
                {{{1.0, 0.5}, {1.5, 0.0}, {0.1, 0.2}}}}}),
        Catch::Matchers::ContainsSubstring(
            "ZernikeB3 radial resolution is insufficient"));
  }
#endif
}
}  // namespace

// [[Timeout, 20]]
SPECTRE_TEST_CASE("Unit.Numerical.Interpolation.Cardinal",
                  "[Unit][NumericalAlgorithms]") {
  MAKE_GENERATOR(generator);
  test_1d(make_not_null(&generator));
  test_2d_cartesian(make_not_null(&generator));
  test_2d_spherical(make_not_null(&generator));
  test_2d_disk(make_not_null(&generator));
  test_3d_cartesian(make_not_null(&generator));
  test_3d_spherical(make_not_null(&generator));
  test_3d_cylinder(make_not_null(&generator));
  test_3d_ball(make_not_null(&generator));
  test_errors();
  {
    INFO("Testing basic construction");
    const intrp::Cardinal<2> interpolant{
        Mesh<2>{{{3, 2}},
                {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
                {{Spectral::Quadrature::Gauss,
                  Spectral::Quadrature::GaussLobatto}}},
        tnsr::I<DataVector, 2, Frame::ElementLogical>{
            {{{1., 2., 3.}, {2., 3., 4.}}}}};
    test_serialization(interpolant);
  }
  {
    INFO("Testing ZernikeB2 serialization");
    const intrp::Cardinal<3> interpolant{
        Mesh<3>{{{3, 9, 4}},
                {{Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2,
                  Spectral::Basis::Legendre}},
                {{Spectral::Quadrature::GaussRadauUpper,
                  Spectral::Quadrature::Equiangular,
                  Spectral::Quadrature::GaussLobatto}}},
        tnsr::I<DataVector, 3, Frame::ElementLogical>{
            {{{1., 2., 3.}, {2., 3., 4.}, {3., 4., 5.}}}}};
    test_serialization(interpolant);
  }
  {
    INFO("Testing ZernikeB3 serialization");
    // Mesh: n_r=3, n_theta=5 (l_max=4), n_phi=9 (m_max=4)
    const intrp::Cardinal<3> interpolant{
        Mesh<3>{
            {{3, 5, 9}},
            {{Spectral::Basis::ZernikeB3, Spectral::Basis::ZernikeB3,
              Spectral::Basis::ZernikeB3}},
            {{Spectral::Quadrature::GaussRadauUpper,
              Spectral::Quadrature::Gauss, Spectral::Quadrature::Equiangular}}},
        tnsr::I<DataVector, 3, Frame::ElementLogical>{
            {{{0.3, -0.6}, {0.8, 2.1}, {1.5, 4.2}}}}};
    test_serialization(interpolant);
  }
}
