// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/StripeIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/LinearOperators/IndefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <size_t Dim, typename T>
DataVector integrand(const std::array<size_t, Dim>& exponent,
                     const T& x) noexcept {
  DataVector integrand(get<0>(x).size(), 1.0);
  for (size_t d = 0; d < Dim; ++d) {
    integrand *= pow(x.get(d), gsl::at(exponent, d));
  }
  return integrand;
}

template <size_t Dim, typename T>
DataVector integral(const std::array<size_t, Dim>& exponent, const T& x,
                    const Mesh<Dim> mesh, const size_t only_this_dim) noexcept {
  DataVector integral(get<0>(x).size(), 1.0);
  for (size_t d = 0; d < Dim; ++d) {
    if (d == only_this_dim) {
      integral *= pow(x.get(d), gsl::at(exponent, d) + 1) /
                  (gsl::at(exponent, d) + 1.0);
      // Adjust to zero integral on lower boundary.
      for (StripeIterator stripe_it(mesh.extents(), d); stripe_it;
           ++stripe_it) {
        for (size_t count = 1, i = stripe_it.offset() + stripe_it.stride();
             count < mesh.extents()[d]; ++count, i += stripe_it.stride()) {
          integral[i] -= integral[stripe_it.offset()];
        }
        integral[stripe_it.offset()] = 0.0;
      }
    } else {
      integral *= pow(x.get(d), gsl::at(exponent, d));
    }
  }
  return integral;
}

template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType>
void test_zero_bc() noexcept {
  const size_t min_pts = 2;
  REQUIRE(5 <= Spectral::maximum_number_of_points<BasisType>);
  for (size_t i = min_pts; i < 5; ++i) {
    Mesh<1> mesh_1d(i, BasisType, QuadratureType);
    const auto coords_1d = logical_coordinates(mesh_1d);
    CHECK_ITERABLE_APPROX(
        indefinite_integral(integrand<1>({{i - 2}}, coords_1d), mesh_1d, 0),
        integral({{i - 2}}, coords_1d, mesh_1d, 0));

    // 2d and 3d cases
    for (size_t j = min_pts; j < 5; ++j) {
      Mesh<2> mesh_2d({{i, j}}, {{BasisType, BasisType}},
                      {{QuadratureType, QuadratureType}});
      const auto coords_2d = logical_coordinates(mesh_2d);
      for (size_t d = 0; d < 2; ++d) {
        CHECK_ITERABLE_APPROX(
            indefinite_integral(integrand<2>({{i - 2, j - 2}}, coords_2d),
                                mesh_2d, d),
            integral({{i - 2, j - 2}}, coords_2d, mesh_2d, d));
      }

      // 3d case
      for (size_t k = min_pts; k < 5; ++k) {
        Mesh<3> mesh_3d({{i, j, k}}, {{BasisType, BasisType, BasisType}},
                        {{QuadratureType, QuadratureType, QuadratureType}});
        const auto coords_3d = logical_coordinates(mesh_3d);
        for (size_t d = 0; d < 3; ++d) {
          CHECK_ITERABLE_APPROX(
              indefinite_integral(
                  integrand<3>({{i - 2, j - 2, k - 2}}, coords_3d), mesh_3d, d),
              integral({{i - 2, j - 2, k - 2}}, coords_3d, mesh_3d, d));
        }
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.IndefiniteIntegral",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  test_zero_bc<Spectral::Basis::Chebyshev,
               Spectral::Quadrature::GaussLobatto>();
  test_zero_bc<Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>();
}
