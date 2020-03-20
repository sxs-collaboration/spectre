// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/StripeIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/LinearOperators/IndefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TypeTraits/GetFundamentalType.hpp"

namespace {
template <typename VectorType, size_t Dim, typename T>
VectorType integrand(
    const std::array<size_t, Dim>& exponent, const T& x,
    const typename VectorType::ElementType overall_factor) noexcept {
  VectorType integrand(get<0>(x).size(), overall_factor);
  for (size_t d = 0; d < Dim; ++d) {
    integrand *= pow(x.get(d), gsl::at(exponent, d));
  }
  return integrand;
}

template <typename VectorType, size_t Dim, typename T>
VectorType integral(
    const std::array<size_t, Dim>& exponent, const T& x, const Mesh<Dim>& mesh,
    const size_t only_this_dim,
    const typename VectorType::ElementType overall_factor) noexcept {
  VectorType integral(get<0>(x).size(), overall_factor);
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

template <typename VectorType, Spectral::Basis BasisType,
          Spectral::Quadrature QuadratureType>
void test_zero_bc() noexcept {
  const size_t min_pts = 2;
  REQUIRE(5 <= Spectral::maximum_number_of_points<BasisType>);

  MAKE_GENERATOR(generator);
  UniformCustomDistribution<
      tt::get_fundamental_type_t<typename VectorType::ElementType>>
      dist{0.5, 2.0};
  const auto overall_factor =
      make_with_random_values<typename VectorType::ElementType>(
          make_not_null(&generator), make_not_null(&dist));

  for (size_t i = min_pts; i < 5; ++i) {
    Mesh<1> mesh_1d(i, BasisType, QuadratureType);
    const auto coords_1d = logical_coordinates(mesh_1d);
    CHECK_ITERABLE_APPROX(
        indefinite_integral(
            integrand<VectorType, 1>({{i - 2}}, coords_1d, overall_factor),
            mesh_1d, 0),
        integral<VectorType>({{i - 2}}, coords_1d, mesh_1d, 0, overall_factor));

    // 2d and 3d cases
    for (size_t j = min_pts; j < 5; ++j) {
      Mesh<2> mesh_2d({{i, j}}, {{BasisType, BasisType}},
                      {{QuadratureType, QuadratureType}});
      const auto coords_2d = logical_coordinates(mesh_2d);
      for (size_t d = 0; d < 2; ++d) {
        CHECK_ITERABLE_APPROX(
            indefinite_integral(
                integrand<VectorType, 2>({{i - 2, j - 2}}, coords_2d,
                                         overall_factor),
                mesh_2d, d),
            integral<VectorType>({{i - 2, j - 2}}, coords_2d, mesh_2d, d,
                                 overall_factor));
      }

      // 3d case
      for (size_t k = min_pts; k < 5; ++k) {
        Mesh<3> mesh_3d({{i, j, k}}, {{BasisType, BasisType, BasisType}},
                        {{QuadratureType, QuadratureType, QuadratureType}});
        const auto coords_3d = logical_coordinates(mesh_3d);
        for (size_t d = 0; d < 3; ++d) {
          CHECK_ITERABLE_APPROX(
              indefinite_integral(
                  integrand<VectorType, 3>({{i - 2, j - 2, k - 2}}, coords_3d,
                                           overall_factor),
                  mesh_3d, d),
              integral<VectorType>({{i - 2, j - 2, k - 2}}, coords_3d, mesh_3d,
                                   d, overall_factor));
        }
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.IndefiniteIntegral",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  test_zero_bc<DataVector, Spectral::Basis::Chebyshev,
               Spectral::Quadrature::GaussLobatto>();
  test_zero_bc<DataVector, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto>();
  test_zero_bc<ComplexDataVector, Spectral::Basis::Chebyshev,
               Spectral::Quadrature::GaussLobatto>();
  test_zero_bc<ComplexDataVector, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto>();
}
