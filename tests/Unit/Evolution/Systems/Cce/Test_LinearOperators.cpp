// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/Systems/Cce/LinearOperators.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "Utilities/VectorAlgebra.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.LinearOperators",
                  "[Unit][Cce]") {
  MAKE_GENERATOR(generator);
  const size_t l_max = 6;
  const size_t number_of_radial_points = 6;
  UniformCustomDistribution<double> dist(0.1, 1.0);

  const ComplexDataVector y = outer_product(
      ComplexDataVector{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max), 1.0},
      Spectral::collocation_points<Spectral::Basis::Legendre,
                                   Spectral::Quadrature::GaussLobatto>(
          number_of_radial_points));
  const size_t polynomial_order = 3;
  const DataVector y_polynomial_coefficients =
      make_with_random_values<DataVector>(
          make_not_null(&generator), make_not_null(&dist), polynomial_order);

  ComplexDataVector to_differentiate{y.size(), 0.0};
  ComplexDataVector expected_derivative{y.size(), 0.0};
  for (size_t i = 0; i < polynomial_order; ++i) {
    to_differentiate += pow(y, i) * y_polynomial_coefficients[i];
    if (i != 0) {
      expected_derivative +=
          static_cast<double>(i) * pow(y, i - 1) * y_polynomial_coefficients[i];
    }
  }
  ComplexDataVector derivative{y.size()};
  Cce::logical_partial_directional_derivative_of_complex(
      make_not_null(&derivative), to_differentiate,
      Mesh<3>{{{Spectral::Swsh::number_of_swsh_theta_collocation_points(l_max),
                Spectral::Swsh::number_of_swsh_phi_collocation_points(l_max),
                number_of_radial_points}},
              Spectral::Basis::Legendre,
              Spectral::Quadrature::GaussLobatto},
      2);
  CHECK_ITERABLE_APPROX(expected_derivative, derivative);
}
