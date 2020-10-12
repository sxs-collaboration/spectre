// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Evolution/Systems/Burgers/System.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Evolution/Systems/Burgers/TimeDerivativeTerms.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

SPECTRE_TEST_CASE("Unit.Burgers.TimeDerivativeTerms", "[Unit][Burgers]") {
  constexpr size_t num_points = 10;
  const Mesh<1> mesh(num_points, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto coords = get<0>(logical_coordinates(mesh));

  // Arbitrary polynomial whose square is exactly representable.
  const Scalar<DataVector> burgers_u{DataVector{
      pow<num_points / 2 - 1>(coords) + pow<num_points / 4>(coords) + 5.0}};

  tnsr::I<DataVector, 1, Frame::Inertial> flux(num_points);
  Scalar<DataVector> dudt{num_points, 0.0};
  Burgers::TimeDerivativeTerms::apply(&dudt, &flux, burgers_u);

  // dudt should be zero since we have no source terms
  const Scalar<DataVector> dudt_expected{num_points, 0.0};
  const tnsr::I<DataVector, 1, Frame::Inertial> flux_expected{
      DataVector{0.5 * square(get(burgers_u))}};

  CHECK_ITERABLE_APPROX(dudt, dudt_expected);
  CHECK_ITERABLE_APPROX(flux, flux_expected);
}
