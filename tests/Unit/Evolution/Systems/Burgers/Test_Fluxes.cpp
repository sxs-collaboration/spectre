// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/Conservative/ConservativeDuDt.hpp"
#include "Evolution/Systems/Burgers/Fluxes.hpp"
#include "Evolution/Systems/Burgers/System.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Tags::deriv
// IWYU pragma: no_forward_declare Tags::div

SPECTRE_TEST_CASE("Unit.Burgers.Fluxes", "[Unit][Burgers]") {
  // Check that the time derivative calculated from the fluxes is the
  // same as from the non-conservative form: d_t u = - u d_x u
  constexpr size_t num_points = 10;
  const Mesh<1> mesh(num_points, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto coords = get<0>(logical_coordinates(mesh));
  const auto identity = make_with_value<
      InverseJacobian<DataVector, 1, Frame::Logical, Frame::Inertial>>(coords,
                                                                       1.);

  Variables<tmpl::list<Burgers::Tags::U>> vars(num_points);
  Variables<tmpl::list<Tags::dt<Burgers::Tags::U>>> dt_vars(num_points);
  const Variables<tmpl::list<Tags::Source<Burgers::Tags::U>>> sources(
      num_points);
  // Arbitrary polynomial whose square is exactly representable.
  get(get<Burgers::Tags::U>(vars)) =
      pow<num_points / 2 - 1>(coords) + pow<num_points / 4>(coords) + 5.;

  const auto deriv_vars =
      partial_derivatives<tmpl::list<Burgers::Tags::U>>(vars, mesh, identity);
  const Scalar<DataVector> dudt_expected{
      -get(get<Burgers::Tags::U>(vars)) *
      get<0>(
          get<Tags::deriv<Burgers::Tags::U, tmpl::size_t<1>, Frame::Inertial>>(
              deriv_vars))};

  using flux_tag =
      Tags::Flux<Burgers::Tags::U, tmpl::size_t<1>, Frame::Inertial>;
  Variables<tmpl::list<flux_tag>> flux(num_points);
  Burgers::Fluxes::apply(&get<flux_tag>(flux), get<Burgers::Tags::U>(vars));
  const auto div_flux = divergence(flux, mesh, identity);
  evolution::dg::ConservativeDuDt<Burgers::System>::apply(
      make_not_null(&dt_vars), mesh, identity, flux, sources);
  CHECK_ITERABLE_APPROX(get<Tags::dt<Burgers::Tags::U>>(dt_vars),
                        dudt_expected);
}
