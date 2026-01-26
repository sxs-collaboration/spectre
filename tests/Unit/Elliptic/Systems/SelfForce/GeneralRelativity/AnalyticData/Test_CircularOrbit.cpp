// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <complex>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/CircularOrbit.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Equations.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace GrSelfForce::AnalyticData {

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GrSelfForce.CircularOrbit",
                  "[PointwiseFunctions][Unit]") {
  // This test checks both the self-force equations and the effective source
  // computation in a very robust way: it ensures that the elliptic operator
  // applied to the singular field gives the effective source.
  // This is done numerically on a rectangular grid in (r_*, theta) near
  // the puncture.
  const double theta_offset = M_PI / 8.;
  const double delta_theta = M_PI / 40.;
  const double rstar_offset = 0.;
  const double delta_rstar = 5.;
  const size_t npoints = 20;
  const domain::creators::Rectangle domain_creator{
      {{rstar_offset, M_PI_2 + theta_offset}},
      {{rstar_offset + delta_rstar, M_PI_2 + theta_offset + delta_theta}},
      {{0, 0}},
      {{npoints, npoints}},
      {{false, false}}};
  const auto domain = domain_creator.create_domain();
  const auto& block = domain.blocks()[0];
  const ElementId<2> element_id{0};
  const ElementMap<2, Frame::Inertial> element_map{element_id, block};
  const Mesh<2> mesh{npoints, Spectral::Basis::Legendre,
                     Spectral::Quadrature::Gauss};
  const auto xi = logical_coordinates(mesh);
  const auto x = element_map(xi);
  const auto inv_jacobian = element_map.inv_jacobian(xi);
  const auto& r_star = get<0>(x);
  const auto& theta = get<1>(x);
  CAPTURE(min(r_star));
  CAPTURE(max(r_star));
  CAPTURE(min(theta));
  CAPTURE(max(theta));

  // Get the analytic fields
  for (int m_mode_number = 0; m_mode_number < 3; ++m_mode_number) {
    CAPTURE(m_mode_number);
    const auto circular_orbit = CircularOrbit{1., 0.9, 6., m_mode_number};
    CAPTURE(circular_orbit.puncture_position());
    const auto background =
        circular_orbit.variables(x, CircularOrbit::background_tags{});
    const auto& alpha = get<Tags::Alpha>(background);
    const auto& beta = get<Tags::Beta>(background);
    const auto& gamma_rstar = get<Tags::GammaRstar>(background);
    const auto& gamma_theta = get<Tags::GammaTheta>(background);
    const auto vars = circular_orbit.variables(x, CircularOrbit::source_tags{});
    const auto& singular_field = get<Tags::SingularField>(vars);
    const auto& deriv_singular_field = get<
        ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>>(
        vars);
    const auto& effective_source = get<::Tags::FixedSource<Tags::MMode>>(vars);

    // Take numeric derivative
    const auto numeric_deriv_singular_field =
        partial_derivative(singular_field, mesh, inv_jacobian);
    const Approx custom_approx = Approx::custom().epsilon(1.e-10).scale(1.);
    for (size_t i = 0; i < deriv_singular_field.size(); ++i) {
      CAPTURE(i);
      CHECK_ITERABLE_CUSTOM_APPROX(numeric_deriv_singular_field[i],
                                   deriv_singular_field[i], custom_approx);
    }

    Variables<
        tmpl::list<::Tags::Flux<Tags::MMode, tmpl::size_t<2>, Frame::Inertial>>>
        fluxes{mesh.number_of_grid_points()};
    auto& flux_singular_field =
        get<::Tags::Flux<Tags::MMode, tmpl::size_t<2>, Frame::Inertial>>(
            fluxes);
    GrSelfForce::Fluxes::apply(make_not_null(&flux_singular_field), alpha, {},
                               deriv_singular_field);
    auto divs = divergence(fluxes, mesh, inv_jacobian);
    auto& scalar_eqn = get<::Tags::div<
        ::Tags::Flux<Tags::MMode, tmpl::size_t<2>, Frame::Inertial>>>(divs);
    for (size_t i = 0; i < scalar_eqn.size(); ++i) {
      scalar_eqn[i] *= -1.;
    }
    GrSelfForce::Sources::apply(make_not_null(&scalar_eqn), beta, gamma_rstar,
                                gamma_theta, singular_field,
                                flux_singular_field);
    for (size_t i = 0; i < scalar_eqn.size(); ++i) {
      CAPTURE(i);
      CHECK_ITERABLE_CUSTOM_APPROX(scalar_eqn[i], -effective_source[i],
                                   custom_approx);
    }
  }
}

}  // namespace GrSelfForce::AnalyticData
