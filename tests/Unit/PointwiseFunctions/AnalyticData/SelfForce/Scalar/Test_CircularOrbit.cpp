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
#include "Elliptic/Systems/SelfForce/Scalar/Equations.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "PointwiseFunctions/AnalyticData/SelfForce/Scalar/CircularOrbit.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ScalarSelfForce::AnalyticData {

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.ScalarSelfForce.CircularOrbit",
                  "[PointwiseFunctions][Unit]") {
  // This test checks both the self-force equations and the effective source
  // computation in a very robust way: it ensures that the elliptic operator
  // applied to the singular field gives the effective source.
  // This is done numerically on a rectangular grid in (r_*, cos(theta)) near
  // the puncture.
  const double costheta_offset = 0.1;
  const double delta_costheta = 0.2;
  const double rstar_offset = 0.;
  const double delta_rstar = 5.;
  const size_t npoints = 20;
  const domain::creators::Rectangle domain_creator{
      {{rstar_offset, costheta_offset}},
      {{rstar_offset + delta_rstar, costheta_offset + delta_costheta}},
      {{0, 0}},
      {{npoints, npoints}},
      {{false, false}}};
  const auto domain = domain_creator.create_domain();
  const auto& block = domain.blocks()[0];
  const ElementId<2> element_id{0};
  const ElementMap<2, Frame::Inertial> element_map{element_id, block};
  const Mesh<2> mesh{npoints, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const auto xi = logical_coordinates(mesh);
  const auto x = element_map(xi);
  const auto inv_jacobian = element_map.inv_jacobian(xi);
  const auto& r_star = get<0>(x);
  const auto& cos_theta = get<1>(x);
  CAPTURE(min(r_star));
  CAPTURE(max(r_star));
  CAPTURE(min(cos_theta));
  CAPTURE(max(cos_theta));

  // Get the analytic fields
  for (int m_mode_number = 0; m_mode_number < 3; ++m_mode_number) {
    CAPTURE(m_mode_number);
    const auto circular_orbit =
        CircularOrbit{1., 0.9, 6., m_mode_number, {{-25., -5., 20., 40.}}};
    CAPTURE(circular_orbit.puncture_position());
    const auto background =
        circular_orbit.variables(x, CircularOrbit::background_tags{});
    const auto& alpha = get<Tags::Alpha>(background);
    const auto& beta = get<Tags::Beta>(background);
    const auto& gamma = get<Tags::Gamma>(background);
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

    tnsr::I<ComplexDataVector, 2> flux_singular_field{};
    ScalarSelfForce::Fluxes::apply(make_not_null(&flux_singular_field), alpha,
                                   {}, deriv_singular_field);
    auto scalar_eqn = divergence(flux_singular_field, mesh, inv_jacobian);
    get(scalar_eqn) *= -1.;
    ScalarSelfForce::Sources::apply(make_not_null(&scalar_eqn), beta, gamma,
                                    singular_field, flux_singular_field);
    // Minus sign is from the definition of the effective source:
    //   \psi = \psi_R + \psi_P = 0
    // where \psi_R is the regular part and \psi_P is the singular part
    //   => -\Delta \psi_R = \Delta \psi_P = S_eff
    // where -Delta represents the elliptic operator. So the effective source
    // for the regular part is is the negative of the elliptic operator
    // acting on the singular part.
    CHECK_ITERABLE_CUSTOM_APPROX(get(scalar_eqn), -get(effective_source),
                                 custom_approx);
  }
}

}  // namespace ScalarSelfForce::AnalyticData
