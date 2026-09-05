// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <complex>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/AnalyticData/CircularOrbit.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Equations.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "Utilities/TMPL.hpp"

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
  for (const bool impose_equatorial_symmetry : {true, false}) {
    CAPTURE(impose_equatorial_symmetry);
    for (int m_mode_number = 0; m_mode_number < 3; ++m_mode_number) {
      CAPTURE(m_mode_number);
      const auto circular_orbit = CircularOrbit{1.,
                                                0.9,
                                                6.,
                                                m_mode_number,
                                                {{-25., -5., 20., 40.}},
                                                false,
                                                impose_equatorial_symmetry};
      CAPTURE(circular_orbit.puncture_position());
      const auto background =
          circular_orbit.variables(x, CircularOrbit::background_tags{});
      const auto& alpha = get<Tags::Alpha>(background);
      const auto& beta = get<Tags::Beta>(background);
      const auto& gamma = get<Tags::Gamma>(background);
      const auto vars =
          circular_orbit.variables(x, CircularOrbit::source_tags{});
      const auto& singular_field = get<Tags::SingularField>(vars);
      const auto& deriv_singular_field = get<
          ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>>(
          vars);
      const auto& effective_source =
          get<::Tags::FixedSource<Tags::MMode>>(vars);

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
                                      singular_field, deriv_singular_field,
                                      flux_singular_field);
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
}

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.ScalarSelfForce.CircularOrbit.Compactification",
    "[PointwiseFunctions][Unit]") {
  // Check that Alpha in the compactified u-region (coordinate sigma) matches
  // the original r-based Alpha formula, rescaled by the chain-rule Jacobian
  // J = dsigma/dr = r_u^2/r^2, for sigma = 2*r_u - r_u^2/r.
  const double black_hole_mass = 1.;
  const double black_hole_spin = 0.9;
  const double r_u = 25.;
  const double r = 2. * r_u;
  // sigma(r=2*r_u) = 2*r_u - r_u^2/(2*r_u) = 1.5*r_u
  const double sigma = 1.5 * r_u;
  const double cos_theta = 0.3;
  const double sin_theta_squared = 1. - square(cos_theta);

  const auto circular_orbit = CircularOrbit{black_hole_mass,
                                            black_hole_spin,
                                            6.,
                                            1,
                                            {{-10., -5., r_u, r_u}},
                                            true,
                                            false};

  tnsr::I<DataVector, 2> x{};
  get<0>(x) = DataVector{sigma};
  get<1>(x) = DataVector{cos_theta};
  const auto background =
      circular_orbit.variables(x, CircularOrbit::background_tags{});
  const auto& alpha = get<Tags::Alpha>(background);

  // Independently compute the reference r-based Alpha and the Jacobian.
  const double a = black_hole_spin * black_hole_mass;
  const double M = black_hole_mass;
  const double r_plus = M * (1. + sqrt(1. - square(black_hole_spin)));
  const double r_minus = M * (1. - sqrt(1. - square(black_hole_spin)));
  const double delta = (r - r_plus) * (r - r_minus);
  const double r_sq_plus_a_sq = square(r) + square(a);
  const double jacobian = square(r_u) / square(r);  // dsigma/dr at r=2*r_u
  const double expected_alpha0 = (delta / r_sq_plus_a_sq) * jacobian;
  const double expected_alpha1 =
      (1. / r_sq_plus_a_sq) / jacobian * sin_theta_squared;

  CHECK_ITERABLE_APPROX(get<0>(alpha)[0], expected_alpha0);
  CHECK_ITERABLE_APPROX(get<1>(alpha)[0], expected_alpha1);
}
}  // namespace ScalarSelfForce::AnalyticData
