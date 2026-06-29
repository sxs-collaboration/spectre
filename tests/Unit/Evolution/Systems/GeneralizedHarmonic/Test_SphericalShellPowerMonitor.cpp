// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/SphericalShells.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ApplyTensorYlmFilter.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/SphericalShellPowerMonitor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace {

Mesh<3> shell_mesh(const size_t radial_extents, const size_t ell_max) {
  return Mesh<3>{
      {{radial_extents, ell_max + 1, 2 * ell_max + 1}},
      {{Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
        Spectral::Basis::SphericalHarmonic}},
      {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
        Spectral::Quadrature::Equiangular}}};
}

InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid> identity_jacobian(
    const size_t size) {
  InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid> jacobian{size,
                                                                        0.0};
  for (size_t i = 0; i < 3; ++i) {
    jacobian.get(i, i) = 1.0;
  }
  return jacobian;
}

void set_constant_angular_radial_profile(
    const gsl::not_null<DataVector*> component, const Mesh<3>& mesh,
    const DataVector& radial_profile) {
  const auto extents = mesh.extents();
  for (size_t r = 0; r < extents[0]; ++r) {
    for (size_t theta = 0; theta < extents[1]; ++theta) {
      for (size_t phi = 0; phi < extents[2]; ++phi) {
        Index<3> index{};
        index[0] = r;
        index[1] = theta;
        index[2] = phi;
        (*component)[collapsed_index(index, extents)] = radial_profile[r];
      }
    }
  }
}

DataVector expected_radial_monitor(const Mesh<3>& mesh,
                                   const DataVector& radial_profile,
                                   const size_t num_tensor_components) {
  ModalVector radial_modes(radial_profile.size(), 0.0);
  to_modal_coefficients(make_not_null(&radial_modes), radial_profile,
                        mesh.slice_through(0));
  DataVector expected(radial_profile.size(), 0.0);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = std::abs(radial_modes[i]) /
                  sqrt(static_cast<double>(num_tensor_components));
  }
  return expected;
}

tnsr::I<DataVector, 3, Frame::Inertial> shell_cartesian_coords(
    const Mesh<3>& mesh, const double inner_radius, const double outer_radius) {
  const auto extents = mesh.extents();
  const domain::creators::SphericalShells shell{
      inner_radius,
      outer_radius,
      0,
      extents[0],
      extents[1] - 1,
      {},
      domain::CoordinateMaps::Distribution::Linear};
  const Domain<3> domain = shell.create_domain();
  const ElementMap<3, Frame::Inertial> logical_to_inertial_map{
      ElementId<3>{0}, domain.blocks()[0].stationary_map().get_clone()};
  return logical_to_inertial_map(logical_coordinates(mesh));
}

void check_monitor(const std::array<DataVector, 2>& monitor,
                   const DataVector& expected_radial,
                   const DataVector& expected_angular) {
  const Approx spec_approx =
      Approx::custom().epsilon(1.0e-11).scale(0.0).margin(1.0e-15);
  CHECK_ITERABLE_CUSTOM_APPROX(monitor[0], expected_radial, spec_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(monitor[1], expected_angular, spec_approx);
}

/*
 * The hard-coded monitor values in `test_kerr_schild_spec_regression` are
 * regression values from a SpEC ApplyObservers run, followed by the explicit
 * normalization postprocessing below.  This is the exact procedure used to
 * produce the constants in this test.
 *
 * Run in a clean temporary directory with SpEC available at
 * `~/work/builds/spec-container/`.  First create `Domain.input`:
 *
 *   SubdomainStructure=
 *       SphericalShells3D(BaseName = Sphere;
 *                         L = 8;
 *                         r-Axis = (Extents  = 7;
 *                                   Bounds   = 4,7;
 *                                   Maps     = Lin;
 *                                   IndexMap = LegendreGaussLobatto;
 *                                   Topology = I1;
 *                                   );
 *                         );
 *
 * and `ApplyObservers.input`:
 *
 *   DataBoxItems=
 *       Subdomain
 *       (Items =
 *        AnalyticEinsteinSolution
 *        (Output    = AnalGr;
 *         Solution = KerrSchild(
 *            Mass     = 1.0;
 *            Center   = 0.0,0.0,0.0;
 *            Spin     = 0.05,-0.1,0.15;
 *         )
 *        ),
 *        AnalyticEinstein::SpacetimeMetric(Input=AnalGr;Output=psi;),
 *        AnalyticEinstein::Kappa(Input=AnalGr;Output=kappa;),
 *        ExtractPiFromKappa(Input=kappa;Output=Pi;),
 *        AnalyticEinstein::SpatialDerivSpacetimeMetric(
 *            Input=AnalGr;Output=dpsi;),
 *        FlattenDeriv(Input=dpsi;Output=Phi;DerivPosition=First;
 *                    ZeroFillOffset=1),
 *        PowerMonitor(Inputs=psi,Pi,Phi;
 *                     UseTensorYlmForS2=yes;
 *                     UseTensorYlmDirect=yes;
 *                     InputPrefixForTensorYlm=;
 *                     AddRadialMonitorForB2B3=yes),
 *        TensorYlmPowerMonitor(Inputs=psi,Pi,Phi;
 *                              Subdomains=Sphere0;)
 *       );
 *   Observers=
 *       PowerDiagnostics(GridDiagnostics=;
 *                        PowerMonitors=Powerpsi,PowerPi,PowerPhi,
 *                                      TensorYlmPowerpsi,TensorYlmPowerPi,
 *                                      TensorYlmPowerPhi;
 *                        Subdomains=Sphere0;
 *                        PowerMonitorOutputFormat=dat);
 *
 * Then run:
 *
 *   /path/to/spec/Support/bin/ApplyObservers \
        -NoDomainHistory -UseTimes 0 ApplyObservers.input
 *
 * This generates `TensorYlmPower*` files above while comparing diagnostics,
 * but the hard-coded values below are produced from the `Power*` files, not
 * the `TensorYlmPower*` files.  The reason for the postprocessing is that
 * SpECTRE's GH shell monitor averages tensor-Ylm coefficients over the
 * physical spherical-basis spin-weighted tensor components, omitting
 * unphysical low-l spin modes.  SpEC's `PowerMonitor` raw component counts
 * differ from SpECTRE's. The `Phi` input is also made with `FlattenDeriv`,
 * which turns SpEC's 30-component `dpsi_iab` into a 40-component object with a
 * zero-filled time-derivative sector.  The following script is the
 * postprocessing step used to convert those raw SpEC rows to the constants
 * checked here:
 *
 *   from math import sqrt
 *
 *   def row(filename):
 *       lines = [line for line in open(filename)
 *                if not line.startswith("#") and line.strip()]
 *       return [float(x) for x in lines[-1].split()[1:]]
 *
 *   metric_pi_angular_factors = [
 *       sqrt(8 / 7), sqrt(32 / 31), sqrt(60 / 59),
 *       sqrt(80 / 79), sqrt(100 / 99), sqrt(120 / 119),
 *       sqrt(140 / 139), sqrt(160 / 159), sqrt(181 / 180)]
 *   phi_angular_factors = [
 *       sqrt(3 / 2), sqrt(15 / 11), sqrt(19 / 14),
 *       sqrt(4 / 3), sqrt(4 / 3), sqrt(4 / 3),
 *       sqrt(4 / 3), sqrt(4 / 3), sqrt(4 / 3)]
 *
 *   print("spacetime_metric radial",
 *         row("Bf0I1Powerpsi_Sphere0.dat"))
 *   print("spacetime_metric angular",
 *         [x * f for x, f in zip(row("Bf1S2Powerpsi_Sphere0.dat"),
 *                                metric_pi_angular_factors)])
 *   print("pi radial", row("Bf0I1PowerPi_Sphere0.dat"))
 *   print("pi angular",
 *         [x * f for x, f in zip(row("Bf1S2PowerPi_Sphere0.dat"),
 *                                metric_pi_angular_factors)])
 *   print("phi radial",
 *         [x * sqrt(4 / 3) for x in row("Bf0I1PowerPhi_Sphere0.dat")])
 *   print("phi angular",
 *         [x * f for x, f in zip(row("Bf1S2PowerPhi_Sphere0.dat"),
 *                                phi_angular_factors)])
 */
void test_kerr_schild_spec_regression() {
  const Mesh<3> mesh = shell_mesh(7, 8);
  const auto coords = shell_cartesian_coords(mesh, 4.0, 7.0);
  const gh::Solutions::WrappedGr<gr::Solutions::KerrSchild> solution{
      1.0, {{0.05, -0.1, 0.15}}, {{0.0, 0.0, 0.0}}};

  const auto analytic_vars = solution.variables(
      coords, 0.0,
      tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3, Frame::Inertial>,
                 gh::Tags::Pi<DataVector, 3, Frame::Inertial>,
                 gh::Tags::Phi<DataVector, 3, Frame::Inertial>>{});
  Variables<ylm::TensorYlm::filter_detail::gh_spacetime_vars_list> gh_vars{
      mesh.number_of_grid_points()};
  get<gr::Tags::SpacetimeMetric<DataVector, 3, Frame::Inertial>>(gh_vars) =
      get<gr::Tags::SpacetimeMetric<DataVector, 3, Frame::Inertial>>(
          analytic_vars);
  get<gh::Tags::Pi<DataVector, 3, Frame::Inertial>>(gh_vars) =
      get<gh::Tags::Pi<DataVector, 3, Frame::Inertial>>(analytic_vars);
  get<gh::Tags::Phi<DataVector, 3, Frame::Inertial>>(gh_vars) =
      get<gh::Tags::Phi<DataVector, 3, Frame::Inertial>>(analytic_vars);

  gh::power_monitor::CartToSphereMatrices cart_to_sphere_matrices{};
  const auto monitors = gh::power_monitor::gh_shell_power_monitors(
      make_not_null(&cart_to_sphere_matrices), gh_vars, mesh,
      identity_jacobian(mesh.number_of_grid_points()));

  check_monitor(
      monitors.spacetime_metric,
      DataVector{0.66272576586915166, 0.055118989877186962,
                 0.010223513029798308, 0.001706566677470182,
                 0.00027144507714939298, 0.000042905006422366406,
                 0.0000063892936284495129},
      DataVector{1.9854935257947095, 0.00806505390676714, 0.00016837816034157,
                 0.00000418589096616, 0.0000001661210045, 0.00000000552502198,
                 0.00000000024433662, 0.00000000000788883,
                 0.00000000000025781});
  check_monitor(
      monitors.pi,
      DataVector{0.012573571910520885, 0.009832206078082233,
                 0.0034601774215344341, 0.00091860759247559592,
                 0.00021004678157227581, 0.000045199806501584563,
                 0.0000085382443326196344},
      DataVector{0.055211802785050945, 0.00068180518923684, 0.00004171340786288,
                 0.00000087524448329, 0.00000007098264593, 0.00000000191450705,
                 0.00000000014586247, 0.00000000000382953,
                 0.00000000000019612});
  check_monitor(
      monitors.phi,
      DataVector{0.03582086057745523, 0.01984177151721315, 0.00550825042019753,
                 0.00122441336098692, 0.00024331209932814, 0.00004644598374878,
                 0.00000801247933444},
      DataVector{0.15154167256606327, 0.00335975215390938, 0.00010001010476677,
                 0.00000325894876587, 0.00000016423770303, 0.00000000657476944,
                 0.00000000029571569, 0.0000000000077388, 0.00000000000002565});
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.SphericalShellPowerMonitor",
    "[Unit][Evolution]") {
  const Mesh<3> mesh = shell_mesh(4, 3);
  Variables<ylm::TensorYlm::filter_detail::gh_spacetime_vars_list> gh_vars(
      mesh.number_of_grid_points(), 0.0);
  const DataVector radial_profile{1.0, -0.5, 0.25, 2.0};
  gh::power_monitor::CartToSphereMatrices cart_to_sphere_matrices{};

  set_constant_angular_radial_profile(
      make_not_null(&get<0, 0>(
          get<gr::Tags::SpacetimeMetric<DataVector, 3, Frame::Inertial>>(
              gh_vars))),
      mesh, radial_profile);

  const auto monitors = gh::power_monitor::gh_shell_power_monitors(
      make_not_null(&cart_to_sphere_matrices), gh_vars, mesh,
      identity_jacobian(mesh.number_of_grid_points()));

  CHECK_ITERABLE_APPROX(monitors.spacetime_metric[0],
                        expected_radial_monitor(mesh, radial_profile, 10));
  CHECK(max(monitors.spacetime_metric[1]) > 0.0);
  CHECK(max(abs(monitors.pi[0])) == 0.0);
  CHECK(max(abs(monitors.pi[1])) == 0.0);
  CHECK(max(abs(monitors.phi[0])) == 0.0);
  CHECK(max(abs(monitors.phi[1])) == 0.0);

  std::fill(gh_vars.data(), gh_vars.data() + gh_vars.size(), 0.0);
  set_constant_angular_radial_profile(
      make_not_null(&get<0, 0>(
          get<gh::Tags::Pi<DataVector, 3, Frame::Inertial>>(gh_vars))),
      mesh, radial_profile);
  const auto pi_monitors = gh::power_monitor::gh_shell_power_monitors(
      make_not_null(&cart_to_sphere_matrices), gh_vars, mesh,
      identity_jacobian(mesh.number_of_grid_points()));
  CHECK_ITERABLE_APPROX(pi_monitors.pi[0],
                        expected_radial_monitor(mesh, radial_profile, 10));
  CHECK(max(pi_monitors.pi[1]) > 0.0);
  CHECK(max(abs(pi_monitors.spacetime_metric[0])) == 0.0);
  CHECK(max(abs(pi_monitors.phi[0])) == 0.0);

  std::fill(gh_vars.data(), gh_vars.data() + gh_vars.size(), 0.0);
  set_constant_angular_radial_profile(
      make_not_null(&get<0, 0, 0>(
          get<gh::Tags::Phi<DataVector, 3, Frame::Inertial>>(gh_vars))),
      mesh, radial_profile);
  const auto phi_monitors = gh::power_monitor::gh_shell_power_monitors(
      make_not_null(&cart_to_sphere_matrices), gh_vars, mesh,
      identity_jacobian(mesh.number_of_grid_points()));
  CHECK_ITERABLE_APPROX(phi_monitors.phi[0],
                        expected_radial_monitor(mesh, radial_profile, 30));
  CHECK(max(phi_monitors.phi[1]) > 0.0);
  CHECK(max(abs(phi_monitors.spacetime_metric[0])) == 0.0);
  CHECK(max(abs(phi_monitors.pi[0])) == 0.0);

  test_kerr_schild_spec_regression();
}
