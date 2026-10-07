// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <pup.h>

#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Domain.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/GrMhd/VerifyGrMhdSolution.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/RotatingStar.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace {
void verify_solution(const RelativisticEuler::Solutions::RotatingStar& solution,
                     const std::array<double, 3>& x,
                     const double error_tolerance) {
  CAPTURE(x);
  const std::array<double, 3> dx{{1.e-4, 1.e-4, 1.e-4}};
  domain::creators::Brick brick(x - dx, x + dx, {{0, 0, 0}}, {{5, 5, 5}},
                                {{false, false, false}});
  Mesh<3> mesh{brick.initial_extents()[0], Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto};
  const auto domain = brick.create_domain();

  verify_grmhd_solution(solution, domain.blocks()[0], mesh, error_tolerance,
                        1.234, 1.e-4);
}

void verify_solution_suite(
    const RelativisticEuler::Solutions::RotatingStar& solution,
    const double error_tolerance) {
  // Near the center the finite difference derivatives cause "large" errors
  // (1e-8) and so the check fails.
  verify_solution(solution, {{1., 0., 0.}}, error_tolerance);
  verify_solution(solution, {{-1., 0., 0.}}, error_tolerance);
  verify_solution(solution, {{0., 1., 0.}}, error_tolerance);
  verify_solution(solution, {{0., -1., 0.}}, error_tolerance);
  verify_solution(solution, {{0., 0., 1.}}, error_tolerance);
  verify_solution(solution, {{0., 0., -1.}}, error_tolerance);
  verify_solution(solution, {{1., 0.528, 0.2}}, error_tolerance);
  verify_solution(solution, {{1., 0.528, -0.2}}, error_tolerance);
  verify_solution(solution, {{1., -0.528, 0.2}}, error_tolerance);
  verify_solution(solution, {{-1., 0.528, 0.2}}, error_tolerance);
  verify_solution(solution, {{1., -0.528, -0.2}}, error_tolerance);
  verify_solution(solution, {{-1., 0.528, -0.2}}, error_tolerance);
  verify_solution(solution, {{-1., -0.528, 0.2}}, error_tolerance);
  verify_solution(solution, {{-1., -0.528, -0.2}}, error_tolerance);
  // Test outside the star
  verify_solution(solution, {{20., 20., 20.}}, error_tolerance);
  verify_solution(solution, {{20., 20., -20.}}, error_tolerance);
  verify_solution(solution, {{20., -20., 20.}}, error_tolerance);
  verify_solution(solution, {{-20., 20., 20.}}, error_tolerance);
  verify_solution(solution, {{20., -20., -20.}}, error_tolerance);
  verify_solution(solution, {{-20., 20., -20.}}, error_tolerance);
  verify_solution(solution, {{-20., -20., 20.}}, error_tolerance);
  verify_solution(solution, {{-20., -20., -20.}}, error_tolerance);
  verify_solution(solution, {{20., 0., 0.}}, error_tolerance);
  verify_solution(solution, {{0., 20., 0.}}, error_tolerance);
  verify_solution(solution, {{0., 0., 20.}}, error_tolerance);
}
}  // namespace

// [[Timeout, 10]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.RelEuler.RotatingStar",
    "[Unit][PointwiseFunctions]") {
  PUPable_reg(RelativisticEuler::Solutions::RotatingStar);
  register_derived_classes_with_charm<
      EquationsOfState::EquationOfState<true, 3>>();
  // You can uncomment the cst_solution code below, include <iostream>,
  // and update the path to the `dat` file in order to figure out what the
  // central rest mass density is.
  // RelativisticEuler::Solutions::detail::CstSolution cst_solution(
  //     "/path/to/InitialData.dat", true, 100);
  // std::cout << cst_solution.interpolate(0.0, 0.0, true) << "\n\n";

  // The tolerance depends on the resolution that the RotNS file used. In order
  // to avoid storing very large files in the repo, we accept a larger
  // tolerance. However, Nils Deppe verified on Jan. 5, 2022 that increasing the
  // RotNS code resolution allows for a stricter tolerance.

  const RelativisticEuler::Solutions::detail::CstSolution cst_solution(
      unit_test_src_path() +
          "/PointwiseFunctions/AnalyticSolutions/RelativisticEuler/"
          "RotatingStarId.dat",
      true, 100.);
  // The following line would fail under the original stencil construction due
  // to rounding errors.
  cst_solution.interpolate(5.85, 0.64999999999, true);

  register_classes_with_charm<RelativisticEuler::Solutions::RotatingStar>();
  const std::unique_ptr<evolution::initial_data::InitialData> option_solution =
      TestHelpers::test_option_tag_factory_creation<
          evolution::initial_data::OptionTags::InitialData,
          RelativisticEuler::Solutions::RotatingStar>(
          "RotatingStar:\n"
          "  RotNsFilename: " +
          unit_test_src_path() +
          "/PointwiseFunctions/AnalyticSolutions/RelativisticEuler/"
          "RotatingStarId.dat\n"
          "  PolytropicConstant: 100.0\n")
          ->get_clone();
  TestHelpers::test_option_tag_factory_creation<
      evolution::initial_data::OptionTags::InitialData,
      RelativisticEuler::Solutions::RotatingStar>(
      "RotatingStar:\n"
      "  RotNsFilename: " +
      unit_test_src_path() +
      "/PointwiseFunctions/AnalyticSolutions/RelativisticEuler/"
      "RotatingStarId_Hybrid.dat\n"
      "  EquationOfState:\n"
      "    Equilibrium3D(HybridEos(PolytropicFluid)):\n"
      "      HybridEos:\n"
      "        PolytropicFluid:\n"
      "          PolytropicConstant: 123.6\n"
      "          PolytropicExponent: 2.0\n"
      "        ThermalAdiabaticIndex: 2.0\n"
      "        MinTemperature: 0.0\n");

  const double error_tolerance = 4.e-6;
  const RelativisticEuler::Solutions::RotatingStar solution(
      dynamic_cast<const RelativisticEuler::Solutions::RotatingStar&>(
          *serialize_and_deserialize(
              RelativisticEuler::Solutions::RotatingStar(
                  unit_test_src_path() +
                      "/PointwiseFunctions/AnalyticSolutions/RelativisticEuler/"
                      "RotatingStarId.dat",
                  100)
                  .get_clone())));

  CHECK(solution.equatorial_radius() == approx(7.155891353887));

  verify_solution_suite(solution, error_tolerance);

  const RelativisticEuler::Solutions::RotatingStar hybrid_solution(
      dynamic_cast<const RelativisticEuler::Solutions::RotatingStar&>(
          *serialize_and_deserialize(
              RelativisticEuler::Solutions::RotatingStar(
                  unit_test_src_path() +
                      "/PointwiseFunctions/AnalyticSolutions/RelativisticEuler/"
                      "RotatingStarId_Hybrid.dat",
                  std::make_unique<EquationsOfState::Equilibrium3D<
                      EquationsOfState::HybridEos<
                          EquationsOfState::PolytropicFluid<true>>>>(
                      EquationsOfState::HybridEos<
                          EquationsOfState::PolytropicFluid<true>>(
                          EquationsOfState::PolytropicFluid<true>(123.6, 2.0),
                          2.0)))
                  .get_clone())));

  CHECK(hybrid_solution.equatorial_radius() == approx(14.7126911145735));

  verify_solution_suite(hybrid_solution, error_tolerance);

  // Check that the central values for density, temperature, and electron
  // fraction match that of the `RotNS` output, indicating successful recovery
  // of all three quantities that are fed in from the file.
  // Note: the original EoS fed into RotNS was defined to have constant T=0.1
  // (SpECTRE units) when density is high. The error in this term is from RotNS,
  // not from SpECTRE.
  // Note: the EoS here is an Equilibrium3D<...> one, meaning that electron
  // fraction doesn't actually have any impact on thermodynamic quantities. This
  // quantity is read in, anyway, so we choose to give this EoS an arbitrary and
  // meaningless Y_e distribution, rather than using an EoS like Tabulated3D,
  // for the sake of reducing the complexity of the test
  const auto vars = hybrid_solution.variables(
      tnsr::I<double, 3>{{0.0, 0.0, 0.0}}, 0.0,
      tmpl::list<hydro::Tags::Temperature<double>,
                 hydro::Tags::ElectronFraction<double>>());
  CHECK(get(get<hydro::Tags::Temperature<double>>(vars)) ==
        approx(1.0000000508433475e-1));
  CHECK(get(get<hydro::Tags::ElectronFraction<double>>(vars)) ==
        approx(4.235069808548e-2));

  CHECK_THROWS_WITH(
      RelativisticEuler::Solutions::RotatingStar(
          unit_test_src_path() +
              "/PointwiseFunctions/AnalyticSolutions/RelativisticEuler/"
              "RotatingStarIdDoesNotExist.dat",
          100),
      Catch::Matchers::ContainsSubstring("Cannot open file"));
}
