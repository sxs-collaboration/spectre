// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>

#include "Evolution/Systems/ForceFree/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/ForceFree/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace ForceFree {

namespace {

SPECTRE_TEST_CASE("Unit.ForceFree.BoundaryCorrections.Rusanov",
                  "[Unit][ForceFree]") {
  PUPable_reg(BoundaryCorrections::Rusanov);
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/ForceFree/BoundaryCorrections"};
  MAKE_GENERATOR(gen);

  using system = System;

  TestHelpers::evolution::dg::test_boundary_correction_conservation<system>(
      make_not_null(&gen), BoundaryCorrections::Rusanov{},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss}, {},
      {});

  TestHelpers::evolution::dg::test_boundary_correction_with_python<system>(
      make_not_null(&gen), "Rusanov",
      {{"dg_package_data_tilde_e", "dg_package_data_tilde_b",
        "dg_package_data_tilde_psi", "dg_package_data_tilde_phi",
        "dg_package_data_tilde_q", "dg_package_data_normal_dot_flux_tilde_e",
        "dg_package_data_normal_dot_flux_tilde_b",
        "dg_package_data_normal_dot_flux_tilde_psi",
        "dg_package_data_normal_dot_flux_tilde_phi",
        "dg_package_data_normal_dot_flux_tilde_q",
        "dg_package_data_abs_char_speed"}},
      {{"dg_boundary_terms_tilde_e", "dg_boundary_terms_tilde_b",
        "dg_boundary_terms_tilde_psi", "dg_boundary_terms_tilde_phi",
        "dg_boundary_terms_tilde_q"}},
      BoundaryCorrections::Rusanov{},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss}, {},
      {});

  const auto rusanov = TestHelpers::test_creation<
      std::unique_ptr<BoundaryCorrections::BoundaryCorrection>>("Rusanov:");

  TestHelpers::evolution::dg::test_boundary_correction_with_python<system>(
      make_not_null(&gen), "Rusanov",
      {{"dg_package_data_tilde_e", "dg_package_data_tilde_b",
        "dg_package_data_tilde_psi", "dg_package_data_tilde_phi",
        "dg_package_data_tilde_q", "dg_package_data_normal_dot_flux_tilde_e",
        "dg_package_data_normal_dot_flux_tilde_b",
        "dg_package_data_normal_dot_flux_tilde_psi",
        "dg_package_data_normal_dot_flux_tilde_phi",
        "dg_package_data_normal_dot_flux_tilde_q",
        "dg_package_data_abs_char_speed"}},
      {{"dg_boundary_terms_tilde_e", "dg_boundary_terms_tilde_b",
        "dg_boundary_terms_tilde_psi", "dg_boundary_terms_tilde_phi",
        "dg_boundary_terms_tilde_q"}},
      dynamic_cast<const BoundaryCorrections::Rusanov&>(*rusanov),
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss}, {},
      {});

  CHECK_FALSE(BoundaryCorrections::Rusanov{} != BoundaryCorrections::Rusanov{});
}

}  // namespace
}  // namespace ForceFree
