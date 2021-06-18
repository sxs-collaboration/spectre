// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
namespace helpers = TestHelpers::evolution::dg;

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.BoundaryCorrections.Hll",
                  "[Unit][GrMhd]") {
  PUPable_reg(grmhd::ValenciaDivClean::BoundaryCorrections::Hll);
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections"};
  MAKE_GENERATOR(gen);

  using system = grmhd::ValenciaDivClean::System;

  const tuples::TaggedTuple<
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<3, Frame::Inertial, DataVector>>>
      ranges{std::array{0.3, 1.0}, std::array{0.01, 0.02}};

  TestHelpers::evolution::dg::test_boundary_correction_conservation<system>(
      make_not_null(&gen), grmhd::ValenciaDivClean::BoundaryCorrections::Hll{},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss}, {},
      ranges);

  TestHelpers::evolution::dg::test_boundary_correction_with_python<system>(
      make_not_null(&gen), "Hll",
      {{"dg_package_data_tilde_d", "dg_package_data_tilde_tau",
        "dg_package_data_tilde_s", "dg_package_data_tilde_b",
        "dg_package_data_tilde_phi", "dg_package_data_normal_dot_flux_tilde_d",
        "dg_package_data_normal_dot_flux_tilde_tau",
        "dg_package_data_normal_dot_flux_tilde_s",
        "dg_package_data_normal_dot_flux_tilde_b",
        "dg_package_data_normal_dot_flux_tilde_phi",
        "dg_package_data_largest_outgoing_char_speed",
        "dg_package_data_largest_ingoing_char_speed"}},
      {{"dg_boundary_terms_tilde_d", "dg_boundary_terms_tilde_tau",
        "dg_boundary_terms_tilde_s", "dg_boundary_terms_tilde_b",
        "dg_boundary_terms_tilde_phi"}},
      grmhd::ValenciaDivClean::BoundaryCorrections::Hll{},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss}, {},
      ranges);

  const auto hll = TestHelpers::test_creation<std::unique_ptr<
      grmhd::ValenciaDivClean::BoundaryCorrections::BoundaryCorrection>>(
      "Hll:");

  TestHelpers::evolution::dg::test_boundary_correction_with_python<system>(
      make_not_null(&gen), "Hll",
      {{"dg_package_data_tilde_d", "dg_package_data_tilde_tau",
        "dg_package_data_tilde_s", "dg_package_data_tilde_b",
        "dg_package_data_tilde_phi", "dg_package_data_normal_dot_flux_tilde_d",
        "dg_package_data_normal_dot_flux_tilde_tau",
        "dg_package_data_normal_dot_flux_tilde_s",
        "dg_package_data_normal_dot_flux_tilde_b",
        "dg_package_data_normal_dot_flux_tilde_phi",
        "dg_package_data_largest_outgoing_char_speed",
        "dg_package_data_largest_ingoing_char_speed"}},
      {{"dg_boundary_terms_tilde_d", "dg_boundary_terms_tilde_tau",
        "dg_boundary_terms_tilde_s", "dg_boundary_terms_tilde_b",
        "dg_boundary_terms_tilde_phi"}},
      dynamic_cast<const grmhd::ValenciaDivClean::BoundaryCorrections::Hll&>(
          *hll),
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss}, {},
      ranges);
}
}  // namespace
