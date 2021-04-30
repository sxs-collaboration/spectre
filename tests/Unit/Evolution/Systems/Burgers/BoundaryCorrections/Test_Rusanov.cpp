// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "Evolution/Systems/Burgers/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/Burgers/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/Burgers/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

SPECTRE_TEST_CASE("Unit.Burgers.Rusanov", "[Unit][Burgers]") {
  PUPable_reg(Burgers::BoundaryCorrections::Rusanov);
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/Burgers/BoundaryCorrections"};
  MAKE_GENERATOR(gen);

  TestHelpers::evolution::dg::test_boundary_correction_conservation<
      Burgers::System>(
      make_not_null(&gen), Burgers::BoundaryCorrections::Rusanov{},
      Mesh<0>{1, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss}, {},
      {});

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      Burgers::System>(
      make_not_null(&gen), "Rusanov",
      {{"dg_package_data_u", "dg_package_data_normal_dot_flux",
        "dg_package_data_abs_char_speed"}},
      {{"dg_boundary_terms_u"}}, Burgers::BoundaryCorrections::Rusanov{},
      Mesh<0>{1, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss}, {},
      {});

  const auto rusanov = TestHelpers::test_creation<
      std::unique_ptr<Burgers::BoundaryCorrections::BoundaryCorrection>>(
      "Rusanov:");

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      Burgers::System>(
      make_not_null(&gen), "Rusanov",
      {{"dg_package_data_u", "dg_package_data_normal_dot_flux",
        "dg_package_data_abs_char_speed"}},
      {{"dg_boundary_terms_u"}},
      dynamic_cast<const Burgers::BoundaryCorrections::Rusanov&>(*rusanov),
      Mesh<0>{1, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss}, {},
      {});
}
