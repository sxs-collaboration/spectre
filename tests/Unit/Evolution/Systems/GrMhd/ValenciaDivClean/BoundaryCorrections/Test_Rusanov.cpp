// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.BoundaryCorrections.Rusanov",
                  "[Unit][GrMhd]") {
  PUPable_reg(grmhd::ValenciaDivClean::BoundaryCorrections::Rusanov);
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections"};
  MAKE_GENERATOR(gen);

  using system = grmhd::ValenciaDivClean::System;

  TestHelpers::evolution::dg::test_boundary_correction_conservation<system>(
      make_not_null(&gen),
      grmhd::ValenciaDivClean::BoundaryCorrections::Rusanov{},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss}, {},
      {});

  TestHelpers::evolution::dg::test_boundary_correction_with_python<system>(
      make_not_null(&gen), "Rusanov", "dg_package_data", "dg_boundary_terms",
      grmhd::ValenciaDivClean::BoundaryCorrections::Rusanov{},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss}, {},
      {});

  const auto rusanov = TestHelpers::test_creation<std::unique_ptr<
      grmhd::ValenciaDivClean::BoundaryCorrections::BoundaryCorrection>>(
      "Rusanov:");

  TestHelpers::evolution::dg::test_boundary_correction_with_python<system>(
      make_not_null(&gen), "Rusanov", "dg_package_data", "dg_boundary_terms",
      dynamic_cast<
          const grmhd::ValenciaDivClean::BoundaryCorrections::Rusanov&>(
          *rusanov),
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss}, {},
      {});

  CHECK_FALSE(grmhd::ValenciaDivClean::BoundaryCorrections::Rusanov{} !=
              grmhd::ValenciaDivClean::BoundaryCorrections::Rusanov{});
}
