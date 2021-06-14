// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/Outflow.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace helpers = TestHelpers::evolution::dg;

SPECTRE_TEST_CASE("Unit.GrMhd.BoundaryConditions.Outflow", "[Unit][GrMhd]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/"};
  MAKE_GENERATOR(gen);
  const std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      boundary_condition = TestHelpers::test_creation<std::unique_ptr<
          grmhd::ValenciaDivClean::BoundaryConditions::BoundaryCondition>>(
          "Outflow:\n");

  helpers::test_boundary_condition_with_python<
      grmhd::ValenciaDivClean::BoundaryConditions::Outflow,
      grmhd::ValenciaDivClean::BoundaryConditions::BoundaryCondition,
      grmhd::ValenciaDivClean::System,
      tmpl::list<grmhd::ValenciaDivClean::BoundaryCorrections::Rusanov>>(
      make_not_null(&gen), "Outflow",
      tuples::TaggedTuple<helpers::Tags::PythonFunctionForErrorMessage<>>{
          "error"},
      "Outflow:\n", Index<2>{5}, db::DataBox<tmpl::list<>>{},
      tuples::TaggedTuple<>{});
}
