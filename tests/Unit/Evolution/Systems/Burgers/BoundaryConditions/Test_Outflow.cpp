// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/Outflow.hpp"
#include "Evolution/Systems/Burgers/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/Burgers/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace helpers = TestHelpers::evolution::dg;

SPECTRE_TEST_CASE("Unit.Burgers.BoundaryConditions.Outflow",
                  "[Unit][Burgers]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/Burgers/BoundaryConditions/"};
  MAKE_GENERATOR(gen);
  const std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      boundary_condition = TestHelpers::test_factory_creation<
          Burgers::BoundaryConditions::BoundaryCondition>("Outflow:\n");

  helpers::test_boundary_condition_with_python<
      Burgers::BoundaryConditions::Outflow,
      Burgers::BoundaryConditions::BoundaryCondition, Burgers::System,
      tmpl::list<Burgers::BoundaryCorrections::Rusanov>>(
      make_not_null(&gen), "Outflow",
      tuples::TaggedTuple<helpers::Tags::PythonFunctionForErrorMessage<>>{
          "error"},
      "Outflow:\n", Index<0>{1}, db::DataBox<tmpl::list<>>{},
      tuples::TaggedTuple<>{});
}
