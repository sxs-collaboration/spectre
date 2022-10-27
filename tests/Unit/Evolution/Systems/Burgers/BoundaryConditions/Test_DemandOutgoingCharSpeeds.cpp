// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/Factory.hpp"
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

SPECTRE_TEST_CASE("Unit.Burgers.BoundaryConditions.DemandOutgoingCharSpeeds",
                  "[Unit][Burgers]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/Burgers/BoundaryConditions/"};
  MAKE_GENERATOR(gen);

  helpers::test_boundary_condition_with_python<
      Burgers::BoundaryConditions::DemandOutgoingCharSpeeds,
      Burgers::BoundaryConditions::BoundaryCondition, Burgers::System,
      tmpl::list<Burgers::BoundaryCorrections::Rusanov>>(
      make_not_null(&gen), "DemandOutgoingCharSpeeds",
      tuples::TaggedTuple<helpers::Tags::PythonFunctionForErrorMessage<>>{
          "error"},
      "DemandOutgoingCharSpeeds:\n", Index<0>{1}, db::DataBox<tmpl::list<>>{},
      tuples::TaggedTuple<>{});
}
