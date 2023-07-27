// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "Evolution/Systems/ForceFree/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"
#include "Evolution/Systems/ForceFree/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/ForceFree/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/ForceFree/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace helpers = TestHelpers::evolution::dg;

SPECTRE_TEST_CASE("Unit.ForceFree.BoundaryConditions.DemandOutgoingCharSpeeds",
                  "[Unit][GrMhd]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/ForceFree/BoundaryConditions/"};
  MAKE_GENERATOR(gen);

  helpers::test_boundary_condition_with_python<
      ForceFree::BoundaryConditions::DemandOutgoingCharSpeeds,
      ForceFree::BoundaryConditions::BoundaryCondition, ForceFree::System,
      tmpl::list<ForceFree::BoundaryCorrections::Rusanov>>(
      make_not_null(&gen), "DemandOutgoingCharSpeeds",
      tuples::TaggedTuple<helpers::Tags::PythonFunctionForErrorMessage<>>{
          "error"},
      "DemandOutgoingCharSpeeds:\n", Index<2>{5}, db::DataBox<tmpl::list<>>{},
      tuples::TaggedTuple<>{});
}
