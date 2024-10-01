// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/BoundaryCorrections/UpwindPenalty.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/UpwindPenalty.hpp"
#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryCorrections/ProductOfCorrections.hpp"
#include "Evolution/Systems/ScalarTensor/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace helpers = TestHelpers::evolution::dg;

namespace {

void test() {
  MAKE_GENERATOR(gen);

  helpers::test_boundary_condition_with_python<
      ScalarTensor::BoundaryConditions::DemandOutgoingCharSpeeds,
      ScalarTensor::BoundaryConditions::BoundaryCondition, ScalarTensor::System,
      tmpl::list<ScalarTensor::BoundaryCorrections::ProductOfCorrections<
          gh::BoundaryCorrections::UpwindPenalty<3>,
          CurvedScalarWave::BoundaryCorrections::UpwindPenalty<3>>>>(
      make_not_null(&gen), "DemandOutgoingCharSpeeds",
      tuples::TaggedTuple<helpers::Tags::PythonFunctionForErrorMessage<>>{
          "error"},
      "DemandOutgoingCharSpeeds:\n", Index<2>{5}, db::DataBox<tmpl::list<>>{},
      tuples::TaggedTuple<
          helpers::Tags::Range<gh::ConstraintDamping::Tags::ConstraintGamma1>>{
          std::array{0.0, 1.0}});
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ScalarTensor.BConds.DemandOutgoingCharSpeeds",
    "[Unit][Evolution]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/ScalarTensor/BoundaryConditions/"};
  test();
}
