// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Outflow.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/UpwindPenalty.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace helpers = TestHelpers::evolution::dg;
namespace {

template <size_t Dim>
void test() noexcept {
  MAKE_GENERATOR(gen);
  const std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      boundary_condition = TestHelpers::test_creation<std::unique_ptr<
          GeneralizedHarmonic::BoundaryConditions::BoundaryCondition<Dim>>>(
          "Outflow:\n");

  helpers::test_boundary_condition_with_python<
      GeneralizedHarmonic::BoundaryConditions::Outflow<Dim>,
      GeneralizedHarmonic::BoundaryConditions::BoundaryCondition<Dim>,
      GeneralizedHarmonic::System<Dim>,
      tmpl::list<GeneralizedHarmonic::BoundaryCorrections::UpwindPenalty<Dim>>>(
      make_not_null(&gen), "Outflow",
      tuples::TaggedTuple<helpers::Tags::PythonFunctionForErrorMessage<>>{
          "error"},
      "Outflow:\n", Index<Dim - 1>{Dim == 1 ? 0 : 5},
      db::DataBox<tmpl::list<>>{},
      tuples::TaggedTuple<helpers::Tags::Range<
          GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1>>{
          std::array{0.0, 1.0}});
}
}  // namespace
SPECTRE_TEST_CASE("Unit.GeneralizedHarmonic.BoundaryConditions.Outflow",
                  "[Unit][GrMhd]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/"};
  test<1>();
  test<2>();
  test<3>();
}
