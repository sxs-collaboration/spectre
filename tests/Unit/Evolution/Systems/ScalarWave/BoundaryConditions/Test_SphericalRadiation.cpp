// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "Evolution/Systems/ScalarWave/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/ScalarWave/BoundaryConditions/SphericalRadiation.hpp"
#include "Evolution/Systems/ScalarWave/BoundaryCorrections/UpwindPenalty.hpp"
#include "Evolution/Systems/ScalarWave/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Range.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace helpers = TestHelpers::evolution::dg;

namespace {
template <size_t Dim>
void test() {
  CAPTURE(Dim);
  MAKE_GENERATOR(gen);
  for (const std::string& bc_string : {"Sommerfeld", "BaylissTurkel"}) {
    CAPTURE(bc_string);
    helpers::test_boundary_condition_with_python<
        ScalarWave::BoundaryConditions::SphericalRadiation<Dim>,
        ScalarWave::BoundaryConditions::BoundaryCondition<Dim>,
        ScalarWave::System<Dim>,
        tmpl::list<ScalarWave::BoundaryCorrections::UpwindPenalty<Dim>>>(
        make_not_null(&gen), "SphericalRadiation",
        tuples::TaggedTuple<
            helpers::Tags::PythonFunctionForErrorMessage<>,
            helpers::Tags::PythonFunctionName<ScalarWave::Tags::Psi>,
            helpers::Tags::PythonFunctionName<ScalarWave::Tags::Pi>,
            helpers::Tags::PythonFunctionName<ScalarWave::Tags::Phi<Dim>>,
            helpers::Tags::PythonFunctionName<
                ScalarWave::Tags::ConstraintGamma2>>{
            "error", "psi", "pi_" + bc_string, "phi", "constraint_gamma2"},
        "SphericalRadiation:\n"
        "  Type: " +
            bc_string,
        Index<Dim - 1>{Dim == 1 ? 1 : 5}, db::DataBox<tmpl::list<>>{},
        tuples::TaggedTuple<
            helpers::Tags::Range<ScalarWave::Tags::ConstraintGamma2>>{
            std::array{0.0, 1.0}});
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ScalarWave.BoundaryConditions.SphericalRadiation",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/ScalarWave/BoundaryConditions/"};
  test<1>();
  test<2>();
  test<3>();
}
