// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/AnalyticConstant.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryCorrections/UpwindPenalty.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
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
void test() {
  MAKE_GENERATOR(gen);

  helpers::test_boundary_condition_with_python<
      CurvedScalarWave::BoundaryConditions::AnalyticConstant<Dim>,
      CurvedScalarWave::BoundaryConditions::BoundaryCondition<Dim>,
      CurvedScalarWave::System<Dim>,
      tmpl::list<CurvedScalarWave::BoundaryCorrections::UpwindPenalty<Dim>>>(
      make_not_null(&gen),
      "Evolution.Systems.CurvedScalarWave.BoundaryConditions."
      "AnalyticConstant",
      tuples::TaggedTuple<
          helpers::Tags::PythonFunctionForErrorMessage<>,
          helpers::Tags::PythonFunctionName<CurvedScalarWave::Tags::Psi>,
          helpers::Tags::PythonFunctionName<CurvedScalarWave::Tags::Pi>,
          helpers::Tags::PythonFunctionName<CurvedScalarWave::Tags::Phi<Dim>>,
          helpers::Tags::PythonFunctionName<gr::Tags::Lapse<DataVector>>,
          helpers::Tags::PythonFunctionName<gr::Tags::Shift<DataVector, Dim>>,
          helpers::Tags::PythonFunctionName<
              CurvedScalarWave::Tags::ConstraintGamma1>,
          helpers::Tags::PythonFunctionName<
              CurvedScalarWave::Tags::ConstraintGamma2>>{
          "error", "psi_analytic_constant", "pi_analytic_constant",
          "phi_analytic_constant", "lapse_analytic_constant",
          "shift_analytic_constant", "constraint_gamma1_analytic_constant",
          "constraint_gamma2_analytic_constant"},
      "AnalyticConstant:\n"
      "  Amplitude: 2.71\n",
      Index<Dim - 1>{Dim == 1 ? 0 : 5}, db::DataBox<tmpl::list<>>{},
      tuples::TaggedTuple<>{});
}
}  // namespace
SPECTRE_TEST_CASE(
    "Unit.CurvedScalarWave.BoundaryConditions.AnalyticConstant",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};
  test<1>();
  test<2>();
  test<3>();
}
