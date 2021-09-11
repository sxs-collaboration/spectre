// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryCorrections/UpwindPenalty.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
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
  CAPTURE(Dim);
  MAKE_GENERATOR(gen);

  helpers::test_boundary_condition_with_python<
      CurvedScalarWave::BoundaryConditions::
          ConstraintPreservingSphericalRadiation<Dim>,
      CurvedScalarWave::BoundaryConditions::BoundaryCondition<Dim>,
      CurvedScalarWave::System<Dim>,
      tmpl::list<CurvedScalarWave::BoundaryCorrections::UpwindPenalty<Dim>>>(
      make_not_null(&gen),
      "Evolution.Systems.CurvedScalarWave.BoundaryConditions."
      "ConstraintPreservingSphericalRadiation",
      tuples::TaggedTuple<
          helpers::Tags::PythonFunctionForErrorMessage<>,
          helpers::Tags::PythonFunctionName<::Tags::dt<CurvedScalarWave::Pi>>,
          helpers::Tags::PythonFunctionName<
              ::Tags::dt<CurvedScalarWave::Phi<Dim>>>,
          helpers::Tags::PythonFunctionName<::Tags::dt<CurvedScalarWave::Psi>>>{
          "error", "dt_pi_constraint_preserving_spherical_radiation",
          "dt_phi_constraint_preserving_spherical_radiation",
          "dt_psi_constraint_preserving_spherical_radiation"},
      "ConstraintPreservingSphericalRadiation:\n",
      Index<Dim - 1>{Dim == 1 ? 1 : 5}, db::DataBox<tmpl::list<>>{},
      tuples::TaggedTuple<>{});
}
}  // namespace
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.CurvedScalarWave.ConstraintPreservingRadiation",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};
  test<1>();
  test<2>();
  test<3>();
}
