// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <limits>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "tests/Unit/PointwiseFunctions/Hydro/EquationsOfState/TestHelpers.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestCreation.hpp"

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.EquationsOfState.IdealFluid",
                  "[Unit][EquationsOfState]") {
  namespace EoS = EquationsOfState;
  Parallel::register_derived_classes_with_charm<
      EoS::EquationOfState<true, 2>>();
  Parallel::register_derived_classes_with_charm<
      EoS::EquationOfState<false, 2>>();
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/Hydro/EquationsOfState/"};
  const double d_for_size = std::numeric_limits<double>::signaling_NaN();
  const DataVector dv_for_size(5);
  TestHelpers::EquationsOfState::check(EoS::IdealFluid<true>{5.0 / 3.0},
                                       "ideal_fluid", d_for_size, 5.0 / 3.0);
  TestHelpers::EquationsOfState::check(EoS::IdealFluid<true>{4.0 / 3.0},
                                       "ideal_fluid", dv_for_size, 4.0 / 3.0);
  TestHelpers::EquationsOfState::check(EoS::IdealFluid<false>{5.0 / 3.0},
                                       "ideal_fluid", d_for_size, 5.0 / 3.0);
  TestHelpers::EquationsOfState::check(EoS::IdealFluid<false>{4.0 / 3.0},
                                       "ideal_fluid", dv_for_size, 4.0 / 3.0);

  TestHelpers::EquationsOfState::check(
      test_factory_creation<EoS::EquationOfState<true, 2>>(
          {"  IdealFluid:\n"
           "    AdiabaticIndex: 1.6666666666666667\n"}),
      "ideal_fluid", d_for_size, 5.0 / 3.0);
  TestHelpers::EquationsOfState::check(
      test_factory_creation<EoS::EquationOfState<true, 2>>(
          {"  IdealFluid:\n"
           "    AdiabaticIndex: 1.3333333333333333\n"}),
      "ideal_fluid", dv_for_size, 4.0 / 3.0);

  TestHelpers::EquationsOfState::check(
      test_factory_creation<EoS::EquationOfState<false, 2>>(
          {"  IdealFluid:\n"
           "    AdiabaticIndex: 1.6666666666666667\n"}),
      "ideal_fluid", d_for_size, 5.0 / 3.0);
  TestHelpers::EquationsOfState::check(
      test_factory_creation<EoS::EquationOfState<false, 2>>(
          {"  IdealFluid:\n"
           "    AdiabaticIndex: 1.3333333333333333\n"}),
      "ideal_fluid", dv_for_size, 4.0 / 3.0);
}
