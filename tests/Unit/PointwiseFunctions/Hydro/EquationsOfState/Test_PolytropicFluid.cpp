// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/PointwiseFunctions/Hydro/EquationsOfState/TestHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.EquationsOfState.PolytropicFluid",
                  "[Unit][EquationsOfState]") {
  namespace EoS = EquationsOfState;
  Parallel::register_derived_classes_with_charm<
      EoS::EquationOfState<true, 1>>();
  Parallel::register_derived_classes_with_charm<
      EoS::EquationOfState<false, 1>>();
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/Hydro/EquationsOfState/"};
  const double d_for_size = std::numeric_limits<double>::signaling_NaN();
  const DataVector dv_for_size(5);
  TestHelpers::EquationsOfState::check(EoS::PolytropicFluid<true>{100.0, 2.0},
                                       "polytropic", d_for_size, 100.0, 2.0);
  TestHelpers::EquationsOfState::check(EoS::PolytropicFluid<true>{134.0, 1.5},
                                       "polytropic", dv_for_size, 134.0, 1.5);
  TestHelpers::EquationsOfState::check(EoS::PolytropicFluid<false>{121.0, 1.2},
                                       "polytropic", d_for_size, 121.0, 1.2);
  TestHelpers::EquationsOfState::check(EoS::PolytropicFluid<false>{117.0, 1.12},
                                       "polytropic", dv_for_size, 117.0, 1.12);

  TestHelpers::EquationsOfState::check(
      TestHelpers::test_creation<
          std::unique_ptr<EoS::EquationOfState<true, 1>>>(
          {"PolytropicFluid:\n"
           "  PolytropicConstant: 100.0\n"
           "  PolytropicExponent: 2.0\n"}),
      "polytropic", d_for_size, 100.0, 2.0);
  TestHelpers::EquationsOfState::check(
      TestHelpers::test_creation<
          std::unique_ptr<EoS::EquationOfState<true, 1>>>(
          {"PolytropicFluid:\n"
           "  PolytropicConstant: 134.0\n"
           "  PolytropicExponent: 1.5\n"}),
      "polytropic", dv_for_size, 134.0, 1.5);

  TestHelpers::EquationsOfState::check(
      TestHelpers::test_creation<
          std::unique_ptr<EoS::EquationOfState<false, 1>>>(
          {"PolytropicFluid:\n"
           "  PolytropicConstant: 121.0\n"
           "  PolytropicExponent: 1.2\n"}),
      "polytropic", d_for_size, 121.0, 1.2);
  TestHelpers::EquationsOfState::check(
      TestHelpers::test_creation<
          std::unique_ptr<EoS::EquationOfState<false, 1>>>(
          {"PolytropicFluid:\n"
           "  PolytropicConstant: 117.0\n"
           "  PolytropicExponent: 1.12\n"}),
      "polytropic", dv_for_size, 117.0, 1.12);
}
