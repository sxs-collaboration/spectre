// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/NewtonianMhd/ConservativeFromPrimitive.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"

namespace {

template <size_t Dim>
void test_conservative_from_primitive(const DataVector& used_for_size) {
  pypp::check_with_random_values<3>(
      &NewtonianMhd::ConservativeFromPrimitive<Dim>::apply,
      "ConservativeFromPrimitive",
      {"mass_density_cons", "momentum_density", "energy_density"},
      {{{-1.0, 1.0}, {-2.0, 2.0}, {-3.0, 3.0}}}, used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianMhd.ConservativeFromPrimitive",
    "[Unit][Evolution]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/NewtonianMhd"};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_conservative_from_primitive, (1, 2, 3))
}
