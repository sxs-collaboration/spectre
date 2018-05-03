// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/PrimitiveFromConservative.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace {

template <size_t Dim, typename DataType>
void test_velocity(const DataType& used_for_size) {
  pypp::check_with_random_values<2>(
      static_cast<tnsr::I<DataType, Dim> (*)(const Scalar<DataType>&,
                                             const tnsr::I<DataType, Dim>&)>(
          &NewtonianEuler::velocity<Dim, DataType>),
      "TestFunctions", "velocity", {{{-1.0, 1.0}, {-2.0, 2.0}}}, used_for_size);
}

template <size_t Dim, typename DataType>
void test_primitive_from_conservative(const DataType& used_for_size) {
  pypp::check_with_random_values<3>(
      &NewtonianEuler::primitive_from_conservative<Dim, DataType>,
      "TestFunctions", {"velocity", "specific_internal_energy"},
      {{{-1.0, 1.0}, {-2.0, 2.0}, {-3.0, 3.0}}}, used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.PrimitiveFromConservative",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/NewtonianEuler"};

  const double d = std::numeric_limits<double>::signaling_NaN();
  test_velocity<1>(d);
  test_velocity<2>(d);
  test_velocity<3>(d);
  test_primitive_from_conservative<1>(d);
  test_primitive_from_conservative<2>(d);
  test_primitive_from_conservative<3>(d);

  const DataVector dv(5);
  test_velocity<1>(dv);
  test_velocity<2>(dv);
  test_velocity<3>(dv);
  test_primitive_from_conservative<1>(dv);
  test_primitive_from_conservative<2>(dv);
  test_primitive_from_conservative<3>(dv);
}
