// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/UniformAcceleration.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"

namespace {

template <size_t Dim>
void test_sources(const std::array<double, Dim>& acceleration_field,
                  const DataVector& used_for_size) noexcept {
  NewtonianEuler::Sources::UniformAcceleration<Dim> source(acceleration_field);
  pypp::check_with_random_values<2>(
      &NewtonianEuler::Sources::UniformAcceleration<Dim>::apply, source,
      "UniformAcceleration",
      {"source_momentum_density", "source_energy_density"},
      {{{0.0, 3.0}, {-1.0, 1.0}}}, std::make_tuple(acceleration_field),
      used_for_size);

  NewtonianEuler::Sources::UniformAcceleration<Dim> source_to_move(
      acceleration_field);
  test_move_semantics(std::move(source_to_move), source);  // NOLINT

  test_serialization(source);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.Sources.UniformAcceleration",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/NewtonianEuler/Sources"};

  const DataVector used_for_size(5);
  test_sources<1>({{-2.0}}, used_for_size);
  test_sources<2>({{-1.2, 8.7}}, used_for_size);
  test_sources<3>({{1.8, -0.05, 5.7}}, used_for_size);
}
