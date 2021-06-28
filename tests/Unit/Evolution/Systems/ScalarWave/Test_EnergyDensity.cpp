// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarWave/EnergyDensity.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"

namespace {
template <size_t SpatialDim>
void test_energy_density(const DataVector& used_for_size) {
  void (*f)(const gsl::not_null<Scalar<DataVector>*>, const Scalar<DataVector>&,
            const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&) =
      &ScalarWave::energy_density<SpatialDim>;
  pypp::check_with_random_values<1>(f, "EnergyDensity", {"energy_density"},
                                    {{{-1., 1.}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarWave.EnergyDensity",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "Evolution/Systems/ScalarWave/");

  const DataVector used_for_size(5);
  test_energy_density<1>(used_for_size);
  test_energy_density<2>(used_for_size);
  test_energy_density<3>(used_for_size);
}
