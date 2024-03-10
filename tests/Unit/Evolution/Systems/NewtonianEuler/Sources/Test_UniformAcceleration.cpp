// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/UniformAcceleration.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"

namespace {
template <size_t Dim>
struct UniformAccelerationProxy
    : public NewtonianEuler::Sources::UniformAcceleration<Dim> {
  void apply(
      const gsl::not_null<tnsr::I<DataVector, Dim>*> source_momentum_density,
      const gsl::not_null<Scalar<DataVector>*> source_energy_density,
      const Scalar<DataVector>& mass_density_cons,
      const tnsr::I<DataVector, Dim>& momentum_density) const {
    get(*source_energy_density) = 0.0;
    for (size_t i = 0; i < Dim; ++i) {
      source_momentum_density->get(i) = 0.0;
    }
    const EquationsOfState::IdealFluid<false> eos{5.0 / 3.0};
    Scalar<DataVector> source_mass_density_cons{};
    static_cast<const NewtonianEuler::Sources::UniformAcceleration<Dim>&>(
        *this)(make_not_null(&source_mass_density_cons),
               source_momentum_density, source_energy_density,
               mass_density_cons, momentum_density, {}, {}, {}, {}, eos, {},
               0.0);
  }

  explicit UniformAccelerationProxy(
      const std::array<double, Dim> acceleration_field)
      : NewtonianEuler::Sources::UniformAcceleration<Dim>(acceleration_field) {}
};

template <size_t Dim>
void test_sources(const std::array<double, Dim>& acceleration_field,
                  const DataVector& used_for_size) {
  UniformAccelerationProxy<Dim> source_proxy(acceleration_field);
  pypp::check_with_random_values<1>(
      &UniformAccelerationProxy<Dim>::apply, source_proxy,
      "UniformAcceleration",
      {"source_momentum_density", "source_energy_density"}, {{{0.0, 3.0}}},
      std::make_tuple(acceleration_field), used_for_size);

  NewtonianEuler::Sources::UniformAcceleration<Dim> source(acceleration_field);
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
