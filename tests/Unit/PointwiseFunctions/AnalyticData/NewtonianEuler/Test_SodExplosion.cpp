// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/NewtonianEuler/SodExplosion.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

template <size_t Dim>
struct SodExplosionProxy : NewtonianEuler::AnalyticData::SodExplosion<Dim> {
  using NewtonianEuler::AnalyticData::SodExplosion<Dim>::SodExplosion;

  template <typename DataType>
  using variables_tags =
      tmpl::list<NewtonianEuler::Tags::MassDensity<DataType>,
                 NewtonianEuler::Tags::Velocity<DataType, Dim, Frame::Inertial>,
                 NewtonianEuler::Tags::SpecificInternalEnergy<DataType>,
                 NewtonianEuler::Tags::Pressure<DataType>>;

  tuples::tagged_tuple_from_typelist<variables_tags<DataVector>>
  primitive_variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x) const noexcept {
    return this->variables(x, variables_tags<DataVector>{});
  }
};

template <size_t Dim>
void test_analytic_data(const DataVector& used_for_size) noexcept {
  const double initial_radius = 0.5;
  const double inner_mass_density = 1.0;
  const double inner_pressure = 1.0;
  const double outer_mass_density = 0.125;
  const double outer_pressure = 0.1;

  const auto members =
      std::make_tuple(initial_radius, inner_mass_density, inner_pressure,
                      outer_mass_density, outer_pressure);

  const SodExplosionProxy<Dim> sod_explosion(initial_radius, inner_mass_density,
                                             inner_pressure, outer_mass_density,
                                             outer_pressure);
  pypp::check_with_random_values<1>(
      &SodExplosionProxy<Dim>::primitive_variables, sod_explosion,
      "SodExplosion",
      {"mass_density", "velocity", "specific_internal_energy", "pressure"},
      {{{0.0, 1.0}}}, members, used_for_size);

  const auto sod_explosion_from_options = TestHelpers::test_creation<
      NewtonianEuler::AnalyticData::SodExplosion<Dim>>(
      "InitialRadius: 0.5\n"
      "InnerMassDensity: 1.0\n"
      "InnerPressure: 1.0\n"
      "OuterMassDensity: 0.125\n"
      "OuterPressure: 0.1\n");
  CHECK(sod_explosion_from_options == sod_explosion);

  // run post-serialized state through checks with random numbers
  pypp::check_with_random_values<1>(
      &SodExplosionProxy<Dim>::primitive_variables,
      serialize_and_deserialize(sod_explosion), "SodExplosion",
      {"mass_density", "velocity", "specific_internal_energy", "pressure"},
      {{{0.0, 1.0}}}, members, used_for_size);

  CHECK_THROWS_WITH(NewtonianEuler::AnalyticData::SodExplosion<Dim>(
                        initial_radius, inner_mass_density, inner_pressure,
                        outer_mass_density + 100.0, outer_pressure,
                        Options::Context{false, {}, 1, 1}),
                    Catch::Matchers::Contains("The inner mass density ("));
  CHECK_THROWS_WITH(NewtonianEuler::AnalyticData::SodExplosion<Dim>(
                        initial_radius, inner_mass_density, inner_pressure,
                        outer_mass_density, outer_pressure + 100.0,
                        Options::Context{false, {}, 1, 1}),
                    Catch::Matchers::Contains("The inner pressure ("));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticData.NewtEuler.SodExplosion",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticData/NewtonianEuler"};

  DataVector used_for_size{5};
  test_analytic_data<2>(used_for_size);
  test_analytic_data<3>(used_for_size);
}
