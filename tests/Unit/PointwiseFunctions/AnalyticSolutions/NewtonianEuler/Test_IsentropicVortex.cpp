// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "ErrorHandling/Error.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"  // IWYU pragma: keep
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/IsentropicVortex.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {

void test_create_from_options() noexcept {
  const auto created_solution =
      test_creation<NewtonianEuler::Solutions::IsentropicVortex>(
          "  AdiabaticIndex: 1.43\n"
          "  Center: [2.3, -1.3, -0.6]\n"
          "  MeanVelocity: [-0.3, 0.1, 0.7]\n"
          "  PerturbAmplitude: 0.5\n"
          "  Strength: 3.76");
  CHECK(created_solution ==
        NewtonianEuler::Solutions::IsentropicVortex(
            1.43, {{2.3, -1.3, -0.6}}, {{-0.3, 0.1, 0.7}}, 0.5, 3.76));
}

void test_move() noexcept {
  NewtonianEuler::Solutions::IsentropicVortex vortex(
      1.32, {{3.2, -4.1, 9.0}}, {{0.43, 0.31, -0.68}}, 0.23, 1.65);
  NewtonianEuler::Solutions::IsentropicVortex vortex_copy(
      1.32, {{3.2, -4.1, 9.0}}, {{0.43, 0.31, -0.68}}, 0.23, 1.65);
  test_move_semantics(std::move(vortex), vortex_copy);  //  NOLINT
}

void test_serialize() noexcept {
  NewtonianEuler::Solutions::IsentropicVortex vortex(
      1.5, {{3.2, 0.1, -1.3}}, {{0.2, -0.25, 0.41}}, 0.13, 1.65);
  test_serialization(vortex);
}

template <typename DataType>
void test_variables(const DataType& used_for_size) noexcept {
  const double adiabatic_index = 1.56;
  const std::array<double, 3> center = {{0.15, -0.02, 1.9}};
  const std::array<double, 3> mean_velocity = {{1.2, 0.43, 0.5}};
  const double perturbation_amplitude = 0.47;
  const double strength = 2.0;

  pypp::check_with_random_values<
      1, tmpl::list<NewtonianEuler::Tags::MassDensity<DataType>,
                    NewtonianEuler::Tags::Velocity<DataType, 3>,
                    NewtonianEuler::Tags::SpecificInternalEnergy<DataType>>>(
      &NewtonianEuler::Solutions::IsentropicVortex::primitive_variables<
          DataType>,
      NewtonianEuler::Solutions::IsentropicVortex(
          adiabatic_index, center, mean_velocity, perturbation_amplitude,
          strength),
      "TestFunctions", {"mass_density", "velocity", "specific_internal_energy"},
      {{{-1.0, 1.0}}}, std::make_tuple(adiabatic_index, center, mean_velocity,
                                       perturbation_amplitude, strength),
      used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.Vortex",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/NewtonianEuler"};

  test_create_from_options();
  test_serialize();
  test_move();

  test_variables(std::numeric_limits<double>::signaling_NaN());
  test_variables(DataVector(5));
}

struct Vortex {
  using type = NewtonianEuler::Solutions::IsentropicVortex;
  static constexpr OptionString help = {"A Newtonian isentropic vortex."};
};

// [[OutputRegex, The strength must be non-negative.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.VortexStrength",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  NewtonianEuler::Solutions::IsentropicVortex test_vortex(
      1.3, {{1.0, 1.0, 1.0}}, {{0.0, 0.0, 0.0}}, -0.15, -1.7);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, In string:.*At line 6 column 13:.Value -0.2 is below the lower
// bound of 0]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.VortexStrengthOpt",
    "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  Options<tmpl::list<Vortex>> test_options("");
  test_options.parse(
      "Vortex:\n"
      "  AdiabaticIndex: 1.4\n"
      "  Center: [-3.9, 1.1, 4.5]\n"
      "  MeanVelocity: [0.1, 0.0, 0.65]\n"
      "  PerturbAmplitude: 0.13\n"
      "  Strength: -0.2");
  test_options.get<Vortex>();
}
