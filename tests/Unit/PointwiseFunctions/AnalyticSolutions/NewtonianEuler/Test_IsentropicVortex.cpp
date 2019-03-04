// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "ErrorHandling/Error.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"  // IWYU pragma: keep
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/IsentropicVortex.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_include <pup.h>

namespace {

template <size_t Dim>
struct IsentropicVortexProxy
    : NewtonianEuler::Solutions::IsentropicVortex<Dim> {
  using NewtonianEuler::Solutions::IsentropicVortex<Dim>::IsentropicVortex;

  template <typename DataType>
  using variables_tags =
      tmpl::list<NewtonianEuler::Tags::MassDensity<DataType>,
                 NewtonianEuler::Tags::Velocity<DataType, Dim, Frame::Inertial>,
                 NewtonianEuler::Tags::SpecificInternalEnergy<DataType>,
                 NewtonianEuler::Tags::Pressure<DataType>>;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<variables_tags<DataType>>
  primitive_variables(const tnsr::I<DataType, Dim, Frame::Inertial>& x,
                      double t) const noexcept {
    return this->variables(x, t, variables_tags<DataType>{});
  }
};

template <size_t Dim, typename DataType>
void test_solution(const DataType& used_for_size,
                   const std::array<double, Dim>& center,
                   const std::string& center_option,
                   const std::array<double, Dim>& mean_velocity,
                   const std::string& mean_velocity_option) noexcept {
  IsentropicVortexProxy<Dim> vortex(1.43, center, mean_velocity, 0.5, 3.76);
  pypp::check_with_random_values<
      1,
      typename IsentropicVortexProxy<Dim>::template variables_tags<DataType>>(
      &IsentropicVortexProxy<Dim>::template primitive_variables<DataType>,
      vortex, "TestFunctions",
      {"mass_density", "velocity", "specific_internal_energy", "pressure"},
      {{{-1., 1.}}}, std::make_tuple(1.43, center, mean_velocity, 0.5, 3.76),
      used_for_size);

  const auto vortex_from_options =
      test_creation<NewtonianEuler::Solutions::IsentropicVortex<Dim>>(
          "  AdiabaticIndex: 1.43\n"
          "  Center: " +
          center_option +
          "\n"
          "  MeanVelocity: " +
          mean_velocity_option +
          "\n"
          "  PerturbAmplitude: 0.5\n"
          "  Strength: 3.76");
  CHECK(vortex_from_options == vortex);

  IsentropicVortexProxy<Dim> vortex_to_move(1.43, center, mean_velocity, 0.5,
                                            3.76);
  test_move_semantics(std::move(vortex_to_move), vortex);  //  NOLINT

  test_serialization(vortex);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.Vortex",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/NewtonianEuler"};

  const std::array<double, 2> center_2d = {{0.12, -0.04}};
  const std::array<double, 2> mean_velocity_2d = {{0.54, -0.02}};
  test_solution<2>(std::numeric_limits<double>::signaling_NaN(), center_2d,
                   "[0.12, -0.04]", mean_velocity_2d, "[0.54, -0.02]");
  test_solution<2>(DataVector(5), center_2d, "[0.12, -0.04]", mean_velocity_2d,
                   "[0.54, -0.02]");

  const std::array<double, 3> center_3d = {{-0.53, -0.1, 1.4}};
  const std::array<double, 3> mean_velocity_3d = {{-0.04, 0.14, 0.3}};
  test_solution<3>(std::numeric_limits<double>::signaling_NaN(), center_3d,
                   "[-0.53, -0.1, 1.4]", mean_velocity_3d,
                   "[-0.04, 0.14, 0.3]");
  test_solution<3>(DataVector(5), center_3d, "[-0.53, -0.1, 1.4]",
                   mean_velocity_3d, "[-0.04, 0.14, 0.3]");
}

// [[OutputRegex, The strength must be non-negative.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.VortexStrength2d",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  NewtonianEuler::Solutions::IsentropicVortex<2> test_vortex(
      1.3, {{3.21, -1.4}}, {{0.12, -0.53}}, -0.15, -1.7);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

    // clang-format off
// [[OutputRegex, The strength must be non-negative.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.VortexStrength3d",
    "[Unit][PointwiseFunctions]") {
  // clang-format on
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  NewtonianEuler::Solutions::IsentropicVortex<3> test_vortex(
      1.65, {{-0.12, 1.542, 3.12}}, {{-0.04, -0.32, 0.003}}, 4.2, -0.5);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

template <size_t Dim>
struct Vortex {
  using type = NewtonianEuler::Solutions::IsentropicVortex<Dim>;
  static constexpr OptionString help = {"A Newtonian isentropic vortex."};
};

// [[OutputRegex, In string:.*At line 6 column 13:.Value -0.2 is below the lower
// bound of 0]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.VortexStrengthOpt2d",
    "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  Options<tmpl::list<Vortex<2>>> test_options("");
  test_options.parse(
      "Vortex:\n"
      "  AdiabaticIndex: 1.4\n"
      "  Center: [-3.9, 1.1]\n"
      "  MeanVelocity: [0.1, -0.032]\n"
      "  PerturbAmplitude: 0.13\n"
      "  Strength: -0.2");
  test_options.get<Vortex<2>>();
}

// [[OutputRegex, In string:.*At line 6 column 13:.Value -0.3 is below the lower
// bound of 0]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.VortexStrengthOpt3d",
    "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  Options<tmpl::list<Vortex<3>>> test_options("");
  test_options.parse(
      "Vortex:\n"
      "  AdiabaticIndex: 1.12\n"
      "  Center: [0.3, -0.12, 4.2]\n"
      "  MeanVelocity: [-0.03, -0.1, 0.09]\n"
      "  PerturbAmplitude: 0.42\n"
      "  Strength: -0.3");
  test_options.get<Vortex<3>>();
}
