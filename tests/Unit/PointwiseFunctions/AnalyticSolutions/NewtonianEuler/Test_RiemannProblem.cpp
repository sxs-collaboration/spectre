// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "ErrorHandling/Error.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/RiemannProblem.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

template <size_t Dim>
struct RiemannProblemProxy : NewtonianEuler::Solutions::RiemannProblem<Dim> {
  using NewtonianEuler::Solutions::RiemannProblem<Dim>::RiemannProblem;

  template <typename DataType>
  using variables_tags =
      tmpl::list<NewtonianEuler::Tags::MassDensity<DataType>,
                 NewtonianEuler::Tags::Velocity<DataType, Dim>,
                 NewtonianEuler::Tags::Pressure<DataType>,
                 NewtonianEuler::Tags::SpecificInternalEnergy<DataType>>;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<variables_tags<DataType>>
  primitive_variables(const tnsr::I<DataType, Dim, Frame::Inertial>& x,
                      double t) const noexcept {
    return this->variables(x, t, variables_tags<DataType>{});
  }
};

template <size_t Dim, typename DataType>
void test_solution(const std::array<double, Dim> left_velocity,
                   const std::string& left_velocity_opt,
                   const std::array<double, Dim> right_velocity,
                   const std::string& right_velocity_opt,
                   const DataType& used_for_size) noexcept {
  // Member variables correspond to Sod tube test. For other test cases,
  // new parameters in the star region must be given in the python modules.
  RiemannProblemProxy<Dim> solution(1.4, 0.5, 1.0, left_velocity, 1.0, 0.125,
                                    right_velocity, 0.1, 1.e-6);
  pypp::check_with_random_values<
      1, typename RiemannProblemProxy<Dim>::template variables_tags<DataType>>(
      &RiemannProblemProxy<Dim>::template primitive_variables<DataType>,
      solution, "RiemannProblem",
      {"mass_density", "velocity", "pressure", "specific_internal_energy"},
      {{{0.0, 1.0}}},
      std::make_tuple(1.4, 0.5, 1.0, left_velocity, 1.0, 0.125, right_velocity,
                      0.1),
      used_for_size, 1.e-9);

  const auto solution_from_options = TestHelpers::test_creation<
      NewtonianEuler::Solutions::RiemannProblem<Dim>>(
      "AdiabaticIndex: 1.4\n"
      "InitialPosition: 0.5\n"
      "LeftMassDensity: 1.0\n"
      "LeftVelocity: " +
      left_velocity_opt +
      "\n"
      "LeftPressure: 1.0\n"
      "RightMassDensity: 0.125\n"
      "RightVelocity: " +
      right_velocity_opt +
      "\n"
      "RightPressure: 0.1\n"
      "PressureStarTol: 1.e-6");
  CHECK(solution_from_options == solution);

  RiemannProblemProxy<Dim> solution_to_move(1.4, 0.5, 1.0, left_velocity, 1.0,
                                            0.125, right_velocity, 0.1, 1.e-6);
  test_move_semantics(std::move(solution_to_move), solution);  //  NOLINT
  test_serialization(solution);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.RiemannProblem",
    "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/NewtonianEuler"};

  test_solution<1>({{0.0}}, "[0.0]", {{0.0}}, "[0.0]",
                   std::numeric_limits<double>::signaling_NaN());
  test_solution<1>({{0.0}}, "[0.0]", {{0.0}}, "[0.0]", DataVector(5));
  test_solution<2>({{0.0, 0.4}}, "[0.0, 0.4]", {{0.0, -0.3}}, "[0.0, -0.3]",
                   std::numeric_limits<double>::signaling_NaN());
  test_solution<2>({{0.0, 0.4}}, "[0.0, 0.4]", {{0.0, -0.3}}, "[0.0, -0.3]",
                   DataVector(5));
  test_solution<3>({{0.0, 0.4, -0.12}}, "[0.0, 0.4, -0.12]",
                   {{0.0, -0.3, 0.53}}, "[0.0, -0.3, 0.53]",
                   std::numeric_limits<double>::signaling_NaN());
  test_solution<3>({{0.0, 0.4, -0.12}}, "[0.0, 0.4, -0.12]",
                   {{0.0, -0.3, 0.53}}, "[0.0, -0.3, 0.53]", DataVector(5));
}

// Test correct evaluation of the variables in the star region for the five
// initial setups listed in Table 4.1 of \Toro2009. The numerical values to
// compare with are listed in Table 4.2 of the same reference.
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.RiemannProblem.StarRg",
    "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/NewtonianEuler"};
  // The accuracy of the numbers provided by Toro isn't too high so
  // we need to adjust the precision of the comparison accordingly.
  Approx larger_approx = Approx::custom().epsilon(1.e-5);

  const double adiabatic_index = 1.4;
  // Initial position doesn't affect the numbers tested below.
  const double initial_position = 0.7;

  // Sod or Shock-Tube problem: one shock wave and one rarefaction wave.
  NewtonianEuler::Solutions::RiemannProblem<1> sod(
      adiabatic_index, initial_position, 1.0, {{0.0}}, 1.0, 0.125, {{0.0}},
      0.1);
  CHECK_ITERABLE_CUSTOM_APPROX(sod.diagnostic_star_region_values(),
                               make_array(0.30313, 0.92745), larger_approx);

  // "123" problem: two strong rarefaction waves. Contact is stationary.
  NewtonianEuler::Solutions::RiemannProblem<2> one_two_three(
      adiabatic_index, initial_position, 1.0, {{-2.0}}, 0.4, 1.0, {{2.0}}, 0.4);
  CHECK_ITERABLE_CUSTOM_APPROX(one_two_three.diagnostic_star_region_values(),
                               make_array(0.00189, 0.),
                               Approx::custom().epsilon(1.e-2));

  // Left half of the blast wave problem of Woodward&Corella.
  NewtonianEuler::Solutions::RiemannProblem<3> left_blast_wave(
      adiabatic_index, initial_position, 1.0, {{0.0}}, 1000.0, 1.0, {{0.0}},
      0.01);
  CHECK_ITERABLE_CUSTOM_APPROX(left_blast_wave.diagnostic_star_region_values(),
                               make_array(460.894, 19.5975), larger_approx);

  // Right half of the blast wave problem of Woodward&Corella.
  NewtonianEuler::Solutions::RiemannProblem<3> right_blast_wave(
      adiabatic_index, initial_position, 1.0, {{0.0}}, 0.01, 1.0, {{0.0}},
      100.0);
  CHECK_ITERABLE_CUSTOM_APPROX(right_blast_wave.diagnostic_star_region_values(),
                               make_array(46.0950, -6.19633), larger_approx);

  // Collision of the two previous shocks.
  NewtonianEuler::Solutions::RiemannProblem<3> shock_collision(
      adiabatic_index, initial_position, 5.99924, {{19.5975}}, 460.894, 5.99242,
      {{-6.19633}}, 46.0950);
  CHECK_ITERABLE_CUSTOM_APPROX(shock_collision.diagnostic_star_region_values(),
                               make_array(1691.64, 8.68975), larger_approx);
}

// [[OutputRegex, The pressure positivity condition must be met.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.RiemannProblem.PositP",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/NewtonianEuler"};
#ifdef SPECTRE_DEBUG
  NewtonianEuler::Solutions::RiemannProblem<1> solution(
      1.4, 0.7, 1.0, {{0.0}}, 1.0, 0.125, {{30.0}}, 1.1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

    // clang-format off
// [[OutputRegex, The mass density must be positive.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.RiemannProblem.Dens",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
  // clang-format on
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/NewtonianEuler"};
#ifdef SPECTRE_DEBUG
  NewtonianEuler::Solutions::RiemannProblem<2> solution(
      1.4, 0.7, -1.0, {{0.0}}, 1.0, 0.125, {{30.0}}, 1.1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The pressure must be positive.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.RiemannProblem.Pres",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/NewtonianEuler"};
#ifdef SPECTRE_DEBUG
  NewtonianEuler::Solutions::RiemannProblem<3> solution(
      1.4, 0.7, 1.0, {{0.0}}, -1.0, 0.125, {{30.0}}, 1.1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, newton_raphson reached max iterations of 50 without
// converging.]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.NewtEuler.RiemannProblem.RootF",
    "[Unit][PointwiseFunctions]") {
  ERROR_TEST();
  TestHelpers::test_creation<NewtonianEuler::Solutions::RiemannProblem<1>>(
      "AdiabaticIndex: 1.4\n"
      "InitialPosition: 0.25\n"
      "LeftMassDensity: 10.0\n"
      "LeftVelocity: [0.0]\n"
      "LeftPressure: 100.0\n"
      "RightMassDensity: 1.0\n"
      "RightVelocity: [0.0]\n"
      "RightPressure: 1.0\n"
      "PressureStarTol: 1.e-10");
}
