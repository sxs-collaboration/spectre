// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <random>
#include <tuple>
#include <utility>

#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/PointwiseFunctions/PostNewtonian/BinaryTrajectories.hpp"

namespace {
// must match value in BinaryTrajectories.py
constexpr double initial_separation{15.366};
const std::array<double, 3> initial_velocity{0.1, -0.2, 0.3};

// Since we can't pass bools to these functions, pass more doubles.
// - newtonian: If it's positive, newtonian is true. If it's negative, newtonian
//   is false.
// - no_expansion: If it's positive, call the `positions_no_expansion` function.
//   If it's negative, just call the `positions` function.
tnsr::I<double, 3> positions1(const double time, const double newtonian,
                              const double no_expansion) {
  const bool newt = newtonian > 0.0 ? true : false;
  BinaryTrajectories expected{initial_separation, initial_velocity, newt};
  if (no_expansion > 0.0) {
    return expected.positions_no_expansion(time)[0];
  } else {
    return expected.positions(time)[0];
  }
}

tnsr::I<double, 3> positions2(const double time, const double newtonian,
                              const double no_expansion) {
  const bool newt = newtonian > 0.0 ? true : false;
  BinaryTrajectories expected{initial_separation, initial_velocity, newt};
  if (no_expansion > 0.0) {
    return expected.positions_no_expansion(time)[1];
  } else {
    return expected.positions(time)[1];
  }
}
}  // namespace

SPECTRE_TEST_CASE("Test.TestHelpers.PostNewtonian.BinaryTrajectories",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "Helpers/Tests/PointwiseFunctions/PostNewtonian/");

  for (const bool newtonian : {true, false}) {
    const BinaryTrajectories expected{initial_separation, initial_velocity,
                                      newtonian};
    pypp::check_with_random_values<1>(
        &BinaryTrajectories::separation<double>,
        BinaryTrajectories{initial_separation, initial_velocity, newtonian},
        "BinaryTrajectories", {"separation"}, {{{-100.0, 100.0}}},
        std::make_tuple(initial_separation, newtonian), initial_separation);
    pypp::check_with_random_values<1>(
        &BinaryTrajectories::orbital_frequency<double>,
        BinaryTrajectories{initial_separation, initial_velocity, newtonian},
        "BinaryTrajectories", {"orbital_frequency"}, {{{-100.0, 100.0}}},
        std::make_tuple(initial_separation, newtonian), initial_separation);
    pypp::check_with_random_values<1>(
        &BinaryTrajectories::angular_velocity<double>,
        BinaryTrajectories{initial_separation, initial_velocity, newtonian},
        "BinaryTrajectories", {"angular_velocity"}, {{{-100.0, 100.0}}},
        std::make_tuple(initial_separation, newtonian), initial_separation);
  }
  // Run these a few times so we get a good mix
  for (size_t i = 0; i < 8; i++) {
    pypp::check_with_random_values<1>(&positions1, "BinaryTrajectories",
                                      "positions1", {{{-100.0, 100.0}}},
                                      initial_separation);
    pypp::check_with_random_values<1>(&positions2, "BinaryTrajectories",
                                      "positions2", {{{-100.0, 100.0}}},
                                      initial_separation);
  }
}
