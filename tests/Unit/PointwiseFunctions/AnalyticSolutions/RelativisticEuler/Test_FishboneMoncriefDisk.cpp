// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "ErrorHandling/Error.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/FishboneMoncriefDisk.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
struct FishboneMoncriefDiskProxy
    : RelativisticEuler::Solutions::FishboneMoncriefDisk {
  using RelativisticEuler::Solutions::FishboneMoncriefDisk::
      FishboneMoncriefDisk;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<variables_tags<DataType>>
  primitive_variables(const tnsr::I<DataType, 3>& x, double t) const noexcept {
    return variables(x, t, variables_tags<DataType>{});
  }
};

void test_create_from_options() noexcept {
  const auto disk =
      test_creation<RelativisticEuler::Solutions::FishboneMoncriefDisk>(
          "  BlackHoleMass: 1.0\n"
          "  BlackHoleSpin: 0.23\n"
          "  InnerEdgeRadius: 6.0\n"
          "  MaxPressureRadius: 12.0\n"
          "  PolytropicConstant: 0.001\n"
          "  PolytropicExponent: 1.4");
  CHECK(disk == RelativisticEuler::Solutions::FishboneMoncriefDisk(
                    1.0, 0.23, 6.0, 12.0, 0.001, 1.4));
}

void test_move() noexcept {
  RelativisticEuler::Solutions::FishboneMoncriefDisk disk(3.45, 0.23, 4.8, 8.6,
                                                          0.02, 1.5);
  RelativisticEuler::Solutions::FishboneMoncriefDisk disk_copy(3.45, 0.23, 4.8,
                                                               8.6, 0.02, 1.5);
  test_move_semantics(std::move(disk), disk_copy);  //  NOLINT
}

void test_serialize() noexcept {
  RelativisticEuler::Solutions::FishboneMoncriefDisk disk(4.21, 0.65, 6.0, 12.0,
                                                          0.001, 1.4);
  test_serialization(disk);
}

template <typename DataType>
void test_variables(const DataType& used_for_size) noexcept {
  const double black_hole_mass = 1.0;
  const double black_hole_spin = 0.9375;
  const double inner_edge_radius = 6.0;
  const double max_pressure_radius = 12.0;
  const double polytropic_constant = 0.001;
  const double polytropic_exponent = 4.0 / 3.0;

  FishboneMoncriefDiskProxy disk(black_hole_mass, black_hole_spin,
                                 inner_edge_radius, max_pressure_radius,
                                 polytropic_constant, polytropic_exponent);
  const auto member_variables = std::make_tuple(
      black_hole_mass, black_hole_spin, inner_edge_radius, max_pressure_radius,
      polytropic_constant, polytropic_exponent);

  pypp::check_with_random_values<
      1, tmpl::list<hydro::Tags::RestMassDensity<DataType>,
                    hydro::Tags::SpatialVelocity<DataType, 3>,
                    hydro::Tags::SpecificInternalEnergy<DataType>,
                    hydro::Tags::Pressure<DataType>,
                    hydro::Tags::SpecificEnthalpy<DataType>>>(
      &FishboneMoncriefDiskProxy::primitive_variables<DataType>, disk,
      "TestFunctions",
      {"rest_mass_density", "spatial_velocity", "specific_internal_energy",
       "pressure", "specific_enthalpy"},
      {{{-15., 15.}}}, member_variables, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.RelEuler.FMDisk",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/RelativisticEuler"};

  test_create_from_options();
  test_serialize();
  test_move();

  test_variables(std::numeric_limits<double>::signaling_NaN());
  test_variables(DataVector(5));
}

// [[OutputRegex, The black hole mass must be positive]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.RelEuler.FMDiskBHMass",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  RelativisticEuler::Solutions::FishboneMoncriefDisk test_disk(
      -0.1, 0.456, 5.0, 13.0, 0.024, 1.53);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The black hole spin magnitude must be in the range \[0, 1\)]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.RelEuler.FMDiskBHSpin",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  RelativisticEuler::Solutions::FishboneMoncriefDisk test_disk(
      2.78, -0.1, 6.5, 12.7, 0.014, 1.32);
  RelativisticEuler::Solutions::FishboneMoncriefDisk another_test_disk(
      2.2, 2.34, 7.5, 13.2, 100.0, 1.78);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The polytropic constant must be positive]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.RelEuler.FMDiskPolytConst",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  RelativisticEuler::Solutions::FishboneMoncriefDisk test_disk(
      0.1, 0.456, 5.0, 13.0, -2.43, 1.53);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The polytropic exponent must be greater than 1]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.RelEuler.FMDiskPolytExp",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  RelativisticEuler::Solutions::FishboneMoncriefDisk test_disk(
      2.4, 0.222, 8.51, 16.3, 4.21, -0.34);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

struct Disk {
  using type = RelativisticEuler::Solutions::FishboneMoncriefDisk;
  static constexpr OptionString help = {
      "A fluid disk orbiting a Kerr black hole."};
};

// [[OutputRegex, In string:.*At line 2 column 18:.Value -1.5 is below the lower
// bound of 0]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.RelEuler.FMDiskBHMassOpt",
    "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  Options<tmpl::list<Disk>> test_options("");
  test_options.parse(
      "Disk:\n"
      "  BlackHoleMass: -1.5\n"
      "  BlackHoleSpin: 0.3\n"
      "  InnerEdgeRadius: 4.3\n"
      "  MaxPressureRadius: 6.7\n"
      "  PolytropicConstant: 0.12\n"
      "  PolytropicExponent: 1.5");
  test_options.get<Disk>();
}

// [[OutputRegex, In string:.*At line 3 column 18:.Value -0.24 is below the
// lower bound of 0]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.RelEuler.FMDiskBHSpinLowerOpt",
    "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  Options<tmpl::list<Disk>> test_options("");
  test_options.parse(
      "Disk:\n"
      "  BlackHoleMass: 1.5\n"
      "  BlackHoleSpin: -0.24\n"
      "  InnerEdgeRadius: 5.76\n"
      "  MaxPressureRadius: 13.2\n"
      "  PolytropicConstant: 0.002\n"
      "  PolytropicExponent: 1.34");
  test_options.get<Disk>();
}

// [[OutputRegex, In string:.*At line 3 column 18:.Value 1.24 is above the
// upper bound of 1.]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.RelEuler.FMDiskBHSpinUpperOpt",
    "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  Options<tmpl::list<Disk>> test_options("");
  test_options.parse(
      "Disk:\n"
      "  BlackHoleMass: 1.5\n"
      "  BlackHoleSpin: 1.24\n"
      "  InnerEdgeRadius: 5.76\n"
      "  MaxPressureRadius: 13.2\n"
      "  PolytropicConstant: 0.002\n"
      "  PolytropicExponent: 1.34");
  test_options.get<Disk>();
}

// [[OutputRegex, In string:.*At line 6 column 23:.Value -0.12 is below the
// lower bound of 0]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.RelEuler.FMDiskPolytConstOpt",
    "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  Options<tmpl::list<Disk>> test_options("");
  test_options.parse(
      "Disk:\n"
      "  BlackHoleMass: 1.5\n"
      "  BlackHoleSpin: 0.3\n"
      "  InnerEdgeRadius: 4.3\n"
      "  MaxPressureRadius: 6.7\n"
      "  PolytropicConstant: -0.12\n"
      "  PolytropicExponent: 1.5");
  test_options.get<Disk>();
}

// [[OutputRegex, In string:.*At line 7 column 23:.Value 0.25 is below the
// lower bound of 1]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.RelEuler.FMDiskPolytExpOpt",
    "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  Options<tmpl::list<Disk>> test_options("");
  test_options.parse(
      "Disk:\n"
      "  BlackHoleMass: 1.5\n"
      "  BlackHoleSpin: 0.3\n"
      "  InnerEdgeRadius: 4.3\n"
      "  MaxPressureRadius: 6.7\n"
      "  PolytropicConstant: 0.123\n"
      "  PolytropicExponent: 0.25");
  test_options.get<Disk>();
}
