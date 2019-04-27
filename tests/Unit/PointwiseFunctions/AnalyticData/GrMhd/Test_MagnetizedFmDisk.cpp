// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"        // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "ErrorHandling/Error.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagnetizedFmDisk.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
struct MagnetizedFmDiskProxy : grmhd::AnalyticData::MagnetizedFmDisk {
  using grmhd::AnalyticData::MagnetizedFmDisk::MagnetizedFmDisk;

  template <typename DataType>
  using hydro_variables_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataType>,
                 hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>,
                 hydro::Tags::SpecificInternalEnergy<DataType>,
                 hydro::Tags::Pressure<DataType>,
                 hydro::Tags::LorentzFactor<DataType>,
                 hydro::Tags::SpecificEnthalpy<DataType>>;

  template <typename DataType>
  using grmhd_variables_tags =
      tmpl::push_back<hydro_variables_tags<DataType>,
                      hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>,
                      hydro::Tags::DivergenceCleaningField<DataType>>;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<hydro_variables_tags<DataType>>
  hydro_variables(const tnsr::I<DataType, 3>& x) const noexcept {
    return variables(x, hydro_variables_tags<DataType>{});
  }

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<grmhd_variables_tags<DataType>>
  grmhd_variables(const tnsr::I<DataType, 3>& x) const noexcept {
    return variables(x, grmhd_variables_tags<DataType>{});
  }
};

void test_create_from_options() noexcept {
  const auto disk = test_creation<grmhd::AnalyticData::MagnetizedFmDisk>(
      "  BhMass: 1.3\n"
      "  BhDimlessSpin: 0.345\n"
      "  InnerEdgeRadius: 6.123\n"
      "  MaxPressureRadius: 14.2\n"
      "  PolytropicConstant: 0.065\n"
      "  PolytropicExponent: 1.654\n"
      "  ThresholdDensity: 0.42\n"
      "  InversePlasmaBeta: 85.0\n"
      "  BFieldNormGridRes: 6");
  CHECK(disk == grmhd::AnalyticData::MagnetizedFmDisk(
                    1.3, 0.345, 6.123, 14.2, 0.065, 1.654, 0.42, 85.0, 6));
}

void test_move() noexcept {
  grmhd::AnalyticData::MagnetizedFmDisk disk(3.51, 0.87, 7.43, 15.3, 42.67,
                                             1.87, 0.13, 0.015, 4);
  grmhd::AnalyticData::MagnetizedFmDisk disk_copy(3.51, 0.87, 7.43, 15.3, 42.67,
                                                  1.87, 0.13, 0.015, 4);
  test_move_semantics(std::move(disk), disk_copy);  //  NOLINT
}

void test_serialize() noexcept {
  grmhd::AnalyticData::MagnetizedFmDisk disk(3.51, 0.87, 7.43, 15.3, 42.67,
                                             1.87, 0.13, 0.015, 4);
  test_serialization(disk);
}

template <typename DataType>
void test_variables(const DataType& used_for_size) {
  const double bh_mass = 1.12;
  const double bh_dimless_spin = 0.97;
  const double inner_edge_radius = 6.2;
  const double max_pressure_radius = 11.6;
  const double polytropic_constant = 0.034;
  const double polytropic_exponent = 1.65;
  const double threshold_density = 0.14;
  const double inverse_plasma_beta = 0.023;

  MagnetizedFmDiskProxy disk(bh_mass, bh_dimless_spin, inner_edge_radius,
                             max_pressure_radius, polytropic_constant,
                             polytropic_exponent, threshold_density,
                             inverse_plasma_beta);
  const auto member_variables = std::make_tuple(
      bh_mass, bh_dimless_spin, inner_edge_radius, max_pressure_radius,
      polytropic_constant, polytropic_exponent, threshold_density,
      inverse_plasma_beta);

  pypp::check_with_random_values<
      1, MagnetizedFmDiskProxy::grmhd_variables_tags<DataType>>(
      &MagnetizedFmDiskProxy::grmhd_variables<DataType>, disk,
      "PointwiseFunctions.AnalyticData.GrMhd.MagnetizedFmDisk",
      {"rest_mass_density", "spatial_velocity", "specific_internal_energy",
       "pressure", "lorentz_factor", "specific_enthalpy", "magnetic_field",
       "divergence_cleaning_field"},
      {{{-20., 20.}}}, member_variables, used_for_size, 1.0e-8);

  // Test a few of the GR components to make sure that the implementation
  // correctly forwards to the background solution of the base class.
  const auto coords = make_with_value<tnsr::I<DataType, 3>>(used_for_size, 1.0);
  const gr::Solutions::KerrSchild ks_soln{
      bh_mass, {{0.0, 0.0, bh_dimless_spin}}, {{0.0, 0.0, 0.0}}};
  CHECK_ITERABLE_APPROX(
      get<gr::Tags::Lapse<DataType>>(ks_soln.variables(
          coords, 0.0, gr::Solutions::KerrSchild::tags<DataType>{})),
      get<gr::Tags::Lapse<DataType>>(
          disk.variables(coords, tmpl::list<gr::Tags::Lapse<DataType>>{})));
  CHECK_ITERABLE_APPROX(
      get<gr::Tags::SqrtDetSpatialMetric<DataType>>(ks_soln.variables(
          coords, 0.0, gr::Solutions::KerrSchild::tags<DataType>{})),
      get<gr::Tags::SqrtDetSpatialMetric<DataType>>(disk.variables(
          coords, tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataType>>{})));
  const auto expected_spatial_metric =
      get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>>(
          ks_soln.variables(coords, 0.0,
                            gr::Solutions::KerrSchild::tags<DataType>{}));
  const auto spatial_metric =
      get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>>(disk.variables(
          coords,
          tmpl::list<gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>>{}));
  CHECK_ITERABLE_APPROX(expected_spatial_metric, spatial_metric);

  // Check that when InversePlasmaBeta = 0, magnetic field vanishes and
  // we recover FishboneMoncriefDisk
  MagnetizedFmDiskProxy another_disk(
      bh_mass, bh_dimless_spin, inner_edge_radius, max_pressure_radius,
      polytropic_constant, polytropic_exponent, threshold_density, 0.0, 4);

  pypp::check_with_random_values<
      1, MagnetizedFmDiskProxy::hydro_variables_tags<DataType>>(
      &MagnetizedFmDiskProxy::hydro_variables<DataType>, another_disk,
      "PointwiseFunctions.AnalyticData.GrMhd.MagnetizedFmDisk",
      {"rest_mass_density", "spatial_velocity", "specific_internal_energy",
       "pressure", "lorentz_factor", "specific_enthalpy"},
      {{{-20., 20.}}}, member_variables, used_for_size, 1.0e-8);
  const auto magnetic_field =
      get<hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>>(
          another_disk.variables(
              coords,
              tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3,
                                                      Frame::Inertial>>{}));
  const auto expected_magnetic_field =
      make_with_value<tnsr::I<DataType, 3>>(used_for_size, 0.0);
  CHECK_ITERABLE_APPROX(magnetic_field, expected_magnetic_field);
  CHECK_ITERABLE_APPROX(
      get<hydro::Tags::DivergenceCleaningField<DataType>>(
          another_disk.variables(
              coords,
              tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>>{})),
      make_with_value<Scalar<DataType>>(used_for_size, 0.0));
}
}  // namespace

// [[TimeOut, 8]]
SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticData.GrMhd.MagFmDisk",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};

  test_create_from_options();
  test_serialize();
  test_move();

  test_variables(std::numeric_limits<double>::signaling_NaN());
  test_variables(DataVector(5));
}

// [[OutputRegex, The threshold density must be in the range \(0, 1\)]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.GrMhd.MagFmDiskThreshLower",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  grmhd::AnalyticData::MagnetizedFmDisk disk(0.7, 0.61, 5.4, 9.182, 11.123,
                                             1.44, -0.2, 0.023, 4);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

    // clang-format off
// [[OutputRegex, The threshold density must be in the range \(0, 1\)]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.GrMhd.MagFmDiskThreshUpper",
    "[Unit][PointwiseFunctions]") {
  // clang-format on
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  grmhd::AnalyticData::MagnetizedFmDisk disk(0.7, 0.61, 5.4, 9.182, 11.123,
                                             1.44, 1.45, 0.023, 4);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The inverse plasma beta must be non-negative.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.GrMhd.MagFmDiskInvBeta",
    "[Unit][PointwiseFunctions]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  grmhd::AnalyticData::MagnetizedFmDisk disk(0.7, 0.61, 5.4, 9.182, 11.123,
                                             1.44, 0.2, -0.153, 4);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

    // clang-format off
// [[OutputRegex, The grid resolution used in the magnetic field
// normalization must be at least 4 points.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.GrMhd.MagFmDiskGridRes",
    "[Unit][PointwiseFunctions]") {
  // clang-format on
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  grmhd::AnalyticData::MagnetizedFmDisk disk(0.7, 0.61, 5.4, 9.182, 11.123,
                                             1.44, 0.2, 0.153, 2);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// Some random member variables might give rise to vanishing magnetic fields.
// clang-format off
// [[OutputRegex, Max b squared is zero.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.GrMhd.MagFmDiskGridUnphysical",
    "[Unit][PointwiseFunctions]") {
  // clang-format on
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  grmhd::AnalyticData::MagnetizedFmDisk disk(0.7, 0.61, 5.4, 9.182, 11.123,
                                             1.44, 0.2, 0.023, 5);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

struct MagFmDisk {
  using type = grmhd::AnalyticData::MagnetizedFmDisk;
  static constexpr OptionString help = {
      "A magnetized fluid disk orbiting a Kerr black hole."};
};

// [[OutputRegex, In string:.*At line 8 column 21:.Value -0.01 is below the
// lower bound of 0]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.GrMhd.MagFmDiskThreshLowerOpt",
    "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  Options<tmpl::list<MagFmDisk>> test_options("");
  test_options.parse(
      "MagFmDisk:\n"
      "  BhMass: 13.45\n"
      "  BhDimlessSpin: 0.45\n"
      "  InnerEdgeRadius: 6.1\n"
      "  MaxPressureRadius: 7.6\n"
      "  PolytropicConstant: 2.42\n"
      "  PolytropicExponent: 1.33\n"
      "  ThresholdDensity: -0.01\n"
      "  InversePlasmaBeta: 0.016\n"
      "  BFieldNormGridRes: 4");
  test_options.get<MagFmDisk>();
}

// [[OutputRegex, In string:.*At line 8 column 21:.Value 4.1 is above the
// upper bound of 1]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.GrMhd.MagFmDiskThreshUpperOpt",
    "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  Options<tmpl::list<MagFmDisk>> test_options("");
  test_options.parse(
      "MagFmDisk:\n"
      "  BhMass: 1.5\n"
      "  BhDimlessSpin: 0.94\n"
      "  InnerEdgeRadius: 6.4\n"
      "  MaxPressureRadius: 8.2\n"
      "  PolytropicConstant: 41.1\n"
      "  PolytropicExponent: 1.8\n"
      "  ThresholdDensity: 4.1\n"
      "  InversePlasmaBeta: 0.03\n"
      "  BFieldNormGridRes: 4");
  test_options.get<MagFmDisk>();
}

// [[OutputRegex, In string:.*At line 9 column 22:.Value -0.03 is below the
// lower bound of 0]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.GrMhd.MagFmDiskInvBetaOpt",
    "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  Options<tmpl::list<MagFmDisk>> test_options("");
  test_options.parse(
      "MagFmDisk:\n"
      "  BhMass: 1.4\n"
      "  BhDimlessSpin: 0.91\n"
      "  InnerEdgeRadius: 6.5\n"
      "  MaxPressureRadius: 7.8\n"
      "  PolytropicConstant: 13.5\n"
      "  PolytropicExponent: 1.54\n"
      "  ThresholdDensity: 0.22\n"
      "  InversePlasmaBeta: -0.03\n"
      "  BFieldNormGridRes: 4");
  test_options.get<MagFmDisk>();
}

// [[OutputRegex, In string:.*At line 10 column 22:.Value 2 is below the
// lower bound of 4]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.GrMhd.MagFmDiskGridResOpt",
    "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  Options<tmpl::list<MagFmDisk>> test_options("");
  test_options.parse(
      "MagFmDisk:\n"
      "  BhMass: 1.4\n"
      "  BhDimlessSpin: 0.91\n"
      "  InnerEdgeRadius: 6.5\n"
      "  MaxPressureRadius: 7.8\n"
      "  PolytropicConstant: 13.5\n"
      "  PolytropicExponent: 1.54\n"
      "  ThresholdDensity: 0.22\n"
      "  InversePlasmaBeta: 0.03\n"
      "  BFieldNormGridRes: 2");
  test_options.get<MagFmDisk>();
}
