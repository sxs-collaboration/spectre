// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagnetizedFmDisk.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphericalKerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

static_assert(
    not is_analytic_solution_v<grmhd::AnalyticData::MagnetizedFmDisk>,
    "MagnetizedFmDisk should be analytic_data, and not an analytic_solution");
static_assert(
    is_analytic_data_v<grmhd::AnalyticData::MagnetizedFmDisk>,
    "MagnetizedFmDisk should be analytic_data, and not an analytic_solution");
struct MagnetizedFmDiskProxy : grmhd::AnalyticData::MagnetizedFmDisk {
  using grmhd::AnalyticData::MagnetizedFmDisk::MagnetizedFmDisk;

  template <typename DataType>
  using hydro_variables_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataType>,
                 hydro::Tags::SpatialVelocity<DataType, 3>,
                 hydro::Tags::SpecificInternalEnergy<DataType>,
                 hydro::Tags::Pressure<DataType>,
                 hydro::Tags::LorentzFactor<DataType>,
                 hydro::Tags::SpecificEnthalpy<DataType>>;

  template <typename DataType>
  using grmhd_variables_tags =
      tmpl::push_back<hydro_variables_tags<DataType>,
                      hydro::Tags::MagneticField<DataType, 3>,
                      hydro::Tags::DivergenceCleaningField<DataType>>;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<hydro_variables_tags<DataType>>
  hydro_variables(const tnsr::I<DataType, 3>& x) const {
    return variables(x, hydro_variables_tags<DataType>{});
  }

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<grmhd_variables_tags<DataType>>
  grmhd_variables(const tnsr::I<DataType, 3>& x) const {
    return variables(x, grmhd_variables_tags<DataType>{});
  }
};

void test_create_from_options() {
  register_classes_with_charm<grmhd::AnalyticData::MagnetizedFmDisk>();
  const std::unique_ptr<evolution::initial_data::InitialData> option_solution =
      TestHelpers::test_option_tag_factory_creation<
          evolution::initial_data::OptionTags::InitialData,
          grmhd::AnalyticData::MagnetizedFmDisk>(
          "MagnetizedFmDisk:\n"
          "  BhMass: 1.3\n"
          "  BhDimlessSpin: 0.345\n"
          "  InnerEdgeRadius: 6.123\n"
          "  MaxPressureRadius: 14.2\n"
          "  PolytropicConstant: 0.065\n"
          "  PolytropicExponent: 1.654\n"
          "  Noise: 0.0\n"
          "  ThresholdDensity: 0.42\n"
          "  InversePlasmaBeta: 85.0\n"
          "  BFieldNormGridRes: 4")
          ->get_clone();
  const auto deserialized_option_solution =
      serialize_and_deserialize(option_solution);
  const auto& disk = dynamic_cast<const grmhd::AnalyticData::MagnetizedFmDisk&>(
      *deserialized_option_solution);

  CHECK(disk == grmhd::AnalyticData::MagnetizedFmDisk(
                    1.3, 0.345, 6.123, 14.2, 0.065, 1.654, 0.0, 0.42, 85.0, 4));
}

void test_move() {
  grmhd::AnalyticData::MagnetizedFmDisk disk(3.51, 0.87, 7.43, 15.3, 42.67,
                                             1.87, 0.0, 0.13, 0.015, 4);
  const grmhd::AnalyticData::MagnetizedFmDisk disk_copy(
      3.51, 0.87, 7.43, 15.3, 42.67, 1.87, 0.0, 0.13, 0.015, 4);
  test_move_semantics(std::move(disk), disk_copy);  //  NOLINT
}

void test_serialize() {
  const grmhd::AnalyticData::MagnetizedFmDisk disk(
      3.51, 0.87, 7.43, 15.3, 42.67, 1.87, 0.0, 0.13, 0.015, 4);
  test_serialization(disk);
}

template <typename DataType>
void test_variables(const DataType& used_for_size) {
  const double bh_mass = 1.23;
  const double bh_dimless_spin = 0.97;
  const double inner_edge_radius = 6.2;
  const double max_pressure_radius = 11.6;
  const double polytropic_constant = 0.034;
  const double polytropic_exponent = 1.65;
  const double noise = 0.0;
  const double threshold_density = 0.14;
  const double inverse_plasma_beta = 0.023;
  const size_t b_field_normalization = 51;  // Using lower than default
                                            //  resolution for faster testing.
  const MagnetizedFmDiskProxy disk(
      bh_mass, bh_dimless_spin, inner_edge_radius, max_pressure_radius,
      polytropic_constant, polytropic_exponent, noise, threshold_density,
      inverse_plasma_beta, b_field_normalization);
  const auto member_variables = std::make_tuple(
      bh_mass, bh_dimless_spin, inner_edge_radius, max_pressure_radius,
      polytropic_constant, polytropic_exponent, noise, threshold_density,
      inverse_plasma_beta);

  pypp::check_with_random_values<1>(
      &MagnetizedFmDiskProxy::grmhd_variables<DataType>, disk,
      "PointwiseFunctions.AnalyticData.GrMhd.MagnetizedFmDisk",
      {"rest_mass_density", "spatial_velocity", "specific_internal_energy",
       "pressure", "lorentz_factor", "specific_enthalpy", "magnetic_field",
       "divergence_cleaning_field"},
      {{{-20., 20.}}}, member_variables, used_for_size, 1.0e-8);

  // Test a few of the GR components to make sure that the implementation
  // correctly forwards to the background solution of the base class.
  const auto coords = make_with_value<tnsr::I<DataType, 3>>(used_for_size, 1.0);
  const gr::Solutions::SphericalKerrSchild sks_soln{
      bh_mass, {{0.0, 0.0, bh_dimless_spin}}, {{0.0, 0.0, 0.0}}};
  const auto [expected_lapse, expected_sqrt_det_gamma] = sks_soln.variables(
      coords, 0.0,
      tmpl::list<gr::Tags::Lapse<DataType>,
                 gr::Tags::SqrtDetSpatialMetric<DataType>>{});
  const auto [lapse, sqrt_det_gamma] = disk.variables(
      coords, tmpl::list<gr::Tags::Lapse<DataType>,
                         gr::Tags::SqrtDetSpatialMetric<DataType>>{});
  CHECK_ITERABLE_APPROX(expected_lapse, lapse);
  CHECK_ITERABLE_APPROX(expected_sqrt_det_gamma, sqrt_det_gamma);
  const auto expected_spatial_metric =
      get<gr::Tags::SpatialMetric<DataType, 3>>(sks_soln.variables(
          coords, 0.0, gr::Solutions::SphericalKerrSchild::tags<DataType>{}));
  const auto spatial_metric =
      get<gr::Tags::SpatialMetric<DataType, 3>>(disk.variables(
          coords, tmpl::list<gr::Tags::SpatialMetric<DataType, 3>>{}));
  CHECK_ITERABLE_APPROX(expected_spatial_metric, spatial_metric);

  // Check that when InversePlasmaBeta = 0, magnetic field vanishes and
  // we recover FishboneMoncriefDisk
  const MagnetizedFmDiskProxy another_disk(
      bh_mass, bh_dimless_spin, inner_edge_radius, max_pressure_radius,
      polytropic_constant, polytropic_exponent, noise, threshold_density, 0.0,
      4);

  pypp::check_with_random_values<1>(
      &MagnetizedFmDiskProxy::hydro_variables<DataType>, another_disk,
      "PointwiseFunctions.AnalyticData.GrMhd.MagnetizedFmDisk",
      {"rest_mass_density", "spatial_velocity", "specific_internal_energy",
       "pressure", "lorentz_factor", "specific_enthalpy"},
      {{{-20., 20.}}}, member_variables, used_for_size, 1.0e-8);
  const auto [magnetic_field, div_clean_field] =
      another_disk.variables(
          coords, tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>,
                             hydro::Tags::DivergenceCleaningField<DataType>>{});
  const auto expected_magnetic_field =
      make_with_value<tnsr::I<DataType, 3>>(used_for_size, 0.0);
  const auto expected_div_clean_field =
      make_with_value<Scalar<DataType>>(used_for_size, 0.0);
  CHECK_ITERABLE_APPROX(magnetic_field, expected_magnetic_field);
  CHECK_ITERABLE_APPROX(div_clean_field, expected_div_clean_field);
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

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      []() {
        grmhd::AnalyticData::MagnetizedFmDisk disk(
            0.7, 0.61, 5.4, 9.182, 11.123, 1.44, 0.0, -0.2, 0.023, 4);
      }(),
      Catch::Matchers::ContainsSubstring(
          "The threshold density must be in the range (0, 1)"));
  CHECK_THROWS_WITH(
      []() {
        grmhd::AnalyticData::MagnetizedFmDisk disk(
            0.7, 0.61, 5.4, 9.182, 11.123, 1.44, 0.0, 1.45, 0.023, 4);
      }(),
      Catch::Matchers::ContainsSubstring(
          "The threshold density must be in the range (0, 1)"));
  CHECK_THROWS_WITH(
      []() {
        grmhd::AnalyticData::MagnetizedFmDisk disk(
            0.7, 0.61, 5.4, 9.182, 11.123, 1.44, 0.0, 0.2, -0.153, 4);
      }(),
      Catch::Matchers::ContainsSubstring(
          "The inverse plasma beta must be non-negative."));
  CHECK_THROWS_WITH(
      []() {
        grmhd::AnalyticData::MagnetizedFmDisk disk(
            0.7, 0.61, 5.4, 9.182, 11.123, 1.44, 0.0, 0.2, 0.153, 2);
      }(),
      Catch::Matchers::ContainsSubstring(
          "The grid resolution used in the magnetic field "
          "normalization must be at least 4 points."));
  CHECK_THROWS_WITH(
      []() {
        grmhd::AnalyticData::MagnetizedFmDisk disk(
            0.7, 0.61, 5.4, 9.182, 11.123, 1.44, 0.0, 0.2, 0.023, 4);
      }(),
      Catch::Matchers::ContainsSubstring("Max b squared is zero."));
#endif
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<grmhd::AnalyticData::MagnetizedFmDisk>(
          "BhMass: 13.45\n"
          "BhDimlessSpin: 0.45\n"
          "InnerEdgeRadius: 6.1\n"
          "MaxPressureRadius: 7.6\n"
          "PolytropicConstant: 2.42\n"
          "PolytropicExponent: 1.33\n"
          "Noise: 0.0\n"
          "ThresholdDensity: -0.01\n"
          "InversePlasmaBeta: 0.016\n"
          "BFieldNormGridRes: 4"),
      Catch::Matchers::ContainsSubstring(
          "Value -0.01 is below the lower bound of 0"));
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<grmhd::AnalyticData::MagnetizedFmDisk>(
          "BhMass: 1.5\n"
          "BhDimlessSpin: 0.94\n"
          "InnerEdgeRadius: 6.4\n"
          "MaxPressureRadius: 8.2\n"
          "PolytropicConstant: 41.1\n"
          "PolytropicExponent: 1.8\n"
          "Noise: 0.0\n"
          "ThresholdDensity: 4.1\n"
          "InversePlasmaBeta: 0.03\n"
          "BFieldNormGridRes: 4"),
      Catch::Matchers::ContainsSubstring(
          "Value 4.1 is above the upper bound of 1"));
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<grmhd::AnalyticData::MagnetizedFmDisk>(
          "BhMass: 1.4\n"
          "BhDimlessSpin: 0.91\n"
          "InnerEdgeRadius: 6.5\n"
          "MaxPressureRadius: 7.8\n"
          "PolytropicConstant: 13.5\n"
          "PolytropicExponent: 1.54\n"
          "Noise : 0.0\n"
          "ThresholdDensity: 0.22\n"
          "InversePlasmaBeta: -0.03\n"
          "BFieldNormGridRes: 4"),
      Catch::Matchers::ContainsSubstring(
          "Value -0.03 is below the lower bound of 0"));
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<grmhd::AnalyticData::MagnetizedFmDisk>(
          "BhMass: 1.4\n"
          "BhDimlessSpin: 0.91\n"
          "InnerEdgeRadius: 6.5\n"
          "MaxPressureRadius: 7.8\n"
          "PolytropicConstant: 13.5\n"
          "PolytropicExponent: 1.54\n"
          "Noise: 0.0\n"
          "ThresholdDensity: 0.22\n"
          "InversePlasmaBeta: 0.03\n"
          "BFieldNormGridRes: 2"),
      Catch::Matchers::ContainsSubstring(
          "Value 2 is below the lower bound of 4"));
}
