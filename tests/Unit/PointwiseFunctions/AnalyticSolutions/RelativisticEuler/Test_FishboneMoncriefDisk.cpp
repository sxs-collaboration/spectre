// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Domain.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/GrMhd/VerifyGrMhdSolution.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphericalKerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/FishboneMoncriefDisk.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct FishboneMoncriefDiskProxy
    : RelativisticEuler::Solutions::FishboneMoncriefDisk {
  using RelativisticEuler::Solutions::FishboneMoncriefDisk::
      FishboneMoncriefDisk;

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
  hydro_variables(const tnsr::I<DataType, 3>& x, double t) const {
    return variables(x, t, hydro_variables_tags<DataType>{});
  }

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<grmhd_variables_tags<DataType>>
  grmhd_variables(const tnsr::I<DataType, 3>& x, double t) const {
    return variables(x, t, grmhd_variables_tags<DataType>{});
  }
};

void test_create_from_options() {
  const auto disk = TestHelpers::test_creation<
      RelativisticEuler::Solutions::FishboneMoncriefDisk>(
      "BhMass: 1.0\n"
      "BhDimlessSpin: 0.23\n"
      "InnerEdgeRadius: 6.0\n"
      "MaxPressureRadius: 12.0\n"
      "PolytropicConstant: 0.001\n"
      "PolytropicExponent: 1.4\n"
      "Noise: 0.0");
  CHECK(disk == RelativisticEuler::Solutions::FishboneMoncriefDisk(
                    1.0, 0.23, 6.0, 12.0, 0.001, 1.4, 0.0));
}

void test_move() {
  RelativisticEuler::Solutions::FishboneMoncriefDisk disk(3.45, 0.23, 4.8, 8.6,
                                                          0.02, 1.5, 0.0);
  const RelativisticEuler::Solutions::FishboneMoncriefDisk disk_copy(
      3.45, 0.23, 4.8, 8.6, 0.02, 1.5, 0.0);
  test_move_semantics(std::move(disk), disk_copy);  //  NOLINT
}

void test_serialize() {
  RelativisticEuler::Solutions::FishboneMoncriefDisk disk(4.21, 0.65, 6.0, 12.0,
                                                          0.001, 1.4, 0.0);
  test_serialization(disk);
}

template <typename DataType>
void test_variables(const DataType& used_for_size) {
  const double bh_mass = 1.23;
  const double bh_dimless_spin = 0.94432;
  const double inner_edge_radius = 6.0;
  const double max_pressure_radius = 12.0;
  const double polytropic_constant = 0.001;
  const double polytropic_exponent = 4.0 / 3.0;
  const double noise = 0.0;

  const FishboneMoncriefDiskProxy disk(
      bh_mass, bh_dimless_spin, inner_edge_radius, max_pressure_radius,
      polytropic_constant, polytropic_exponent, noise);
  const auto member_variables = std::make_tuple(
      bh_mass, bh_dimless_spin, inner_edge_radius, max_pressure_radius,
      polytropic_constant, polytropic_exponent, noise);

  pypp::check_with_random_values<1>(
      &FishboneMoncriefDiskProxy::hydro_variables<DataType>, disk,
      "FishboneMoncriefDisk",
      {"rest_mass_density", "spatial_velocity", "specific_internal_energy",
       "pressure", "lorentz_factor", "specific_enthalpy"},
      {{{-15., 15.}}}, member_variables, used_for_size);

  pypp::check_with_random_values<1>(
      &FishboneMoncriefDiskProxy::grmhd_variables<DataType>, disk,
      "FishboneMoncriefDisk",
      {"rest_mass_density", "spatial_velocity", "specific_internal_energy",
       "pressure", "lorentz_factor", "specific_enthalpy", "magnetic_field",
       "divergence_cleaning_field"},
      {{{-15., 15.}}}, member_variables, used_for_size);

  // Test a few of the GR components to make sure that the implementation
  // correctly forwards to the background solution. Not meant to be extensive.
  const auto coords = make_with_value<tnsr::I<DataType, 3>>(used_for_size, 1.0);
  const gr::Solutions::SphericalKerrSchild sks_soln{
      bh_mass, {{0.0, 0.0, bh_dimless_spin}}, {{0.0, 0.0, 0.0}}};
  CHECK_ITERABLE_APPROX(
      get<gr::Tags::Lapse<DataType>>(sks_soln.variables(
          coords, 0.0, gr::Solutions::SphericalKerrSchild::tags<DataType>{})),
      get<gr::Tags::Lapse<DataType>>(disk.variables(
          coords, 0.0, tmpl::list<gr::Tags::Lapse<DataType>>{})));
  CHECK_ITERABLE_APPROX(
      get<gr::Tags::SqrtDetSpatialMetric<DataType>>(sks_soln.variables(
          coords, 0.0, gr::Solutions::SphericalKerrSchild::tags<DataType>{})),
      get<gr::Tags::SqrtDetSpatialMetric<DataType>>(disk.variables(
          coords, 0.0,
          tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataType>>{})));
  const auto expected_spatial_metric =
      get<gr::Tags::SpatialMetric<DataType, 3>>(sks_soln.variables(
          coords, 0.0, gr::Solutions::SphericalKerrSchild::tags<DataType>{}));
  const auto spatial_metric =
      get<gr::Tags::SpatialMetric<DataType, 3>>(disk.variables(
          coords, 0.0, tmpl::list<gr::Tags::SpatialMetric<DataType, 3>>{}));
  CHECK_ITERABLE_APPROX(expected_spatial_metric, spatial_metric);
}

//  A former implementation of the `IntermediateVariables` function in
//  `FishboneMoncriefDisk.cpp` included an ill-defined calculation of
//  sin_theta_squared, which resulted in negative numbers due to round-off
//  errors. This would induce FPEs after taking the square root of
//  sin_theta_squared. This test evaluates the most recent implementation at
//  those points in order to ensure that the FPEs are no longer induced.
template <typename DataType>
void test_sin_theta_squared(const DataType& used_for_size) {
  // Numbers below reproduce the initial data the bug was spotted with,
  // along with the points where the FPEs were found.
  const FishboneMoncriefDiskProxy disk(1.0, 0.9375, 6.0, 12.0, 0.001,
                                       1.3333333333333333333333, 0.0);
  using variables_tags =
      FishboneMoncriefDiskProxy::grmhd_variables_tags<DataType>;
  auto coords = make_with_value<tnsr::I<DataType, 3>>(used_for_size, 0.0);
  get<2>(coords) = -1.8427400003643177317514;
  disk.variables(coords, 0.0, variables_tags{});
  get<2>(coords) = 1.8427400003643177317514;
  disk.variables(coords, 0.0, variables_tags{});
  get<2>(coords) = -1.9588918957550884858421;
  disk.variables(coords, 0.0, variables_tags{});
  get<2>(coords) = 1.9588918957550884858421;
  disk.variables(coords, 0.0, variables_tags{});
  CHECK(true);
}

void test_solution() {
  register_classes_with_charm<
      RelativisticEuler::Solutions::FishboneMoncriefDisk>();
  const std::unique_ptr<evolution::initial_data::InitialData> option_solution =
      TestHelpers::test_option_tag_factory_creation<
          evolution::initial_data::OptionTags::InitialData,
          RelativisticEuler::Solutions::FishboneMoncriefDisk>(
          "FishboneMoncriefDisk:\n"
          "  BhMass: 1.0\n"
          "  BhDimlessSpin: 0.9375\n"
          "  InnerEdgeRadius: 6.0\n"
          "  MaxPressureRadius: 12.0\n"
          "  PolytropicConstant: 0.001\n"
          "  PolytropicExponent: 1.33333333333333333\n"
          "  Noise: 0.0\n")
          ->get_clone();
  const auto deserialized_option_solution =
      serialize_and_deserialize(option_solution);
  const auto& solution =
      dynamic_cast<const RelativisticEuler::Solutions::FishboneMoncriefDisk&>(
          *deserialized_option_solution);

  const std::array<double, 3> x{{5.0, 5.0, 0.0}};
  const std::array<double, 3> dx{{1.e-1, 1.e-1, 1.e-1}};

  domain::creators::Brick brick(x - dx, x + dx, {{0, 0, 0}}, {{8, 8, 8}},
                                {{false, false, false}});
  Mesh<3> mesh{brick.initial_extents()[0], Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto};
  const auto domain = brick.create_domain();
  verify_grmhd_solution(solution, domain.blocks()[0], mesh, 1.e-9, 1.234,
                        1.e-1);
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
  test_sin_theta_squared(std::numeric_limits<double>::signaling_NaN());
  test_sin_theta_squared(DataVector(5));

  test_solution();

  CHECK_THROWS_WITH(TestHelpers::test_creation<
                        RelativisticEuler::Solutions::FishboneMoncriefDisk>(
                        "BhMass: -1.5\n"
                        "BhDimlessSpin: 0.3\n"
                        "InnerEdgeRadius: 4.3\n"
                        "MaxPressureRadius: 6.7\n"
                        "PolytropicConstant: 0.12\n"
                        "PolytropicExponent: 1.5\n"
                        "Noise : 0.0"),
                    Catch::Matchers::ContainsSubstring(
                        "Value -1.5 is below the lower bound of 0"));
  CHECK_THROWS_WITH(TestHelpers::test_creation<
                        RelativisticEuler::Solutions::FishboneMoncriefDisk>(
                        "BhMass: 1.5\n"
                        "BhDimlessSpin: -0.24\n"
                        "InnerEdgeRadius: 5.76\n"
                        "MaxPressureRadius: 13.2\n"
                        "PolytropicConstant: 0.002\n"
                        "PolytropicExponent: 1.34\n"
                        "Noise: 0.0"),
                    Catch::Matchers::ContainsSubstring(
                        "Value -0.24 is below the lower bound of 0"));
  CHECK_THROWS_WITH(TestHelpers::test_creation<
                        RelativisticEuler::Solutions::FishboneMoncriefDisk>(
                        "BhMass: 1.5\n"
                        "BhDimlessSpin: 1.24\n"
                        "InnerEdgeRadius: 5.76\n"
                        "MaxPressureRadius: 13.2\n"
                        "PolytropicConstant: 0.002\n"
                        "PolytropicExponent: 1.34\n"
                        "Noise: 0.0"),
                    Catch::Matchers::ContainsSubstring(
                        "Value 1.24 is above the upper bound of 1"));
  CHECK_THROWS_WITH(TestHelpers::test_creation<
                        RelativisticEuler::Solutions::FishboneMoncriefDisk>(
                        "BhMass: 1.5\n"
                        "BhDimlessSpin: 0.3\n"
                        "InnerEdgeRadius: 4.3\n"
                        "MaxPressureRadius: 6.7\n"
                        "PolytropicConstant: -0.12\n"
                        "PolytropicExponent: 1.5\n"
                        "Noise: 0.0"),
                    Catch::Matchers::ContainsSubstring(
                        "Value -0.12 is below the lower bound of 0"));
  CHECK_THROWS_WITH(TestHelpers::test_creation<
                        RelativisticEuler::Solutions::FishboneMoncriefDisk>(
                        "BhMass: 1.5\n"
                        "BhDimlessSpin: 0.3\n"
                        "InnerEdgeRadius: 4.3\n"
                        "MaxPressureRadius: 6.7\n"
                        "PolytropicConstant: 0.123\n"
                        "PolytropicExponent: 0.25\n"
                        "Noise: 0.0"),
                    Catch::Matchers::ContainsSubstring(
                        "Value 0.25 is below the lower bound of 1"));
  CHECK_THROWS_WITH(TestHelpers::test_creation<
                        RelativisticEuler::Solutions::FishboneMoncriefDisk>(
                        "BhMass: 1.5\n"
                        "BhDimlessSpin: 0.3\n"
                        "InnerEdgeRadius: 4.3\n"
                        "MaxPressureRadius: 6.7\n"
                        "PolytropicConstant: 0.123\n"
                        "PolytropicExponent: 1.5\n"
                        "Noise: -1.25"),
                    Catch::Matchers::ContainsSubstring(
                        "Value -1.25 is below the lower bound of 0"));
  CHECK_THROWS_WITH(TestHelpers::test_creation<
                        RelativisticEuler::Solutions::FishboneMoncriefDisk>(
                        "BhMass: 1.5\n"
                        "BhDimlessSpin: 0.3\n"
                        "InnerEdgeRadius: 4.3\n"
                        "MaxPressureRadius: 6.7\n"
                        "PolytropicConstant: 0.123\n"
                        "PolytropicExponent: 1.5\n"
                        "Noise: 1.5"),
                    Catch::Matchers::ContainsSubstring(
                        "Value 1.5 is above the upper bound of 1"));
}
