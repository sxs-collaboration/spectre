// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"    // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"          // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Error.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagneticFieldLoop.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {

struct MagneticFieldLoopProxy : grmhd::AnalyticData::MagneticFieldLoop {
  using grmhd::AnalyticData::MagneticFieldLoop::MagneticFieldLoop;

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
  hydro_variables(const tnsr::I<DataType, 3, Frame::Inertial>& x) const
      noexcept {
    return variables(x, hydro_variables_tags<DataType>{});
  }

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<grmhd_variables_tags<DataType>>
  grmhd_variables(const tnsr::I<DataType, 3, Frame::Inertial>& x) const
      noexcept {
    return variables(x, grmhd_variables_tags<DataType>{});
  }
};

void test_create_from_options() noexcept {
  const auto magnetic_field_loop =
      TestHelpers::test_creation<grmhd::AnalyticData::MagneticFieldLoop>(
          "Pressure: 3.0\n"
          "RestMassDensity: 1.0\n"
          "AdiabaticIndex: 1.66666666666666667\n"
          "AdvectionVelocity: [0.5, 0.04166666666666667, 0.0]\n"
          "MagFieldStrength: 0.001\n"
          "InnerRadius: 0.06\n"
          "OuterRadius: 0.3\n");
  CHECK(magnetic_field_loop ==
        grmhd::AnalyticData::MagneticFieldLoop(
            3.0, 1.0, 1.66666666666666667,
            std::array<double, 3>{{0.5, 0.04166666666666667, 0.0}}, 0.001, 0.06,
            0.3));
}

void test_move() noexcept {
  grmhd::AnalyticData::MagneticFieldLoop magnetic_field_loop(
      3.0, 1.0, 1.66666666666666667,
      std::array<double, 3>{{0.5, 0.04166666666666667, 0.0}}, 0.001, 0.06, 0.3);
  grmhd::AnalyticData::MagneticFieldLoop magnetic_field_loop_copy(
      3.0, 1.0, 1.66666666666666667,
      std::array<double, 3>{{0.5, 0.04166666666666667, 0.0}}, 0.001, 0.06, 0.3);
  test_move_semantics(std::move(magnetic_field_loop),
                      magnetic_field_loop_copy);  //  NOLINT
}

void test_serialize() noexcept {
  grmhd::AnalyticData::MagneticFieldLoop magnetic_field_loop(
      3.0, 1.0, 1.66666666666666667,
      std::array<double, 3>{{0.5, 0.04166666666666667, 0.0}}, 0.001, 0.06, 0.3);
  test_serialization(magnetic_field_loop);
}

template <typename DataType>
void test_variables(const DataType& used_for_size) noexcept {
  const double pressure = 3.0;
  const double rest_mass_density = 1.0;
  const double adiabatic_index = 1.6666666666666666;
  const std::array<double, 3> advection_velocity{
      {0.5, 0.04166666666666667, 0.0}};
  const double magnetic_field_magnitude = 0.001;
  const double inner_radius = 0.06;
  const double outer_radius = 0.3;

  const auto member_variables = std::make_tuple(
      pressure, rest_mass_density, adiabatic_index, advection_velocity,
      magnetic_field_magnitude, inner_radius, outer_radius);

  MagneticFieldLoopProxy magnetic_field_loop(
      pressure, rest_mass_density, adiabatic_index, advection_velocity,
      magnetic_field_magnitude, inner_radius, outer_radius);

  pypp::check_with_random_values<
      1, MagneticFieldLoopProxy::hydro_variables_tags<DataType>>(
      &MagneticFieldLoopProxy::hydro_variables<DataType>, magnetic_field_loop,
      "MagneticFieldLoop",
      {"compute_rest_mass_density", "spatial_velocity",
       "specific_internal_energy", "compute_pressure", "lorentz_factor",
       "specific_enthalpy"},
      {{{-1.0, 1.0}}}, member_variables, used_for_size);

  pypp::check_with_random_values<
      1, MagneticFieldLoopProxy::grmhd_variables_tags<DataType>>(
      &MagneticFieldLoopProxy::grmhd_variables<DataType>, magnetic_field_loop,
      "MagneticFieldLoop",
      {"compute_rest_mass_density", "spatial_velocity",
       "specific_internal_energy", "compute_pressure", "lorentz_factor",
       "specific_enthalpy", "magnetic_field", "divergence_cleaning_field"},
      {{{-1.0, 1.0}}}, member_variables, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.GrMhd.MagneticFieldLoop",
    "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticData/GrMhd"};

  test_create_from_options();
  test_serialize();
  test_move();

  test_variables(std::numeric_limits<double>::signaling_NaN());
  test_variables(DataVector(5));
}

// [[OutputRegex, MagneticFieldLoop: superluminal AdvectionVelocity]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.GrMhd.MagneticFieldLoopBadVelocity",
    "[Unit][PointwiseFunctions]") {
  ERROR_TEST();
  grmhd::AnalyticData::MagneticFieldLoop magnetic_loop(
      3.0, 1.0, 1.66666666666666667,
      std::array<double, 3>{{0.5, 0.04166666666666667, -0.9}}, 0.001, 0.06,
      0.3);
  ERROR("Failed to trigger PARSE_ERROR in a parse error test");
}
