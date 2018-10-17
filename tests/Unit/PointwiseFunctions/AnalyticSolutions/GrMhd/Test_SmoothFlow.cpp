// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/SmoothFlow.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_forward_declare Tags::dt
// IWYU pragma: no_include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"

namespace {

struct SmoothFlowProxy : grmhd::Solutions::SmoothFlow {
  using grmhd::Solutions::SmoothFlow::SmoothFlow;

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
  hydro_variables(const tnsr::I<DataType, 3>& x, double t) const noexcept {
    return variables(x, t, hydro_variables_tags<DataType>{});
  }

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<grmhd_variables_tags<DataType>>
  grmhd_variables(const tnsr::I<DataType, 3>& x, double t) const noexcept {
    return variables(x, t, grmhd_variables_tags<DataType>{});
  }
};

void test_create_from_options() noexcept {
  const auto flow = test_creation<grmhd::Solutions::SmoothFlow>(
      "  MeanVelocity: [0.1, -0.2, 0.3]\n"
      "  WaveVector: [-0.13, -0.54, 0.04]\n"
      "  Pressure: 1.23\n"
      "  AdiabaticIndex: 1.4\n"
      "  PerturbationSize: 0.75");
  CHECK(flow.mean_velocity() == std::array<double, 3>{{0.1, -0.2, 0.3}});
  CHECK(flow.wavevector() == std::array<double, 3>{{-0.13, -0.54, 0.04}});
  CHECK(flow.pressure() == 1.23);
  CHECK(flow.adiabatic_index() == 1.4);
  CHECK(flow.perturbation_size() == 0.75);
}

void test_move() noexcept {
  grmhd::Solutions::SmoothFlow flow({{0.24, 0.11, 0.04}}, {{0.14, 0.42, -0.03}},
                                    1.3, 1.5, 0.24);
  grmhd::Solutions::SmoothFlow flow_copy({{0.24, 0.11, 0.04}},
                                         {{0.14, 0.42, -0.03}}, 1.3, 1.5, 0.24);
  test_move_semantics(std::move(flow), flow_copy);  //  NOLINT
}

void test_serialize() noexcept {
  grmhd::Solutions::SmoothFlow flow({{0.24, 0.11, 0.04}}, {{0.14, 0.42, -0.03}},
                                    1.3, 1.5, 0.24);
  test_serialization(flow);
}

template <typename DataType>
void test_variables(const DataType& used_for_size) {
  const std::array<double, 3> mean_velocity = {{0.23, 0.01, 0.31}};
  const std::array<double, 3> wave_vector = {{0.11, 0.23, 0.32}};
  const double pressure = 1.3;
  const double adiabatic_index = 4. / 3.;
  const double perturbation_size = 0.78;

  pypp::check_with_random_values<
      1, SmoothFlowProxy::hydro_variables_tags<DataType>>(
      &SmoothFlowProxy::hydro_variables<DataType>,
      SmoothFlowProxy(mean_velocity, wave_vector, pressure, adiabatic_index,
                      perturbation_size),
      "TestFunctions",
      {"smooth_flow_rest_mass_density", "smooth_flow_spatial_velocity",
       "smooth_flow_specific_internal_energy", "smooth_flow_pressure",
       "smooth_flow_lorentz_factor", "smooth_flow_specific_enthalpy"},
      {{{-15., 15.}}},
      std::make_tuple(mean_velocity, wave_vector, pressure, adiabatic_index,
                      perturbation_size),
      used_for_size);

  pypp::check_with_random_values<
      1, SmoothFlowProxy::grmhd_variables_tags<DataType>>(
      &SmoothFlowProxy::grmhd_variables<DataType>,
      SmoothFlowProxy(mean_velocity, wave_vector, pressure, adiabatic_index,
                      perturbation_size),
      "TestFunctions",
      {"smooth_flow_rest_mass_density", "smooth_flow_spatial_velocity",
       "smooth_flow_specific_internal_energy", "smooth_flow_pressure",
       "smooth_flow_lorentz_factor", "smooth_flow_specific_enthalpy",
       "smooth_flow_magnetic_field", "smooth_flow_divergence_cleaning_field"},
      {{{-15., 15.}}},
      std::make_tuple(mean_velocity, wave_vector, pressure, adiabatic_index,
                      perturbation_size),
      used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.GrMhd.SmoothFlow",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/GrMhd"};

  test_create_from_options();
  test_serialize();
  test_move();

  test_variables(std::numeric_limits<double>::signaling_NaN());
  test_variables(DataVector(5));
}
