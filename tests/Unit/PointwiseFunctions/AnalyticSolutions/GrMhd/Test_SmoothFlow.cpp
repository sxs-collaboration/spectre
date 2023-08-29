// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Domain.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/GrMhd/VerifyGrMhdSolution.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/SmoothFlow.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_include <vector>
// IWYU pragma: no_include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"

// IWYU pragma: no_forward_declare Tags::dt

namespace {

struct SmoothFlowProxy : grmhd::Solutions::SmoothFlow {
  using grmhd::Solutions::SmoothFlow::SmoothFlow;

  template <typename DataType>
  using hydro_variables_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataType>,
                 hydro::Tags::ElectronFraction<DataType>,
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
  const auto flow = TestHelpers::test_creation<grmhd::Solutions::SmoothFlow>(
      "MeanVelocity: [0.1, -0.2, 0.3]\n"
      "WaveVector: [-0.13, -0.54, 0.04]\n"
      "Pressure: 1.23\n"
      "AdiabaticIndex: 1.4\n"
      "PerturbationSize: 0.75");
  CHECK(flow == grmhd::Solutions::SmoothFlow({{0.1, -0.2, 0.3}},
                                             {{-0.13, -0.54, 0.04}}, 1.23, 1.4,
                                             0.75));
}

void test_move() {
  grmhd::Solutions::SmoothFlow flow({{0.24, 0.11, 0.04}}, {{0.14, 0.42, -0.03}},
                                    1.3, 1.5, 0.24);
  grmhd::Solutions::SmoothFlow flow_copy({{0.24, 0.11, 0.04}},
                                         {{0.14, 0.42, -0.03}}, 1.3, 1.5, 0.24);
  test_move_semantics(std::move(flow), flow_copy);  //  NOLINT
}

void test_serialize() {
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

  pypp::check_with_random_values<1>(
      &SmoothFlowProxy::hydro_variables<DataType>,
      SmoothFlowProxy(mean_velocity, wave_vector, pressure, adiabatic_index,
                      perturbation_size),
      "GrMhd.SmoothFlow",
      {"rest_mass_density", "electron_fraction", "spatial_velocity",
       "specific_internal_energy", "pressure", "lorentz_factor",
       "specific_enthalpy_relativistic"},
      {{{-15., 15.}}},
      std::make_tuple(mean_velocity, wave_vector, pressure, adiabatic_index,
                      perturbation_size),
      used_for_size);

  pypp::check_with_random_values<1>(
      &SmoothFlowProxy::grmhd_variables<DataType>,
      SmoothFlowProxy(mean_velocity, wave_vector, pressure, adiabatic_index,
                      perturbation_size),
      "GrMhd.SmoothFlow",
      {"rest_mass_density", "electron_fraction", "spatial_velocity",
       "specific_internal_energy", "pressure", "lorentz_factor",
       "specific_enthalpy_relativistic", "magnetic_field",
       "divergence_cleaning_field"},
      {{{-15., 15.}}},
      std::make_tuple(mean_velocity, wave_vector, pressure, adiabatic_index,
                      perturbation_size),
      used_for_size);

  // Test a few of the GR components to make sure that the implementation
  // correctly forwards to the background solution. Not meant to be extensive.
  grmhd::Solutions::SmoothFlow soln(mean_velocity, wave_vector, pressure,
                                    adiabatic_index, perturbation_size);
  const auto coords = make_with_value<tnsr::I<DataType, 3>>(used_for_size, 1.0);
  CHECK_ITERABLE_APPROX(
      make_with_value<Scalar<DataType>>(used_for_size, 1.0),
      get<gr::Tags::Lapse<DataType>>(soln.variables(
          coords, 0.0, tmpl::list<gr::Tags::Lapse<DataType>>{})));
  CHECK_ITERABLE_APPROX(
      make_with_value<Scalar<DataType>>(used_for_size, 1.0),
      get<gr::Tags::SqrtDetSpatialMetric<DataType>>(soln.variables(
          coords, 0.0,
          tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataType>>{})));
  auto expected_spatial_metric =
      make_with_value<tnsr::ii<DataType, 3, Frame::Inertial>>(used_for_size,
                                                              0.0);
  for (size_t i = 0; i < 3; ++i) {
    expected_spatial_metric.get(i, i) = 1.0;
  }
  const auto spatial_metric =
      get<gr::Tags::SpatialMetric<DataType, 3>>(soln.variables(
          coords, 0.0, tmpl::list<gr::Tags::SpatialMetric<DataType, 3>>{}));
  CHECK_ITERABLE_APPROX(expected_spatial_metric, spatial_metric);
}

void test_solution() {
  register_classes_with_charm<grmhd::Solutions::SmoothFlow>();
  const std::unique_ptr<evolution::initial_data::InitialData> option_solution =
      TestHelpers::test_option_tag_factory_creation<
          evolution::initial_data::OptionTags::InitialData,
          grmhd::Solutions::SmoothFlow>(
          "SmoothFlow:\n"
          "  MeanVelocity: [0.1, -0.2, 0.3]\n"
          "  WaveVector: [-0.13, -0.54, 0.04]\n"
          "  Pressure: 1.23\n"
          "  AdiabaticIndex: 1.4\n"
          "  PerturbationSize: 0.75\n")
          ->get_clone();
  const auto deserialized_option_solution =
      serialize_and_deserialize(option_solution);
  const auto& solution = dynamic_cast<const grmhd::Solutions::SmoothFlow&>(
      *deserialized_option_solution);

  const std::array<double, 3> x{{4.0, 4.0, 4.0}};
  const std::array<double, 3> dx{{1.e-3, 1.e-3, 1.e-3}};

  domain::creators::Brick brick(x - dx, x + dx, {{0, 0, 0}}, {{4, 4, 4}},
                                {{false, false, false}});
  Mesh<3> mesh{brick.initial_extents()[0],
               SpatialDiscretization::Basis::Legendre,
               SpatialDiscretization::Quadrature::GaussLobatto};
  const auto domain = brick.create_domain();
  verify_grmhd_solution(solution, domain.blocks()[0], mesh, 1.e-10, 1.234,
                        1.e-4);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.GrMhd.SmoothFlow",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/"};

  test_create_from_options();
  test_serialize();
  test_move();

  test_variables(std::numeric_limits<double>::signaling_NaN());
  test_variables(DataVector(5));

  test_solution();
}
