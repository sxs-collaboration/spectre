// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"
#include "Evolution/Systems/RadiationTransport/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RadiationTransport/M1Grey/ConstantM1.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using InitialData = evolution::initial_data::InitialData;
using ConstantM1 = RadiationTransport::M1Grey::Solutions::ConstantM1;

struct ConstantM1Proxy : RadiationTransport::M1Grey::Solutions::ConstantM1 {
  using ConstantM1::ConstantM1;
  using hydro_variables_tags =
      tmpl::list<hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::LorentzFactor<DataVector>>;

  using m1_variables_tags =
      tmpl::list<RadiationTransport::M1Grey::Tags::TildeE<
                     Frame::Inertial, neutrinos::ElectronNeutrinos<0>>,
                 RadiationTransport::M1Grey::Tags::TildeS<
                     Frame::Inertial, neutrinos::ElectronNeutrinos<0>>>;

  tuples::tagged_tuple_from_typelist<hydro_variables_tags> hydro_variables(
      const tnsr::I<DataVector, 3>& x, double t) const {
    return variables(x, t, hydro_variables_tags{});
  }

  tuples::tagged_tuple_from_typelist<m1_variables_tags> m1_variables(
      const tnsr::I<DataVector, 3>& x, double t) const {
    return variables(x, t, m1_variables_tags{});
  }
};

void test_create_from_options() {
  register_classes_with_charm<ConstantM1>();
  const std::unique_ptr<evolution::initial_data::InitialData> option_solution =
      TestHelpers::test_option_tag_factory_creation<
          evolution::initial_data::OptionTags::InitialData, ConstantM1>(
          "ConstantM1:\n"
          "  MeanVelocity: [0.0, 0.2, 0.1]\n"
          "  ComovingEnergyDensity: 1.3");
  const auto deserialized_option_solution =
      serialize_and_deserialize(option_solution);
  const auto& created_solution =
      dynamic_cast<const ConstantM1&>(*deserialized_option_solution);
  CHECK(created_solution == ConstantM1({{0.0, 0.2, 0.1}}, 1.3));
}

void test_move() {
  ConstantM1 flow({{0.24, 0.11, 0.04}}, 1.3);
  const ConstantM1 flow_copy({{0.24, 0.11, 0.04}}, 1.3);
  test_move_semantics(std::move(flow), flow_copy);  //  NOLINT
}

void test_serialize() {
  const ConstantM1 flow({{0.24, 0.11, 0.04}}, 1.3);
  test_serialization(flow);
}

void test_derived() {
  register_classes_with_charm<ConstantM1>();
  const std::unique_ptr<InitialData> initial_data_ptr =
      std::make_unique<ConstantM1>(std::array{0.24, 0.11, 0.04}, 1.3);
  const std::unique_ptr<InitialData> deserialized_initial_data_ptr =
      serialize_and_deserialize(initial_data_ptr)->get_clone();
  CHECK(dynamic_cast<ConstantM1*>(deserialized_initial_data_ptr.get()) !=
        nullptr);
}

void test_variables(const DataVector& used_for_size) {
  const std::array<double, 3> mean_velocity = {{0.23, 0.01, 0.31}};
  const double comoving_energy_density = 1.3;

  // Test M1 variables
  pypp::check_with_random_values<1>(
      &ConstantM1Proxy::m1_variables,
      ConstantM1Proxy(mean_velocity, comoving_energy_density), "ConstantM1",
      {"tildeE", "tildeS"}, {{{-15., 15.}}},
      std::make_tuple(mean_velocity, comoving_energy_density), used_for_size);

  // Test hydro variables
  pypp::check_with_random_values<1>(
      &ConstantM1Proxy::hydro_variables,
      ConstantM1Proxy(mean_velocity, comoving_energy_density), "ConstantM1",
      {"spatial_velocity", "lorentz_factor"}, {{{-15., 15.}}},
      std::make_tuple(mean_velocity, comoving_energy_density), used_for_size);

  // Test a few of the GR components to make sure that the implementation
  // correctly forwards to the background solution. Not meant to be extensive.
  const RadiationTransport::M1Grey::Solutions::ConstantM1 soln(
      mean_velocity, comoving_energy_density);
  const auto coords =
      make_with_value<tnsr::I<DataVector, 3>>(used_for_size, 1.0);
  CHECK_ITERABLE_APPROX(
      make_with_value<Scalar<DataVector>>(used_for_size, 1.0),
      get<gr::Tags::Lapse<DataVector>>(soln.variables(
          coords, 0.0, tmpl::list<gr::Tags::Lapse<DataVector>>{})));
  CHECK_ITERABLE_APPROX(
      make_with_value<Scalar<DataVector>>(used_for_size, 1.0),
      get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(soln.variables(
          coords, 0.0,
          tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataVector>>{})));
  auto expected_spatial_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                0.0);
  for (size_t i = 0; i < 3; ++i) {
    expected_spatial_metric.get(i, i) = 1.0;
  }
  const auto spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3>>(soln.variables(
          coords, 0.0, tmpl::list<gr::Tags::SpatialMetric<DataVector, 3>>{}));
  CHECK_ITERABLE_APPROX(expected_spatial_metric, spatial_metric);
}

}  // end namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.M1Grey.ConstantM1",
    "[Unit][PointwiseFunctions]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/RadiationTransport/M1Grey/"};

  test_create_from_options();
  test_serialize();
  test_move();
  test_derived();
  test_variables(DataVector(5));
}
