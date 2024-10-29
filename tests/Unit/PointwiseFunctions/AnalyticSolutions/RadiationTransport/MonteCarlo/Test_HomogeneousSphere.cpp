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
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Particles/MonteCarlo/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RadiationTransport/MonteCarlo/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RadiationTransport/MonteCarlo/HomogeneousSphere.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Tabulated3d.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

using InitialData = evolution::initial_data::InitialData;
using HomogeneousSphere =
    RadiationTransport::MonteCarlo::Solutions::HomogeneousSphere;
using TabulatedEoS = EquationsOfState::Tabulated3D<true>;

const std::string h5_file_name_compose{
    unit_test_src_path() +
    "PointwiseFunctions/Hydro/EquationsOfState/dd2_unit_test.h5"};

void test_create_from_options() {
  register_classes_with_charm<HomogeneousSphere>();
  const std::unique_ptr<InitialData> option_solution =
      TestHelpers::test_option_tag_factory_creation<
          evolution::initial_data::OptionTags::InitialData, HomogeneousSphere>(
          "HomogeneousSphere:\n"
          "  Radius: 1.0\n"
          "  Densities: [1.e-12, 1.e-3]\n"
          "  Temperatures: [0.01, 10.0]\n"
          "  ElectronFractions: [0.1,0.1]\n"
          "  EquationOfState:\n"
          "    Tabulated3D:\n"
          "       TableFilename: " +
          unit_test_src_path() +
          "PointwiseFunctions/Hydro/EquationsOfState/dd2_unit_test.h5\n"
          "       TableSubFilename: 'dd2'");

  register_derived_classes_with_charm<TabulatedEoS>();
  Parallel::printf("%s", h5_file_name_compose);
  const auto deserialized_option_solution =
      serialize_and_deserialize(option_solution);
  const auto& created_solution =
      dynamic_cast<const HomogeneousSphere&>(*deserialized_option_solution);
  CHECK(created_solution ==
        HomogeneousSphere(
            1.0, {{1.e-12, 1.e-3}}, {{0.01, 10.0}}, {{0.1, 0.1}},
            std::make_unique<TabulatedEoS>(h5_file_name_compose, "/dd2")));
  Parallel::printf("Testing HomogeneousSphere\n");
}

void test_move() {
  HomogeneousSphere sphere(
      1.0, {{1.e-12, 1.e-3}}, {{0.01, 10.0}}, {{0.1, 0.1}},
      std::make_unique<TabulatedEoS>(h5_file_name_compose, "/dd2"));
  const HomogeneousSphere sphere_copy(
      1.0, {{1.e-12, 1.e-3}}, {{0.01, 10.0}}, {{0.1, 0.1}},
      std::make_unique<TabulatedEoS>(h5_file_name_compose, "/dd2"));
  test_move_semantics(std::move(sphere), sphere_copy);  //  NOLINT
}

void test_serialize() {
  const HomogeneousSphere sphere(
      1.0, {{1.e-12, 1.e-3}}, {{0.01, 10.0}}, {{0.1, 0.1}},
      std::make_unique<TabulatedEoS>(h5_file_name_compose, "/dd2"));
  test_serialization(sphere);
}

void test_derived() {
  register_classes_with_charm<HomogeneousSphere>();
  register_derived_classes_with_charm<TabulatedEoS>();

  const std::unique_ptr<InitialData> initial_data_ptr =
      std::make_unique<HomogeneousSphere>(
          1.0, std::array{1.e-12, 1.e-3}, std::array{0.01, 10.0},
          std::array{0.1, 0.1},
          std::make_unique<TabulatedEoS>(h5_file_name_compose, "/dd2"));
  const std::unique_ptr<InitialData> deserialized_initial_data_ptr =
      serialize_and_deserialize(initial_data_ptr)->get_clone();
  CHECK(dynamic_cast<HomogeneousSphere*>(deserialized_initial_data_ptr.get()) !=
        nullptr);
}

void test_variables(const DataVector& used_for_size) {
  const double radius = 1.0;
  const std::array<double, 2> densities = {{1.e-12, 1.e-3}};
  const std::array<double, 2> temperatures = {{0.01, 10.0}};
  const std::array<double, 2> electron_fractions = {{0.06, 0.08}};
  const HomogeneousSphere soln(
      radius, densities, temperatures, electron_fractions,
      std::make_unique<TabulatedEoS>(h5_file_name_compose, "/dd2"));

  // Test a few of the GR components to make sure that the implementation
  // correctly forwards to the background solution. Not meant to be extensive.
  auto coords = make_with_value<tnsr::I<DataVector, 3>>(used_for_size, 1.0);
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<double> rng_uniform(-1.5 * radius,
                                                     1.5 * radius);
  for (size_t p = 0; p < used_for_size.size(); p++) {
    get<0>(coords)[p] = rng_uniform(generator);
    get<1>(coords)[p] = rng_uniform(generator);
    get<2>(coords)[p] = rng_uniform(generator);
  }
  auto coord_radius = get(magnitude(coords));
  auto expected_temperature =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto expected_density =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto expected_electron_fraction =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  for (size_t p = 0; p < used_for_size.size(); p++) {
    get(expected_temperature)[p] =
        (coord_radius[p] > radius) ? temperatures[1] : temperatures[0];
    get(expected_density)[p] =
        (coord_radius[p] > radius) ? densities[1] : densities[0];
    get(expected_electron_fraction)[p] = (coord_radius[p] > radius)
                                             ? electron_fractions[1]
                                             : electron_fractions[0];
  }
  CHECK_ITERABLE_APPROX(
      expected_temperature,
      get<hydro::Tags::Temperature<DataVector>>(soln.variables(
          coords, 0.0, tmpl::list<hydro::Tags::Temperature<DataVector>>{})));
  CHECK_ITERABLE_APPROX(
      expected_density,
      get<hydro::Tags::RestMassDensity<DataVector>>(soln.variables(
          coords, 0.0,
          tmpl::list<hydro::Tags::RestMassDensity<DataVector>>{})));
  CHECK_ITERABLE_APPROX(
      expected_electron_fraction,
      get<hydro::Tags::ElectronFraction<DataVector>>(soln.variables(
          coords, 0.0,
          tmpl::list<hydro::Tags::ElectronFraction<DataVector>>{})));
  CHECK_ITERABLE_APPROX(
      make_with_value<Scalar<DataVector>>(used_for_size, 1.0),
      get<hydro::Tags::LorentzFactor<DataVector>>(soln.variables(
          coords, 0.0, tmpl::list<hydro::Tags::LorentzFactor<DataVector>>{})));
  const auto spatial_velocity =
      get<hydro::Tags::SpatialVelocity<DataVector, 3>>(soln.variables(
          coords, 0.0,
          tmpl::list<hydro::Tags::SpatialVelocity<DataVector, 3>>{}));
  const auto expected_spatial_velocity =
      make_with_value<tnsr::I<DataVector, 3>>(used_for_size, 0.0);
  CHECK_ITERABLE_APPROX(expected_spatial_velocity, spatial_velocity);
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
    "Unit.PointwiseFunctions.AnalyticSolutions.MonteCarlo.HomogeneousSphere",
    "[Unit][PointwiseFunctions]") {
  test_create_from_options();
  test_serialize();
  test_move();
  test_derived();
  test_variables(DataVector(5));
}
