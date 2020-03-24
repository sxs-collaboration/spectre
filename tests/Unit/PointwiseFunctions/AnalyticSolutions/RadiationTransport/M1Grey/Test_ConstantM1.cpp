// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/RadiationTransport/Tags.hpp"  // IWYU pragma: keep
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RadiationTransport/M1Grey/ConstantM1.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_include <vector>
// IWYU pragma: no_include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"

// IWYU pragma: no_forward_declare Tags::dt
// IWYU pragma: no_forward_declare Tensor

namespace {

struct ConstantM1Proxy : RadiationTransport::M1Grey::Solutions::ConstantM1 {
  using RadiationTransport::M1Grey::Solutions::ConstantM1::ConstantM1;

  using hydro_variables_tags =
      tmpl::list<hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>,
                 hydro::Tags::LorentzFactor<DataVector>>;

  using m1_variables_tags =
      tmpl::list<RadiationTransport::M1Grey::Tags::TildeE<
                     Frame::Inertial, neutrinos::ElectronNeutrinos<0>>,
                 RadiationTransport::M1Grey::Tags::TildeS<
                     Frame::Inertial, neutrinos::ElectronNeutrinos<0>>>;

  tuples::tagged_tuple_from_typelist<hydro_variables_tags> hydro_variables(
      const tnsr::I<DataVector, 3>& x, double t) const noexcept {
    return variables(x, t, hydro_variables_tags{});
  }

  tuples::tagged_tuple_from_typelist<m1_variables_tags> m1_variables(
      const tnsr::I<DataVector, 3>& x, double t) const noexcept {
    return variables(x, t, m1_variables_tags{});
  }
};

void test_create_from_options() noexcept {
  const auto flow = TestHelpers::test_creation<
      RadiationTransport::M1Grey::Solutions::ConstantM1>(
      "MeanVelocity: [0.0, 0.2, 0.1]\n"
      "ComovingEnergyDensity: 1.3");
  CHECK(flow == RadiationTransport::M1Grey::Solutions::ConstantM1(
                    {{0.0, 0.2, 0.1}}, 1.3));
}

void test_move() noexcept {
  RadiationTransport::M1Grey::Solutions::ConstantM1 flow({{0.24, 0.11, 0.04}},
                                                         1.3);
  RadiationTransport::M1Grey::Solutions::ConstantM1 flow_copy(
      {{0.24, 0.11, 0.04}}, 1.3);
  test_move_semantics(std::move(flow), flow_copy);  //  NOLINT
}

void test_serialize() noexcept {
  RadiationTransport::M1Grey::Solutions::ConstantM1 flow({{0.24, 0.11, 0.04}},
                                                         1.3);
  test_serialization(flow);
}

void test_variables(const DataVector& used_for_size) {
  const std::array<double, 3> mean_velocity = {{0.23, 0.01, 0.31}};
  const double comoving_energy_density = 1.3;

  // Test M1 variables
  pypp::check_with_random_values<1, ConstantM1Proxy::m1_variables_tags>(
      &ConstantM1Proxy::m1_variables,
      ConstantM1Proxy(mean_velocity, comoving_energy_density), "TestFunctions",
      {"constant_m1_tildeE", "constant_m1_tildeS"}, {{{-15., 15.}}},
      std::make_tuple(mean_velocity, comoving_energy_density), used_for_size);

  // Test hydro variables
  pypp::check_with_random_values<1, ConstantM1Proxy::hydro_variables_tags>(
      &ConstantM1Proxy::hydro_variables,
      ConstantM1Proxy(mean_velocity, comoving_energy_density), "TestFunctions",
      {"constant_m1_spatial_velocity", "constant_m1_lorentz_factor"},
      {{{-15., 15.}}}, std::make_tuple(mean_velocity, comoving_energy_density),
      used_for_size);

  // Test a few of the GR components to make sure that the implementation
  // correctly forwards to the background solution. Not meant to be extensive.
  RadiationTransport::M1Grey::Solutions::ConstantM1 soln(
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
  const auto spatial_metric = get<
      gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(soln.variables(
      coords, 0.0,
      tmpl::list<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>{}));
  CHECK_ITERABLE_APPROX(expected_spatial_metric, spatial_metric);
}

}  // end namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.M1Grey.ConstantM1",
    "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/RadiationTransport/M1Grey/"};

  test_create_from_options();
  test_serialize();
  test_move();

  test_variables(DataVector(5));
}
