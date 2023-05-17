// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Dispatch.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Gauges.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/SetPiFromGauge.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Tags/GaugeCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpacetimeDerivativeOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim>
void test(const gsl::not_null<std::mt19937*> generator) {
  CAPTURE(Dim);
  std::uniform_real_distribution<> metric_dist(0.1, 1.);
  std::uniform_real_distribution<> deriv_dist(-1.e-5, 1.e-5);

  using evolved_vars_tags =
      tmpl::list<gr::Tags::SpacetimeMetric<DataVector, Dim>,
                 gh::Tags::Pi<DataVector, Dim>, gh::Tags::Phi<DataVector, Dim>>;

  const Mesh<Dim> mesh{5, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const size_t num_points = mesh.number_of_grid_points();
  Variables<evolved_vars_tags> evolved_vars{mesh.number_of_grid_points()};
  get<gh::Tags::Pi<DataVector, Dim>>(evolved_vars) =
      make_with_random_values<tnsr::aa<DataVector, Dim, Frame::Inertial>>(
          generator, make_not_null(&deriv_dist), num_points);
  get<gh::Tags::Phi<DataVector, Dim>>(evolved_vars) =
      make_with_random_values<tnsr::iaa<DataVector, Dim, Frame::Inertial>>(
          generator, make_not_null(&deriv_dist), num_points);
  get<gr::Tags::SpacetimeMetric<DataVector, Dim>>(evolved_vars) =
      make_with_random_values<tnsr::aa<DataVector, Dim, Frame::Inertial>>(
          generator, make_not_null(&metric_dist), num_points);
  get<0, 0>(get<gr::Tags::SpacetimeMetric<DataVector, Dim>>(evolved_vars)) +=
      -2.0;
  for (size_t i = 0; i < Dim; ++i) {
    get<gr::Tags::SpacetimeMetric<DataVector, Dim>>(evolved_vars)
        .get(i + 1, i + 1) += 4.0;
    get<gr::Tags::SpacetimeMetric<DataVector, Dim>>(evolved_vars)
        .get(i + 1, 0) *= 0.01;
  }

  auto box = db::create<db::AddSimpleTags<
      ::Tags::Time, ::Tags::Variables<evolved_vars_tags>,
      domain::Tags::Mesh<Dim>, domain::Tags::ElementMap<Dim, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                  Frame::Inertial>,
      domain::Tags::FunctionsOfTimeInitialize,
      domain::Tags::Coordinates<Dim, Frame::ElementLogical>,
      gh::gauges::Tags::GaugeCondition>>(
      0., evolved_vars, mesh,
      ElementMap<Dim, Frame::Grid>{
          ElementId<Dim>{0},
          domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
              domain::CoordinateMaps::Identity<Dim>{})},
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<Dim>{}),
      std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{},
      logical_coordinates(mesh),
      std::unique_ptr<gh::gauges::GaugeCondition>(
          std::make_unique<gh::gauges::DampedHarmonic>(
              100., std::array{1.2, 1.5, 1.7}, std::array{2, 4, 6})));
  db::mutate_apply<gh::gauges::SetPiFromGauge<Dim>>(make_not_null(&box));

  // Verify that the gauge constraint is satisfied
  const auto& spacetime_metric =
      db::get<gr::Tags::SpacetimeMetric<DataVector, Dim>>(box);
  const auto& pi = db::get<gh::Tags::Pi<DataVector, Dim>>(box);
  const auto& phi = db::get<gh::Tags::Phi<DataVector, Dim>>(box);

  const auto spatial_metric = gr::spatial_metric(spacetime_metric);
  auto [sqrt_det_spatial_metric, inverse_spatial_metric] =
      determinant_and_inverse(spatial_metric);
  get(sqrt_det_spatial_metric) = sqrt(get(sqrt_det_spatial_metric));
  const auto shift = gr::shift(spacetime_metric, inverse_spatial_metric);
  const auto lapse = gr::lapse(shift, spacetime_metric);
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);
  const auto spacetime_normal_one_form =
      gr::spacetime_normal_one_form<DataVector, Dim, Frame::Inertial>(lapse);
  const auto spacetime_normal_vector =
      gr::spacetime_normal_vector(lapse, shift);
  tnsr::abb<DataVector, Dim, Frame::Inertial> d4_spacetime_metric{};
  gh::spacetime_derivative_of_spacetime_metric(
      make_not_null(&d4_spacetime_metric), lapse, shift, pi, phi);

  Scalar<DataVector> half_pi_two_normals{get(lapse).size(), 0.0};
  tnsr::i<DataVector, Dim, Frame::Inertial> half_phi_two_normals{
      get(lapse).size(), 0.0};
  for (size_t a = 0; a < Dim + 1; ++a) {
    get(half_pi_two_normals) += spacetime_normal_vector.get(a) *
                                spacetime_normal_vector.get(a) * pi.get(a, a);
    for (size_t i = 0; i < Dim; ++i) {
      half_phi_two_normals.get(i) += 0.5 * spacetime_normal_vector.get(a) *
                                     spacetime_normal_vector.get(a) *
                                     phi.get(i, a, a);
    }
    for (size_t b = a + 1; b < Dim + 1; ++b) {
      get(half_pi_two_normals) += 2.0 * spacetime_normal_vector.get(a) *
                                  spacetime_normal_vector.get(b) * pi.get(a, b);
      for (size_t i = 0; i < Dim; ++i) {
        half_phi_two_normals.get(i) += spacetime_normal_vector.get(a) *
                                       spacetime_normal_vector.get(b) *
                                       phi.get(i, a, b);
      }
    }
  }
  get(half_pi_two_normals) *= 0.5;

  tnsr::a<DataVector, Dim, Frame::Inertial> gauge_h(num_points);
  tnsr::ab<DataVector, Dim, Frame::Inertial> d4_gauge_h(num_points);
  gh::gauges::dispatch(
      make_not_null(&gauge_h), make_not_null(&d4_gauge_h), lapse, shift,
      spacetime_normal_one_form, spacetime_normal_vector,
      sqrt_det_spatial_metric, inverse_spatial_metric, d4_spacetime_metric,
      half_pi_two_normals, half_phi_two_normals, spacetime_metric, pi, phi,
      mesh, db::get<::Tags::Time>(box),
      db::get<domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                          Frame::Inertial>>(
          box)(db::get<domain::Tags::ElementMap<Dim, Frame::Grid>>(box)(
          db::get<domain::Tags::Coordinates<Dim, Frame::ElementLogical>>(box))),
      {}, db::get<gh::gauges::Tags::GaugeCondition>(box));

  const auto gauge_constraint = gh::gauge_constraint(
      gauge_h, spacetime_normal_one_form, spacetime_normal_vector,
      inverse_spatial_metric, inverse_spacetime_metric, pi, phi);
  const tnsr::a<DataVector, Dim, Frame::Inertial> expected_gauge_constraint{
      get<0>(gauge_constraint).size(), 0.};

  Approx local_approx = Approx::custom().epsilon(1.e-10).scale(1.);
  CHECK_ITERABLE_CUSTOM_APPROX(gauge_constraint, expected_gauge_constraint,
                               local_approx);
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GH.Gauge.SetPiFromGauge",
                  "[Unit][Evolution][Actions]") {
  MAKE_GENERATOR(generator);
  test<1>(make_not_null(&generator));
  test<2>(make_not_null(&generator));
  test<3>(make_not_null(&generator));
}
}  // namespace
