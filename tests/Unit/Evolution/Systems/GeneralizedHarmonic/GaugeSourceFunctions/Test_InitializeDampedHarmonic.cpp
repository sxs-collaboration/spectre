// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/InitializeDampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Initialize.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim, typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;

  using initial_tags = tmpl::list<
      Tags::Time, domain::Tags::Mesh<Dim>,
      domain::Tags::Coordinates<Dim, Frame::Inertial>,
      domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                  Frame::Inertial>,
      domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Inertial>,
      typename Metavariables::variables_tag>;
  using initial_compute_tags = db::AddComputeTags<::Tags::DerivCompute<
      typename Metavariables::variables_tag,
      domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Inertial>,
      typename Metavariables::evolved_vars>>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<
          ActionTesting::InitializeDataBox<initial_tags, initial_compute_tags>,
          GeneralizedHarmonic::Actions::InitializeGhAnd3Plus1Variables<Dim>,
          GeneralizedHarmonic::gauges::Actions::InitializeDampedHarmonic<
              Dim>>>>;
};

template <size_t Dim>
struct Metavariables {
  using evolved_vars =
      tmpl::list<gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>,
                 GeneralizedHarmonic::Tags::Pi<Dim, Frame::Inertial>,
                 GeneralizedHarmonic::Tags::Phi<Dim, Frame::Inertial>>;
  using variables_tag = Tags::Variables<evolved_vars>;
  static constexpr size_t volume_dim = Dim;
  using component_list = tmpl::list<component<Dim, Metavariables>>;
  enum class Phase { Initialization, Exit };
};

template <size_t Dim>
void test(const gsl::not_null<std::mt19937*> generator) noexcept {
  CAPTURE(Dim);
  using metavars = Metavariables<Dim>;
  using comp = component<Dim, metavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  PUPable_reg(
      SINGLE_ARG(domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                                       domain::CoordinateMaps::Identity<Dim>>));

  std::uniform_real_distribution<> pdist(0.1, 1.);
  std::uniform_int_distribution<> idist(2, 7);

  // roll on function parameters for lapse / shift terms
  const double t_start = pdist(*generator) * 0.1;
  const double sigma_t = pdist(*generator) * 0.2;
  const double r_max = pdist(*generator) * 0.7;

  MockRuntimeSystem runner{{t_start, sigma_t, r_max}};
  Mesh<Dim> mesh{5, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};
  InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial> inv_jac{
      mesh.number_of_grid_points(), 0.};
  for (size_t i = 0; i < Dim; ++i) {
    inv_jac.get(i, i) = 1.;
  }
  const auto inertial_coords =
      make_with_random_values<tnsr::I<DataVector, Dim, Frame::Inertial>>(
          generator, make_not_null(&pdist), mesh.number_of_grid_points());
  Variables<typename metavars::evolved_vars> evolved_vars{
      mesh.number_of_grid_points()};
  tmpl::for_each<typename metavars::evolved_vars>(
      [&evolved_vars, &generator, &pdist](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        fill_with_random_values(make_not_null(&get<tag>(evolved_vars)),
                                generator, make_not_null(&pdist));
      });

  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&pdist), mesh.number_of_grid_points());
  const auto shift =
      make_with_random_values<tnsr::I<DataVector, Dim, Frame::Inertial>>(
          generator, make_not_null(&pdist), mesh.number_of_grid_points());
  tnsr::ii<DataVector, Dim, Frame::Inertial> spatial_metric{
      mesh.number_of_grid_points(), 0.};
  for (size_t i = 0; i < Dim; ++i) {
    spatial_metric.get(i, i) = 1.;
  }
  get<gr::Tags::SpacetimeMetric<Dim, Frame::Inertial>>(evolved_vars) =
      gr::spacetime_metric(lapse, shift, spatial_metric);

  const double time = 2.0;
  ActionTesting::emplace_component_and_initialize<comp>(
      &runner, 0,
      {time, mesh, inertial_coords,
       domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
           domain::CoordinateMaps::Identity<Dim>{}),
       inv_jac, evolved_vars});

  // Invoke the InitializeGhAnd3Plus1Variables action
  ActionTesting::next_action<comp>(make_not_null(&runner), 0);
  // Invoke the InitializeDampedHarmonic action
  ActionTesting::next_action<comp>(make_not_null(&runner), 0);

  tnsr::a<DataVector, Dim, Frame::Inertial> expected_gauge_h{};
  tnsr::ab<DataVector, Dim, Frame::Inertial> expected_d4_gauge_h{};

  GeneralizedHarmonic::gauges::damped_harmonic_rollon(
      make_not_null(&expected_gauge_h), make_not_null(&expected_d4_gauge_h),
      ActionTesting::get_databox_tag<
          comp, GeneralizedHarmonic::Tags::InitialGaugeH<Dim, Frame::Inertial>>(
          runner, 0),
      ActionTesting::get_databox_tag<
          comp, GeneralizedHarmonic::Tags::SpacetimeDerivInitialGaugeH<
                    Dim, Frame::Inertial>>(runner, 0),
      ActionTesting::get_databox_tag<comp, gr::Tags::Lapse<DataVector>>(runner,
                                                                        0),
      ActionTesting::get_databox_tag<comp,
                                     gr::Tags::Shift<Dim, Frame::Inertial>>(
          runner, 0),
      ActionTesting::get_databox_tag<
          comp, gr::Tags::SpacetimeNormalOneForm<Dim, Frame::Inertial>>(runner,
                                                                        0),
      ActionTesting::get_databox_tag<
          comp, gr::Tags::SqrtDetSpatialMetric<DataVector>>(runner, 0),
      ActionTesting::get_databox_tag<
          comp, gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial>>(runner,
                                                                      0),
      ActionTesting::get_databox_tag<
          comp, gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>>(
          runner, 0),
      ActionTesting::get_databox_tag<
          comp, GeneralizedHarmonic::Tags::Pi<Dim, Frame::Inertial>>(runner, 0),
      ActionTesting::get_databox_tag<
          comp, GeneralizedHarmonic::Tags::Phi<Dim, Frame::Inertial>>(runner,
                                                                      0),
      time,
      ActionTesting::get_databox_tag<
          comp, domain::Tags::Coordinates<Dim, Frame::Inertial>>(runner, 0),

      1., 1., 1.,  // amp_coef_{L1, L2, S}
      4, 4, 4,     // exp_{L1, L2, S}
      t_start, sigma_t, r_max);

  CHECK_ITERABLE_APPROX(
      SINGLE_ARG(ActionTesting::get_databox_tag<
                 comp, GeneralizedHarmonic::Tags::GaugeH<Dim, Frame::Inertial>>(
          runner, 0)),
      expected_gauge_h);
  CHECK_ITERABLE_APPROX(
      SINGLE_ARG(ActionTesting::get_databox_tag<
                 comp, GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<
                           Dim, Frame::Inertial>>(runner, 0)),
      expected_d4_gauge_h);
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GH.Gauge.InitializeDampedHarmonic",
                  "[Unit][Evolution][Actions]") {
  MAKE_GENERATOR(generator);
  test<1>(make_not_null(&generator));
  test<2>(make_not_null(&generator));
  test<3>(make_not_null(&generator));
}
}  // namespace
