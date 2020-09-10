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
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/InitializeDampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Initialize.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
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

  using initial_tags = tmpl::conditional_t<
      metavariables::use_rollon,
      tmpl::list<
          Tags::Time, domain::Tags::Mesh<Dim>,
          domain::Tags::Coordinates<Dim, Frame::Inertial>,
          domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                      Frame::Inertial>,
          domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Inertial>,
          typename Metavariables::variables_tag>,
      tmpl::list<
          ::Initialization::Tags::InitialTime, domain::Tags::Mesh<Dim>,
          domain::Tags::Coordinates<Dim, Frame::Logical>,
          domain::Tags::Coordinates<Dim, Frame::Inertial>,
          domain::Tags::ElementMap<Metavariables::volume_dim, Frame::Grid>,
          domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                      Frame::Inertial>,
          domain::Tags::FunctionsOfTime,
          domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Inertial>,
          typename Metavariables::variables_tag>>;
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
              Dim, metavariables::use_rollon>>>>;
};

template <size_t Dim, bool UseRollon>
struct Metavariables {
  static constexpr bool use_rollon = UseRollon;
  using evolved_vars =
      tmpl::list<gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>,
                 GeneralizedHarmonic::Tags::Pi<Dim, Frame::Inertial>,
                 GeneralizedHarmonic::Tags::Phi<Dim, Frame::Inertial>>;
  using variables_tag = Tags::Variables<evolved_vars>;
  static constexpr size_t volume_dim = Dim;
  using component_list = tmpl::list<component<Dim, Metavariables>>;
  enum class Phase { Initialization, Exit };
};

template <size_t Dim, bool UseRollon>
void test(const gsl::not_null<std::mt19937*> generator) noexcept {
  CAPTURE(Dim);
  CAPTURE(UseRollon);
  using metavars = Metavariables<Dim, UseRollon>;
  using comp = component<Dim, metavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  PUPable_reg(
      SINGLE_ARG(domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                                       domain::CoordinateMaps::Identity<Dim>>));
  PUPable_reg(
      SINGLE_ARG(domain::CoordinateMap<Frame::Logical, Frame::Grid,
                                       domain::CoordinateMaps::Identity<Dim>>));

  std::uniform_real_distribution<> pdist(0.1, 1.);
  std::uniform_int_distribution<> idist(2, 7);

  // roll on function parameters for lapse / shift terms
  const double t_start = pdist(*generator) * 0.1;
  const double sigma_t = pdist(*generator) * 0.2;
  const double r_max = pdist(*generator) * 0.7;

  MockRuntimeSystem runner = [r_max, t_start, sigma_t]() {
    if constexpr (UseRollon) {
      return MockRuntimeSystem{{r_max, t_start, sigma_t}};
    } else {
      (void)t_start;
      (void)sigma_t;
      return MockRuntimeSystem{{r_max}};
    }
  }();
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

  const double time = 0.0;
  if constexpr (UseRollon) {
    ActionTesting::emplace_component_and_initialize<comp>(
        &runner, 0,
        {time, mesh, inertial_coords,
         domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
             domain::CoordinateMaps::Identity<Dim>{}),
         inv_jac, evolved_vars});
  } else {
    tnsr::I<DataVector, Dim, Frame::Logical> logical_coords;
    for (size_t i = 0; i < Dim; ++i) {
      logical_coords.get(i) = inertial_coords.get(i);
    }

    ActionTesting::emplace_component_and_initialize<comp>(
        &runner, 0,
        {time, mesh, logical_coords, inertial_coords,
         ElementMap<Dim, Frame::Grid>{
             ElementId<Dim>{0},
             domain::make_coordinate_map_base<Frame::Logical, Frame::Grid>(
                 domain::CoordinateMaps::Identity<Dim>{})},
         domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
             domain::CoordinateMaps::Identity<Dim>{}),
         std::unordered_map<
             std::string,
             std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{},
         inv_jac, evolved_vars});
  }

  // Invoke the InitializeGhAnd3Plus1Variables action
  ActionTesting::next_action<comp>(make_not_null(&runner), 0);
  // Invoke the InitializeDampedHarmonic action
  ActionTesting::next_action<comp>(make_not_null(&runner), 0);

  tnsr::a<DataVector, Dim, Frame::Inertial> expected_gauge_h{};
  tnsr::ab<DataVector, Dim, Frame::Inertial> expected_d4_gauge_h{};

  if constexpr (UseRollon) {
    GeneralizedHarmonic::gauges::damped_harmonic_rollon(
        make_not_null(&expected_gauge_h), make_not_null(&expected_d4_gauge_h),
        ActionTesting::get_databox_tag<
            comp,
            GeneralizedHarmonic::Tags::InitialGaugeH<Dim, Frame::Inertial>>(
            runner, 0),
        ActionTesting::get_databox_tag<
            comp, GeneralizedHarmonic::Tags::SpacetimeDerivInitialGaugeH<
                      Dim, Frame::Inertial>>(runner, 0),
        ActionTesting::get_databox_tag<comp, gr::Tags::Lapse<DataVector>>(
            runner, 0),
        ActionTesting::get_databox_tag<comp,
                                       gr::Tags::Shift<Dim, Frame::Inertial>>(
            runner, 0),
        ActionTesting::get_databox_tag<
            comp, gr::Tags::SpacetimeNormalOneForm<Dim, Frame::Inertial>>(
            runner, 0),
        ActionTesting::get_databox_tag<
            comp, gr::Tags::SqrtDetSpatialMetric<DataVector>>(runner, 0),
        ActionTesting::get_databox_tag<
            comp, gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial>>(runner,
                                                                        0),
        ActionTesting::get_databox_tag<
            comp, gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>>(
            runner, 0),
        ActionTesting::get_databox_tag<
            comp, GeneralizedHarmonic::Tags::Pi<Dim, Frame::Inertial>>(runner,
                                                                       0),
        ActionTesting::get_databox_tag<
            comp, GeneralizedHarmonic::Tags::Phi<Dim, Frame::Inertial>>(runner,
                                                                        0),
        time,
        ActionTesting::get_databox_tag<
            comp, domain::Tags::Coordinates<Dim, Frame::Inertial>>(runner, 0),
        1., 1., 1.,  // amp_coef_{L1, L2, S}
        4, 4, 4,     // exp_{L1, L2, S}
        t_start, sigma_t, r_max);
  } else {
    GeneralizedHarmonic::gauges::damped_harmonic(
        make_not_null(&expected_gauge_h), make_not_null(&expected_d4_gauge_h),
        ActionTesting::get_databox_tag<comp, gr::Tags::Lapse<DataVector>>(
            runner, 0),
        ActionTesting::get_databox_tag<comp,
                                       gr::Tags::Shift<Dim, Frame::Inertial>>(
            runner, 0),
        ActionTesting::get_databox_tag<
            comp, gr::Tags::SpacetimeNormalOneForm<Dim, Frame::Inertial>>(
            runner, 0),
        ActionTesting::get_databox_tag<
            comp, gr::Tags::SqrtDetSpatialMetric<DataVector>>(runner, 0),
        ActionTesting::get_databox_tag<
            comp, gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial>>(runner,
                                                                        0),
        ActionTesting::get_databox_tag<
            comp, gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>>(
            runner, 0),
        ActionTesting::get_databox_tag<
            comp, GeneralizedHarmonic::Tags::Pi<Dim, Frame::Inertial>>(runner,
                                                                       0),
        ActionTesting::get_databox_tag<
            comp, GeneralizedHarmonic::Tags::Phi<Dim, Frame::Inertial>>(runner,
                                                                        0),
        ActionTesting::get_databox_tag<
            comp, domain::Tags::Coordinates<Dim, Frame::Inertial>>(runner, 0),
        1., 1., 1.,  // amp_coef_{L1, L2, S}
        4, 4, 4,     // exp_{L1, L2, S}
        r_max);
  }

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

  // Verify that the gauge constraint is satisfied
  const auto& spacetime_metric = ActionTesting::get_databox_tag<
      comp, gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>>(runner,
                                                                         0);
  const auto& pi = ActionTesting::get_databox_tag<
      comp, GeneralizedHarmonic::Tags::Pi<Dim, Frame::Inertial>>(runner, 0);
  const auto& phi = ActionTesting::get_databox_tag<
      comp, GeneralizedHarmonic::Tags::Phi<Dim, Frame::Inertial>>(runner, 0);
  const auto& gauge_h = ActionTesting::get_databox_tag<
      comp, GeneralizedHarmonic::Tags::GaugeH<Dim, Frame::Inertial>>(runner, 0);
  const auto inverse_spatial_metric =
      determinant_and_inverse(gr::spatial_metric(spacetime_metric)).second;
  const auto inverse_spacetime_metric = gr::inverse_spacetime_metric(
      gr::lapse(shift, spacetime_metric),
      gr::shift(spacetime_metric, inverse_spatial_metric),
      inverse_spatial_metric);
  const auto spacetime_unit_normal_one_form =
      gr::spacetime_normal_one_form<Dim, Frame::Inertial>(
          gr::lapse(shift, spacetime_metric));
  const auto spacetime_unit_normal_vector = gr::spacetime_normal_vector(
      gr::lapse(shift, spacetime_metric),
      gr::shift(spacetime_metric, inverse_spatial_metric));
  const auto gauge_constraint = GeneralizedHarmonic::gauge_constraint(
      gauge_h, spacetime_unit_normal_one_form, spacetime_unit_normal_vector,
      inverse_spatial_metric, inverse_spacetime_metric, pi, phi);
  const tnsr::a<DataVector, Dim, Frame::Inertial> expected_gauge_constraint{
      get<0>(gauge_constraint).size(), 0.};

  Approx local_approx = Approx::custom().epsilon(1.e-10).scale(1.);
  CHECK_ITERABLE_CUSTOM_APPROX(gauge_constraint, expected_gauge_constraint,
                               local_approx);
}

void test_options() noexcept {
  Options::Parser<
      tmpl::list<GeneralizedHarmonic::gauges::Actions::OptionTags::
                     DampedHarmonicRollOnStart,
                 GeneralizedHarmonic::gauges::Actions::OptionTags::
                     DampedHarmonicRollOnWindow,
                 GeneralizedHarmonic::gauges::Actions::OptionTags::
                     DampedHarmonicSpatialDecayWidth<Frame::Inertial>>>
      opts("");
  opts.parse(
      "EvolutionSystem:\n"
      "  GeneralizedHarmonic:\n"
      "    Gauge:\n"
      "      RollOnStartTime : 0.\n"
      "      RollOnTimeWindow : 100.\n"
      "      SpatialDecayWidth : 50.\n");
  CHECK(opts.template get<GeneralizedHarmonic::gauges::Actions::OptionTags::
                              DampedHarmonicRollOnStart>() == 0.);
  CHECK(opts.template get<GeneralizedHarmonic::gauges::Actions::OptionTags::
                              DampedHarmonicRollOnWindow>() == 100.);
  CHECK(opts.template get<
            GeneralizedHarmonic::gauges::Actions::OptionTags::
                DampedHarmonicSpatialDecayWidth<Frame::Inertial>>() == 50.);
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GH.Gauge.InitializeDampedHarmonic",
                  "[Unit][Evolution][Actions]") {
  MAKE_GENERATOR(generator);
  test<1, true>(make_not_null(&generator));
  test<2, true>(make_not_null(&generator));
  test<3, true>(make_not_null(&generator));

  test<1, false>(make_not_null(&generator));
  test<2, false>(make_not_null(&generator));
  test<3, false>(make_not_null(&generator));

  test_options();
}
}  // namespace
