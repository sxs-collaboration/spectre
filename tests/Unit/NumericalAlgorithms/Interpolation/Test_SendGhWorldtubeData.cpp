// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/Actions/BoundaryComputeAndSendToEvolution.hpp"
#include "Evolution/Systems/Cce/Callbacks/SendGhWorldtubeData.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/PostInterpolationCallback.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/Sphere.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <typename InterpolationTargetTag, bool LocalTimeStepping>
struct dispatch_to_send_gh_worldtube_data {
  template <typename ParallelComponent, typename... DbTags, typename ArrayIndex,
            typename Metavariables>
  static void apply(const db::DataBox<tmpl::list<DbTags...>>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/) {
    if constexpr (tmpl2::flat_any_v<std::is_same_v<::Tags::Time, DbTags>...>) {
      using post_intrp_callback = intrp::callbacks::SendGhWorldtubeData<
          Cce::CharacteristicEvolution<Metavariables>, InterpolationTargetTag,
          false, LocalTimeStepping>;
      static_assert(tt::assert_conforms_to_v<
                    post_intrp_callback,
                    intrp::protocols::PostInterpolationCallback>);
      post_intrp_callback::apply(box, cache, db::get<::Tags::Time>(box));
    } else {
      ERROR("Missing required tag ::Tag::Time");
    }
  }
};

tnsr::aa<DataVector, 3> received_spacetime_metric;
tnsr::iaa<DataVector, 3> received_phi;
tnsr::aa<DataVector, 3> received_pi;
struct test_receive_gh_data {
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void apply(const db::DataBox<DbTagList>& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& /* cache*/,
                    const ArrayIndex& /*array_index*/, const double /*time*/,
                    const tnsr::aa<DataVector, 3>& spacetime_metric,
                    const tnsr::iaa<DataVector, 3>& phi,
                    const tnsr::aa<DataVector, 3>& pi) {
    received_spacetime_metric = spacetime_metric;
    received_phi = phi;
    received_pi = pi;
  }
};

template <typename Metavariables, typename InterpolationTargetTag>
struct mock_interpolation_target {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tags =
      tmpl::list<intrp::Tags::Sphere<InterpolationTargetTag>>;
  using simple_tags = tmpl::list<
      ::Tags::Variables<tmpl::list<::gr::Tags::SpacetimeMetric<DataVector, 3>,
                                   gh::Tags::Phi<DataVector, 3>,
                                   gh::Tags::Pi<DataVector, 3>>>,
      ::Tags::Time>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags, tmpl::list<>>>>>;
};

template <typename Metavariables>
struct mock_gh_worldtube_boundary {
  using metavariables = Metavariables;
  using component_being_mocked = Cce::GhWorldtubeBoundary<Metavariables>;
  using replace_these_simple_actions = tmpl::conditional_t<
      Metavariables::local_time_stepping,
      tmpl::list<Cce::Actions::ReceiveGhWorldtubeData<
          Cce::CharacteristicEvolution<Metavariables>, false>>,
      tmpl::list<Cce::Actions::SendToEvolution<
          Cce::GhWorldtubeBoundary<Metavariables>,
          Cce::CharacteristicEvolution<Metavariables>>>>;
  using with_these_simple_actions = tmpl::list<test_receive_gh_data>;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

template <bool LocalTimeStepping>
struct test_metavariables {
  struct Target : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::Time;
    using vars_to_interpolate_to_target =
        typename mock_interpolation_target<test_metavariables,
                                           Target>::simple_tags;
    using compute_target_points =
        intrp::TargetPoints::Sphere<Target, ::Frame::Inertial>;
    using post_interpolation_callback = intrp::callbacks::SendGhWorldtubeData<
        Cce::CharacteristicEvolution<test_metavariables>, Target, false,
        LocalTimeStepping>;
    using compute_items_on_target = tmpl::list<>;
  };

  static constexpr bool local_time_stepping = LocalTimeStepping;
  using component_list = tmpl::list<
      mock_gh_worldtube_boundary<test_metavariables<LocalTimeStepping>>,
      mock_interpolation_target<test_metavariables<LocalTimeStepping>, Target>>;
};

template <bool LocalTimeStepping, typename Generator>
void test_callback_function(const gsl::not_null<Generator*> gen) {
  UniformCustomDistribution<size_t> resolution_distribution{7, 10};
  const size_t l_max = resolution_distribution(*gen);
  UniformCustomDistribution<double> value_distribution{0.1, 1.0};
  using spacetime_tags =
      tmpl::list<::gr::Tags::SpacetimeMetric<DataVector, 3>,
                 gh::Tags::Phi<DataVector, 3>, gh::Tags::Pi<DataVector, 3>>;
  using target = typename test_metavariables<LocalTimeStepping>::Target;
  const intrp::AngularOrdering angular_ordering = intrp::AngularOrdering::Cce;
  const double radius = 3.6;
  const std::array<double, 3> center = {{0.05, 0.06, 0.07}};
  // Options for Sphere
  intrp::OptionHolders::Sphere sphere_opts(l_max, center, radius,
                                           angular_ordering);
  Variables<spacetime_tags> spacetime_variables{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  tmpl::for_each<spacetime_tags>(
      [&gen, &value_distribution, &spacetime_variables](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        fill_with_random_values(make_not_null(&get<tag>(spacetime_variables)),
                                gen, make_not_null(&value_distribution));
      });
  ActionTesting::MockRuntimeSystem<test_metavariables<LocalTimeStepping>>
      runner{{std::move(sphere_opts)}};
  runner.set_phase(Parallel::Phase::Initialization);
  ActionTesting::emplace_component_and_initialize<
      mock_interpolation_target<test_metavariables<LocalTimeStepping>, target>>(
      &runner, 0_st, {spacetime_variables, 0.05});
  ActionTesting::emplace_component<
      mock_gh_worldtube_boundary<test_metavariables<LocalTimeStepping>>>(
      &runner, 0_st);
  runner.set_phase(Parallel::Phase::Testing);
  ActionTesting::simple_action<
      mock_interpolation_target<test_metavariables<LocalTimeStepping>, target>,
      dispatch_to_send_gh_worldtube_data<target, LocalTimeStepping>>(
      make_not_null(&runner), 0_st);
  ActionTesting::invoke_queued_simple_action<
      mock_gh_worldtube_boundary<test_metavariables<LocalTimeStepping>>>(
      make_not_null(&runner), 0_st);
  // check that the tags have been communicated properly (here they propagate
  // through to the replaced simple action that stores them in the globals)
  CHECK(get<::gr::Tags::SpacetimeMetric<DataVector, 3>>(spacetime_variables) ==
        received_spacetime_metric);
  CHECK(get<gh::Tags::Phi<DataVector, 3>>(spacetime_variables) == received_phi);
  CHECK(get<gh::Tags::Pi<DataVector, 3>>(spacetime_variables) == received_pi);

  // Error test
  intrp::OptionHolders::Sphere sphere_opts2(
      l_max, center, std::vector<double>{3.6, 3.7}, angular_ordering);
  ActionTesting::MockRuntimeSystem<test_metavariables<LocalTimeStepping>>
      runner2{{std::move(sphere_opts2)}};
  runner2.set_phase(Parallel::Phase::Initialization);
  ActionTesting::emplace_component_and_initialize<
      mock_interpolation_target<test_metavariables<LocalTimeStepping>, target>>(
      &runner2, 0_st, {spacetime_variables, 0.05});
  ActionTesting::emplace_component<
      mock_gh_worldtube_boundary<test_metavariables<LocalTimeStepping>>>(
      &runner2, 0_st);
  runner2.set_phase(Parallel::Phase::Testing);
  CHECK_THROWS_WITH(
      ([&runner2]() {
        ActionTesting::simple_action<
            mock_interpolation_target<test_metavariables<LocalTimeStepping>,
                                      target>,
            dispatch_to_send_gh_worldtube_data<target, LocalTimeStepping>>(
            make_not_null(&runner2), 0_st);
      })(),
      Catch::Contains(
          "SendGhWorldtubeData expects a single worldtube radius, not"));
}

SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.Interpolation.SendGhWorldtubeDataCallback",
    "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  // For local time stepping
  test_callback_function<true>(make_not_null(&gen));
  // For global time stepping
  test_callback_function<false>(make_not_null(&gen));
}
}  // namespace
