// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/SendGhWorldtubeData.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct dispatch_to_send_gh_worldtube_data {
  template <typename ParallelComponent, typename... DbTags, typename ArrayIndex,
            typename Metavariables,
            Requires<tmpl2::flat_any_v<
                std::is_same_v<::Tags::TimeStepId, DbTags>...>> = nullptr>
  static void apply(const db::DataBox<tmpl::list<DbTags...>>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/) noexcept {
    intrp::callbacks::SendGhWorldtubeData<
        Cce::CharacteristicEvolution<Metavariables>, ::Tags::TimeStepId,
        false>::apply(box, cache, db::get<::Tags::TimeStepId>(box));
  }
};

tnsr::aa<DataVector, 3> received_spacetime_metric;
tnsr::iaa<DataVector, 3> received_phi;
tnsr::aa<DataVector, 3> received_pi;
tnsr::aa<DataVector, 3> received_dt_spacetime_metric;
tnsr::iaa<DataVector, 3> received_dt_phi;
tnsr::aa<DataVector, 3> received_dt_pi;
TimeStepId received_time_step_id;
struct test_receive_gh_data {
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void apply(const db::DataBox<DbTagList>& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& /* cache*/,
                    const ArrayIndex& /*array_index*/,
                    const TimeStepId& time_step_id,
                    const tnsr::aa<DataVector, 3>& spacetime_metric,
                    const tnsr::iaa<DataVector, 3>& phi,
                    const tnsr::aa<DataVector, 3>& pi,
                    const tnsr::aa<DataVector, 3>& dt_spacetime_metric,
                    const tnsr::iaa<DataVector, 3>& dt_phi,
                    const tnsr::aa<DataVector, 3>& dt_pi) noexcept {
    received_time_step_id = time_step_id;
    received_spacetime_metric = spacetime_metric;
    received_phi = phi;
    received_pi = pi;
    received_dt_spacetime_metric = dt_spacetime_metric;
    received_dt_phi = dt_phi;
    received_dt_pi = dt_pi;
  }
};

template <typename Metavariables>
struct mock_interpolation_target {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using simple_tags = tmpl::list<
      ::Tags::Variables<tmpl::list<
          ::gr::Tags::SpacetimeMetric<3, Frame::Inertial>,
          GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>,
          GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>,
          ::Tags::dt<::gr::Tags::SpacetimeMetric<3, Frame::Inertial>>,
          ::Tags::dt<::GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>,
          ::Tags::dt<::GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>>>>,
      ::Tags::TimeStepId>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags, tmpl::list<>>>>>;
};

template <typename Metavariables>
struct mock_gh_worldtube_boundary {
  using metavariables = Metavariables;
  using component_being_mocked = Cce::GhWorldtubeBoundary<Metavariables>;
  using replace_these_simple_actions =
      tmpl::list<Cce::Actions::ReceiveGhWorldtubeData<
          Cce::CharacteristicEvolution<Metavariables>, false>>;
  using with_these_simple_actions = tmpl::list<test_receive_gh_data>;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
};

struct test_metavariables {
  using temporal_id = ::Tags::TimeStepId;
  using component_list =
      tmpl::list<mock_gh_worldtube_boundary<test_metavariables>,
                 mock_interpolation_target<test_metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.Interpolation.SendGhWorldtubeDataCallback",
    "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> resolution_distribution{7, 10};
  const size_t l_max = resolution_distribution(gen);
  UniformCustomDistribution<double> value_distribution{0.1, 1.0};
  using spacetime_tags = tmpl::list<
      ::gr::Tags::SpacetimeMetric<3, Frame::Inertial>,
      GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>,
      GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>,
      ::Tags::dt<::gr::Tags::SpacetimeMetric<3, Frame::Inertial>>,
      ::Tags::dt<::GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>,
      ::Tags::dt<::GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>>>;
  Variables<spacetime_tags> spacetime_variables{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  tmpl::for_each<spacetime_tags>([&gen, &value_distribution,
                                  &spacetime_variables](auto tag_v) noexcept {
    using tag = typename decltype(tag_v)::type;
    fill_with_random_values(make_not_null(&get<tag>(spacetime_variables)),
                            make_not_null(&gen),
                            make_not_null(&value_distribution));
  });
  ActionTesting::MockRuntimeSystem<test_metavariables> runner{{}};
  runner.set_phase(test_metavariables::Phase::Initialization);
  ActionTesting::emplace_component_and_initialize<
      mock_interpolation_target<test_metavariables>>(
      &runner, 0_st,
      {spacetime_variables, TimeStepId{true, 0_st, {{0.0, 0.1}, {1, 2}}}});
  ActionTesting::emplace_component<
      mock_gh_worldtube_boundary<test_metavariables>>(&runner, 0_st);
  runner.set_phase(test_metavariables::Phase::Testing);
  ActionTesting::simple_action<mock_interpolation_target<test_metavariables>,
                               dispatch_to_send_gh_worldtube_data>(
      make_not_null(&runner), 0_st);
  ActionTesting::invoke_queued_simple_action<
      mock_gh_worldtube_boundary<test_metavariables>>(make_not_null(&runner),
                                                      0_st);
  // check that the tags have been communicated properly (here they propagate
  // through to the replaced simple action that stores them in the globals)
  CHECK(get<::gr::Tags::SpacetimeMetric<3, Frame::Inertial>>(
            spacetime_variables) == received_spacetime_metric);
  CHECK(get<GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>(
            spacetime_variables) == received_phi);
  CHECK(get<GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>>(
            spacetime_variables) == received_pi);
  CHECK(get<::Tags::dt<::gr::Tags::SpacetimeMetric<3, Frame::Inertial>>>(
            spacetime_variables) == received_dt_spacetime_metric);
  CHECK(get<::Tags::dt<GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>>(
            spacetime_variables) == received_dt_phi);
  CHECK(get<::Tags::dt<GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>>>(
            spacetime_variables) == received_dt_pi);
}
}  // namespace
