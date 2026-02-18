// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/CompletionCriteria.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/CompletionSingleton.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename Metavariables>
struct MockSingletonComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using simple_tags =
      tmpl::list<gh::bbh::Tags::GaugeConstraintExceeded,
                 gh::bbh::Tags::ThreeIndexConstraintExceeded,
                 gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold,
                 gh::bbh::Tags::CommonHorizonSuccessCount,
                 gh::bbh::Tags::CompletionRequested,
                 gh::bbh::Tags::CommonHorizonSuccessRecords,
                 gh::bbh::Tags::ConstraintCheckRecords,
                 gh::bbh::Tags::ReportedConstraintCheckRecords,
                 gh::bbh::Tags::ElementCompletionRequested>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
};

template <typename Metavariables>
struct MockElementComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using simple_tags = tmpl::list<gh::bbh::Tags::ElementCompletionRequested>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
};

struct MockMetavariables {
  using component_list = tmpl::list<MockSingletonComponent<MockMetavariables>,
                                    MockElementComponent<MockMetavariables>>;
  using gh_dg_element_array = MockElementComponent<MockMetavariables>;
  using const_global_cache_tags =
      tmpl::list<gh::bbh::Tags::MinCommonHorizonSuccessesBeforeChecks,
                 gh::bbh::Tags::MaxCommonHorizonSuccesses,
                 gh::bbh::Tags::GaugeConstraintLinfThreshold,
                 gh::bbh::Tags::ThreeIndexConstraintLinfThreshold,
                 gh::bbh::Tags::CommonHorizonLMaxThreshold,
                 gh::bbh::Tags::ConstraintCheckVerbose>;
  using mutable_global_cache_tags = tmpl::list<>;
};

using mock_component = MockSingletonComponent<MockMetavariables>;
using mock_element_component = MockElementComponent<MockMetavariables>;

auto make_runner(const size_t min_common_horizon_successes_before_checks = 2,
                 const size_t max_common_horizon_successes = 100,
                 const double gauge_constraint_linf_threshold = 10.0,
                 const double three_index_constraint_linf_threshold = 20.0,
                 const size_t common_horizon_lmax_threshold = 6,
                 const bool constraint_check_verbose = false) {
  return ActionTesting::MockRuntimeSystem<MockMetavariables>{
      {min_common_horizon_successes_before_checks, max_common_horizon_successes,
       gauge_constraint_linf_threshold, three_index_constraint_linf_threshold,
       common_horizon_lmax_threshold, constraint_check_verbose}};
}

void initialize_components(
    const gsl::not_null<ActionTesting::MockRuntimeSystem<MockMetavariables>*>
        runner) {
  ActionTesting::emplace_component_and_initialize<mock_component>(
      runner, 0,
      {false, false, false, 0_st, false,
       gh::bbh::Tags::CommonHorizonSuccessRecords::type{},
       gh::bbh::Tags::ConstraintCheckRecords::type{},
       gh::bbh::Tags::ReportedConstraintCheckRecords::type{}, false});
  ActionTesting::emplace_component_and_initialize<mock_element_component>(
      runner, 0, {false});
}

SPECTRE_TEST_CASE("Unit.GeneralizedHarmonic.BbhCompletionSingleton",
                  "[Unit][Evolution]") {
  {
    INFO("Constraint checks remain gated by min successes at their check time");
    auto runner = make_runner();
    initialize_components(make_not_null(&runner));
    auto& box = ActionTesting::get_databox<mock_component>(runner, 0);
    auto& element_box =
        ActionTesting::get_databox<mock_element_component>(runner, 0);

    ActionTesting::simple_action<mock_component,
                                 gh::bbh::Actions::ProcessConstraintMaxima>(
        make_not_null(&runner), 0, 1.5, 11.0, 1.0);
    CHECK_FALSE(db::get<gh::bbh::Tags::GaugeConstraintExceeded>(box));
    CHECK_FALSE(db::get<gh::bbh::Tags::CompletionRequested>(box));

    ActionTesting::simple_action<mock_component,
                                 gh::bbh::Actions::RecordCommonHorizonSuccess>(
        make_not_null(&runner), 0, LinkedMessageId<double>{2.0, std::nullopt},
        8_st);
    ActionTesting::simple_action<mock_component,
                                 gh::bbh::Actions::RecordCommonHorizonSuccess>(
        make_not_null(&runner), 0, LinkedMessageId<double>{3.0, std::nullopt},
        8_st);
    CHECK(db::get<gh::bbh::Tags::CommonHorizonSuccessCount>(box) == 2_st);
    CHECK_FALSE(db::get<gh::bbh::Tags::GaugeConstraintExceeded>(box));
    CHECK_FALSE(db::get<gh::bbh::Tags::CompletionRequested>(box));

    // Delayed older successes still don't arm the t=1.5 constraint check until
    // enough successes exist at or before that check time.
    ActionTesting::simple_action<mock_component,
                                 gh::bbh::Actions::RecordCommonHorizonSuccess>(
        make_not_null(&runner), 0, LinkedMessageId<double>{1.0, std::nullopt},
        8_st);
    CHECK_FALSE(db::get<gh::bbh::Tags::GaugeConstraintExceeded>(box));
    CHECK_FALSE(db::get<gh::bbh::Tags::CompletionRequested>(box));

    ActionTesting::simple_action<mock_component,
                                 gh::bbh::Actions::RecordCommonHorizonSuccess>(
        make_not_null(&runner), 0, LinkedMessageId<double>{1.2, std::nullopt},
        8_st);
    CHECK(db::get<gh::bbh::Tags::CommonHorizonSuccessCount>(box) == 4_st);
    CHECK(db::get<gh::bbh::Tags::GaugeConstraintExceeded>(box));
    CHECK(db::get<gh::bbh::Tags::CompletionRequested>(box));
    CHECK_FALSE(
        ActionTesting::is_simple_action_queue_empty<mock_element_component>(
            runner, 0));
    ActionTesting::invoke_queued_simple_action<mock_element_component>(
        make_not_null(&runner), 0);
    CHECK(db::get<gh::bbh::Tags::ElementCompletionRequested>(element_box));
  }

  {
    INFO("Delayed older AhC success can retroactively satisfy the LMax path");
    auto runner = make_runner();
    initialize_components(make_not_null(&runner));
    auto& box = ActionTesting::get_databox<mock_component>(runner, 0);
    auto& element_box =
        ActionTesting::get_databox<mock_element_component>(runner, 0);

    ActionTesting::simple_action<mock_component,
                                 gh::bbh::Actions::RecordCommonHorizonSuccess>(
        make_not_null(&runner), 0, LinkedMessageId<double>{2.0, std::nullopt},
        8_st);
    ActionTesting::simple_action<mock_component,
                                 gh::bbh::Actions::RecordCommonHorizonSuccess>(
        make_not_null(&runner), 0, LinkedMessageId<double>{3.0, std::nullopt},
        8_st);
    CHECK(db::get<gh::bbh::Tags::CommonHorizonSuccessCount>(box) == 2_st);
    CHECK_FALSE(
        db::get<gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold>(box));
    CHECK_FALSE(db::get<gh::bbh::Tags::CompletionRequested>(box));

    ActionTesting::simple_action<mock_component,
                                 gh::bbh::Actions::RecordCommonHorizonSuccess>(
        make_not_null(&runner), 0, LinkedMessageId<double>{1.0, std::nullopt},
        6_st);
    CHECK(db::get<gh::bbh::Tags::CommonHorizonSuccessCount>(box) == 3_st);
    CHECK(db::get<gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold>(box));
    CHECK(db::get<gh::bbh::Tags::CompletionRequested>(box));
    CHECK_FALSE(
        ActionTesting::is_simple_action_queue_empty<mock_element_component>(
            runner, 0));
    ActionTesting::invoke_queued_simple_action<mock_element_component>(
        make_not_null(&runner), 0);
    CHECK(db::get<gh::bbh::Tags::ElementCompletionRequested>(element_box));
  }

  {
    INFO("Max AhC successes can request completion");
    auto runner = make_runner(2, 3, 1.0e6, 1.0e6, 0, false);
    initialize_components(make_not_null(&runner));
    auto& box = ActionTesting::get_databox<mock_component>(runner, 0);
    auto& element_box =
        ActionTesting::get_databox<mock_element_component>(runner, 0);

    ActionTesting::simple_action<mock_component,
                                 gh::bbh::Actions::RecordCommonHorizonSuccess>(
        make_not_null(&runner), 0, LinkedMessageId<double>{1.0, std::nullopt},
        8_st);
    ActionTesting::simple_action<mock_component,
                                 gh::bbh::Actions::RecordCommonHorizonSuccess>(
        make_not_null(&runner), 0, LinkedMessageId<double>{2.0, std::nullopt},
        8_st);
    CHECK(db::get<gh::bbh::Tags::CommonHorizonSuccessCount>(box) == 2_st);
    CHECK_FALSE(db::get<gh::bbh::Tags::CompletionRequested>(box));

    ActionTesting::simple_action<mock_component,
                                 gh::bbh::Actions::RecordCommonHorizonSuccess>(
        make_not_null(&runner), 0, LinkedMessageId<double>{3.0, std::nullopt},
        8_st);
    CHECK(db::get<gh::bbh::Tags::CommonHorizonSuccessCount>(box) == 3_st);
    CHECK_FALSE(
        db::get<gh::bbh::Tags::CommonHorizonLMaxBelowOrEqualThreshold>(box));
    CHECK_FALSE(db::get<gh::bbh::Tags::GaugeConstraintExceeded>(box));
    CHECK_FALSE(db::get<gh::bbh::Tags::ThreeIndexConstraintExceeded>(box));
    CHECK(db::get<gh::bbh::Tags::CompletionRequested>(box));
    CHECK_FALSE(
        ActionTesting::is_simple_action_queue_empty<mock_element_component>(
            runner, 0));
    ActionTesting::invoke_queued_simple_action<mock_element_component>(
        make_not_null(&runner), 0);
    CHECK(db::get<gh::bbh::Tags::ElementCompletionRequested>(element_box));
  }

  {
    INFO("Three-index threshold can request completion");
    auto runner = make_runner(2, 100, 1.0e6, 20.0, 0, false);
    initialize_components(make_not_null(&runner));
    auto& box = ActionTesting::get_databox<mock_component>(runner, 0);
    auto& element_box =
        ActionTesting::get_databox<mock_element_component>(runner, 0);

    ActionTesting::simple_action<mock_component,
                                 gh::bbh::Actions::RecordCommonHorizonSuccess>(
        make_not_null(&runner), 0, LinkedMessageId<double>{1.0, std::nullopt},
        8_st);
    ActionTesting::simple_action<mock_component,
                                 gh::bbh::Actions::RecordCommonHorizonSuccess>(
        make_not_null(&runner), 0, LinkedMessageId<double>{1.2, std::nullopt},
        8_st);
    CHECK_FALSE(db::get<gh::bbh::Tags::CompletionRequested>(box));

    ActionTesting::simple_action<mock_component,
                                 gh::bbh::Actions::ProcessConstraintMaxima>(
        make_not_null(&runner), 0, 1.5, 1.0, 21.0);
    CHECK_FALSE(db::get<gh::bbh::Tags::GaugeConstraintExceeded>(box));
    CHECK(db::get<gh::bbh::Tags::ThreeIndexConstraintExceeded>(box));
    CHECK(db::get<gh::bbh::Tags::CompletionRequested>(box));
    CHECK_FALSE(
        ActionTesting::is_simple_action_queue_empty<mock_element_component>(
            runner, 0));
    ActionTesting::invoke_queued_simple_action<mock_element_component>(
        make_not_null(&runner), 0);
    CHECK(db::get<gh::bbh::Tags::ElementCompletionRequested>(element_box));
  }

  {
    INFO("Invalid min/max input relation errors in release mode");
    auto runner = make_runner(3, 2, 1.0e6, 1.0e6, 0, false);
    initialize_components(make_not_null(&runner));

    CHECK_THROWS_WITH(
        (ActionTesting::simple_action<
            mock_component, gh::bbh::Actions::RecordCommonHorizonSuccess>(
            make_not_null(&runner), 0,
            LinkedMessageId<double>{1.0, std::nullopt}, 8_st)),
        Catch::Matchers::ContainsSubstring("MaxCommonHorizonSuccesses"));
  }

  {
    INFO("Duplicate AhC records are rejected");
    auto runner = make_runner(10, 100, 1.0e6, 1.0e6, 0, false);
    initialize_components(make_not_null(&runner));

    ActionTesting::simple_action<mock_component,
                                 gh::bbh::Actions::RecordCommonHorizonSuccess>(
        make_not_null(&runner), 0, LinkedMessageId<double>{2.0, std::nullopt},
        8_st);
    CHECK_THROWS_WITH(
        (ActionTesting::simple_action<
            mock_component, gh::bbh::Actions::RecordCommonHorizonSuccess>(
            make_not_null(&runner), 0,
            LinkedMessageId<double>{2.0, std::nullopt}, 7_st)),
        Catch::Matchers::ContainsSubstring(
            "Duplicate common-horizon completion record"));
  }

  {
    INFO("Duplicate constraint records are rejected");
    auto runner = make_runner(10, 100, 1.0e6, 1.0e6, 0, false);
    initialize_components(make_not_null(&runner));

    ActionTesting::simple_action<mock_component,
                                 gh::bbh::Actions::ProcessConstraintMaxima>(
        make_not_null(&runner), 0, 2.0, 5.0, 6.0);
    CHECK_THROWS_WITH(
        (ActionTesting::simple_action<
            mock_component, gh::bbh::Actions::ProcessConstraintMaxima>(
            make_not_null(&runner), 0, 2.0, 8.0, 9.0)),
        Catch::Matchers::ContainsSubstring(
            "Duplicate BBH completion constraint-max record"));
  }

  {
    INFO("Element completion-request simple action latches true");
    auto runner = make_runner();
    initialize_components(make_not_null(&runner));
    auto& box = ActionTesting::get_databox<mock_element_component>(runner, 0);
    ActionTesting::simple_action<
        mock_element_component,
        gh::bbh::Actions::SetElementCompletionRequested>(make_not_null(&runner),
                                                         0);
    CHECK(db::get<gh::bbh::Tags::ElementCompletionRequested>(box));
  }
}
}  // namespace
