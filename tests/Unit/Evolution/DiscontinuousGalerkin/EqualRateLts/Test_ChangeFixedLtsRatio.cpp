// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/ChangeFixedLtsRatio.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/Tags/ChangeFixedLtsRatioTags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/FixedLtsRatio.hpp"
#include "Time/Tags/StepNumberWithinSlab.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Component;

struct Metavariables {
  using component_list = tmpl::list<Component>;
};

struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;

  using simple_tags = tmpl::list<
      ::Tags::FixedLtsRatio, ::Tags::TimeStepId, ::Tags::StepNumberWithinSlab,
      evolution::dg::Tags::ChangeFixedLtsRatio::NumberOfExpectedMessages,
      evolution::dg::Tags::ChangeFixedLtsRatio::NewStepSize>;
  using compute_tags = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, compute_tags>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<evolution::dg::Actions::ChangeFixedLtsRatio>>>;
};

using StepId = std::pair<int64_t, uint64_t>;

const Slab slab(1.2, 7.5);
const double slab_size = slab.duration().value();

using RunResult = std::pair<bool, std::optional<size_t>>;
// Returns (was_ready, post_run_fixed_ratio)
RunResult run_action(
    const std::optional<size_t>& original_fixed_ratio,
    const StepId& current_step,
    const std::map<StepId, size_t>& number_of_expected_messages,
    const std::map<StepId, std::vector<double>>& messages) {
  const TimeStepId time_step_id_step(true, current_step.first,
                                     slab.start() + slab.duration() / 2);
  const TimeStepId time_step_id_substep(
      true, current_step.first, slab.start() + slab.duration() / 2, 1,
      slab.duration() / 4, slab.start().value() + slab_size * 5.0 / 8.0);

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};

  ActionTesting::emplace_component_and_initialize<Component>(
      &runner, 0,
      {original_fixed_ratio, time_step_id_substep, current_step.second,
       number_of_expected_messages, messages});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  auto& box = ActionTesting::get_databox<Component>(make_not_null(&runner), 0);

  // First run on a substep.  Should never do anything.
  REQUIRE(ActionTesting::next_action_if_ready<Component>(make_not_null(&runner),
                                                         0));
  CHECK(original_fixed_ratio == db::get<::Tags::FixedLtsRatio>(box));
  CHECK(number_of_expected_messages ==
        db::get<
            evolution::dg::Tags::ChangeFixedLtsRatio::NumberOfExpectedMessages>(
            box));
  CHECK(messages ==
        db::get<evolution::dg::Tags::ChangeFixedLtsRatio::NewStepSize>(box));

  db::mutate<::Tags::TimeStepId>(
      [&](const gsl::not_null<TimeStepId*> id) { *id = time_step_id_step; },
      make_not_null(&box));

  // Now run on a full step for the actual test.
  const bool was_ready =
      ActionTesting::next_action_if_ready<Component>(make_not_null(&runner), 0);

  if (was_ready) {
    const auto check_cleaned = [&current_step](const auto& orig_map,
                                               const auto& new_map) {
      size_t expected_size = 0;
      for (const auto& [id, expected] : orig_map) {
        if (id > current_step) {
          CHECK(new_map.at(id) == expected);
          ++expected_size;
        }
      }
      CHECK(new_map.size() == expected_size);
    };

    check_cleaned(
        number_of_expected_messages,
        db::get<
            evolution::dg::Tags::ChangeFixedLtsRatio::NumberOfExpectedMessages>(
            box));
    check_cleaned(
        messages,
        db::get<evolution::dg::Tags::ChangeFixedLtsRatio::NewStepSize>(box));
  } else {
    CHECK(
        number_of_expected_messages ==
        db::get<
            evolution::dg::Tags::ChangeFixedLtsRatio::NumberOfExpectedMessages>(
            box));
    CHECK(messages ==
          db::get<evolution::dg::Tags::ChangeFixedLtsRatio::NewStepSize>(box));
  }

  return {was_ready, db::get<::Tags::FixedLtsRatio>(box)};
}

SPECTRE_TEST_CASE("Unit.Evolution.DG.EqualRateLts.ChangeFixedLtsRatio",
                  "[Unit][Evolution]") {
  // Non-fixed element does nothing
  CHECK(run_action(std::nullopt, {3, 4}, {}, {}) ==
        RunResult(true, std::nullopt));

  // No change actions does nothing
  CHECK(run_action(std::optional{8_st}, {3, 4}, {}, {}) ==
        RunResult(true, std::optional{8_st}));

  // All messages after current time does nothing
  CHECK(run_action(std::optional{8_st}, {3, 4}, {{{3, 5}, 1}}, {}) ==
        RunResult(true, std::optional{8_st}));
  CHECK(run_action(std::optional{8_st}, {3, 4}, {{{4, 0}, 1}}, {}) ==
        RunResult(true, std::optional{8_st}));

  // Normal case processing messages for current time
  CHECK(run_action(std::optional{8_st}, {3, 4}, {{{3, 4}, 1}}, {}) ==
        RunResult(false, std::optional{8_st}));
  CHECK(run_action(std::optional{8_st}, {3, 4}, {{{3, 4}, 1}},
                   {{{3, 4}, {slab_size / 4.0}}}) ==
        RunResult(true, std::optional{4_st}));
  CHECK(run_action(std::optional{8_st}, {3, 4}, {{{3, 4}, 1}},
                   {{{3, 4}, {slab_size / 3.9}}}) ==
        RunResult(true, std::optional{4_st}));
  CHECK(run_action(std::optional{8_st}, {3, 4}, {{{3, 4}, 1}},
                   {{{3, 4}, {slab_size / 4.1}}}) ==
        RunResult(true, std::optional{8_st}));
  CHECK(run_action(std::optional{8_st}, {3, 4}, {{{3, 4}, 2}}, {}) ==
        RunResult(false, std::optional{8_st}));
  CHECK(run_action(std::optional{8_st}, {3, 4}, {{{3, 4}, 2}},
                   {{{3, 4}, {slab_size / 4.0}}}) ==
        RunResult(false, std::optional{8_st}));
  CHECK(run_action(std::optional{8_st}, {3, 4}, {{{3, 4}, 2}},
                   {{{3, 4}, {slab_size / 4.0, slab_size / 2.0}}}) ==
        RunResult(true, std::optional{4_st}));
  CHECK(run_action(std::optional{8_st}, {3, 4}, {{{3, 4}, 2}},
                   {{{3, 4}, {slab_size / 2.0, slab_size / 4.0}}}) ==
        RunResult(true, std::optional{4_st}));
  // Future messages should be ignored.
  CHECK(run_action(std::optional{8_st}, {3, 4}, {{{3, 4}, 1}, {{3, 5}, 1}},
                   {}) == RunResult(false, std::optional{8_st}));
  CHECK(run_action(std::optional{8_st}, {3, 4}, {{{3, 4}, 1}, {{3, 5}, 1}},
                   {{{3, 4}, {slab_size / 4.0}}}) ==
        RunResult(true, std::optional{4_st}));
  CHECK(run_action(std::optional{8_st}, {3, 4}, {{{3, 4}, 1}, {{3, 5}, 1}},
                   {{{3, 5}, {slab_size / 8.0}}}) ==
        RunResult(false, std::optional{8_st}));
  CHECK(
      run_action(std::optional{8_st}, {3, 4}, {{{3, 4}, 1}, {{3, 5}, 1}},
                 {{{3, 4}, {slab_size / 4.0}}, {{3, 5}, {slab_size / 16.0}}}) ==
      RunResult(true, std::optional{4_st}));

  // Overrun of step number at a slab boundary
  CHECK(run_action(std::optional{8_st}, {4, 0}, {{{3, 4}, 1}}, {}) ==
        RunResult(false, std::optional{8_st}));
  CHECK(run_action(std::optional{8_st}, {4, 0}, {{{3, 4}, 1}},
                   {{{3, 4}, {slab_size / 4.0}}}) ==
        RunResult(true, std::optional{4_st}));
  CHECK(run_action(std::optional{8_st}, {4, 0}, {{{3, 4}, 1}, {{4, 0}, 1}},
                   {}) == RunResult(false, std::optional{8_st}));
  CHECK(run_action(std::optional{8_st}, {4, 0}, {{{3, 4}, 1}, {{4, 0}, 1}},
                   {{{3, 4}, {slab_size / 4.0}}}) ==
        RunResult(false, std::optional{8_st}));
  CHECK(
      run_action(std::optional{8_st}, {4, 0}, {{{3, 4}, 1}, {{4, 0}, 1}},
                 {{{3, 4}, {slab_size / 4.0}}, {{4, 0}, {slab_size / 2.0}}}) ==
      RunResult(true, std::optional{4_st}));
  CHECK(
      run_action(std::optional{8_st}, {4, 0}, {{{3, 4}, 1}, {{4, 0}, 1}},
                 {{{3, 4}, {slab_size / 2.0}}, {{4, 0}, {slab_size / 4.0}}}) ==
      RunResult(true, std::optional{4_st}));

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(run_action(std::nullopt, {3, 4}, {{{3, 4}, 1}},
                               {{{3, 4}, {slab_size / 4.0}}}),
                    Catch::Matchers::ContainsSubstring(
                        "Attempting to adjust FixedLtsRatio when not set"));
  CHECK_THROWS_WITH(
      run_action(std::optional{8_st}, {3, 4}, {{{3, 4}, 0}}, {}),
      Catch::Matchers::ContainsSubstring(
          "Should only create map entries when sending messages"));
  CHECK_THROWS_WITH(run_action(std::optional{8_st}, {3, 4}, {{{3, 4}, 1}},
                               {{{3, 4}, {slab_size / 4.0, slab_size / 2.0}}}),
                    Catch::Matchers::ContainsSubstring(
                        "Received too many messages at step (3, 4)"));
  CHECK_THROWS_WITH(
      run_action(std::optional{8_st}, {3, 4}, {{{3, 4}, 1}},
                 {{{3, 3}, {slab_size / 2.0}}, {{3, 4}, {slab_size / 4.0}}}),
      Catch::Matchers::ContainsSubstring(
          "Received unexpected change for step (3, 3)"));
#endif
}
}  // namespace
