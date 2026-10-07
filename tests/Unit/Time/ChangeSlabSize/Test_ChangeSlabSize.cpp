// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Time/ChangeSlabSize/ChangeSlabSize.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/MinimumTimeStep.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/DormandPrince5.hpp"
#include "Time/TimeSteppers/Rk3HesthavenSsp.hpp"
#include "Utilities/Gsl.hpp"

namespace {
struct Vars1 : db::SimpleTag {
  using type = double;
};

struct Vars2 : db::SimpleTag {
  using type = double;
};

template <size_t Dim>
void test_with_mortar_next_temporal_id() {
  const TimeSteppers::DormandPrince5 time_stepper{};
  const Slab initial_slab(2.0, 3.0);
  const TimeStepId initial_id(true, 5, initial_slab.start());
  const TimeDelta initial_step = initial_slab.duration();
  const TimeStepId next_id =
      time_stepper.next_time_id(initial_id, initial_step);
  const AdaptiveSteppingDiagnostics diagnostics{
      .number_of_slabs = 20,
      .number_of_slab_size_changes = 30,
      .number_of_steps = 40,
      .number_of_step_fraction_changes = 50,
      .number_of_step_rejections = 60};
  TimeSteppers::History<Vars1::type> history{};
  history.insert(initial_id, 1.23, 4.56);

  const ElementId<Dim> neighbor(5);
  const DirectionalId<Dim> mortar1{Direction<Dim>::lower_xi(), neighbor};
  const DirectionalId<Dim> mortar2{Direction<Dim>::upper_xi(), neighbor};

  // In real use these should all be the same, but they could be
  // either the current ID or the next one so we test both here.
  const DirectionalIdMap<Dim, TimeStepId> mortar_next_temporal_ids{
      {mortar1, initial_id}, {mortar2, next_id}};

  auto box = db::create<
      db::AddSimpleTags<
          Tags::ConcreteTimeStepper<TimeStepper>, Tags::TimeStepId,
          Tags::TimeStep, Tags::Next<Tags::TimeStepId>,
          Tags::AdaptiveSteppingDiagnostics,
          Tags::HistoryEvolvedVariables<Vars1>, ::Tags::MinimumTimeStep,
          evolution::dg::Tags::MortarNextTemporalId<Dim>>,
      time_stepper_ref_tags<TimeStepper>>(
      static_cast<std::unique_ptr<TimeStepper>>(
          std::make_unique<TimeSteppers::DormandPrince5>(time_stepper)),
      initial_id, initial_step, next_id, diagnostics, std::move(history), 1e-8,
      mortar_next_temporal_ids);

  change_slab_size(make_not_null(&box), 5.0);

  const Slab new_slab(2.0, 5.0);
  const TimeStepId expected_id(true, 5, new_slab.start());
  const TimeDelta expected_step = new_slab.duration();
  const TimeStepId expected_next_id =
      time_stepper.next_time_id(expected_id, expected_step);
  auto expected_diagnostics = diagnostics;
  ++expected_diagnostics.number_of_slab_size_changes;
  TimeSteppers::History<Vars1::type> expected_history{};
  expected_history.insert(expected_id, 1.23, 4.56);
  const DirectionalIdMap<Dim, TimeStepId> expected_mortar_next_temporal_ids{
      {mortar1, expected_id}, {mortar2, expected_next_id}};

  CHECK(db::get<Tags::TimeStepId>(box) == expected_id);
  CHECK(db::get<Tags::TimeStep>(box) == expected_step);
  CHECK(db::get<Tags::Next<Tags::TimeStepId>>(box) == expected_next_id);
  CHECK(db::get<Tags::AdaptiveSteppingDiagnostics>(box) ==
        expected_diagnostics);
  CHECK(db::get<Tags::HistoryEvolvedVariables<Vars1>>(box) == expected_history);
  CHECK(db::get<evolution::dg::Tags::MortarNextTemporalId<Dim>>(box) ==
        expected_mortar_next_temporal_ids);
}

SPECTRE_TEST_CASE("Unit.Time.ChangeSlabSize", "[Unit][Time]") {
  // Forward in time, no substeps
  {
    const TimeSteppers::AdamsBashforth time_stepper(1);
    const Slab initial_slab(2.0, 3.0);
    const TimeStepId initial_id(true, 5, initial_slab.start());
    const TimeDelta initial_step = initial_slab.duration();
    const TimeStepId next_id =
        time_stepper.next_time_id(initial_id, initial_step);
    const AdaptiveSteppingDiagnostics diagnostics{20, 30, 40, 50, 60};
    TimeSteppers::History<Vars1::type> history1{};
    TimeSteppers::History<Vars2::type> history2{};
    history2.insert(initial_id, 1.23, 4.56);

    auto box = db::create<
        db::AddSimpleTags<
            Tags::ConcreteTimeStepper<TimeStepper>, Tags::TimeStepId,
            Tags::TimeStep, Tags::Next<Tags::TimeStepId>,
            Tags::AdaptiveSteppingDiagnostics,
            Tags::HistoryEvolvedVariables<Vars1>,
            Tags::HistoryEvolvedVariables<Vars2>, ::Tags::MinimumTimeStep>,
        time_stepper_ref_tags<TimeStepper>>(
        static_cast<std::unique_ptr<TimeStepper>>(
            std::make_unique<TimeSteppers::AdamsBashforth>(time_stepper)),
        initial_id, initial_step, next_id, diagnostics, std::move(history1),
        std::move(history2), 1e-8);

    change_slab_size(make_not_null(&box), 5.0);

    const Slab new_slab(2.0, 5.0);
    const TimeStepId expected_id(true, 5, new_slab.start());
    const TimeDelta expected_step = new_slab.duration();
    const TimeStepId expected_next_id =
        time_stepper.next_time_id(expected_id, expected_step);
    auto expected_diagnostics = diagnostics;
    ++expected_diagnostics.number_of_slab_size_changes;
    TimeSteppers::History<Vars1::type> expected_history1{};
    TimeSteppers::History<Vars2::type> expected_history2{};
    expected_history2.insert(expected_id, 1.23, 4.56);

    CHECK(db::get<Tags::TimeStepId>(box) == expected_id);
    CHECK(db::get<Tags::TimeStep>(box) == expected_step);
    CHECK(db::get<Tags::Next<Tags::TimeStepId>>(box) == expected_next_id);
    CHECK(db::get<Tags::AdaptiveSteppingDiagnostics>(box) ==
          expected_diagnostics);
    CHECK(db::get<Tags::HistoryEvolvedVariables<Vars1>>(box) ==
          expected_history1);
    CHECK(db::get<Tags::HistoryEvolvedVariables<Vars2>>(box) ==
          expected_history2);
  }

  // Backward in time, substep method
  {
    const TimeSteppers::Rk3HesthavenSsp time_stepper{};
    const Slab initial_slab(2.0, 3.0);
    const TimeStepId initial_id(false, 5, initial_slab.end());
    const TimeDelta initial_step = -initial_slab.duration();
    const TimeStepId next_id =
        time_stepper.next_time_id(initial_id, initial_step);
    const AdaptiveSteppingDiagnostics diagnostics{20, 30, 40, 50, 60};
    TimeSteppers::History<Vars1::type> history1{};
    TimeSteppers::History<Vars2::type> history2{};
    history2.insert(initial_id, 1.23, 4.56);

    auto box = db::create<
        db::AddSimpleTags<
            Tags::ConcreteTimeStepper<TimeStepper>, Tags::TimeStepId,
            Tags::TimeStep, Tags::Next<Tags::TimeStepId>,
            Tags::AdaptiveSteppingDiagnostics,
            Tags::HistoryEvolvedVariables<Vars1>,
            Tags::HistoryEvolvedVariables<Vars2>, ::Tags::MinimumTimeStep>,
        time_stepper_ref_tags<TimeStepper>>(
        static_cast<std::unique_ptr<TimeStepper>>(
            std::make_unique<TimeSteppers::Rk3HesthavenSsp>(time_stepper)),
        initial_id, initial_step, next_id, diagnostics, std::move(history1),
        std::move(history2), 1e-8);

    change_slab_size(make_not_null(&box), -1.0);

    const Slab new_slab(-1.0, 3.0);
    const TimeStepId expected_id(false, 5, new_slab.end());
    const TimeDelta expected_step = -new_slab.duration();
    const TimeStepId expected_next_id =
        time_stepper.next_time_id(expected_id, expected_step);
    auto expected_diagnostics = diagnostics;
    ++expected_diagnostics.number_of_slab_size_changes;
    TimeSteppers::History<Vars1::type> expected_history1{};
    TimeSteppers::History<Vars2::type> expected_history2{};
    expected_history2.insert(expected_id, 1.23, 4.56);

    CHECK(db::get<Tags::TimeStepId>(box) == expected_id);
    CHECK(db::get<Tags::TimeStep>(box) == expected_step);
    CHECK(db::get<Tags::Next<Tags::TimeStepId>>(box) == expected_next_id);
    CHECK(db::get<Tags::AdaptiveSteppingDiagnostics>(box) ==
          expected_diagnostics);
    CHECK(db::get<Tags::HistoryEvolvedVariables<Vars1>>(box) ==
          expected_history1);
    CHECK(db::get<Tags::HistoryEvolvedVariables<Vars2>>(box) ==
          expected_history2);
  }

  // No change
  {
    const TimeSteppers::AdamsBashforth time_stepper(1);
    const Slab initial_slab(2.0, 3.0);
    const TimeStepId initial_id(true, 5, initial_slab.start());
    const TimeDelta initial_step = initial_slab.duration();
    const TimeStepId next_id =
        time_stepper.next_time_id(initial_id, initial_step);
    const AdaptiveSteppingDiagnostics diagnostics{20, 30, 40, 50, 60};
    TimeSteppers::History<Vars1::type> history1{};
    TimeSteppers::History<Vars2::type> history2{};
    history2.insert(initial_id, 1.23, 4.56);

    auto box = db::create<
        db::AddSimpleTags<
            Tags::ConcreteTimeStepper<TimeStepper>, Tags::TimeStepId,
            Tags::TimeStep, Tags::Next<Tags::TimeStepId>,
            Tags::AdaptiveSteppingDiagnostics,
            Tags::HistoryEvolvedVariables<Vars1>,
            Tags::HistoryEvolvedVariables<Vars2>, ::Tags::MinimumTimeStep>,
        time_stepper_ref_tags<TimeStepper>>(
        static_cast<std::unique_ptr<TimeStepper>>(
            std::make_unique<TimeSteppers::AdamsBashforth>(time_stepper)),
        initial_id, initial_step, next_id, diagnostics, history1, history2,
        1e-8);

    change_slab_size(make_not_null(&box), 3.0);

    CHECK(db::get<Tags::TimeStepId>(box) == initial_id);
    CHECK(db::get<Tags::TimeStep>(box) == initial_step);
    CHECK(db::get<Tags::Next<Tags::TimeStepId>>(box) == next_id);
    CHECK(db::get<Tags::AdaptiveSteppingDiagnostics>(box) == diagnostics);
    CHECK(db::get<Tags::HistoryEvolvedVariables<Vars1>>(box) == history1);
    CHECK(db::get<Tags::HistoryEvolvedVariables<Vars2>>(box) == history2);
  }

  test_with_mortar_next_temporal_id<1>();
  test_with_mortar_next_temporal_id<2>();
  test_with_mortar_next_temporal_id<3>();
}
}  // namespace
