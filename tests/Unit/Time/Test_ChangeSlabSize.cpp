// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Time/ChangeSlabSize.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/Rk3HesthavenSsp.hpp"
#include "Utilities/Gsl.hpp"

namespace {
struct Vars1 : db::SimpleTag {
  using type = double;
};

struct Vars2 : db::SimpleTag {
  using type = double;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.ChangeSlabSize", "[Unit][Time]") {
  // Forward in time, no substeps
  {
    const TimeSteppers::AdamsBashforth time_stepper(1);
    const Slab initial_slab(2.0, 3.0);
    const TimeStepId initial_id(true, 5, initial_slab.start());
    const TimeDelta initial_step = initial_slab.duration();
    const TimeStepId next_id =
        time_stepper.next_time_id(initial_id, initial_step);
    const TimeDelta next_step{};  // overwritten by function
    const AdaptiveSteppingDiagnostics diagnostics{20, 30, 40, 50, 60};
    TimeSteppers::History<Vars1::type> history1{};
    TimeSteppers::History<Vars2::type> history2{};
    history2.insert(initial_id, 1.23, 4.56);

    auto box = db::create<db::AddSimpleTags<
        Tags::TimeStepper<TimeSteppers::AdamsBashforth>, Tags::TimeStepId,
        Tags::TimeStep, Tags::Next<Tags::TimeStepId>,
        Tags::Next<Tags::TimeStep>, Tags::AdaptiveSteppingDiagnostics,
        Tags::HistoryEvolvedVariables<Vars1>,
        Tags::HistoryEvolvedVariables<Vars2>>>(
        std::make_unique<TimeSteppers::AdamsBashforth>(time_stepper),
        initial_id, initial_step, next_id, next_step, diagnostics,
        std::move(history1), std::move(history2));

    change_slab_size(make_not_null(&box), 5.0);

    const Slab new_slab(2.0, 5.0);
    const TimeStepId expected_id(true, 5, new_slab.start());
    const TimeDelta expected_step = new_slab.duration();
    const TimeStepId expected_next_id =
        time_stepper.next_time_id(expected_id, expected_step);
    const TimeDelta expected_next_step = expected_step;
    auto expected_diagnostics = diagnostics;
    ++expected_diagnostics.number_of_slab_size_changes;
    TimeSteppers::History<Vars1::type> expected_history1{};
    TimeSteppers::History<Vars2::type> expected_history2{};
    expected_history2.insert(expected_id, 1.23, 4.56);

    CHECK(db::get<Tags::TimeStepId>(box) == expected_id);
    CHECK(db::get<Tags::TimeStep>(box) == expected_step);
    CHECK(db::get<Tags::Next<Tags::TimeStepId>>(box) == expected_next_id);
    CHECK(db::get<Tags::Next<Tags::TimeStep>>(box) == expected_next_step);
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
    const TimeDelta next_step{};  // overwritten by function
    const AdaptiveSteppingDiagnostics diagnostics{20, 30, 40, 50, 60};
    TimeSteppers::History<Vars1::type> history1{};
    TimeSteppers::History<Vars2::type> history2{};
    history2.insert(initial_id, 1.23, 4.56);

    auto box = db::create<db::AddSimpleTags<
        Tags::TimeStepper<TimeSteppers::Rk3HesthavenSsp>, Tags::TimeStepId,
        Tags::TimeStep, Tags::Next<Tags::TimeStepId>,
        Tags::Next<Tags::TimeStep>, Tags::AdaptiveSteppingDiagnostics,
        Tags::HistoryEvolvedVariables<Vars1>,
        Tags::HistoryEvolvedVariables<Vars2>>>(
        std::make_unique<TimeSteppers::Rk3HesthavenSsp>(time_stepper),
        initial_id, initial_step, next_id, next_step, diagnostics,
        std::move(history1), std::move(history2));

    change_slab_size(make_not_null(&box), -1.0);

    const Slab new_slab(-1.0, 3.0);
    const TimeStepId expected_id(false, 5, new_slab.end());
    const TimeDelta expected_step = -new_slab.duration();
    const TimeStepId expected_next_id =
        time_stepper.next_time_id(expected_id, expected_step);
    const TimeDelta expected_next_step = expected_step;
    auto expected_diagnostics = diagnostics;
    ++expected_diagnostics.number_of_slab_size_changes;
    TimeSteppers::History<Vars1::type> expected_history1{};
    TimeSteppers::History<Vars2::type> expected_history2{};
    expected_history2.insert(expected_id, 1.23, 4.56);

    CHECK(db::get<Tags::TimeStepId>(box) == expected_id);
    CHECK(db::get<Tags::TimeStep>(box) == expected_step);
    CHECK(db::get<Tags::Next<Tags::TimeStepId>>(box) == expected_next_id);
    CHECK(db::get<Tags::Next<Tags::TimeStep>>(box) == expected_next_step);
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
    const TimeDelta next_step = initial_step / 2;
    const AdaptiveSteppingDiagnostics diagnostics{20, 30, 40, 50, 60};
    TimeSteppers::History<Vars1::type> history1{};
    TimeSteppers::History<Vars2::type> history2{};
    history2.insert(initial_id, 1.23, 4.56);

    auto box = db::create<db::AddSimpleTags<
        Tags::TimeStepper<TimeSteppers::AdamsBashforth>, Tags::TimeStepId,
        Tags::TimeStep, Tags::Next<Tags::TimeStepId>,
        Tags::Next<Tags::TimeStep>, Tags::AdaptiveSteppingDiagnostics,
        Tags::HistoryEvolvedVariables<Vars1>,
        Tags::HistoryEvolvedVariables<Vars2>>>(
        std::make_unique<TimeSteppers::AdamsBashforth>(time_stepper),
        initial_id, initial_step, next_id, next_step, diagnostics, history1,
        history2);

    change_slab_size(make_not_null(&box), 3.0);

    CHECK(db::get<Tags::TimeStepId>(box) == initial_id);
    CHECK(db::get<Tags::TimeStep>(box) == initial_step);
    CHECK(db::get<Tags::Next<Tags::TimeStepId>>(box) == next_id);
    CHECK(db::get<Tags::Next<Tags::TimeStep>>(box) == next_step);
    CHECK(db::get<Tags::AdaptiveSteppingDiagnostics>(box) == diagnostics);
    CHECK(db::get<Tags::HistoryEvolvedVariables<Vars1>>(box) == history1);
    CHECK(db::get<Tags::HistoryEvolvedVariables<Vars2>>(box) == history2);
  }
}
