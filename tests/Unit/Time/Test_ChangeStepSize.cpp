// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Time/ChangeStepSize.hpp"
#include "Time/History.hpp"
#include "Time/LtsMode.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Tags/FixedLtsRatio.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/LtsMode.hpp"
#include "Time/Tags/LtsStepChoosers.hpp"
#include "Time/Tags/MinimumTimeStep.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/Rk3HesthavenSsp.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Var : db::SimpleTag {
  using type = double;
};

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                             tmpl::list<StepChoosers::Constant>>>;
  };
};

template <typename StepChoosersToUse = AllStepChoosers>
void check(const bool time_runs_forward,
           std::unique_ptr<LtsTimeStepper> time_stepper,
           TimeSteppers::History<double> history, const Time& time,
           const double request, const TimeDelta& expected_step) {
  CAPTURE(time);
  CAPTURE(request);

  const TimeDelta initial_step_size = (time_runs_forward ? 1 : -1) *
                                      time.slab().duration() /
                                      time.fraction().denominator();

  auto choosers =
      make_vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>(
          std::make_unique<StepChoosers::Constant>(2. * request),
          std::make_unique<StepChoosers::Constant>(request),
          std::make_unique<StepChoosers::Constant>(2. * request));

  auto box = db::create<
      db::AddSimpleTags<
          Parallel::Tags::MetavariablesImpl<Metavariables>, Tags::LtsMode,
          Tags::ConcreteTimeStepper<LtsTimeStepper>, Tags::MinimumTimeStep,
          Tags::TimeStepId, Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
          Tags::LtsStepChoosers, Tags::HistoryEvolvedVariables<Var>,
          Tags::AdaptiveSteppingDiagnostics>,
      db::AddComputeTags<time_stepper_ref_tags<LtsTimeStepper>>>(
      Metavariables{}, LtsMode::Conservative, std::move(time_stepper), 1e-8,
      TimeStepId(time_runs_forward, 0, time, 1, initial_step_size,
                 time.value()),
      TimeStepId(time_runs_forward, 0, time + initial_step_size),
      initial_step_size, std::move(choosers), std::move(history),
      AdaptiveSteppingDiagnostics{1, 2, 3, 4, 5});

  // Nothing should happen on a substep
  db::mutate_apply<ChangeStepSize<StepChoosersToUse>>(make_not_null(&box));
  CHECK(db::get<Tags::TimeStep>(box) == initial_step_size);
  CHECK(db::get<Tags::AdaptiveSteppingDiagnostics>(box) ==
        AdaptiveSteppingDiagnostics{1, 2, 3, 4, 5});

  db::mutate<Tags::TimeStepId>(
      [&](const gsl::not_null<TimeStepId*> id) {
        *id = TimeStepId(time_runs_forward, 0, time);
      },
      make_not_null(&box));

  db::mutate_apply<ChangeStepSize<StepChoosersToUse>>(make_not_null(&box));

  CHECK(db::get<Tags::TimeStep>(box) == expected_step);
  CHECK(db::get<Tags::AdaptiveSteppingDiagnostics>(box) ==
        AdaptiveSteppingDiagnostics{
            1, 2, 3,
            db::get<Tags::TimeStep>(box) == initial_step_size ? 4_st : 5_st,
            5});
}

void test_fixed_lts_ratio() {
  std::unique_ptr<LtsTimeStepper> time_stepper =
      std::make_unique<TimeSteppers::AdamsBashforth>(3);
  const Slab slab(2.3, 4.5);
  const TimeStepId initial_id(true, 0, slab.start());
  const auto initial_step = slab.duration() / 4;
  const auto next_id = time_stepper->next_time_id(initial_id, initial_step);
  TimeSteppers::History<double> history(3);
  history.insert(TimeStepId(true, -1, slab.start() + initial_step), 0.0, 0.0);
  history.insert(TimeStepId(true, -1, slab.start() + 2 * initial_step), 0.0,
                 0.0);
  history.insert(initial_id, 0.0, 0.0);

  auto box = db::create<
      db::AddSimpleTags<
          Parallel::Tags::MetavariablesImpl<Metavariables>, Tags::LtsMode,
          Tags::ConcreteTimeStepper<LtsTimeStepper>, Tags::LtsStepChoosers,
          Tags::MinimumTimeStep, Tags::FixedLtsRatio, Tags::TimeStepId,
          Tags::TimeStep, Tags::Next<Tags::TimeStepId>,
          Tags::HistoryEvolvedVariables<Var>,
          Tags::AdaptiveSteppingDiagnostics>,
      db::AddComputeTags<time_stepper_ref_tags<LtsTimeStepper>>>(
      Metavariables{}, LtsMode::Conservative, std::move(time_stepper),
      Tags::LtsStepChoosers::type{}, 1e-10, std::optional<size_t>(8),
      initial_id, initial_step, next_id, std::move(history),
      AdaptiveSteppingDiagnostics{1, 2, 3, 4, 5});

  db::mutate_apply<ChangeStepSize<>>(make_not_null(&box));
  // Step size change forbidden after self-start
  CHECK(db::get<Tags::TimeStep>(box) == initial_step);
  CHECK(db::get<Tags::AdaptiveSteppingDiagnostics>(box) ==
        AdaptiveSteppingDiagnostics{1, 2, 3, 4, 5});

  db::mutate<Tags::HistoryEvolvedVariables<Var>>(
      [&](const gsl::not_null<TimeSteppers::History<double>*> local_history) {
        const auto old_step = initial_step.with_slab(slab.retreat());
        local_history->clear();
        local_history->insert(TimeStepId(true, -1, slab.start() - 2 * old_step),
                              0.0, 0.0);
        local_history->insert(TimeStepId(true, -1, slab.start() - old_step),
                              0.0, 0.0);
        local_history->insert(initial_id, 0.0, 0.0);
      },
      make_not_null(&box));

  db::mutate_apply<ChangeStepSize<>>(make_not_null(&box));
  CHECK(db::get<Tags::TimeStep>(box).fraction() == Rational(1, 8));
  CHECK(db::get<Tags::AdaptiveSteppingDiagnostics>(box) ==
        AdaptiveSteppingDiagnostics{1, 2, 3, 5, 5});
}

void test_gts() {
  // NOLINTNEXTLINE(misc-const-correctness)
  std::unique_ptr<TimeStepper> time_stepper =
      std::make_unique<TimeSteppers::Rk3HesthavenSsp>();
  REQUIRE(dynamic_cast<const LtsTimeStepper*>(time_stepper.get()) == nullptr);

  // Everything should be ignored with LTS off, so set some nonsense values.
  const TimeStepId initial_id(false, 0, Slab(2.3, 4.5).start());
  const auto initial_step = Slab(1.7, 9.5).duration() / 4;
  const TimeStepId next_id(true, 3, Slab(7.8, 9.0).start());
  // NOLINTNEXTLINE(misc-const-correctness)
  TimeSteppers::History<double> history(6);

  auto choosers =
      make_vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>(
          std::make_unique<StepChoosers::Constant>(1e-3));

  auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>,
                        Tags::LtsMode, Tags::ConcreteTimeStepper<TimeStepper>,
                        Tags::LtsStepChoosers, Tags::MinimumTimeStep,
                        Tags::FixedLtsRatio, Tags::TimeStepId, Tags::TimeStep,
                        Tags::Next<Tags::TimeStepId>,
                        Tags::HistoryEvolvedVariables<Var>,
                        Tags::AdaptiveSteppingDiagnostics>,
      db::AddComputeTags<time_stepper_ref_tags<TimeStepper>>>(
      Metavariables{}, LtsMode::Off, std::move(time_stepper),
      std::move(choosers), 300.0, std::optional<size_t>(8), initial_id,
      initial_step, next_id, std::move(history),
      AdaptiveSteppingDiagnostics{1, 2, 3, 4, 5});

  db::mutate_apply<ChangeStepSize<>>(make_not_null(&box));

  CHECK(db::get<Tags::TimeStepId>(box) == initial_id);
  CHECK(db::get<Tags::TimeStep>(box) == initial_step);
  CHECK(db::get<Tags::Next<Tags::TimeStepId>>(box) == next_id);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.ChangeStepSize", "[Unit][Time][Actions]") {
  register_classes_with_charm<TimeSteppers::AdamsBashforth>();
  register_factory_classes_with_charm<Metavariables>();
  const Slab slab(-5., -2.);
  const double slab_length = slab.duration().value();
  check(true, std::make_unique<TimeSteppers::AdamsBashforth>(1), {},
        slab.start() + slab.duration() / 4, slab_length / 5.,
        slab.duration() / 8);
  check(true, std::make_unique<TimeSteppers::AdamsBashforth>(1), {},
        slab.start() + slab.duration() / 4, slab_length, slab.duration() / 4);
  check(false, std::make_unique<TimeSteppers::AdamsBashforth>(1), {},
        slab.end() - slab.duration() / 4, slab_length / 5.,
        -slab.duration() / 8);
  check(false, std::make_unique<TimeSteppers::AdamsBashforth>(1), {},
        slab.end() - slab.duration() / 4, slab_length, -slab.duration() / 4);

  // Check for roundoff issues
  check(true, std::make_unique<TimeSteppers::AdamsBashforth>(1), {},
        slab.start() + slab.duration() / 4,
        slab_length / 16. / (1.0 + std::numeric_limits<double>::epsilon()),
        slab.duration() / 32);
  check(false, std::make_unique<TimeSteppers::AdamsBashforth>(1), {},
        slab.end() - slab.duration() / 4,
        slab_length / 16. / (1.0 + std::numeric_limits<double>::epsilon()),
        -slab.duration() / 32);

  {
    // History out of order, as if just after self-start.
    TimeSteppers::History<double> history{};
    history.insert(TimeStepId(true, -1, slab.start() + slab.duration() / 8),
                   0.0, 0.0);
    history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
    check(true, std::make_unique<TimeSteppers::AdamsBashforth>(3),
          std::move(history), slab.start(), 1.0e-3, slab.duration());
  }

  CHECK_THROWS_WITH(
      check(true, std::make_unique<TimeSteppers::AdamsBashforth>(1), {},
            slab.start() + slab.duration() / 4, 1e-9, slab.duration() / 4),
      Catch::Matchers::ContainsSubstring("smaller than the MinimumTimeStep"));

  test_fixed_lts_ratio();
  test_gts();
}
