// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "Time/ChangeStepSize.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/FixedLtsRatio.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/MinimumTimeStep.hpp"
#include "Time/Tags/StepChoosers.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// a silly step chooser that just always rejects, to test the step rejection
// control-flow.
struct StepRejector : public StepChooser<StepChooserUse::LtsStep> {
  using argument_tags = tmpl::list<>;
  using compute_tags = tmpl::list<>;
  using PUP::able::register_constructor;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(StepRejector);  // NOLINT
#pragma GCC diagnostic pop
  explicit StepRejector(CkMigrateMessage* /*unused*/) {}
  StepRejector() = default;
  explicit StepRejector(const double decrease) : decrease_(decrease) {}

  std::pair<TimeStepRequest, bool> operator()(const double last_step) const {
    return {{.size_goal = last_step * decrease_}, false};
  }

  bool uses_local_data() const override { return false; }
  bool can_be_delayed() const override { return true; }

  void pup(PUP::er& p) override { p | decrease_; }

 private:
  double decrease_ = 1.0;
};

PUP::able::PUP_ID StepRejector::my_PUP_ID = 0;

struct Var : db::SimpleTag {
  using type = double;
};

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                             tmpl::list<StepChoosers::Constant, StepRejector>>>;
  };
};

template <typename StepChoosersToUse = AllStepChoosers>
void check(const bool time_runs_forward,
           std::unique_ptr<LtsTimeStepper> time_stepper,
           TimeSteppers::History<double> history, const Time& time,
           const double request, const TimeDelta& expected_step,
           std::optional<std::unique_ptr<StepRejector>> rejector) {
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
  if (rejector.has_value()) {
    choosers.emplace_back(std::move(*rejector));
  }

  auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>,
                        Tags::ConcreteTimeStepper<LtsTimeStepper>,
                        Tags::MinimumTimeStep, Tags::TimeStepId,
                        Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
                        Tags::Next<Tags::TimeStep>, Tags::StepChoosers,
                        Tags::HistoryEvolvedVariables<Var>>,
      db::AddComputeTags<time_stepper_ref_tags<LtsTimeStepper>>>(
      Metavariables{}, std::move(time_stepper), 1e-8,
      TimeStepId(time_runs_forward, 0,
                 time_runs_forward ? time.slab().start() : time.slab().end()),
      TimeStepId(time_runs_forward, 0, time), initial_step_size,
      initial_step_size, std::move(choosers), std::move(history));

  const bool accepted =
      change_step_size<StepChoosersToUse>(make_not_null(&box));

  CHECK(accepted != rejector.has_value());
  CHECK(db::get<Tags::Next<Tags::TimeStep>>(box) == expected_step);
}

struct FixedRatioMetavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                             tmpl::list<StepChoosers::Constant>>>;
  };
};

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
          Parallel::Tags::MetavariablesImpl<FixedRatioMetavariables>,
          Tags::ConcreteTimeStepper<LtsTimeStepper>, Tags::StepChoosers,
          Tags::MinimumTimeStep, Tags::FixedLtsRatio, Tags::TimeStepId,
          Tags::TimeStep, Tags::Next<Tags::TimeStepId>,
          Tags::Next<Tags::TimeStep>, Tags::HistoryEvolvedVariables<Var>>,
      db::AddComputeTags<time_stepper_ref_tags<LtsTimeStepper>>>(
      FixedRatioMetavariables{}, std::move(time_stepper),
      Tags::StepChoosers::type{}, 1e-10, std::optional<size_t>(8), initial_id,
      initial_step, next_id, initial_step, std::move(history));

  change_step_size(make_not_null(&box));
  // Step size change forbidden after self-start
  CHECK(db::get<Tags::Next<Tags::TimeStep>>(box) == initial_step);

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

  change_step_size(make_not_null(&box));
  CHECK(db::get<Tags::Next<Tags::TimeStep>>(box).fraction() == Rational(1, 8));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.ChangeStepSize", "[Unit][Time][Actions]") {
  register_classes_with_charm<TimeSteppers::AdamsBashforth>();
  register_factory_classes_with_charm<Metavariables>();
  const Slab slab(-5., -2.);
  const double slab_length = slab.duration().value();
  for (auto reject_step : {true, false}) {
    check(true, std::make_unique<TimeSteppers::AdamsBashforth>(1), {},
          slab.start() + slab.duration() / 4, slab_length / 5.,
          slab.duration() / 8,
          reject_step ? std::optional{std::make_unique<StepRejector>(0.5)}
                      : std::nullopt);
    check(true, std::make_unique<TimeSteppers::AdamsBashforth>(1), {},
          slab.start() + slab.duration() / 4, slab_length,
          reject_step ? slab.duration() / 8 : slab.duration() / 4,
          reject_step ? std::optional{std::make_unique<StepRejector>(0.5)}
                      : std::nullopt);
    check(false, std::make_unique<TimeSteppers::AdamsBashforth>(1), {},
          slab.end() - slab.duration() / 4, slab_length / 5.,
          -slab.duration() / 8,
          reject_step ? std::optional{std::make_unique<StepRejector>(0.5)}
                      : std::nullopt);
    check(false, std::make_unique<TimeSteppers::AdamsBashforth>(1), {},
          slab.end() - slab.duration() / 4, slab_length,
          reject_step ? -slab.duration() / 8 : -slab.duration() / 4,
          reject_step ? std::optional{std::make_unique<StepRejector>(0.5)}
                      : std::nullopt);

    // Check for roundoff issues
    check(true, std::make_unique<TimeSteppers::AdamsBashforth>(1), {},
          slab.start() + slab.duration() / 4,
          slab_length / 16. / (1.0 + std::numeric_limits<double>::epsilon()),
          slab.duration() / 32,
          reject_step ? std::optional{std::make_unique<StepRejector>(0.5)}
                      : std::nullopt);
    check(false, std::make_unique<TimeSteppers::AdamsBashforth>(1), {},
          slab.end() - slab.duration() / 4,
          slab_length / 16. / (1.0 + std::numeric_limits<double>::epsilon()),
          -slab.duration() / 32,
          reject_step ? std::optional{std::make_unique<StepRejector>(0.5)}
                      : std::nullopt);
  }

  {
    // History out of order, as if just after self-start.
    TimeSteppers::History<double> history{};
    history.insert(TimeStepId(true, -1, slab.start() + slab.duration() / 8),
                   0.0, 0.0);
    check(true, std::make_unique<TimeSteppers::AdamsBashforth>(1),
          std::move(history), slab.start() + slab.duration() / 4, 1.0e-3,
          slab.duration() / 4, std::nullopt);
  }

  CHECK_THROWS_WITH(
      ([&slab, &slab_length]() {
        check<tmpl::list<StepChoosers::Constant>>(
            true, std::make_unique<TimeSteppers::AdamsBashforth>(1), {},
            slab.start() + slab.duration() / 4, slab_length / 5.,
            slab.duration() / 8, std::make_unique<StepRejector>(0.5));
      })(),
      Catch::Matchers::ContainsSubstring("is not registered"));

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      check(true, std::make_unique<TimeSteppers::AdamsBashforth>(1), {},
            slab.start() + slab.duration() / 4, slab_length,
            slab.duration() / 4, std::make_unique<StepRejector>(1.0)),
      Catch::Matchers::ContainsSubstring("Step was rejected, but not changed"));
#endif

  CHECK_THROWS_WITH(
      check(true, std::make_unique<TimeSteppers::AdamsBashforth>(1), {},
            slab.start() + slab.duration() / 4, 1e-9, slab.duration() / 4,
            std::nullopt),
      Catch::Matchers::ContainsSubstring("smaller than the MinimumTimeStep"));

  test_fixed_lts_ratio();
}
