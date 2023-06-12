// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/MinimumGridSpacing.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Time/TimeSteppers/TimeStepperTestUtils.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Time/ChooseLtsStepSize.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/Cfl.hpp"
#include "Time/StepChoosers/Increase.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/TakeStep.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace {
struct EvolvedVariable : db::SimpleTag {
  using type = DataVector;
};

struct Metavariables {
  struct system {
    static constexpr size_t volume_dim = 1;
    using variables_tag = EvolvedVariable;
    struct largest_characteristic_speed : db::SimpleTag {
      using type = double;
    };

    struct compute_largest_characteristic_speed : largest_characteristic_speed,
                                                  db::ComputeTag {
      using argument_tags = tmpl::list<>;
      using return_type = double;
      using base = largest_characteristic_speed;
      SPECTRE_ALWAYS_INLINE static constexpr void function(
          const gsl::not_null<double*> speed) {
        *speed = 1.0;
      }
    };
  };

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                   tmpl::list<StepChoosers::Increase<StepChooserUse::LtsStep>,
                              StepChoosers::Cfl<
                                  StepChooserUse::LtsStep, Frame::Inertial,
                                  typename Metavariables::system>>>>;
  };

  using component_list = tmpl::list<>;
};

void test_gts() {
  const Slab slab{0.0, 1.00};
  const TimeDelta time_step = slab.duration() / 4;

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist{-1.0, 1.0};

  const auto initial_values = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&dist), DataVector{5});

  // exponential function
  const auto update_rhs = [](const gsl::not_null<DataVector*> dt_y,
                             const DataVector& y) { *dt_y = 1.0e-2 * y; };

  typename ::Tags::HistoryEvolvedVariables<EvolvedVariable>::type history{5};
  // prepare history so that the Adams-Bashforth is ready to take steps
  TimeStepperTestUtils::initialize_history(
      slab.start(), make_not_null(&history),
      [&initial_values](const auto t) {
        return initial_values * exp(1.0e-2 * t);
      },
      [](const auto y, const auto /*t*/) { return 1.0e-2 * y; }, time_step, 4);

  auto box = db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<Metavariables>, Tags::TimeStepId,
      Tags::Next<Tags::TimeStepId>, Tags::TimeStep, Tags::Next<Tags::TimeStep>,
      EvolvedVariable, Tags::dt<EvolvedVariable>,
      Tags::HistoryEvolvedVariables<EvolvedVariable>,
      Tags::TimeStepper<TimeStepper>,
      ::Tags::IsUsingTimeSteppingErrorControl>>(
      Metavariables{}, TimeStepId{true, 0_st, slab.start()},
      TimeStepId{true, 0_st, Time{slab, {1, 4}}}, time_step, time_step,
      initial_values, DataVector{5, 0.0}, std::move(history),
      static_cast<std::unique_ptr<TimeStepper>>(
          std::make_unique<TimeSteppers::AdamsBashforth>(5)),
      false);
  // update the rhs
  db::mutate<Tags::dt<EvolvedVariable>>(update_rhs, make_not_null(&box),
                                        db::get<EvolvedVariable>(box));
  take_step<typename Metavariables::system, false>(make_not_null(&box));
  // check that the state is as expected
  CHECK(db::get<Tags::TimeStepId>(box).substep_time() == 0.0);
  CHECK(db::get<Tags::Next<Tags::TimeStepId>>(box).substep_time() ==
        approx(0.25));
  CHECK(db::get<Tags::Next<Tags::TimeStep>>(box) == TimeDelta{slab, {1, 4}});
  CHECK_ITERABLE_APPROX(db::get<EvolvedVariable>(box),
                        initial_values * exp(0.0025));
  CHECK_ITERABLE_APPROX(db::get<Tags::dt<EvolvedVariable>>(box),
                        1.0e-2 * initial_values);
}

void test_lts() {
  const Slab slab{0.0, 1.00};
  const TimeDelta time_step = slab.duration() / 4;

  std::vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>
      step_choosers;
  step_choosers.emplace_back(
      std::make_unique<StepChoosers::Increase<StepChooserUse::LtsStep>>(2.0));
  step_choosers.emplace_back(
      std::make_unique<
          StepChoosers::Cfl<StepChooserUse::LtsStep, Frame::Inertial,
                            typename Metavariables::system>>(1.0));

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist{-1.0, 1.0};

  const auto initial_values = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&dist), DataVector{5});

  // exponential function
  const auto update_rhs = [](const gsl::not_null<DataVector*> dt_y,
                             const DataVector& y) { *dt_y = 1.0e-2 * y; };

  typename ::Tags::HistoryEvolvedVariables<EvolvedVariable>::type history{5};
  // prepare history so that the Adams-Bashforth is ready to take steps
  TimeStepperTestUtils::initialize_history(
      slab.start(), make_not_null(&history),
      [&initial_values](const auto t) {
        return initial_values * exp(1.0e-2 * t);
      },
      [](const auto y, const auto /*t*/) { return 1.0e-2 * y; }, time_step, 4);

  auto box = db::create<
      db::AddSimpleTags<
          Parallel::Tags::MetavariablesImpl<Metavariables>, Tags::TimeStepId,
          Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
          Tags::Next<Tags::TimeStep>, EvolvedVariable,
          Tags::dt<EvolvedVariable>, Tags::StepperError<EvolvedVariable>,
          Tags::PreviousStepperError<EvolvedVariable>,
          Tags::HistoryEvolvedVariables<EvolvedVariable>,
          Tags::TimeStepper<LtsTimeStepper>, Tags::StepChoosers,
          domain::Tags::MinimumGridSpacing<1, Frame::Inertial>,
          ::Tags::IsUsingTimeSteppingErrorControl, Tags::StepperErrorUpdated,
          Tags::AdaptiveSteppingDiagnostics>,
      db::AddComputeTags<typename Metavariables::system::
                             compute_largest_characteristic_speed>>(
      Metavariables{}, TimeStepId{true, 0_st, slab.start()},
      TimeStepId{true, 0_st, Time{slab, {1, 4}}}, time_step, time_step,
      initial_values, DataVector{5, 0.0}, DataVector{}, DataVector{},
      std::move(history),
      static_cast<std::unique_ptr<LtsTimeStepper>>(
          std::make_unique<TimeSteppers::AdamsBashforth>(5)),
      std::move(step_choosers),
      1.0 / TimeSteppers::AdamsBashforth{5}.stable_step(), true, false,
      AdaptiveSteppingDiagnostics{1, 2, 3, 4, 5});

  // update the rhs
  db::mutate<Tags::dt<EvolvedVariable>>(update_rhs, make_not_null(&box),
                                        db::get<EvolvedVariable>(box));
  take_step<typename Metavariables::system, true>(make_not_null(&box));
  // check that the state is as expected
  CHECK(db::get<Tags::TimeStepId>(box).substep_time() == 0.0);
  CHECK(db::get<Tags::Next<Tags::TimeStepId>>(box).substep_time() ==
        approx(0.25));
  CHECK(db::get<Tags::Next<Tags::TimeStep>>(box) == TimeDelta{slab, {1, 4}});
  CHECK_ITERABLE_APPROX(db::get<EvolvedVariable>(box),
                        initial_values * exp(0.0025));
  CHECK_ITERABLE_APPROX(db::get<Tags::dt<EvolvedVariable>>(box),
                        1.0e-2 * initial_values);
  CHECK(db::get<Tags::AdaptiveSteppingDiagnostics>(box) ==
        AdaptiveSteppingDiagnostics{1, 2, 3, 4, 5});

  // advance time
  db::mutate<Tags::TimeStepId, Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
             Tags::Next<Tags::TimeStep>>(
      [](const gsl::not_null<TimeStepId*> time_id,
         const gsl::not_null<TimeStepId*> next_time_id,
         const gsl::not_null<TimeDelta*> local_time_step,
         const gsl::not_null<TimeDelta*> next_time_step,
         const LtsTimeStepper& time_stepper) {
        *time_id = *next_time_id;
        *local_time_step =
            next_time_step->with_slab(time_id->step_time().slab());

        *next_time_id =
            time_stepper.next_time_id(*next_time_id, *local_time_step);
        *next_time_step =
            local_time_step->with_slab(next_time_id->step_time().slab());
      },
      make_not_null(&box), db::get<Tags::TimeStepper<>>(box));

  db::mutate<Tags::dt<EvolvedVariable>>(update_rhs, make_not_null(&box),
                                        db::get<EvolvedVariable>(box));
  // alter the grid spacing so that the CFL condition will cause rejection.
  db::mutate<domain::Tags::MinimumGridSpacing<1, Frame::Inertial>>(
      [](const gsl::not_null<double*> grid_spacing) {
        *grid_spacing = 0.15 / TimeSteppers::AdamsBashforth{5}.stable_step();
      },
      make_not_null(&box));
  take_step<typename Metavariables::system, true>(make_not_null(&box));
  // check that the state is as expected
  CHECK(db::get<Tags::TimeStepId>(box).substep_time() == approx(0.25));
  CHECK(db::get<Tags::TimeStep>(box) == TimeDelta{slab, {1, 8}});
  CHECK(db::get<Tags::Next<Tags::TimeStep>>(box) == TimeDelta{slab, {1, 8}});
  CHECK_ITERABLE_APPROX(db::get<EvolvedVariable>(box),
                        initial_values * exp(0.00375));
  CHECK_ITERABLE_APPROX(db::get<Tags::dt<EvolvedVariable>>(box),
                        1.0e-2 * initial_values * exp(0.0025));
  CHECK(db::get<Tags::AdaptiveSteppingDiagnostics>(box) ==
        AdaptiveSteppingDiagnostics{1, 2, 3, 5, 6});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.TakeStep", "[Unit][Time]") {
  test_gts();
  test_lts();
}
