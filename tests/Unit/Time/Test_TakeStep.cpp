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
#include "Time/StepChoosers/Cfl.hpp"
#include "Time/StepChoosers/Increase.hpp"
#include "Time/StepControllers/BinaryFraction.hpp"
#include "Time/Tags.hpp"
#include "Time/TakeStep.hpp"
#include "Time/Time.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Literals.hpp"

namespace {
struct EvolvedVariable : db::SimpleTag {
  using type = DataVector;
};

struct Metavariables {
  struct system {
    using variables_tag = EvolvedVariable;
    struct compute_largest_characteristic_speed {
      using argument_tags = tmpl::list<>;
      static constexpr double apply() noexcept { return 1.0; }
    };
  };
  static constexpr bool local_time_stepping = true;
  using time_stepper_tag = Tags::TimeStepper<LtsTimeStepper>;
  using component_list = tmpl::list<>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.TakeStep", "[Unit][Time]") {
  using step_chooser_list =
      tmpl::list<StepChoosers::Registrars::Increase,
                 StepChoosers::Registrars::Cfl<1, Frame::Inertial>>;
  const Slab slab{0.0, 1.00};
  const TimeDelta time_step = slab.duration() / 4;

  const Parallel::GlobalCache<Metavariables> cache{};
  std::vector<std::unique_ptr<StepChooser<step_chooser_list>>> step_choosers;
  step_choosers.emplace_back(
      std::make_unique<StepChoosers::Increase<step_chooser_list>>(2.0));
  step_choosers.emplace_back(
      std::make_unique<
          StepChoosers::Cfl<1, Frame::Inertial, step_chooser_list>>(1.0));

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist{-1.0, 1.0};

  const auto initial_values = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&dist), DataVector{5});

  // exponential function
  const auto update_rhs = [](const gsl::not_null<DataVector*> dt_y,
                             const DataVector& y) noexcept {
    *dt_y = 1.0e-2 * y;
  };

  typename ::Tags::HistoryEvolvedVariables<EvolvedVariable>::type history{5};
  // prepare history so that the Adams-Bashforth is ready to take steps
  TimeStepperTestUtils::initialize_history(
      slab.start(), make_not_null(&history),
      [&initial_values](const auto t) noexcept {
        return initial_values * exp(1.0e-2 * t);
      },
      [](const auto y, const auto /*t*/) noexcept { return 1.0e-2 * y; },
      time_step, 4);

  auto box = db::create<db::AddSimpleTags<
      Tags::TimeStepId, Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
      Tags::Next<Tags::TimeStep>, EvolvedVariable, Tags::dt<EvolvedVariable>,
      Tags::HistoryEvolvedVariables<EvolvedVariable>,
      Tags::TimeStepper<LtsTimeStepper>, Tags::StepChoosers<step_chooser_list>,
      domain::Tags::MinimumGridSpacing<1, Frame::Inertial>,
      Tags::StepController>>(
      TimeStepId{true, 0_st, slab.start()},
      TimeStepId{true, 0_st, Time{slab, {1, 4}}}, time_step, time_step,
      initial_values, DataVector{5, 0.0}, std::move(history),
      static_cast<std::unique_ptr<LtsTimeStepper>>(
          std::make_unique<TimeSteppers::AdamsBashforthN>(5)),
      std::move(step_choosers),
      1.0 / TimeSteppers::AdamsBashforthN{5}.stable_step(),
      static_cast<std::unique_ptr<StepController>>(
          std::make_unique<StepControllers::BinaryFraction>()));

  // update the rhs
  db::mutate<Tags::dt<EvolvedVariable>>(make_not_null(&box), update_rhs,
                                         db::get<EvolvedVariable>(box));
  take_step<step_chooser_list>(make_not_null(&box), cache);
  // check that the state is as expected
  CHECK(db::get<Tags::TimeStepId>(box).substep_time().value() == 0.0);
  CHECK(db::get<Tags::Next<Tags::TimeStepId>>(box).substep_time().value() ==
        approx(0.25));
  CHECK(db::get<Tags::Next<Tags::TimeStep>>(box) == TimeDelta{slab, {1, 4}});
  CHECK_ITERABLE_APPROX(db::get<EvolvedVariable>(box),
                        initial_values * exp(0.0025));
  CHECK_ITERABLE_APPROX(db::get<Tags::dt<EvolvedVariable>>(box),
                        1.0e-2 * initial_values);

  // advance time
  db::mutate<Tags::TimeStepId, Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
             Tags::Next<Tags::TimeStep>>(
      make_not_null(&box),
      [](const gsl::not_null<TimeStepId*> time_id,
         const gsl::not_null<TimeStepId*> next_time_id,
         const gsl::not_null<TimeDelta*> local_time_step,
         const gsl::not_null<TimeDelta*> next_time_step,
         const TimeStepper& time_stepper) noexcept {
        *time_id = *next_time_id;
        *local_time_step =
            next_time_step->with_slab(time_id->step_time().slab());

        *next_time_id =
            time_stepper.next_time_id(*next_time_id, *local_time_step);
        *next_time_step =
            local_time_step->with_slab(next_time_id->step_time().slab());
      },
      db::get<Tags::TimeStepper<>>(box));

  db::mutate<Tags::dt<EvolvedVariable>>(make_not_null(&box), update_rhs,
                                        db::get<EvolvedVariable>(box));
  // alter the grid spacing so that the CFL condition will cause rejection.
  db::mutate<domain::Tags::MinimumGridSpacing<1, Frame::Inertial>>(
      make_not_null(&box), [](const gsl::not_null<double*> grid_spacing) {
        *grid_spacing = 0.15 / TimeSteppers::AdamsBashforthN{5}.stable_step();
      });
  take_step<step_chooser_list>(make_not_null(&box), cache);
  // check that the state is as expected
  CHECK(db::get<Tags::TimeStepId>(box).substep_time().value() == approx(0.25));
  CHECK(db::get<Tags::TimeStep>(box) == TimeDelta{slab, {1, 8}});
  CHECK(db::get<Tags::Next<Tags::TimeStep>>(box) == TimeDelta{slab, {1, 8}});
  CHECK_ITERABLE_APPROX(db::get<EvolvedVariable>(box),
                        initial_values * exp(0.00375));
  CHECK_ITERABLE_APPROX(db::get<Tags::dt<EvolvedVariable>>(box),
                        1.0e-2 * initial_values * exp(0.0025));
}
