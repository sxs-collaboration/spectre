// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Tags/Flags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Tags/ArrayIndex.hpp"
#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Time/ChooseLtsStepSize.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/Increase.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Rational.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
template <typename TimeStepperType>
struct TestMetavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        StepChooser<StepChooserUse::LtsStep>,
        tmpl::list<StepChoosers::Increase<StepChooserUse::LtsStep>>>>;
  };
  using component_list = tmpl::list<>;
  using const_global_cache_tags =
      tmpl::list<::Tags::TimeStepper<TimeStepperType>>;
};

void test_gts() {
  const double initial_time = 1.5;
  const double initial_dt = 0.5;
  const double initial_slab_size = initial_dt;
  std::unique_ptr<TimeStepper> time_stepper =
      std::make_unique<TimeSteppers::AdamsBashforth>(5);

  const Slab initial_slab =
      Slab::with_duration_from_start(initial_time, initial_slab_size);
  const Time time = initial_slab.start();
  const TimeStepId expected_next_time_step_id = TimeStepId(
      true, -static_cast<int64_t>(time_stepper->number_of_past_steps()), time);
  const TimeDelta expected_time_step = time.slab().duration();
  const TimeDelta expected_next_time_step = expected_time_step;

  tuples::TaggedTuple<::Tags::TimeStepper<TimeStepper>>
      const_global_cache_items(std::move(time_stepper));

  Parallel::GlobalCache<TestMetavariables<TimeStepper>> global_cache(
      std::move(const_global_cache_items));
  auto box = db::create<
      db::AddSimpleTags<
          Parallel::Tags::GlobalCacheImpl<TestMetavariables<TimeStepper>>,
          ::Tags::Time, Initialization::Tags::InitialTimeDelta,
          Initialization::Tags::InitialSlabSize<false>,
          ::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep,
          ::Tags::Next<::Tags::TimeStep>>,
      tmpl::list<
          Parallel::Tags::FromGlobalCache<::Tags::TimeStepper<TimeStepper>>>>(
      &global_cache, initial_time, initial_dt, initial_slab_size, TimeStepId{},
      TimeDelta{}, TimeDelta{});

  db::mutate_apply<
      Initialization::TimeStepping<TestMetavariables<TimeStepper>, false>>(
      make_not_null(&box));

  CHECK(db::get<::Tags::Next<::Tags::TimeStepId>>(box) ==
        expected_next_time_step_id);
  CHECK(db::get<::Tags::TimeStep>(box) == expected_time_step);
  CHECK(db::get<::Tags::Next<::Tags::TimeStep>>(box) ==
        expected_next_time_step);
}

void test_lts() {
  const double initial_time = 1.5;
  const double initial_dt = 0.5;
  const double initial_slab_size = 4.5;
  std::unique_ptr<LtsTimeStepper> lts_time_stepper =
      std::make_unique<TimeSteppers::AdamsBashforth>(5);

  const Slab initial_slab =
      Slab::with_duration_from_start(initial_time, initial_slab_size);
  const Time time = initial_slab.start();
  const TimeStepId expected_next_time_step_id = TimeStepId(
      true, -static_cast<int64_t>(lts_time_stepper->number_of_past_steps()),
      time);
  const TimeDelta expected_time_step = choose_lts_step_size(time, initial_dt);
  const TimeDelta expected_next_time_step = expected_time_step;

  tuples::TaggedTuple<::Tags::TimeStepper<LtsTimeStepper>>
      const_global_cache_items(std::move(lts_time_stepper));

  Parallel::GlobalCache<TestMetavariables<LtsTimeStepper>> global_cache(
      std::move(const_global_cache_items));

  auto box = db::create<
      db::AddSimpleTags<
          Parallel::Tags::GlobalCacheImpl<TestMetavariables<LtsTimeStepper>>,
          ::Tags::Time, Initialization::Tags::InitialTimeDelta,
          Initialization::Tags::InitialSlabSize<true>,
          ::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep,
          ::Tags::Next<::Tags::TimeStep>>,
      tmpl::list<Parallel::Tags::FromGlobalCache<
          ::Tags::TimeStepper<LtsTimeStepper>>>>(
      &global_cache, initial_time, initial_dt, initial_slab_size, TimeStepId{},
      TimeDelta{}, TimeDelta{});

  db::mutate_apply<
      Initialization::TimeStepping<TestMetavariables<LtsTimeStepper>, true>>(
      make_not_null(&box));

  CHECK(db::get<::Tags::Next<::Tags::TimeStepId>>(box) ==
        expected_next_time_step_id);
  CHECK(db::get<::Tags::TimeStep>(box) == expected_time_step);
  CHECK(db::get<::Tags::Next<::Tags::TimeStep>>(box) ==
        expected_next_time_step);
}
using items_type =
    tuples::TaggedTuple<Parallel::Tags::ArrayIndexImpl<ElementId<1>>,
                        ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
                        ::Tags::TimeStep, ::Tags::Next<::Tags::TimeStep>,
                        ::Tags::Time, ::Tags::AdaptiveSteppingDiagnostics>;

using parent_items_type =
    tuples::TaggedTuple<Parallel::Tags::ArrayIndexImpl<ElementId<1>>,
                        ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
                        ::Tags::TimeStep, ::Tags::Next<::Tags::TimeStep>,
                        ::Tags::Time, ::Tags::AdaptiveSteppingDiagnostics,
                        ::amr::Tags::Flags<1>>;

template <typename DbTagList>
void check(const db::DataBox<DbTagList>& box,
           const TimeStepId& expected_time_step_id,
           const TimeStepId& expected_next_time_step_id,
           const TimeDelta& expected_time_step,
           const TimeDelta& expected_next_time_step, const double expected_time,
           const AdaptiveSteppingDiagnostics& expected_diagnostics) {
  CHECK(db::get<::Tags::TimeStepId>(box) == expected_time_step_id);
  CHECK(db::get<::Tags::Next<::Tags::TimeStepId>>(box) ==
        expected_next_time_step_id);
  CHECK(db::get<::Tags::TimeStep>(box) == expected_time_step);
  CHECK(db::get<::Tags::Next<::Tags::TimeStep>>(box) ==
        expected_next_time_step);
  CHECK(db::get<::Tags::Time>(box) == expected_time);
  CHECK(db::get<::Tags::AdaptiveSteppingDiagnostics>(box) ==
        expected_diagnostics);
}

void test_p_refine() {
  const ElementId<1> element_id{0};
  const Element<1> element{element_id, DirectionMap<1, Neighbors<1>>{}};
  const Mesh<1> mesh{2, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const Slab slab(0., 1.);
  const Time start{slab.start()};
  const TimeDelta time_step{slab.duration()};
  const TimeDelta next_time_step = time_step;
  const TimeStepId time_step_id{time_step.is_positive(), 8, start};
  const TimeStepId next_time_step_id{time_step.is_positive(), 8,
                                     start + time_step};
  const double time = start.value();
  const AdaptiveSteppingDiagnostics diagnostics{7, 2, 13, 4, 5};

  auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::ArrayIndexImpl<ElementId<1>>,
                        ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
                        ::Tags::TimeStep, ::Tags::Next<::Tags::TimeStep>,
                        ::Tags::Time, ::Tags::AdaptiveSteppingDiagnostics>>(
      element_id, time_step_id, next_time_step_id, time_step, next_time_step,
      time, diagnostics);

  db::mutate_apply<Initialization::ProjectTimeStepping<1>>(
      make_not_null(&box), std::make_pair(mesh, element));

  check(box, time_step_id, next_time_step_id, time_step, next_time_step, time,
        diagnostics);
}

void test_split() {
  const ElementId<1> parent_id{0};
  const ElementId<1> child_1_id{0, std::array{SegmentId{1, 0}}};
  const ElementId<1> child_2_id{0, std::array{SegmentId{1, 1}}};

  const Slab slab(1., 1.5);
  const Time start{slab.start()};
  const TimeDelta time_step{slab.duration()};
  const TimeDelta next_time_step = time_step;
  const TimeStepId time_step_id{time_step.is_positive(), 8, start};
  const TimeStepId next_time_step_id{time_step.is_positive(), 8,
                                     start + time_step};
  const double time = start.value();
  const AdaptiveSteppingDiagnostics diagnostics{7, 2, 13, 4, 5};

  const parent_items_type parent_items{
      parent_id,         time_step_id,
      next_time_step_id, time_step,
      next_time_step,    time,
      diagnostics,       std::array{::amr::Flag::Split}};

  auto child_1_box = db::create<
      db::AddSimpleTags<Parallel::Tags::ArrayIndexImpl<ElementId<1>>,
                        ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
                        ::Tags::TimeStep, ::Tags::Next<::Tags::TimeStep>,
                        ::Tags::Time, ::Tags::AdaptiveSteppingDiagnostics>>(
      child_1_id, TimeStepId{}, TimeStepId{}, TimeDelta{}, TimeDelta{}, 0.0,
      AdaptiveSteppingDiagnostics{});

  auto child_2_box = db::create<
      db::AddSimpleTags<Parallel::Tags::ArrayIndexImpl<ElementId<1>>,
                        ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
                        ::Tags::TimeStep, ::Tags::Next<::Tags::TimeStep>,
                        ::Tags::Time, ::Tags::AdaptiveSteppingDiagnostics>>(
      child_2_id, TimeStepId{}, TimeStepId{}, TimeDelta{}, TimeDelta{}, 0.0,
      AdaptiveSteppingDiagnostics{});

  db::mutate_apply<Initialization::ProjectTimeStepping<1>>(
      make_not_null(&child_1_box), parent_items);

  check(child_1_box, time_step_id, next_time_step_id, time_step, next_time_step,
        time, diagnostics);

  db::mutate_apply<Initialization::ProjectTimeStepping<1>>(
      make_not_null(&child_2_box), parent_items);

  check(child_2_box, time_step_id, next_time_step_id, time_step, next_time_step,
        time, AdaptiveSteppingDiagnostics{7, 2, 0, 0, 0});
}

template <bool ForwardInTime>
void test_join() {
  const ElementId<1> parent_id{0};
  const ElementId<1> child_1_id{0, std::array{SegmentId{1, 0}}};
  const ElementId<1> child_2_id{0, std::array{SegmentId{1, 1}}};

  const Slab slab_1(1., 1.5);
  const Time start_1{ForwardInTime ? slab_1.start() : slab_1.end()};
  const TimeDelta time_step_1{ForwardInTime ? slab_1.duration()
                                            : -slab_1.duration()};
  const TimeDelta next_time_step_1 = time_step_1;
  const TimeStepId time_step_id_1{time_step_1.is_positive(), 8, start_1};
  const TimeStepId next_time_step_id_1{time_step_1.is_positive(), 8,
                                       start_1 + time_step_1};
  const double time_1 = start_1.value();
  const AdaptiveSteppingDiagnostics diagnostics_1{7, 2, 13, 4, 5};

  const Slab slab_2(1., 1.5);
  const Time start_2{ForwardInTime ? slab_2.start() : slab_2.end()};
  const TimeDelta time_step_2{slab_2, Rational{ForwardInTime ? 1 : -1, 2}};
  const TimeDelta next_time_step_2 = time_step_2;
  const TimeStepId time_step_id_2{time_step_2.is_positive(), 8, start_2};
  const TimeStepId next_time_step_id_2{time_step_2.is_positive(), 8,
                                       start_2 + time_step_2};
  const double time_2 = start_2.value();
  const AdaptiveSteppingDiagnostics diagnostics_2{7, 2, 27, 2, 8};

  std::unordered_map<ElementId<1>, items_type> children_items;
  children_items.emplace(
      child_1_id,
      items_type{child_1_id, time_step_id_1, next_time_step_id_1, time_step_1,
                 next_time_step_1, time_1, diagnostics_1});
  children_items.emplace(
      child_2_id,
      items_type{child_2_id, time_step_id_2, next_time_step_id_2, time_step_2,
                 next_time_step_2, time_2, diagnostics_2});

  auto parent_box = db::create<
      db::AddSimpleTags<Parallel::Tags::ArrayIndexImpl<ElementId<1>>,
                        ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
                        ::Tags::TimeStep, ::Tags::Next<::Tags::TimeStep>,
                        ::Tags::Time, ::Tags::AdaptiveSteppingDiagnostics>>(
      parent_id, TimeStepId{}, TimeStepId{}, TimeDelta{}, TimeDelta{}, 0.0,
      AdaptiveSteppingDiagnostics{});

  db::mutate_apply<Initialization::ProjectTimeStepping<1>>(
      make_not_null(&parent_box), children_items);

  check(parent_box, time_step_id_2, next_time_step_id_2, time_step_2,
        next_time_step_2, time_2, AdaptiveSteppingDiagnostics{7, 2, 40, 6, 13});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Initialization.TimeStepping",
                  "[Evolution][Unit]") {
  test_gts();
  test_lts();
  test_p_refine();
  test_split();
  test_join<true>();
  test_join<false>();
}
