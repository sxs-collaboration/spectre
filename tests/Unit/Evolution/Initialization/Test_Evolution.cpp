// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Tags/Flags.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/TestTags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Tags/ArrayIndex.hpp"
#include "ParallelAlgorithms/Amr/Projectors/Variables.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Time/ChangeSlabSize/Tags.hpp"
#include "Time/ChooseLtsStepSize.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/LimitIncrease.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/StepNumberWithinSlab.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
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
    using factory_classes =
        tmpl::map<tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                             tmpl::list<StepChoosers::LimitIncrease>>>;
  };
  using component_list = tmpl::list<>;
  using const_global_cache_tags =
      tmpl::list<::Tags::ConcreteTimeStepper<TimeStepperType>>;
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

  tuples::TaggedTuple<::Tags::ConcreteTimeStepper<TimeStepper>>
      const_global_cache_items(std::move(time_stepper));

  Parallel::GlobalCache<TestMetavariables<TimeStepper>> global_cache(
      std::move(const_global_cache_items));
  auto box = db::create<
      db::AddSimpleTags<
          Parallel::Tags::GlobalCacheImpl<TestMetavariables<TimeStepper>>,
          ::Tags::Time, Initialization::Tags::InitialTimeDelta,
          Initialization::Tags::InitialSlabSize<false>,
          ::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep,
          ::Tags::ChangeSlabSize::SlabSizeGoal>,
      tmpl::list<Parallel::Tags::FromGlobalCache<
          ::Tags::ConcreteTimeStepper<TimeStepper>>>>(
      &global_cache, initial_time, initial_dt, initial_slab_size, TimeStepId{},
      TimeDelta{}, std::numeric_limits<double>::signaling_NaN());

  db::mutate_apply<Initialization::TimeStepping<TestMetavariables<TimeStepper>,
                                                TimeStepper>>(
      make_not_null(&box));

  CHECK(db::get<::Tags::Next<::Tags::TimeStepId>>(box) ==
        expected_next_time_step_id);
  CHECK(db::get<::Tags::TimeStep>(box) == expected_time_step);
  CHECK(db::get<::Tags::ChangeSlabSize::SlabSizeGoal>(box) ==
        initial_slab_size);
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

  tuples::TaggedTuple<::Tags::ConcreteTimeStepper<LtsTimeStepper>>
      const_global_cache_items(std::move(lts_time_stepper));

  Parallel::GlobalCache<TestMetavariables<LtsTimeStepper>> global_cache(
      std::move(const_global_cache_items));

  auto box = db::create<
      db::AddSimpleTags<
          Parallel::Tags::GlobalCacheImpl<TestMetavariables<LtsTimeStepper>>,
          ::Tags::Time, Initialization::Tags::InitialTimeDelta,
          Initialization::Tags::InitialSlabSize<true>,
          ::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep,
          ::Tags::ChangeSlabSize::SlabSizeGoal>,
      tmpl::list<Parallel::Tags::FromGlobalCache<
          ::Tags::ConcreteTimeStepper<LtsTimeStepper>>>>(
      &global_cache, initial_time, initial_dt, initial_slab_size, TimeStepId{},
      TimeDelta{}, std::numeric_limits<double>::signaling_NaN());

  db::mutate_apply<Initialization::TimeStepping<
      TestMetavariables<LtsTimeStepper>, LtsTimeStepper>>(make_not_null(&box));

  CHECK(db::get<::Tags::Next<::Tags::TimeStepId>>(box) ==
        expected_next_time_step_id);
  CHECK(db::get<::Tags::TimeStep>(box) == expected_time_step);
  CHECK(db::get<::Tags::ChangeSlabSize::SlabSizeGoal>(box) ==
        initial_slab_size);
}
using items_type = tuples::TaggedTuple<
    Parallel::Tags::ArrayIndexImpl<ElementId<1>>, ::Tags::TimeStepId,
    ::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep, ::Tags::Time,
    ::Tags::StepNumberWithinSlab, ::Tags::AdaptiveSteppingDiagnostics,
    ::Tags::ChangeSlabSize::SlabSizeGoal>;

using parent_items_type = tuples::TaggedTuple<
    Parallel::Tags::ArrayIndexImpl<ElementId<1>>, ::Tags::TimeStepId,
    ::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep, ::Tags::Time,
    ::Tags::StepNumberWithinSlab, ::Tags::AdaptiveSteppingDiagnostics,
    ::Tags::ChangeSlabSize::SlabSizeGoal, ::amr::Tags::Info<1>>;

template <typename DbTagList>
void check(const db::DataBox<DbTagList>& box,
           const TimeStepId& expected_time_step_id,
           const TimeStepId& expected_next_time_step_id,
           const TimeDelta& expected_time_step, const double expected_time,
           const uint64_t expected_step_number_within_slab,
           const AdaptiveSteppingDiagnostics& expected_diagnostics,
           const double expected_slab_size_goal) {
  CHECK(db::get<::Tags::TimeStepId>(box) == expected_time_step_id);
  CHECK(db::get<::Tags::Next<::Tags::TimeStepId>>(box) ==
        expected_next_time_step_id);
  CHECK(db::get<::Tags::TimeStep>(box) == expected_time_step);
  CHECK(db::get<::Tags::Time>(box) == expected_time);
  CHECK(db::get<::Tags::StepNumberWithinSlab>(box) ==
        expected_step_number_within_slab);
  CHECK(db::get<::Tags::AdaptiveSteppingDiagnostics>(box) ==
        expected_diagnostics);
  CHECK(db::get<::Tags::ChangeSlabSize::SlabSizeGoal>(box) ==
        expected_slab_size_goal);
}

void test_p_refine() {
  const ElementId<1> element_id{0};
  const Element<1> element{element_id, DirectionMap<1, Neighbors<1>>{}};
  const Mesh<1> mesh{2, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const Slab slab(0., 1.);
  const Time start{slab.start()};
  const TimeDelta time_step{slab.duration()};
  const TimeStepId time_step_id{time_step.is_positive(), 8, start};
  const TimeStepId next_time_step_id{time_step.is_positive(), 8,
                                     start + time_step};
  const double time = start.value();
  const uint64_t step_number_within_slab{0};
  const AdaptiveSteppingDiagnostics diagnostics{7, 2, 13, 4, 5};
  const double slab_size_goal = 1.34;

  auto box = db::create<db::AddSimpleTags<
      Parallel::Tags::ArrayIndexImpl<ElementId<1>>, ::Tags::TimeStepId,
      ::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep, ::Tags::Time,
      ::Tags::StepNumberWithinSlab, ::Tags::AdaptiveSteppingDiagnostics,
      ::Tags::ChangeSlabSize::SlabSizeGoal>>(
      element_id, time_step_id, next_time_step_id, time_step, time,
      step_number_within_slab, diagnostics, slab_size_goal);

  db::mutate_apply<Initialization::ProjectTimeStepping<1>>(
      make_not_null(&box), std::make_pair(mesh, element));

  check(box, time_step_id, next_time_step_id, time_step, time,
        step_number_within_slab, diagnostics, slab_size_goal);
}

void test_split() {
  const ElementId<1> parent_id{0};
  const ElementId<1> child_1_id{0, std::array{SegmentId{1, 0}}};
  const ElementId<1> child_2_id{0, std::array{SegmentId{1, 1}}};

  const Slab slab(1., 1.5);
  const Time start{slab.start()};
  const TimeDelta time_step{slab.duration()};
  const TimeStepId time_step_id{time_step.is_positive(), 8, start};
  const TimeStepId next_time_step_id{time_step.is_positive(), 8,
                                     start + time_step};
  const double time = start.value();
  const uint64_t step_number_within_slab{0};
  const AdaptiveSteppingDiagnostics diagnostics{7, 2, 13, 4, 5};
  const double slab_size_goal = 1.34;

  const parent_items_type parent_items{
      parent_id,
      time_step_id,
      next_time_step_id,
      time_step,
      time,
      step_number_within_slab,
      diagnostics,
      slab_size_goal,
      ::amr::Info<1>{std::array{::amr::Flag::Split}, Mesh<1>{}}};

  auto child_1_box = db::create<db::AddSimpleTags<
      Parallel::Tags::ArrayIndexImpl<ElementId<1>>, ::Tags::TimeStepId,
      ::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep, ::Tags::Time,
      ::Tags::StepNumberWithinSlab, ::Tags::AdaptiveSteppingDiagnostics,
      ::Tags::ChangeSlabSize::SlabSizeGoal>>(
      child_1_id, TimeStepId{}, TimeStepId{}, TimeDelta{}, 0.0,
      std::numeric_limits<uint64_t>::max(), AdaptiveSteppingDiagnostics{},
      std::numeric_limits<double>::signaling_NaN());

  auto child_2_box = db::create<db::AddSimpleTags<
      Parallel::Tags::ArrayIndexImpl<ElementId<1>>, ::Tags::TimeStepId,
      ::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep, ::Tags::Time,
      ::Tags::StepNumberWithinSlab, ::Tags::AdaptiveSteppingDiagnostics,
      ::Tags::ChangeSlabSize::SlabSizeGoal>>(
      child_2_id, TimeStepId{}, TimeStepId{}, TimeDelta{}, 0.0,
      std::numeric_limits<uint64_t>::max(), AdaptiveSteppingDiagnostics{},
      std::numeric_limits<double>::signaling_NaN());

  db::mutate_apply<Initialization::ProjectTimeStepping<1>>(
      make_not_null(&child_1_box), parent_items);

  check(child_1_box, time_step_id, next_time_step_id, time_step, time,
        step_number_within_slab, diagnostics, slab_size_goal);

  db::mutate_apply<Initialization::ProjectTimeStepping<1>>(
      make_not_null(&child_2_box), parent_items);

  check(child_2_box, time_step_id, next_time_step_id, time_step, time,
        step_number_within_slab, AdaptiveSteppingDiagnostics{7, 2, 0, 0, 0},
        slab_size_goal);
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
  const TimeStepId time_step_id_1{time_step_1.is_positive(), 8, start_1};
  const TimeStepId next_time_step_id_1{time_step_1.is_positive(), 8,
                                       start_1 + time_step_1};
  const double time_1 = start_1.value();
  const uint64_t step_number_within_slab_1{0};
  const AdaptiveSteppingDiagnostics diagnostics_1{7, 2, 13, 4, 5};
  const double slab_size_goal_1 = 1.34;

  const Slab slab_2(1., 1.5);
  const Time start_2{ForwardInTime ? slab_2.start() : slab_2.end()};
  const TimeDelta time_step_2{slab_2, Rational{ForwardInTime ? 1 : -1, 2}};
  const TimeStepId time_step_id_2{time_step_2.is_positive(), 8, start_2};
  const TimeStepId next_time_step_id_2{time_step_2.is_positive(), 8,
                                       start_2 + time_step_2};
  const double time_2 = start_2.value();
  const uint64_t step_number_within_slab_2 = step_number_within_slab_1;
  const AdaptiveSteppingDiagnostics diagnostics_2{7, 2, 27, 2, 8};
  const double slab_size_goal_2 = slab_size_goal_1;

  std::unordered_map<ElementId<1>, items_type> children_items;
  children_items.emplace(
      child_1_id,
      items_type{child_1_id, time_step_id_1, next_time_step_id_1, time_step_1,
                 time_1, step_number_within_slab_1, diagnostics_1,
                 slab_size_goal_1});
  children_items.emplace(
      child_2_id,
      items_type{child_2_id, time_step_id_2, next_time_step_id_2, time_step_2,
                 time_2, step_number_within_slab_2, diagnostics_2,
                 slab_size_goal_2});

  auto parent_box = db::create<db::AddSimpleTags<
      Parallel::Tags::ArrayIndexImpl<ElementId<1>>, ::Tags::TimeStepId,
      ::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep, ::Tags::Time,
      ::Tags::StepNumberWithinSlab, ::Tags::AdaptiveSteppingDiagnostics,
      ::Tags::ChangeSlabSize::SlabSizeGoal>>(
      parent_id, TimeStepId{}, TimeStepId{}, TimeDelta{}, 0.0,
      std::numeric_limits<uint64_t>::max(), AdaptiveSteppingDiagnostics{},
      std::numeric_limits<double>::signaling_NaN());

  db::mutate_apply<Initialization::ProjectTimeStepping<1>>(
      make_not_null(&parent_box), children_items);

  check(parent_box, time_step_id_2, next_time_step_id_2, time_step_2, time_2,
        step_number_within_slab_2, AdaptiveSteppingDiagnostics{7, 2, 40, 6, 13},
        slab_size_goal_2);
}

SPECTRE_TEST_CASE("Unit.Evolution.Initialization.TimeStepping",
                  "[Evolution][Unit]") {
  test_gts();
  test_lts();
  static_assert(tt::assert_conforms_to_v<Initialization::ProjectTimeStepping<1>,
                                         amr::protocols::Projector>);
  test_p_refine();
  test_split();
  test_join<true>();
  test_join<false>();
}

namespace time_stepper_history {
using VariablesType =
    Variables<tmpl::list<TestHelpers::Tags::Scalar<DataVector>>>;

using DtVariablesType =
    Variables<tmpl::list<::Tags::dt<TestHelpers::Tags::Scalar<DataVector>>>>;

template <size_t Dim>
struct TestSystem {
  using variables_tag =
      Tags::Variables<tmpl::list<TestHelpers::Tags::Scalar<DataVector>>>;
};

template <size_t Dim>
struct TestMetavariables {
  static constexpr size_t volume_dim = Dim;
  using system = TestSystem<Dim>;
};

template <typename T>
T f(const T& x, const std::array<double, 3>& c) {
  return c[0] + c[1] * x + c[2] * square(x);
}

template <size_t Dim>
VariablesType make_vars(
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& x, const double t) {
  const auto t_coeffs = std::array{0.5, 1.5, 2.5};
  const auto number_of_points = get<0>(x).size();
  VariablesType result{number_of_points, f(t, t_coeffs)};
  const auto x_coeffs = std::array{0.75, -1.75, 2.75};
  DataVector& s = get(get<TestHelpers::Tags::Scalar<DataVector>>(result));
  s *= f(x[0], x_coeffs);
  if constexpr (Dim > 1) {
    const auto y_coeffs = std::array{-0.25, 1.25, -2.25};
    s *= f(x[1], y_coeffs);
  }
  if constexpr (Dim > 2) {
    const auto z_coeffs = std::array{0.125, -1.625, -2.875};
    s *= f(x[2], z_coeffs);
  }
  return result;
}

template <size_t Dim>
DtVariablesType make_dt_vars(
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& x, const double t) {
  const auto dt_coeffs = std::array{1.5, 5.0, 0.0};
  const auto number_of_points = get<0>(x).size();
  DtVariablesType result{number_of_points, f(t, dt_coeffs)};
  const auto x_coeffs = std::array{0.75, -1.75, 2.75};
  DataVector& s =
      get(get<::Tags::dt<TestHelpers::Tags::Scalar<DataVector>>>(result));
  s *= f(x[0], x_coeffs);
  if constexpr (Dim > 1) {
    const auto y_coeffs = std::array{-0.25, 1.25, -2.25};
    s *= f(x[1], y_coeffs);
  }
  if constexpr (Dim > 2) {
    const auto z_coeffs = std::array{0.125, -1.625, -2.875};
    s *= f(x[2], z_coeffs);
  }
  return result;
}

template <size_t Dim>
void test_initialization() {
  const TimeSteppers::AdamsBashforth ab2{2};
  const Mesh<Dim> mesh{3, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  DtVariablesType dt_vars{};
  DtVariablesType expected_dt_vars{mesh.number_of_grid_points()};
  TimeSteppers::History<VariablesType> history{};
  TimeSteppers::History<VariablesType> expected_history{1};
  Initialization::TimeStepperHistory<TestMetavariables<Dim>>::apply(
      make_not_null(&dt_vars), make_not_null(&history), ab2, mesh);
  CHECK(dt_vars.size() == expected_dt_vars.size());
  CHECK(history == expected_history);
}

void check_history(
    const TimeSteppers::History<VariablesType>& history,
    const TimeSteppers::History<VariablesType>& expected_history) {
  for (size_t i = 0; i < history.size(); ++i) {
    CHECK(history[i].time_step_id == expected_history[i].time_step_id);
    CHECK(history[i].value.has_value() ==
          expected_history[i].value.has_value());
    if (history[i].value.has_value()) {
      CHECK_VARIABLES_APPROX(*history[i].value, *expected_history[i].value);
    }
    CHECK_VARIABLES_APPROX(history[i].derivative,
                           expected_history[i].derivative);
  }
  const auto& substeps = history.substeps();
  const auto& expected_substeps = expected_history.substeps();
  for (size_t i = 0; i < substeps.size(); ++i) {
    CHECK(substeps[i].time_step_id == expected_substeps[i].time_step_id);
    CHECK(substeps[i].value.has_value() ==
          expected_substeps[i].value.has_value());
    if (substeps[i].value.has_value()) {
      CHECK_VARIABLES_APPROX(*substeps[i].value, *expected_substeps[i].value);
    }
    CHECK_VARIABLES_APPROX(substeps[i].derivative,
                           expected_substeps[i].derivative);
  }
}

template <size_t Dim>
void check(const TimeSteppers::History<VariablesType>& original_history,
           const TimeSteppers::History<VariablesType>& expected_history,
           const Mesh<Dim>& new_mesh, const ElementId<Dim>& element_id,
           const Mesh<Dim>& old_mesh, const Element<Dim>& element) {
  DtVariablesType dt_vars{};
  TimeSteppers::History<VariablesType> history = original_history;
  Initialization::ProjectTimeStepperHistory<TestMetavariables<Dim>>::apply(
      make_not_null(&dt_vars), make_not_null(&history), new_mesh, element_id,
      std::make_pair(old_mesh, element));
  CHECK(dt_vars.size() == new_mesh.number_of_grid_points());
  check_history(history, expected_history);
  Initialization::ProjectTimeStepperHistory<TestMetavariables<Dim>>::apply(
      make_not_null(&dt_vars), make_not_null(&history), old_mesh, element_id,
      std::make_pair(new_mesh, element));
  CHECK(dt_vars.size() == old_mesh.number_of_grid_points());
  check_history(history, original_history);
}

template <size_t Dim>
void test_p_refine() {
  const ElementId<Dim> element_id{0};
  const Element<Dim> element{element_id, DirectionMap<Dim, Neighbors<Dim>>{}};
  const Mesh<Dim> old_mesh{4, Spectral::Basis::Legendre,
                           Spectral::Quadrature::GaussLobatto};
  std::array<size_t, Dim> new_extents{};
  std::iota(new_extents.begin(), new_extents.end(), 3_st);
  const Mesh<Dim> new_mesh{new_extents, Spectral::Basis::Legendre,
                           Spectral::Quadrature::GaussLobatto};
  const auto x_old = logical_coordinates(old_mesh);
  const auto x_new = logical_coordinates(new_mesh);
  TimeSteppers::History<VariablesType> history{};
  TimeSteppers::History<VariablesType> expected_history{};
  check(history, expected_history, new_mesh, element_id, old_mesh, element);
  const Slab slab(0.0, 1.0);
  history.integration_order(4);
  expected_history.integration_order(4);
  TimeStepId time_step_id{true, 0, slab.start()};
  double t = time_step_id.substep_time();
  history.insert_initial(time_step_id, make_vars(x_old, t),
                         make_dt_vars(x_old, t));
  expected_history.insert_initial(time_step_id, make_vars(x_new, t),
                                  make_dt_vars(x_new, t));
  check(history, expected_history, new_mesh, element_id, old_mesh, element);
  time_step_id =
      TimeStepId{true, -1, slab.start() - Slab(-1.0, 0.0).duration() / 4};
  t = time_step_id.substep_time();
  history.insert_initial(time_step_id, make_vars(x_old, t),
                         make_dt_vars(x_old, t));
  expected_history.insert_initial(time_step_id, make_vars(x_new, t),
                                  make_dt_vars(x_new, t));
  check(history, expected_history, new_mesh, element_id, old_mesh, element);
  time_step_id =
      TimeStepId{true, -1, slab.start() - Slab(-1.0, 0.0).duration() / 2};
  t = time_step_id.substep_time();
  history.insert_initial(time_step_id, make_vars(x_old, t),
                         make_dt_vars(x_old, t));
  expected_history.insert_initial(time_step_id, make_vars(x_new, t),
                                  make_dt_vars(x_new, t));
  check(history, expected_history, new_mesh, element_id, old_mesh, element);
  time_step_id = TimeStepId{true, 0, slab.start() + slab.duration() / 4};
  t = time_step_id.substep_time();
  history.insert(time_step_id, make_vars(x_old, t), make_dt_vars(x_old, t));
  expected_history.insert(time_step_id, make_vars(x_new, t),
                          make_dt_vars(x_new, t));
  check(history, expected_history, new_mesh, element_id, old_mesh, element);
  const auto step_time = history.back().time_step_id.step_time();
  const auto step_size = slab.duration() / 4;
  time_step_id =
      TimeStepId{true, 0,         step_time,
                 1,    step_size, (step_time + slab.duration() / 4).value()};
  t = time_step_id.substep_time();
  history.insert(time_step_id, make_vars(x_old, t), make_dt_vars(x_old, t));
  expected_history.insert(time_step_id, make_vars(x_new, t),
                          make_dt_vars(x_new, t));
  check(history, expected_history, new_mesh, element_id, old_mesh, element);
}

struct Metavariables {
  static constexpr size_t volume_dim = 3;
  struct system {
    using variables_tag =
        Tags::Variables<tmpl::list<TestHelpers::Tags::Vector<DataVector>>>;
  };
};

using variables_tag = Metavariables::system::variables_tag;
using dt_variables_tag = db::add_tag_prefix<Tags::dt, variables_tag>;

template <size_t Label>
struct HistoryEntry : db::SimpleTag {
  using type = variables_tag::type;
};

template <size_t Label>
struct HistoryDeriv : db::SimpleTag {
  using type = dt_variables_tag::type;
};

using ElementData =
    tuples::TaggedTuple<domain::Tags::Element<3>, domain::Tags::Mesh<3>,
                        Tags::HistoryEvolvedVariables<variables_tag>,
                        HistoryEntry<0>, HistoryDeriv<0>, HistoryDeriv<1>>;

ElementData element_data(gsl::not_null<std::mt19937*> gen,
                         const ElementId<3>& element_id, const Mesh<3>& mesh,
                         const TimeStepId& time_step_id0,
                         const TimeStepId& time_step_id1) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  auto value0 = make_with_random_values<variables_tag::type>(
      gen, make_not_null(&dist), mesh.number_of_grid_points());
  auto deriv0 = make_with_random_values<dt_variables_tag::type>(
      gen, make_not_null(&dist), mesh.number_of_grid_points());
  auto deriv1 = make_with_random_values<dt_variables_tag::type>(
      gen, make_not_null(&dist), mesh.number_of_grid_points());
  TimeSteppers::History<variables_tag::type> history(4);
  history.insert(time_step_id0, value0, deriv0);
  history.insert(time_step_id1, decltype(history)::no_value, deriv1);
  return {Element<3>(element_id, {}), mesh,
          std::move(history),         std::move(value0),
          std::move(deriv0),          std::move(deriv1)};
}

void compare_p_refine() {
  MAKE_GENERATOR(gen);
  const Slab slab(0.0, 1.0);
  const TimeStepId time_step_id0(true, 1, slab.start());
  const TimeStepId time_step_id1 =
      time_step_id0.next_substep(slab.duration(), 0.5);
  const ElementId<3> element_id(3, {});
  const Mesh<3> old_mesh(4, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto);
  const Mesh<3> new_mesh(5, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto);

  const auto old_data =
      element_data(&gen, element_id, old_mesh, time_step_id0, time_step_id1);
  auto box = db::create<db::AddSimpleTags<
      Parallel::Tags::ArrayIndexImpl<ElementId<3>>, domain::Tags::Element<3>,
      domain::Tags::Mesh<3>, dt_variables_tag,
      Tags::HistoryEvolvedVariables<variables_tag>, HistoryEntry<0>,
      HistoryDeriv<0>, HistoryDeriv<1>>>(
      element_id, get<domain::Tags::Element<3>>(old_data), new_mesh,
      dt_variables_tag::type{},
      get<Tags::HistoryEvolvedVariables<variables_tag>>(old_data),
      get<HistoryEntry<0>>(old_data), get<HistoryDeriv<0>>(old_data),
      get<HistoryDeriv<1>>(old_data));

  // Compare with the Variables projector
  db::mutate_apply<Initialization::ProjectTimeStepperHistory<Metavariables>>(
      make_not_null(&box),
      std::pair(old_mesh, get<domain::Tags::Element<3>>(old_data)));
  db::mutate_apply<amr::projectors::ProjectVariables<
      3, HistoryEntry<0>, HistoryDeriv<0>, HistoryDeriv<1>>>(
      make_not_null(&box),
      std::pair(old_mesh, get<domain::Tags::Element<3>>(old_data)));

  CHECK(db::get<dt_variables_tag>(box).number_of_grid_points() ==
        new_mesh.number_of_grid_points());
  const auto& history =
      db::get<Tags::HistoryEvolvedVariables<variables_tag>>(box);
  CHECK(history.integration_order() == 4);
  CHECK_VARIABLES_APPROX(history[time_step_id0].value.value(),
                         db::get<HistoryEntry<0>>(box));
  CHECK_VARIABLES_APPROX(history[time_step_id0].derivative,
                         db::get<HistoryDeriv<0>>(box));
  CHECK(not history[time_step_id1].value.has_value());
  CHECK_VARIABLES_APPROX(history[time_step_id1].derivative,
                         db::get<HistoryDeriv<1>>(box));
}

void compare_h_refine() {
  MAKE_GENERATOR(gen);
  const Slab slab(0.0, 1.0);
  const TimeStepId time_step_id0(true, 1, slab.start());
  const TimeStepId time_step_id1 =
      time_step_id0.next_substep(slab.duration(), 0.5);
  const ElementId<3> parent_id(3, {});
  const ElementId<3> child0_id = parent_id.id_of_child(1, Side::Lower);
  const ElementId<3> child1_id = parent_id.id_of_child(1, Side::Upper);
  const Mesh<3> mesh(4, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);

  const auto parent_data =
      element_data(&gen, parent_id, mesh, time_step_id0, time_step_id1);
  const auto child0_data =
      element_data(&gen, child0_id, mesh, time_step_id0, time_step_id1);
  const auto child1_data =
      element_data(&gen, child1_id, mesh, time_step_id0, time_step_id1);

  {
    auto box = db::create<db::AddSimpleTags<
        Parallel::Tags::ArrayIndexImpl<ElementId<3>>, domain::Tags::Element<3>,
        domain::Tags::Mesh<3>, dt_variables_tag,
        Tags::HistoryEvolvedVariables<variables_tag>, HistoryEntry<0>,
        HistoryDeriv<0>, HistoryDeriv<1>>>(
        child0_id, get<domain::Tags::Element<3>>(child0_data), mesh,
        dt_variables_tag::type{},
        Tags::HistoryEvolvedVariables<variables_tag>::type{},
        HistoryEntry<0>::type{}, HistoryDeriv<0>::type{},
        HistoryDeriv<1>::type{});

    // Compare with the Variables projector
    db::mutate_apply<Initialization::ProjectTimeStepperHistory<Metavariables>>(
        make_not_null(&box), parent_data);
    db::mutate_apply<amr::projectors::ProjectVariables<
        3, HistoryEntry<0>, HistoryDeriv<0>, HistoryDeriv<1>>>(
        make_not_null(&box), parent_data);

    CHECK(db::get<dt_variables_tag>(box).number_of_grid_points() ==
          mesh.number_of_grid_points());
    const auto& history =
        db::get<Tags::HistoryEvolvedVariables<variables_tag>>(box);
    CHECK(history.integration_order() == 4);
    CHECK_VARIABLES_APPROX(history[time_step_id0].value.value(),
                           db::get<HistoryEntry<0>>(box));
    CHECK_VARIABLES_APPROX(history[time_step_id0].derivative,
                           db::get<HistoryDeriv<0>>(box));
    CHECK(not history[time_step_id1].value.has_value());
    CHECK_VARIABLES_APPROX(history[time_step_id1].derivative,
                           db::get<HistoryDeriv<1>>(box));
  }

  {
    auto box = db::create<db::AddSimpleTags<
        Parallel::Tags::ArrayIndexImpl<ElementId<3>>, domain::Tags::Element<3>,
        domain::Tags::Mesh<3>, dt_variables_tag,
        Tags::HistoryEvolvedVariables<variables_tag>, HistoryEntry<0>,
        HistoryDeriv<0>, HistoryDeriv<1>>>(
        parent_id, get<domain::Tags::Element<3>>(parent_data), mesh,
        dt_variables_tag::type{},
        Tags::HistoryEvolvedVariables<variables_tag>::type{},
        HistoryEntry<0>::type{}, HistoryDeriv<0>::type{},
        HistoryDeriv<1>::type{});

    std::unordered_map<ElementId<3>, ElementData> children_data{};
    children_data.emplace(child0_id, child0_data);
    children_data.emplace(child1_id, child1_data);

    // Compare with the Variables projector
    db::mutate_apply<Initialization::ProjectTimeStepperHistory<Metavariables>>(
        make_not_null(&box), children_data);
    db::mutate_apply<amr::projectors::ProjectVariables<
        3, HistoryEntry<0>, HistoryDeriv<0>, HistoryDeriv<1>>>(
        make_not_null(&box), children_data);

    CHECK(db::get<dt_variables_tag>(box).number_of_grid_points() ==
          mesh.number_of_grid_points());
    const auto& history =
        db::get<Tags::HistoryEvolvedVariables<variables_tag>>(box);
    CHECK(history.integration_order() == 4);
    CHECK_VARIABLES_APPROX(history[time_step_id0].value.value(),
                           db::get<HistoryEntry<0>>(box));
    CHECK_VARIABLES_APPROX(history[time_step_id0].derivative,
                           db::get<HistoryDeriv<0>>(box));
    CHECK(not history[time_step_id1].value.has_value());
    CHECK_VARIABLES_APPROX(history[time_step_id1].derivative,
                           db::get<HistoryDeriv<1>>(box));
  }
}

void compare_nonuniform_join() {
  MAKE_GENERATOR(gen);
  const Slab slab(0.0, 1.0);
  const TimeStepId time_step_id0(true, 1, slab.start());
  const TimeStepId time_step_id1 =
      time_step_id0.next_substep(slab.duration(), 0.5);
  const ElementId<3> parent_id(3, {});
  const ElementId<3> child0_id = parent_id.id_of_child(1, Side::Lower);
  const ElementId<3> child1_id = parent_id.id_of_child(1, Side::Upper);
  const Mesh<3> child0_mesh(4, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto);
  const Mesh<3> child1_mesh(3, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto);
  const auto& parent_mesh = child0_mesh;

  const auto parent_data =
      element_data(&gen, parent_id, parent_mesh, time_step_id0, time_step_id1);
  const auto child0_data =
      element_data(&gen, child0_id, child0_mesh, time_step_id0, time_step_id1);
  const auto child1_data =
      element_data(&gen, child1_id, child1_mesh, time_step_id0, time_step_id1);

  auto box = db::create<db::AddSimpleTags<
      Parallel::Tags::ArrayIndexImpl<ElementId<3>>, domain::Tags::Element<3>,
      domain::Tags::Mesh<3>, dt_variables_tag,
      Tags::HistoryEvolvedVariables<variables_tag>, HistoryEntry<0>,
      HistoryDeriv<0>, HistoryDeriv<1>>>(
      parent_id, get<domain::Tags::Element<3>>(parent_data), parent_mesh,
      dt_variables_tag::type{},
      Tags::HistoryEvolvedVariables<variables_tag>::type{},
      HistoryEntry<0>::type{}, HistoryDeriv<0>::type{},
      HistoryDeriv<1>::type{});

  std::unordered_map<ElementId<3>, ElementData> children_data{};
  children_data.emplace(child0_id, child0_data);
  children_data.emplace(child1_id, child1_data);

  // Compare with the Variables projector
  db::mutate_apply<Initialization::ProjectTimeStepperHistory<Metavariables>>(
      make_not_null(&box), children_data);
  db::mutate_apply<amr::projectors::ProjectVariables<
      3, HistoryEntry<0>, HistoryDeriv<0>, HistoryDeriv<1>>>(
      make_not_null(&box), children_data);

  CHECK(db::get<dt_variables_tag>(box).number_of_grid_points() ==
        parent_mesh.number_of_grid_points());
  const auto& history =
      db::get<Tags::HistoryEvolvedVariables<variables_tag>>(box);
  CHECK(history.integration_order() == 4);
  CHECK_VARIABLES_APPROX(history[time_step_id0].value.value(),
                         db::get<HistoryEntry<0>>(box));
  CHECK_VARIABLES_APPROX(history[time_step_id0].derivative,
                         db::get<HistoryDeriv<0>>(box));
  CHECK(not history[time_step_id1].value.has_value());
  CHECK_VARIABLES_APPROX(history[time_step_id1].derivative,
                         db::get<HistoryDeriv<1>>(box));
}

SPECTRE_TEST_CASE("Unit.Evolution.Initialization.TimeStepperHistory",
                  "[Evolution][Unit]") {
  test_initialization<1>();
  test_initialization<2>();
  test_initialization<3>();
  static_assert(tt::assert_conforms_to_v<
                Initialization::ProjectTimeStepperHistory<TestMetavariables<1>>,
                amr::protocols::Projector>);
  test_p_refine<1>();
  test_p_refine<2>();
  test_p_refine<3>();

  compare_p_refine();
  compare_h_refine();
  compare_nonuniform_join();
}
}  // namespace time_stepper_history
}  // namespace
