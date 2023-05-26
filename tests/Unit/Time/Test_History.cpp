// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/TestHelpers.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

namespace {
constexpr size_t num_points = 3;

template <typename T>
const T& as_const(const T& t) {
  return t;
}

struct VarTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

void test_untyped_step_record() {
  const Slab slab(0.0, 1.0);
  const TimeStepId time_a(true, 0, slab.start());
  const TimeStepId time_b(true, 0, slab.start(), 1, slab.duration(),
                          slab.start().value());
  const TimeSteppers::UntypedStepRecord<double> a{time_a, std::optional{5.0},
                                                  6.0};
  auto b = a;
  CHECK(a == b);
  CHECK_FALSE(a != b);

  b.time_step_id = time_b;
  CHECK(a != b);
  CHECK_FALSE(a == b);

  b = a;
  b.value = std::nullopt;
  CHECK(a != b);
  CHECK_FALSE(a == b);

  b = a;
  b.derivative = -10.0;
  CHECK(a != b);
  CHECK_FALSE(a == b);
}

template <typename Vars, typename Derivs>
void test_step_record() {
  using Record = TimeSteppers::StepRecord<Vars>;
  static_assert(std::is_same_v<typename Record::DerivVars, Derivs>);

  const Slab slab(0.0, 1.0);
  const TimeStepId time_a(true, 0, slab.start());
  const TimeStepId time_b(true, 0, slab.start(), 1, slab.duration(),
                          slab.start().value());

  const Record a{time_a, std::optional{make_with_value<Vars>(num_points, 2.0)},
                 make_with_value<Derivs>(num_points, 3.0)};
  auto b = a;
  CHECK(a == b);
  CHECK_FALSE(a != b);

  b.time_step_id = time_b;
  CHECK(a != b);
  CHECK_FALSE(a == b);

  b = a;
  b.value = std::nullopt;
  CHECK(a != b);
  CHECK_FALSE(a == b);

  b = a;
  b.derivative *= 2.0;
  CHECK(a != b);
  CHECK_FALSE(a == b);

  test_serialization(a);
}

template <typename Vars, typename Derivs>
void test_history() {
  const auto make_value = [](const double v) {
    return make_with_value<Vars>(num_points, v);
  };
  const auto make_deriv = [](const double v) {
    return make_with_value<Derivs>(num_points, v);
  };

  using History = TimeSteppers::History<Vars>;
  static_assert(std::is_same_v<typename History::DerivVars, Derivs>);
  using UntypedVars = typename History::UntypedVars;
  using ConstUntyped = TimeSteppers::ConstUntypedHistory<UntypedVars>;
  using MutableUntyped = TimeSteppers::MutableUntypedHistory<UntypedVars>;

  const Slab slab(0.0, 1.0);

  History history(5);
  const History& const_history = history;
  CHECK(const_history.integration_order() == 5);
  history.integration_order(4);
  CHECK(const_history.integration_order() == 4);

  // Initial state: [steps] [step_time: substeps] = [] []
  CHECK(const_history.empty());
  history.insert_initial(TimeStepId(true, 0, slab.start()), make_value(-1.0),
                         make_deriv(-10.0));
  // [(0, -1, -10)] []
  CHECK(const_history.size() == 1);
  history.insert_initial(
      TimeStepId(true, -1, slab.start() - Slab(-1.0, 0.0).duration() / 4),
      History::no_value, make_deriv(-20.0));
  // [(-1/4, X, -20), (0, -1, -10)] []
  CHECK(const_history.size() == 2);
  history.insert_initial(
      TimeStepId(true, -1, slab.start() - Slab(-1.0, 0.0).duration() / 2),
      make_value(-3.0), make_deriv(-30.0));
  // [(-1/2, -3, -30), (-1/4, X, -20), (0, -1, -10)] []
  CHECK(const_history.size() == 3);

  CHECK(const_history.max_size() < 20);
  CHECK(const_history.max_size() >= const_history.size());

  CHECK(const_history[0].time_step_id ==
        TimeStepId(true, -1, slab.start() - Slab(-1.0, 0.0).duration() / 2));
  CHECK(const_history[0].value == std::optional{make_value(-3.0)});
  CHECK(const_history[0].derivative == make_deriv(-30.0));
  CHECK(not const_history[1].value.has_value());
  CHECK(const_history[1].derivative == make_deriv(-20.0));
  CHECK(const_history[2].value == std::optional{make_value(-1.0)});
  CHECK(const_history[2].derivative == make_deriv(-10.0));
  CHECK(const_history.latest_value() == *const_history.back().value);
  CHECK(&const_history.latest_value() == &*const_history.back().value);

  history[0].derivative = make_deriv(-300.0);
  // [(-1/2, -3, -300), (-1/4, X, -20), (0, -1, -10)] []
  CHECK(const_history[0].derivative == make_deriv(-300.0));

  history.insert(TimeStepId(true, 0, slab.start() + slab.duration() / 4),
                 make_value(1.0), make_deriv(10.0));
  // [(-1/2, -3, -300), (-1/4, X, -20), (0, -1, -10), (1/4, 1, 10)] []
  CHECK(const_history.size() == 4);
  CHECK(const_history[0].derivative == make_deriv(-300.0));
  CHECK(const_history[3].derivative == make_deriv(10.0));

  CHECK(&const_history[2] == &const_history[const_history[2].time_step_id]);
  history[const_history[2].time_step_id].derivative = make_deriv(-100.0);
  // [(-1/2, -3, -300), (-1/4, X, -20), (0, -1, -100), (1/4, 1, 10)] []
  CHECK(const_history[2].derivative == make_deriv(-100.0));

  [[maybe_unused]] const double* cached_value = nullptr;
  if constexpr (tt::is_a_v<Variables, Vars>) {
    cached_value = const_history[2].value->data();
  }
  history.discard_value(const_history[2].time_step_id);
  // [(-1/2, -3, -300), (-1/4, X, -20), (0, X, -100), (1/4, 1, 10)] []
  CHECK(not const_history[2].value.has_value());
  CHECK(const_history[0].value.has_value());
  CHECK(const_history[3].value.has_value());
  history.insert_in_place(
      TimeStepId(true, 0, slab.start() + slab.duration() / 2),
      [&](const auto v) {
        if constexpr (tt::is_a_v<Variables, Vars>) {
          CHECK(v->data() == cached_value);
        }
        // Avoid overwriting the cached allocation for testing.
        *v = as_const(make_value(2.0));
      },
      [&](const auto d) { *d = as_const(make_deriv(20.0)); });
  // [(-1/2, -3, -300), (-1/4, X, -20), (0, X, -100), (1/4, 1, 10),
  //  (1/2, 2, 20)] []
  CHECK(*const_history.back().value == make_value(2.0));
  CHECK(const_history.back().derivative == make_deriv(20.0));

  {
    const auto check_const_untyped = [&const_history](const auto& untyped) {
      CHECK(untyped.integration_order() == const_history.integration_order());
      CHECK(untyped.size() == const_history.size());
      CHECK(untyped.max_size() == const_history.max_size());
      CHECK(untyped[0].time_step_id == const_history[0].time_step_id);
      CHECK(untyped[0].value.has_value());
      CHECK(*untyped[0].value == *const_history[0].value);
      CHECK(untyped[0].derivative == const_history[0].derivative);
      if constexpr (tt::is_a_v<Variables, Vars>) {
        CHECK(untyped[0].value->data() == const_history[0].value->data());
        CHECK(untyped[0].derivative.data() ==
              const_history[0].derivative.data());
      }
      CHECK(untyped[1].time_step_id == const_history[1].time_step_id);
      CHECK(not untyped[1].value.has_value());
      CHECK(untyped[1].derivative == const_history[1].derivative);
      CHECK(&untyped[1] == &untyped[untyped[1].time_step_id]);
    };

    // Capture rvalue temporaries by const reference to the base class
    // to verify all functionality is correctly declared virtual.
    const ConstUntyped& const_untyped = const_history.untyped();
    check_const_untyped(const_untyped);
    const MutableUntyped& mutable_untyped = history.untyped();
    check_const_untyped(mutable_untyped);
    const auto implicitly_convert =
        [](const ConstUntyped& x) -> const ConstUntyped& { return x; };
    check_const_untyped(implicitly_convert(mutable_untyped));
    mutable_untyped.discard_value(mutable_untyped[3].time_step_id);
    // [(-1/2, -3, -300), (-1/4, X, -20), (0, X, -100), (1/4, X, 10),
    //  (1/2, 2, 20)] []
    CHECK(not const_history[3].value.has_value());
    CHECK(not mutable_untyped[3].value.has_value());
  }

  [[maybe_unused]] const double* cached_deriv = nullptr;
  if constexpr (tt::is_a_v<Variables, Vars>) {
    cached_value = const_history.front().value->data();
    cached_deriv = const_history.front().derivative.data();
  }
  history.pop_front();
  // [(-1/4, X, -20), (0, X, -100), (1/4, X, 10), (1/2, 2, 20)] []
  CHECK(const_history.size() == 4);
  CHECK(const_history.front().derivative == make_deriv(-20.0));
  history.insert_in_place(
      TimeStepId(true, 0, slab.start() + slab.duration() * 3 / 4),
      [&](const auto v) {
        if constexpr (tt::is_a_v<Variables, Vars>) {
          CHECK(v->data() == cached_value);
        }
        *v = as_const(make_value(3.0));
      },
      [&](const auto d) {
        if constexpr (tt::is_a_v<Variables, Vars>) {
          CHECK(d->data() == cached_deriv);
        }
        *d = as_const(make_deriv(30.0));
      });
  // [(-1/4, X, -20), (0, X, -100), (1/4, X, 10), (1/2, 2, 20), (3/4, 3, 30)]
  // []

  history.pop_front();
  // [(0, X, -100), (1/4, X, 10), (1/2, 2, 20), (3/4, 3, 30)] []

  {
    const MutableUntyped& untyped = history.untyped();
    untyped.pop_front();
    // [(1/4, X, 10), (1/2, 2, 20), (3/4, 3, 30)] []
    CHECK(const_history.size() == 3);
    CHECK(const_history.front().derivative == make_deriv(10.0));
    // Verify that `untyped` reflects the new state.
    CHECK(untyped.size() == 3);
    CHECK(untyped.front().derivative == make_deriv(10.0));
  }

  CHECK(const_history.at_step_start());
  CHECK(static_cast<const ConstUntyped&>(const_history.untyped())
            .at_step_start());
  history.undo_latest();
  // [(1/4, X, 10), (1/2, 2, 20)] []
  CHECK(const_history.size() == 2);
  CHECK(const_history.front().derivative == make_deriv(10.0));
  CHECK(const_history.at_step_start());
  CHECK(static_cast<const ConstUntyped&>(const_history.untyped())
            .at_step_start());

  // Now test substeps

  CHECK(as_const(const_history.substeps()).empty());
  CHECK(as_const(history.substeps()).empty());

  const auto step_time = const_history.back().time_step_id.step_time();
  const auto step_size = slab.duration() / 4;
  history.insert(TimeStepId(true, 0, step_time, 1, step_size,
                            (step_time + slab.duration() / 4).value()),
                 make_value(4.0), make_deriv(40.0));
  // [(1/4, X, 10), (1/2, 2, 20)] [1/2: (1, 4, 40)]
  CHECK(const_history.back().derivative == make_deriv(20.0));
  CHECK(as_const(const_history.substeps()).size() == 1);
  CHECK(as_const(const_history.substeps())[0].derivative == make_deriv(40.0));
  CHECK(as_const(history.substeps()).size() == 1);
  as_const(history.substeps())[0].derivative = make_deriv(400.0);
  // [(1/4, X, 10), (1/2, 2, 20)] [1/2: (1, 4, 400)]
  CHECK(as_const(const_history.substeps())[0].derivative == make_deriv(400.0));

  CHECK(not const_history.at_step_start());
  CHECK(not static_cast<const ConstUntyped&>(const_history.untyped())
                .at_step_start());
  history.undo_latest();
  // [(1/4, X, 10), (1/2, 2, 20)] []
  CHECK(as_const(const_history.substeps()).empty());
  CHECK(const_history.at_step_start());
  CHECK(static_cast<const ConstUntyped&>(const_history.untyped())
            .at_step_start());

  history.insert(TimeStepId(true, 0, step_time, 1, step_size,
                            (step_time + slab.duration() / 4).value()),
                 make_value(4.0), make_deriv(40.0));
  // [(1/4, X, 10), (1/2, 2, 20)] [1/2: (1, 4, 40)]
  history.insert(TimeStepId(true, 0, step_time, 2, step_size,
                            (step_time + slab.duration() / 4).value()),
                 make_value(5.0), make_deriv(50.0));
  // [(1/4, X, 10), (1/2, 2, 20)] [1/2: (1, 4, 40), (2, 5, 50)]

  {
    const auto expected_latest = *const_history.substeps().back().value;
    CHECK(const_history.latest_value() == expected_latest);
    CHECK(&const_history.latest_value() ==
          &*const_history.substeps().back().value);
    history.discard_value(const_history.substeps().back().time_step_id);
    // [(1/4, X, 10), (1/2, 2, 20)] [1/2: (1, 4, 40), (2, X, 50)]
    CHECK(not const_history.substeps().back().value.has_value());
    CHECK(const_history.latest_value() == expected_latest);
  }

  {
    const auto typed = const_history.substeps();
    const ConstUntyped& untyped_interface = const_history.untyped();
    const auto untyped = untyped_interface.substeps();
    CHECK(untyped.size() == typed.size());
    CHECK(untyped.max_size() == typed.max_size());
    CHECK(typed.max_size() < 20);
    CHECK(typed.max_size() >= typed.size());

    CHECK(untyped[0].time_step_id == typed[0].time_step_id);
    CHECK(typed[0].value.has_value());
    CHECK(untyped[0].value.has_value());
    CHECK(*untyped[0].value == *typed[0].value);
    CHECK(untyped[0].derivative == typed[0].derivative);
    CHECK(untyped[1].time_step_id == typed[1].time_step_id);
    CHECK(not typed[1].value.has_value());
    CHECK(not untyped[1].value.has_value());
    CHECK(untyped[1].derivative == typed[1].derivative);
  }

  static_cast<const MutableUntyped&>(history.untyped())
      .discard_value(const_history.substeps().front().time_step_id);
  // [(1/4, X, 10), (1/2, 2, 20)] [1/2: (1, X, 40), (2, X, 50)]
  CHECK(not const_history.substeps().front().value.has_value());

  CHECK(not const_history.at_step_start());
  CHECK(not static_cast<const ConstUntyped&>(const_history.untyped())
                .at_step_start());
  history.undo_latest();
  // [(1/4, X, 10), (1/2, 2, 20)] [1/2: (1, X, 40)]
  CHECK(const_history.substeps().size() == 1);
  CHECK(const_history.substeps().front().derivative == make_deriv(40.0));
  CHECK(not const_history.at_step_start());
  CHECK(not static_cast<const ConstUntyped&>(const_history.untyped())
                .at_step_start());

  history.clear_substeps();
  // [(1/4, X, 10), (1/2, 2, 20)] []
  CHECK(const_history.substeps().empty());
  CHECK(const_history.size() == 2);
  CHECK(const_history.at_step_start());
  CHECK(static_cast<const ConstUntyped&>(const_history.untyped())
            .at_step_start());

  history.insert(TimeStepId(true, 0, step_time, 1, step_size,
                            (step_time + slab.duration() / 4).value()),
                 make_value(4.0), make_deriv(40.0));
  // [(1/4, X, 10), (1/2, 2, 20)] [1/2: (1, 4, 40)]
  history.insert(TimeStepId(true, 0, step_time, 2, step_size,
                            (step_time + slab.duration() / 4).value()),
                 make_value(5.0), make_deriv(50.0));
  // [(1/4, X, 10), (1/2, 2, 20)] [(1, 4, 40), (2, 5, 50)]
  history.insert(TimeStepId(true, 1, slab.end()), make_value(6.0),
                 make_deriv(60.0));
  // [(1/4, X, 10), (1/2, 2, 20), (1, 6, 60)] [1/2: (1, 4, 40), (2, 5, 50)]

  CHECK(const_history.size() == 3);
  CHECK(as_const(const_history.substeps()).size() == 2);
  CHECK(const_history.at_step_start());
  CHECK(static_cast<const ConstUntyped&>(const_history.untyped())
            .at_step_start());

  history.undo_latest();
  // [(1/4, X, 10), (1/2, 2, 20)] [1/2: (1, 4, 40), (2, 5, 50)]

  CHECK(const_history.size() == 2);
  CHECK(as_const(const_history.substeps()).size() == 2);
  CHECK(not const_history.at_step_start());
  CHECK(not static_cast<const ConstUntyped&>(const_history.untyped())
                .at_step_start());

  const auto step_time2 = slab.start() + slab.duration() * 3 / 4;
  ASSERT(step_time2 - step_time == step_size, "Test bug");
  const auto step_size2 = slab.duration() / 4;
  history.insert(TimeStepId(true, 1, step_time2), make_value(6.0),
                 make_deriv(60.0));
  // [(1/4, X, 10), (1/2, 2, 20), (3/4, 6, 60)] [1/2: (1, 4, 40), (2, 5, 50)]

  static_cast<const MutableUntyped&>(history.untyped()).clear_substeps();
  // [(1/4, X, 10), (1/2, 2, 20), (3/4, 6, 60)] []

  CHECK(const_history.size() == 3);
  CHECK(as_const(const_history.substeps()).empty());
  CHECK(const_history.at_step_start());
  CHECK(static_cast<const ConstUntyped&>(const_history.untyped())
            .at_step_start());

  history.insert(
      TimeStepId(true, 1, step_time2, 1, step_size2, slab.end().value()),
      make_value(7.0), make_deriv(70.0));
  // [(1/4, X, 10), (1/2, 2, 20), (3/4, 6, 60)] [3/4: (1, 7, 70)]

  history.shrink_to_fit();
  CHECK(const_history.size() == 3);
  CHECK(as_const(const_history.substeps()).size() == 1);

  if constexpr (tt::is_a_v<Variables, Vars>) {
    cached_value = const_history.substeps().back().value->data();
    cached_deriv = const_history.substeps().back().derivative.data();
  }
  history.undo_latest();
  // [(1/4, X, 10), (1/2, 2, 20), (3/4, 6, 60)] []
  CHECK(const_history.size() == 3);
  CHECK(as_const(const_history.substeps()).empty());
  history.insert_in_place(
      TimeStepId(true, 1, step_time2, 1, step_size2, slab.end().value()),
      History::no_value, [&](const auto d) {
        if constexpr (tt::is_a_v<Variables, Vars>) {
          CHECK(d->data() == cached_deriv);
        }
        *d = as_const(make_deriv(70.0));
      });
  // [(1/4, X, 10), (1/2, 2, 20), (3/4, 6, 60)] [3/4: (1, X, 70)]
  history.insert(
      TimeStepId(true, 1, step_time2, 2, step_size2, slab.end().value()),
      History::no_value, make_deriv(80.0));
  // [(1/4, X, 10), (1/2, 2, 20), (3/4, 6, 60)] [3/4: (1, X, 70), (1, X, 80)]
  history.insert_in_place(
      TimeStepId(true, 1, step_time2, 3, step_size2, slab.end().value()),
      [&](const auto v) {
        if constexpr (tt::is_a_v<Variables, Vars>) {
          CHECK(v->data() == cached_value);
        }
        *v = as_const(make_value(9.0));
      },
      [&](const auto d) { *d = as_const(make_deriv(90.0)); });
  // [(1/4, X, 10), (1/2, 2, 20), (3/4, 6, 60)]
  //     [3/4: (1, X, 70), (1, X, 80), (1, 9, 90)]

  history.undo_latest();
  // [(1/4, X, 10), (1/2, 2, 20), (3/4, 6, 60)] [3/4: (1, X, 70), (1, X, 80)]
  history.shrink_to_fit();

  history.insert_in_place(
      TimeStepId(true, 1, step_time2, 3, step_size2, slab.end().value()),
      [&](const auto v) {
        if constexpr (tt::is_a_v<Variables, Vars>) {
          CHECK(v->size() == 0);
        }
        *v = as_const(make_value(9.0));
      },
      [&](const auto d) {
        if constexpr (tt::is_a_v<Variables, Vars>) {
          CHECK(d->size() == 0);
        }
        *d = as_const(make_deriv(90.0));
      });
  // [(1/4, X, 10), (1/2, 2, 20), (3/4, 6, 60)]
  //     [3/4: (1, X, 70), (1, X, 80), (1, 8, 80)]

  const auto check_copy = [&history](const auto& copy) {
    CHECK(copy.integration_order() == history.integration_order());
    CHECK(copy.size() == history.size());
    CHECK(copy.front() == history.front());
    CHECK(copy.substeps().size() == history.substeps().size());
    CHECK(copy.substeps().front() == history.substeps().front());
  };
  {
    const auto copy_constructed = history;
    check_copy(copy_constructed);
    History copy_assigned{};
    copy_assigned = history;
    check_copy(copy_assigned);
  }
  const auto copy = serialize_and_deserialize(history);
  check_copy(copy);

  history.map_entries([](const auto entry) { *entry *= 10.0; });
  // [(1/4, X, 100), (1/2, 20, 200), (3/4, 60, 600)]
  // [3/4: (1, X, 700), (1, X, 800), (1, 9, 900)]
  CHECK(history.size() == 3);
  for (size_t i = 0; i < history.size(); ++i) {
    CHECK(history[i].time_step_id == copy[i].time_step_id);
    CHECK(history[i].value.has_value() == copy[i].value.has_value());
    if (history[i].value.has_value()) {
      CHECK(*history[i].value == Vars(10.0 * *copy[i].value));
    }
    CHECK(history[i].derivative == Derivs(10.0 * copy[i].derivative));
  }
  CHECK(history.substeps().size() == 3);
  for (size_t i = 0; i < history.substeps().size(); ++i) {
    CHECK(history.substeps()[i].time_step_id ==
          copy.substeps()[i].time_step_id);
    CHECK(history.substeps()[i].value.has_value() ==
          copy.substeps()[i].value.has_value());
    if (history.substeps()[i].value.has_value()) {
      CHECK(*history.substeps()[i].value ==
            Vars(10.0 * *copy.substeps()[i].value));
    }
    CHECK(history.substeps()[i].derivative ==
          Derivs(10.0 * copy.substeps()[i].derivative));
  }

  history.clear();
  // [] []
  CHECK(history.integration_order() == copy.integration_order());
  CHECK(history.empty());
  CHECK(history.substeps().empty());

  {
    History a{2};
    History b{3};
    History c = b;
    c.insert(TimeStepId(true, 0, slab.start()), make_value(1.0),
             make_deriv(10.0));
    History d = c;
    d.insert(TimeStepId(true, 0, slab.start(), 1, slab.duration(),
                        slab.start().value()),
             make_value(2.0), make_deriv(20.0));

    CHECK(a == a);
    CHECK_FALSE(a != a);
    CHECK(b == b);
    CHECK_FALSE(b != b);
    CHECK(c == c);
    CHECK_FALSE(c != c);
    CHECK(d == d);
    CHECK_FALSE(d != d);
    CHECK(a != b);
    CHECK_FALSE(a == b);
    CHECK(b != c);
    CHECK_FALSE(b == c);
    CHECK(c != d);
    CHECK_FALSE(c == d);
  }
}

void test_history_assertions() {
#ifdef SPECTRE_DEBUG
  const Slab slab(0.0, 1.0);
  const auto slab_half = slab.start() + slab.duration() / 2;
  const auto step = slab_half - slab.start();

  // Insertion errors
  {
    TimeSteppers::History<double> history(1);
    history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
    CHECK_THROWS_WITH(
        history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0),
        Catch::Contains("must be later"));
  }
  {
    TimeSteppers::History<double> history(1);
    history.insert_initial(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
    CHECK_THROWS_WITH(
        history.insert_initial(TimeStepId(true, 0, slab.start()), 0.0, 0.0),
        Catch::Contains("must be earlier"));
  }
  {
    TimeSteppers::History<double> history(1);
    CHECK_THROWS_WITH(
        history.insert(
            TimeStepId(true, 0, slab.start(), 1, step, slab.start().value()),
            0.0, 0.0),
        Catch::Contains("Cannot insert substep into empty history"));
  }
  {
    TimeSteppers::History<double> history(1);
    history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
    CHECK_THROWS_WITH(history.insert(TimeStepId(true, 0, slab.start(), 2, step,
                                                slab.start().value()),
                                     0.0, 0.0),
                      Catch::Contains("Cannot insert substep 2 following 0"));
  }
  {
    TimeSteppers::History<double> history(1);
    history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
    CHECK_THROWS_WITH(history.insert(TimeStepId(true, 0, slab_half, 1, step,
                                                slab.end().value()),
                                     0.0, 0.0),
                      Catch::Contains("Cannot insert substep ") and
                          Catch::Contains(" of different step "));
  }
  {
    TimeSteppers::History<double> history(1);
    history.insert_initial(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
    CHECK_THROWS_WITH(
        history.insert_initial(
            TimeStepId(true, 0, slab.start(), 2, step, slab.start().value()),
            0.0, 0.0),
        Catch::Contains("Cannot use insert_initial for substeps"));
  }
  {
    TimeSteppers::History<double> history(1);
    // +1 because 0 isn't a substep.
    for (size_t i = 0; i < history.substeps().max_size() + 1; ++i) {
      history.insert(
          TimeStepId(true, 0, slab.start(), i, step, slab.start().value()), 0.0,
          0.0);
    }
    CHECK_THROWS_WITH(
        history.insert(
            TimeStepId(true, 0, slab.start(), history.substeps().max_size() + 1,
                       step, slab.start().value()),
            0.0, 0.0),
        Catch::Contains(
            "Cannot insert new substep because the History is full"));
  }

  // Indexing errors
  {
    TimeSteppers::History<double> history(1);
    history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
    CHECK_THROWS_WITH(std::as_const(history)[TimeStepId(true, 1, slab.start())],
                      Catch::Contains("not present"));
  }
  {
    TimeSteppers::History<double> history(1);
    history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
    CHECK_THROWS_WITH(history[TimeStepId(true, 1, slab.start())],
                      Catch::Contains("not present"));
  }
  {
    TimeSteppers::History<double> history(1);
    history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
    CHECK_THROWS_WITH(
        std::as_const(history)[TimeStepId(true, 0, slab.start(), 1, step,
                                          slab.start().value())],
        Catch::Contains("not present"));
  }
  {
    TimeSteppers::History<double> history(1);
    history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
    CHECK_THROWS_WITH(history[TimeStepId(true, 0, slab.start(), 1, step,
                                         slab.start().value())],
                      Catch::Contains("not present"));
  }
  {
    TimeSteppers::History<double> history(1);
    history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
    history.insert(
        TimeStepId(true, 0, slab.start(), 1, step, slab.end().value()), 0.0,
        0.0);
    CHECK_THROWS_WITH(
        history[TimeStepId(true, 0, slab_half, 1, step, slab.end().value())],
        Catch::Contains("not present"));
  }
  {
    const TimeSteppers::History<double> history(1);
    CHECK_THROWS_WITH(history.substeps()[0],
                      Catch::Contains("Requested substep 0 but only have 0"));
  }
  {
    const TimeSteppers::History<double> history(1);
    CHECK_THROWS_WITH(history.substeps()[1],
                      Catch::Contains("Requested substep 1 but only have 0"));
  }
  {
    TimeSteppers::History<double> history(1);
    CHECK_THROWS_WITH(history.substeps()[0],
                      Catch::Contains("Requested substep 0 but only have 0"));
  }

  // Untyped indexing errors
  {
    const TimeSteppers::History<double> history(1);
    CHECK_THROWS_WITH(history.untyped()[0],
                      Catch::Contains("Requested step 0 but only have 0"));
  }
  {
    TimeSteppers::History<double> history(1);
    history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
    CHECK_THROWS_WITH(history.untyped()[TimeStepId(true, 1, slab.start())],
                      Catch::Contains("not present"));
  }
  {
    TimeSteppers::History<double> history(1);
    history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
    CHECK_THROWS_WITH(history.untyped()[TimeStepId(true, 0, slab.start(), 1,
                                                   step, slab.start().value())],
                      Catch::Contains("not present"));
  }
  {
    TimeSteppers::History<double> history(1);
    history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
    history.insert(
        TimeStepId(true, 0, slab.start(), 1, step, slab.end().value()), 0.0,
        0.0);
    CHECK_THROWS_WITH(history.untyped()[TimeStepId(true, 0, slab_half, 1, step,
                                                   slab.end().value())],
                      Catch::Contains("not present"));
  }
  {
    const TimeSteppers::History<double> history(1);
    CHECK_THROWS_WITH(history.untyped().substeps()[0],
                      Catch::Contains("Requested substep 0 but only have 0"));
  }

  // Record removal errors
  {
    TimeSteppers::History<double> history(1);
    CHECK_THROWS_WITH(history.pop_front(), Catch::Contains("History is empty"));
  }
  {
    TimeSteppers::History<double> history(1);
    CHECK_THROWS_WITH(history.undo_latest(),
                      Catch::Contains("History is empty"));
  }
  {
    TimeSteppers::History<double> history(1);
    history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
    history.insert(
        TimeStepId(true, 0, slab.start(), 1, step, slab.start().value()), 0.0,
        0.0);
    CHECK_THROWS_WITH(history.pop_front(),
                      Catch::Contains("Cannot remove a step with substeps.  "
                                      "Call clear_substeps() first"));
  }

  // Value discarding errors
  {
    TimeSteppers::History<double> history(1);
    history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
    CHECK_THROWS_WITH(history.discard_value(TimeStepId(true, 1, slab.start())),
                      Catch::Contains("not present"));
  }
  {
    TimeSteppers::History<double> history(1);
    history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
    CHECK_THROWS_WITH(
        history.discard_value(
            TimeStepId(true, 0, slab.start(), 1, step, slab.start().value())),
        Catch::Contains("not present"));
  }

  // Untyped value discarding errors
  {
    TimeSteppers::History<double> history(1);
    history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
    CHECK_THROWS_WITH(
        history.untyped().discard_value(TimeStepId(true, 1, slab.start())),
        Catch::Contains("not present"));
  }
  {
    TimeSteppers::History<double> history(1);
    history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
    CHECK_THROWS_WITH(
        history.untyped().discard_value(
            TimeStepId(true, 0, slab.start(), 1, step, slab.start().value())),
        Catch::Contains("not present"));
  }

  // latest_value errors
  {
    TimeSteppers::History<double> history(1);
    history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
    history.discard_value(TimeStepId(true, 0, slab.start()));
    history.insert(TimeStepId(true, 1, slab.start()), 0.0, 0.0);
    history.undo_latest();
    CHECK_THROWS_WITH(history.latest_value(),
                      Catch::Contains("Latest value unavailable"));
  }
  {
    TimeSteppers::History<double> history(1);
    history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
    history.discard_value(TimeStepId(true, 0, slab.start()));
    history.insert(TimeStepId(true, 1, slab.start()), 0.0, 0.0);
    history.discard_value(TimeStepId(true, 1, slab.start()));
    history.undo_latest();
    CHECK_THROWS_WITH(history.latest_value(),
                      Catch::Contains("Latest value unavailable"));
  }

#endif  // SPECTRE_DEBUG
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.History", "[Unit][Time]") {
  test_untyped_step_record();

  test_step_record<double, double>();
  test_step_record<Variables<tmpl::list<VarTag>>,
                   Variables<tmpl::list<Tags::dt<VarTag>>>>();

  test_history<double, double>();
  test_history<Variables<tmpl::list<VarTag>>,
               Variables<tmpl::list<Tags::dt<VarTag>>>>();

  test_history_assertions();
}
