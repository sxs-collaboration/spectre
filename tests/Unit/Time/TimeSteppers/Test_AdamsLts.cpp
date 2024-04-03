// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/StdHelpers.hpp"

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <ostream>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Time/ApproximateTime.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsLts.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Rational.hpp"

namespace {
namespace adams_lts = TimeSteppers::adams_lts;

void test_exact_substep_time() {
  const Slab slab(1.2, 3.4);
  const Time quarter = slab.start() + slab.duration() / 4;
  const Time middle = slab.start() + slab.duration() / 2;
  CHECK(adams_lts::exact_substep_time(TimeStepId(true, 1, middle)) == middle);
  CHECK(adams_lts::exact_substep_time(TimeStepId(false, 1, middle)) == middle);
  CHECK(adams_lts::exact_substep_time(TimeStepId(
            true, 1, quarter, 1, slab.duration() / 4, middle.value())) ==
        middle);
  CHECK(adams_lts::exact_substep_time(TimeStepId(
            false, 1, middle, 1, -slab.duration() / 4, quarter.value())) ==
        quarter);
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      adams_lts::exact_substep_time(TimeStepId(
          true, 1, slab.start(), 1, slab.duration(), middle.value())),
      Catch::Matchers::ContainsSubstring("Substep not at expected time"));
#endif  // SPECTRE_DEBUG
}

void test_lts_coefficients_struct() {
  const Slab slab(0.0, 1.0);
  const TimeStepId id1(true, 0, slab.start());
  const TimeStepId id2 = id1.next_substep(slab.duration() / 2, 1.0);
  const TimeStepId id3 = id2.next_step(slab.duration() / 2);

  const adams_lts::LtsCoefficients coefs1{
      {id1, id1, 2.0}, {id1, id2, 3.0}, {id1, id3, 4.0}};
  const adams_lts::LtsCoefficients coefs2{
      {id1, id1, 5.0}, {id1, id3, 7.0}, {id2, id1, 9.0}};

  const adams_lts::LtsCoefficients sum{
      {id1, id1, 7.0}, {id1, id2, 3.0}, {id1, id3, 11.0}, {id2, id1, 9.0}};
  const adams_lts::LtsCoefficients difference{
      {id1, id1, -3.0}, {id1, id2, 3.0}, {id1, id3, -3.0}, {id2, id1, -9.0}};

  {
    auto s = coefs1;
    CHECK(&(s += coefs2) == &s);
    CHECK(s == sum);
  }
  {
    auto d = coefs1;
    CHECK(&(d -= coefs2) == &d);
    CHECK(d == difference);
  }

  CHECK(coefs1 + coefs2 == sum);
  CHECK(coefs1 - coefs2 == difference);
  CHECK(adams_lts::LtsCoefficients(coefs1) + coefs2 == sum);
  CHECK(adams_lts::LtsCoefficients(coefs1) - coefs2 == difference);
  CHECK(coefs1 + adams_lts::LtsCoefficients(coefs2) == sum);
  CHECK(coefs1 - adams_lts::LtsCoefficients(coefs2) == difference);
  CHECK(adams_lts::LtsCoefficients(coefs1) +
            adams_lts::LtsCoefficients(coefs2) ==
        sum);
  CHECK(adams_lts::LtsCoefficients(coefs1) -
            adams_lts::LtsCoefficients(coefs2) ==
        difference);

  adams_lts::LtsCoefficients allocated_coefs{};
  for (size_t i = 0; i < adams_lts::lts_coefficients_static_size + 1; ++i) {
    allocated_coefs.emplace_back(
        id1,
        TimeStepId(
            true, static_cast<int64_t>(i),
            Slab(static_cast<double>(i), static_cast<double>(i + 1)).start()),
        i);
  }

  {
    auto a = allocated_coefs;
    const auto* const data = a.data();
    const auto res = std::move(a) + allocated_coefs;
    CHECK(res.data() == data);
    CHECK(res == allocated_coefs + allocated_coefs);
  }
  {
    auto a = allocated_coefs;
    const auto* const data = a.data();
    const auto res = std::move(a) - allocated_coefs;
    CHECK(res.data() == data);
    CHECK(res == allocated_coefs - allocated_coefs);
  }
  {
    auto a = allocated_coefs;
    const auto* const data = a.data();
    const auto res = allocated_coefs + std::move(a);
    CHECK(res.data() == data);
    CHECK(res == allocated_coefs + allocated_coefs);
  }
  {
    auto a = allocated_coefs;
    const auto* const data = a.data();
    const auto res = allocated_coefs - std::move(a);
    CHECK(res.data() == data);
    CHECK(res == allocated_coefs - allocated_coefs);
  }
  {
    auto a = allocated_coefs;
    auto b = allocated_coefs;
    const auto* const a_data = a.data();
    const auto* const b_data = a.data();
    const auto res = std::move(a) + std::move(b);
    CHECK((res.data() == a_data or res.data() == b_data));
    CHECK(res == allocated_coefs + allocated_coefs);
  }
  {
    auto a = allocated_coefs;
    auto b = allocated_coefs;
    const auto* const a_data = a.data();
    const auto* const b_data = a.data();
    const auto res = std::move(a) - std::move(b);
    CHECK((res.data() == a_data or res.data() == b_data));
    CHECK(res == allocated_coefs - allocated_coefs);
  }
}

template <typename T>
void test_apply_coefficients(const T& used_for_size) {
  const Slab slab(1.2, 3.4);

  TimeSteppers::BoundaryHistory<double, double, T> history{};
  history.local().insert(TimeStepId(true, 0, slab.start()), 1, 1.0);
  history.local().insert(TimeStepId(true, 0, slab.end()), 1, 10.0);
  history.remote().insert(TimeStepId(true, 0, slab.start()), 1, 100.0);
  history.remote().insert(TimeStepId(true, 0, slab.end()), 1, 10000.0);

  const adams_lts::LtsCoefficients coefficients{
      {history.local()[0], history.remote()[0], 1.0},
      {history.local()[0], history.remote()[1], 2.0},
      {history.local()[1], history.remote()[1], 3.0}};

  const auto coupling = [&used_for_size](const double local,
                                         const double remote) {
    return make_with_value<T>(used_for_size, local * remote);
  };

  auto result = make_with_value<T>(used_for_size, 2.0);
  adams_lts::apply_coefficients(make_not_null(&result), coefficients,
                                history.evaluator(coupling));

  CHECK(result == make_with_value<T>(used_for_size, 320102.0));
}

struct FakeId {
  int step_time;
  std::optional<int> step_size_if_substep{};
  int64_t slab = 0;
};

bool operator<(const FakeId& a, const FakeId& b) {
  return a.slab < b.slab or
         (a.slab == b.slab and (a.step_time < b.step_time or
                                (a.step_time == b.step_time and
                                 not a.step_size_if_substep.has_value() and
                                 b.step_size_if_substep.has_value())));
}

std::ostream& operator<<(std::ostream& s, const FakeId& id) {
  using ::operator<<;
  return s << "[" << id.step_time << " " << id.step_size_if_substep << " "
           << id.slab << "]";
}

using ExpectedCoefficients = std::map<std::pair<FakeId, FakeId>, double>;

ExpectedCoefficients flip_sides(const ExpectedCoefficients& coefs) {
  ExpectedCoefficients flipped{};
  for (const auto& coef : coefs) {
    flipped.insert({{coef.first.second, coef.first.first}, coef.second});
  }
  return flipped;
}

namespace step_coefficients_detail {
constexpr int max_time = 16;

Time make_time(const int time) {
  return {Slab(-max_time, max_time), Rational(time + max_time, 2 * max_time)};
}

ApproximateTime make_time(const double time) { return {time}; }

TimeStepId make_id(const FakeId& fake_id) {
  if (fake_id.step_size_if_substep.has_value()) {
    ASSERT(*fake_id.step_size_if_substep > 0, "Zero step size");
    const auto unit_step =
        Slab(-max_time, max_time).duration() / (2 * max_time);
    return {
        true,
        fake_id.slab,
        make_time(fake_id.step_time),
        1,
        *fake_id.step_size_if_substep * unit_step,
        static_cast<double>(fake_id.step_time + *fake_id.step_size_if_substep)};
  } else {
    return {true, fake_id.slab, make_time(fake_id.step_time)};
  }
}

FakeId fake_from_id(const TimeStepId& id) {
  return {static_cast<int>(id.step_time().value()),
          id.substep() == 1
              ? std::optional(static_cast<int>(id.step_size().value()))
              : std::nullopt,
          id.slab_number()};
}

// Map t -> -t/2.  As our test slab is symmetrical about 0, this maps
// the fraction as [0, 1] -> [3/4, 1/4].
Time alternate_time(const Time& time) {
  return {time.slab(), Rational(3, 4) - time.fraction() / 2};
}

ApproximateTime alternate_time(const ApproximateTime& time) {
  return {-0.5 * time.value()};
}

TimeStepId alternate_id(const TimeStepId& id) {
  if (id.substep() == 0) {
    return {not id.time_runs_forward(), id.slab_number(),
            alternate_time(id.step_time())};
  } else {
    return {not id.time_runs_forward(),
            id.slab_number(),
            alternate_time(id.step_time()),
            id.substep(),
            id.step_size() / -2,
            id.substep_time() / -2.0};
  }
}

template <typename T>
ExpectedCoefficients step_coefficients_impl(
    const std::vector<FakeId>& local_steps,
    const std::vector<FakeId>& remote_steps,
    const adams_lts::AdamsScheme& local_scheme,
    const adams_lts::AdamsScheme& remote_scheme,
    const adams_lts::AdamsScheme& small_step_scheme, const int step_start,
    const T step_end) {
  const auto history_order = std::numeric_limits<size_t>::max();  // unused

  TimeSteppers::BoundaryHistory<double, double, double> history{};
  TimeSteppers::BoundaryHistory<double, double, double> alt_history{};

  for (const auto& step : local_steps) {
    const auto id = make_id(step);
    history.local().insert(id, history_order, 0.0);
    alt_history.local().insert(alternate_id(id), history_order, 0.0);
  }
  for (const auto& step : remote_steps) {
    const auto id = make_id(step);
    history.remote().insert(id, history_order, 0.0);
    alt_history.remote().insert(alternate_id(id), history_order, 0.0);
  }

  const auto coefficients = adams_lts::lts_coefficients(
      history.local(), history.remote(), make_time(step_start),
      make_time(step_end), local_scheme, remote_scheme, small_step_scheme);

  // Compare with step scaled by -1/2.
  const auto alt_coefficients = adams_lts::lts_coefficients(
      alt_history.local(), alt_history.remote(),
      alternate_time(make_time(step_start)),
      alternate_time(make_time(step_end)), local_scheme, remote_scheme,
      small_step_scheme);
  REQUIRE(coefficients.size() == alt_coefficients.size());
  for (size_t i = 0; i < coefficients.size(); ++i) {
    const auto& coef = coefficients[i];
    const auto& alt_coef = alt_coefficients[i];
    CHECK(get<0>(alt_coef) == alternate_id(get<0>(coef)));
    CHECK(get<1>(alt_coef) == alternate_id(get<1>(coef)));
    CHECK(get<2>(alt_coef) == approx(get<2>(coef) / -2.0));
  }

  ExpectedCoefficients coefficients_map{};
  for (const auto& entry : coefficients) {
    const auto insert_success = coefficients_map.insert(
        {{fake_from_id(get<0>(entry)), fake_from_id(get<1>(entry))},
         get<2>(entry)});
    if (not insert_success.second) {
      // Duplicate entry.
      CAPTURE(entry);
      CHECK(false);
      insert_success.first->second += get<2>(entry);
    }
  }
  return coefficients_map;
}
}  // namespace step_coefficients_detail

ExpectedCoefficients dense_coefficients(
    const std::vector<FakeId>& local_steps,
    const std::vector<FakeId>& remote_steps,
    const adams_lts::AdamsScheme& local_scheme,
    const adams_lts::AdamsScheme& remote_scheme,
    const adams_lts::AdamsScheme& small_step_scheme, const int step_start,
    const double step_end) {
  auto result = step_coefficients_detail::step_coefficients_impl(
      local_steps, remote_steps, local_scheme, remote_scheme, small_step_scheme,
      step_start, step_end);
  // NOLINTNEXTLINE(readability-suspicious-call-argument)
  const auto reversed = step_coefficients_detail::step_coefficients_impl(
      remote_steps, local_steps, remote_scheme, local_scheme, small_step_scheme,
      step_start, step_end);
  CHECK_ITERABLE_APPROX(result, flip_sides(reversed));
  return result;
}

ExpectedCoefficients step_coefficients(
    const std::vector<FakeId>& local_steps,
    const std::vector<FakeId>& remote_steps,
    const adams_lts::AdamsScheme& local_scheme,
    const adams_lts::AdamsScheme& remote_scheme,
    const adams_lts::AdamsScheme& small_step_scheme, const int step_start,
    const int step_end, const bool also_check_dense = true) {
  auto result = step_coefficients_detail::step_coefficients_impl(
      local_steps, remote_steps, local_scheme, remote_scheme, small_step_scheme,
      step_start, step_end);
  // NOLINTNEXTLINE(readability-suspicious-call-argument)
  const auto reversed = step_coefficients_detail::step_coefficients_impl(
      remote_steps, local_steps, remote_scheme, local_scheme, small_step_scheme,
      step_start, step_end);
  CHECK_ITERABLE_APPROX(result, flip_sides(reversed));
  if (also_check_dense) {
    const auto dense = dense_coefficients(
        local_steps, remote_steps, local_scheme, remote_scheme,
        small_step_scheme, step_start, static_cast<double>(step_end));
    CHECK_ITERABLE_APPROX(result, dense);
  }
  return result;
}

void test_lts_coefficients() {
  // AB2 [0 1] step to 2
  const std::array coefs_ab2{-1.0 / 2.0, 3.0 / 2.0};
  // AB2 [0 1] step to 3/2
  const std::array coefs_ab2_32{-1.0 / 8.0, 5.0 / 8.0};
  // AB3 [0 1 2] step to 3
  const std::array coefs_ab3{5.0 / 12.0, -4.0 / 3.0, 23.0 / 12.0};
  // AB3 [0 1 2] step to 5/2
  const std::array coefs_ab3_52{1.0 / 12.0, -7.0 / 24.0, 17.0 / 24.0};
  // AM3 [0 1 (2)] step to 2
  const std::array coefs_am3{-1.0 / 12.0, 2.0 / 3.0, 5.0 / 12.0};
  // AM3 [0 1 (2)] dense to 3/2
  const std::array coefs_am3_32{-1.0 / 24.0, 11.0 / 24.0, 1.0 / 12.0};

  {
    INFO("AB GTS order 1");
    const adams_lts::AdamsScheme ab1{adams_lts::SchemeType::Explicit, 1};
    const std::vector<FakeId> steps{{0}};
    // clang-format off
    const ExpectedCoefficients expected{
        {{{0}, {0}}, 1.0}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps, steps, ab1, ab1, ab1, 0, 1), expected);
    const ExpectedCoefficients expected_dense_12{
        {{{0}, {0}}, 0.5}};
    CHECK_ITERABLE_APPROX(
        dense_coefficients(steps, steps, ab1, ab1, ab1, 0, 0.5),
        expected_dense_12);
    // clang-format on
  }

  {
    INFO("AB GTS order 3");
    const adams_lts::AdamsScheme ab3{adams_lts::SchemeType::Explicit, 3};
    const std::vector<FakeId> steps{{0}, {1}, {2}};
    // clang-format off
    const ExpectedCoefficients expected{
        {{{0}, {0}}, coefs_ab3[0]},
        {{{1}, {1}}, coefs_ab3[1]},
        {{{2}, {2}}, coefs_ab3[2]}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps, steps, ab3, ab3, ab3, 2, 3), expected);
    const ExpectedCoefficients expected_dense_52{
        {{{0}, {0}}, coefs_ab3_52[0]},
        {{{1}, {1}}, coefs_ab3_52[1]},
        {{{2}, {2}}, coefs_ab3_52[2]}};
    CHECK_ITERABLE_APPROX(
        dense_coefficients(steps, steps, ab3, ab3, ab3, 2, 2.5),
        expected_dense_52);
    // clang-format on
  }

  {
    INFO("AM GTS order 3");
    const adams_lts::AdamsScheme ab2{adams_lts::SchemeType::Explicit, 2};
    const adams_lts::AdamsScheme am3{adams_lts::SchemeType::Implicit, 3};
    const std::vector<FakeId> steps{{0}, {1}, {1, 1}};
    // clang-format off
    const ExpectedCoefficients expected{
        {{{0}, {0}}, coefs_am3[0]},
        {{{1}, {1}}, coefs_am3[1]},
        {{{1, 1}, {1, 1}}, coefs_am3[2]}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps, steps, am3, am3, am3, 1, 2), expected);
    const ExpectedCoefficients expected_predictor{
        {{{0}, {0}}, coefs_ab2[0]},
        {{{1}, {1}}, coefs_ab2[1]}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps, steps, ab2, ab2, ab2, 1, 2),
        expected_predictor);
    const ExpectedCoefficients expected_dense_32_nonmonotonic{
        {{{0}, {0}}, coefs_am3_32[0]},
        {{{1}, {1}}, coefs_am3_32[1]},
        {{{1, 1}, {1, 1}}, coefs_am3_32[2]}};
    CHECK_ITERABLE_APPROX(
        dense_coefficients(steps, steps, am3, am3, am3, 1, 1.5),
        expected_dense_32_nonmonotonic);
    const ExpectedCoefficients expected_dense_32_monotonic{
        {{{0}, {0}}, coefs_ab2_32[0]},
        {{{1}, {1}}, coefs_ab2_32[1]}};
    CHECK_ITERABLE_APPROX(
        dense_coefficients(steps, steps, ab2, ab2, ab2, 1, 1.5),
        expected_dense_32_monotonic);
    // clang-format on
  }

  {
    INFO("AB single-side order 3");
    const adams_lts::AdamsScheme ab1{adams_lts::SchemeType::Explicit, 1};
    const adams_lts::AdamsScheme ab3{adams_lts::SchemeType::Explicit, 3};
    const std::vector<FakeId> steps{{0}, {1}, {2}};
    const std::vector<FakeId> single_step{{0}};
    // clang-format off
    const ExpectedCoefficients expected{
        {{{0}, {0}}, coefs_ab3[0]},
        {{{1}, {0}}, coefs_ab3[1]},
        {{{2}, {0}}, coefs_ab3[2]}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps, single_step, ab3, ab1, ab3, 2, 3), expected);
    const ExpectedCoefficients expected_dense_52{
        {{{0}, {0}}, coefs_ab3_52[0]},
        {{{1}, {0}}, coefs_ab3_52[1]},
        {{{2}, {0}}, coefs_ab3_52[2]}};
    CHECK_ITERABLE_APPROX(
        dense_coefficients(steps, single_step, ab3, ab1, ab3, 2, 2.5),
        expected_dense_52);
    // clang-format on
  }

  {
    INFO("AM single-side order 3");
    const adams_lts::AdamsScheme ab1{adams_lts::SchemeType::Explicit, 1};
    const adams_lts::AdamsScheme ab2{adams_lts::SchemeType::Explicit, 2};
    const adams_lts::AdamsScheme am3{adams_lts::SchemeType::Implicit, 3};
    const std::vector<FakeId> steps{{0}, {1}, {1, 1}};
    const std::vector<FakeId> single_step{{0}};
    // clang-format off
    const ExpectedCoefficients expected{
        {{{0}, {0}}, coefs_am3[0]},
        {{{1}, {0}}, coefs_am3[1]},
        {{{1, 1}, {0}}, coefs_am3[2]}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps, single_step, am3, ab1, am3, 1, 2),
        expected);
    const ExpectedCoefficients expected_predictor{
        {{{0}, {0}}, coefs_ab2[0]},
        {{{1}, {0}}, coefs_ab2[1]}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps, single_step, ab2, ab1, ab2, 1, 2),
        expected_predictor);
    // Nonmonotonic requires additional future data.
    const ExpectedCoefficients expected_dense_32_monotonic{
        {{{0}, {0}}, coefs_ab2_32[0]},
        {{{1}, {0}}, coefs_ab2_32[1]}};
    CHECK_ITERABLE_APPROX(
        dense_coefficients(steps, single_step, ab2, ab1, ab2, 1, 1.5),
        expected_dense_32_monotonic);
    // clang-format on
  }

  {
    INFO("AB 2:1 order 3");
    // -8          -4           0           4
    //             -4    -2     0     2     4
    const adams_lts::AdamsScheme ab3{adams_lts::SchemeType::Explicit, 3};
    const std::vector<FakeId> steps_large{{-8}, {-4}, {0}};
    const std::vector<FakeId> steps_small{{-4}, {-2}, {0}, {2}};
    // clang-format off
    const ExpectedCoefficients expected_large{
        {{{0}, {2}}, 115.0 / 16.0},
        {{{0}, {0}}, 7.0 / 6.0},
        {{{0}, {-2}}, -11.0 / 16.0},
        {{{-4}, {2}}, -115.0 / 24.0},
        {{{-4}, {-2}}, -11.0 / 8.0},
        {{{-4}, {-4}}, 5.0 / 6.0},
        {{{-8}, {2}}, 23.0 / 16.0},
        {{{-8}, {-2}}, 11.0 / 48.0}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps_large, steps_small, ab3, ab3, ab3, 0, 4, false),
        expected_large);
    const ExpectedCoefficients expected_small_1{
        {{{0}, {0}}, 23.0 / 6.0},
        {{{-2}, {0}}, -1.0},
        {{{-2}, {-4}}, -2.0},
        {{{-4}, {-4}}, 5.0 / 6.0},
        {{{-2}, {-8}}, 1.0 / 3.0}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps_small, steps_large, ab3, ab3, ab3, 0, 2),
        expected_small_1);
    const ExpectedCoefficients expected_small_2{
        {{{2}, {0}}, 115.0 / 16.0},
        {{{0}, {0}}, -8.0 / 3.0},
        {{{-2}, {0}}, 5.0 / 16.0},
        {{{2}, {-4}}, -115.0 / 24.0},
        {{{-2}, {-4}}, 5.0 / 8.0},
        {{{2}, {-8}}, 23.0 / 16.0},
        {{{-2}, {-8}}, -5.0 / 48.0}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps_small, steps_large, ab3, ab3, ab3, 2, 4),
        expected_small_2);
    // clang-format on
  }

  {
    INFO("AB LTS -> GTS order 2");
    // -2     0  1
    //    -1  0  1
    const adams_lts::AdamsScheme ab2{adams_lts::SchemeType::Explicit, 2};
    const std::vector<FakeId> steps_large{{-2}, {0}};
    const std::vector<FakeId> steps_small{{-1}, {0}};
    // clang-format off
    const ExpectedCoefficients expected{
        {{{0}, {0}}, 3.0 / 2.0},
        {{{0}, {-1}}, -1.0 / 4.0},
        {{{-2}, {-1}}, -1.0 / 4.0}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps_large, steps_small, ab2, ab2, ab2, 0, 1),
        expected);
    // clang-format on
  }

  {
    INFO("AB 3:1 order 2");
    // -3        0        3
    //       -1  0  1  2  3
    const adams_lts::AdamsScheme ab2{adams_lts::SchemeType::Explicit, 2};
    const std::vector<FakeId> steps_large{{-3}, {0}};
    const std::vector<FakeId> steps_small{{-1}, {0}, {1}, {2}};
    // clang-format off
    const ExpectedCoefficients expected_large{
        {{{-3}, {-1}}, -1.0 / 6.0},
        {{{-3}, {1}}, -1.0 / 3.0},
        {{{-3}, {2}}, -1.0},
        {{{0}, {-1}}, -1.0 / 3.0},
        {{{0}, {0}}, 1.0},
        {{{0}, {1}}, 4.0 / 3.0},
        {{{0}, {2}}, 5.0 / 2.0}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps_large, steps_small, ab2, ab2, ab2, 0, 3, false),
        expected_large);
    const ExpectedCoefficients expected_small_1{
        {{{-1}, {-3}}, -1.0 / 6.0},
        {{{-1}, {0}}, -1.0 / 3.0},
        {{{0}, {0}}, 3.0 / 2.0}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps_small, steps_large, ab2, ab2, ab2, 0, 1),
        expected_small_1);
    const ExpectedCoefficients expected_small_2{
        {{{0}, {0}}, -1.0 / 2.0},
        {{{1}, {-3}}, -1.0 / 2.0},
        {{{1}, {0}}, 2.0}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps_small, steps_large, ab2, ab2, ab2, 1, 2),
        expected_small_2);
    const ExpectedCoefficients expected_small_3{
        {{{1}, {-3}}, 1.0 / 6.0},
        {{{1}, {0}}, -2.0 / 3.0},
        {{{2}, {-3}}, -1.0},
        {{{2}, {0}}, 5.0 / 2.0}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps_small, steps_large, ab2, ab2, ab2, 2, 3),
        expected_small_3);

    const ExpectedCoefficients expected_dense_12{
        {{{-3}, {-1}}, -1.0 / 24.0},
        {{{0}, {-1}}, -1.0 / 12.0},
        {{{0}, {0}}, 5.0 / 8.0}};
    CHECK_ITERABLE_APPROX(
        dense_coefficients(steps_large, steps_small, ab2, ab2, ab2, 0, 0.5),
        expected_dense_12);
    const ExpectedCoefficients expected_dense_32{
        {{{0}, {0}}, -1.0 / 8.0},
        {{{-3}, {1}}, -5.0 / 24.0},
        {{{0}, {1}}, 5.0 / 6.0}};
    CHECK_ITERABLE_APPROX(
        dense_coefficients(steps_large, steps_small, ab2, ab2, ab2, 1, 1.5),
        expected_dense_32);
    const ExpectedCoefficients expected_dense_52{
        {{{-3}, {1}}, 1.0 / 24.0},
        {{{0}, {1}}, -1.0 / 6.0},
        {{{-3}, {2}}, -5.0 / 12.0},
        {{{0}, {2}}, 25.0 / 24.0}};
    CHECK_ITERABLE_APPROX(
        dense_coefficients(steps_large, steps_small, ab2, ab2, ab2, 2, 2.5),
        expected_dense_52);
    // clang-format on
  }

  {
    INFO("AB unaligned order 2");
    // 0  1     3  4     6
    // 0     2  3     5  6
    const adams_lts::AdamsScheme ab2{adams_lts::SchemeType::Explicit, 2};
    const std::vector<FakeId> steps_a{{1}, {3}, {4}};
    const std::vector<FakeId> steps_b{{2}, {3}, {5}};
    // clang-format off
    const ExpectedCoefficients expected_a_1{
        {{{1}, {2}}, -1.0 / 4.0},
        {{{3}, {2}}, -1.0 / 4.0},
        {{{3}, {3}}, 3.0 / 2.0}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps_a, steps_b, ab2, ab2, ab2, 3, 4),
        expected_a_1);
    const ExpectedCoefficients expected_a_2{
        {{{3}, {3}}, -1.0 / 2.0},
        {{{3}, {5}}, -3.0 / 2.0},
        {{{4}, {2}}, -3.0 / 2.0},
        {{{4}, {3}}, 11.0 / 4.0},
        {{{4}, {5}}, 11.0 / 4.0}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps_a, steps_b, ab2, ab2, ab2, 4, 6, false),
        expected_a_2);
    const ExpectedCoefficients expected_b_1{
        {{{2}, {1}}, -1.0 / 4.0},
        {{{2}, {3}}, -1.0 / 4.0},
        {{{2}, {4}}, -3.0 / 2.0},
        {{{3}, {3}}, 1.0},
        {{{3}, {4}}, 3.0}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps_b, steps_a, ab2, ab2, ab2, 3, 5, false),
        expected_b_1);
    const ExpectedCoefficients expected_b_2{
        {{{3}, {4}}, -1.0 / 4.0},
        {{{5}, {3}}, -3.0 / 2.0},
        {{{5}, {4}}, 11.0 / 4.0}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps_b, steps_a, ab2, ab2, ab2, 5, 6),
        expected_b_2);
    // clang-format on
  }

  {
    INFO("AM 2:1 order 2");
    // 0     2
    // 0  1  2
    const adams_lts::AdamsScheme ab1{adams_lts::SchemeType::Explicit, 1};
    const adams_lts::AdamsScheme am2{adams_lts::SchemeType::Implicit, 2};
    const std::vector<FakeId> steps_large{{0}, {0, 2}};
    const std::vector<FakeId> steps_small{{0}, {0, 1}, {1}, {1, 1}};
    const std::vector<FakeId> steps_small_for_nonmonotonic_predictor{{0}};
    // clang-format off
    // Monotonic predictor requires two calls, both of which are
    // small-side tests below.
    const ExpectedCoefficients expected_large_predictor_nonmonotonic{
        {{{0}, {0}}, 2.0}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps_large, steps_small_for_nonmonotonic_predictor,
                          ab1, ab1, ab1, 0, 2),
        expected_large_predictor_nonmonotonic);
    const ExpectedCoefficients expected_large_corrector{
        {{{0}, {0}}, 1.0 / 2.0},
        {{{0}, {0, 1}}, 1.0 / 4.0},
        {{{0, 2}, {0, 1}}, 1.0 / 4.0},
        {{{0}, {1}}, 1.0 / 4.0},
        {{{0, 2}, {1}}, 1.0 / 4.0},
        {{{0, 2}, {1, 1}}, 1.0 / 2.0}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps_large, steps_small, am2, am2, am2, 0, 2, false),
        expected_large_corrector);
    const ExpectedCoefficients expected_small_predictor_1{
        {{{0}, {0}}, 1.0}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps_small, steps_large, ab1, ab1, ab1, 0, 1),
        expected_small_predictor_1);
    const ExpectedCoefficients expected_small_corrector_1{
        {{{0}, {0}}, 1.0 / 2.0},
        {{{0, 1}, {0}}, 1.0 / 4.0},
        {{{0, 1}, {0, 2}}, 1.0 / 4.0}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps_small, steps_large, am2, am2, am2, 0, 1),
        expected_small_corrector_1);
    const ExpectedCoefficients expected_small_predictor_2{
        {{{1}, {0}}, 1.0}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps_small, steps_large, ab1, ab1, ab1, 1, 2),
        expected_small_predictor_2);
    const ExpectedCoefficients expected_small_corrector_2{
        {{{1}, {0}}, 1.0 / 4.0},
        {{{1}, {0, 2}}, 1.0 / 4.0},
        {{{1, 1}, {0, 2}}, 1.0 / 2.0}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps_small, steps_large, am2, am2, am2, 1, 2),
        expected_small_corrector_2);
    // clang-format on
  }

  {
    INFO("AB self-start order 4");
    // 0 1 2 3 -- 0 1
    // 0 1 2 3 -- 0 1
    const adams_lts::AdamsScheme ab4{adams_lts::SchemeType::Explicit, 4};
    const std::vector<FakeId> steps{{2, {}, -1}, {3, {}, -1}, {0}, {1}};
    // clang-format off
    const ExpectedCoefficients expected{
        {{{2, {}, -1}, {2, {}, -1}}, 13.0 / 24.0},
        {{{3, {}, -1}, {3, {}, -1}}, -1.0 / 24.0},
        {{{0}, {0}}, -1.0 / 24.0},
        {{{1}, {1}}, 13.0 / 24.0}};
    CHECK_ITERABLE_APPROX(
        step_coefficients(steps, steps, ab4, ab4, ab4, 1, 2),
        expected);
    // clang-format on
  }
}

SPECTRE_TEST_CASE("Unit.Time.TimeSteppers.AdamsLts", "[Unit][Time]") {
  test_exact_substep_time();
  test_lts_coefficients_struct();
  test_apply_coefficients(0.0);
  test_apply_coefficients(DataVector(5, 0.0));
  test_lts_coefficients();
}
}  // namespace
