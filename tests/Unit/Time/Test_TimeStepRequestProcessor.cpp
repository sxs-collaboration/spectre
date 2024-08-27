// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <vector>

#include "Framework/TestHelpers.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Time/TimeStepRequestProcessor.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Numeric.hpp"

namespace {
// Test processing requests in all orders, with different subsets
// combined first.
void test_permutations_roundoff(const std::vector<TimeStepRequest>& requests,
                                const double start, const double goal,
                                const std::optional<double>& expected_goal,
                                const double expected_step,
                                const double expected_end) {
  std::vector<size_t> ordering(requests.size());
  alg::iota(ordering, 0_st);
  std::vector<size_t> partition{};
  partition.reserve(requests.size());
  do {
    partition.clear();
    partition.push_back(requests.size());
    for (;;) {
      {
        TimeStepRequestProcessor processor(goal > 0.0);
        TimeStepRequestProcessor processor2(goal > 0.0);
        size_t index = 0;
        for (const size_t partition_element : partition) {
          TimeStepRequestProcessor element_processor(goal > 0.0);
          for (size_t i = 0; i < partition_element; ++i) {
            element_processor.process(requests[ordering[index]]);
            ++index;
          }
          processor2 = processor + element_processor;
          processor += serialize_and_deserialize(element_processor);
          CHECK(processor == processor2);
          CHECK_FALSE(processor != processor2);
        }
        CHECK(processor.new_step_size_goal() == expected_goal);
        CHECK(processor.step_size(start, goal) == expected_step);
        CHECK(processor.step_end(start, goal) == expected_end);
        CHECK(processor2.new_step_size_goal() == expected_goal);
        CHECK(processor2.step_size(start, goal) == expected_step);
        CHECK(processor2.step_end(start, goal) == expected_end);
      }

      if (requests.empty() or partition.size() == requests.size()) {
        break;
      }
      size_t trailing_ones = 0;
      while (partition.back() == 1) {
        ++trailing_ones;
        partition.pop_back();
      }
      --partition.back();
      partition.push_back(trailing_ones + 1);
    }
  } while (std::next_permutation(ordering.begin(), ordering.end()));
}

// Test processing requests in all orders, with different subsets
// combined first.  This version computes the step end from the step
// size, so only works for the non-roundoff-sensitive tests, but it
// removes a lot of redundancy in those.
void test_permutations(const std::vector<TimeStepRequest>& requests,
                       const double start, const double goal,
                       const std::optional<double>& expected_goal,
                       const double expected_step) {
  test_permutations_roundoff(requests, start, goal, expected_goal,
                             expected_step, start + expected_step);
}

void test_with_roundoff(const double start, const double step,
                        const double end) {
  CAPTURE(start);
  CAPTURE(step);
  CAPTURE(end);
  test_permutations_roundoff({}, start, step, {}, step, start + step);

  test_permutations_roundoff({{.size_goal = step}}, start, 0.5 * step, {step},
                             step, start + step);
  test_permutations_roundoff({{.size = step}}, start, 2.0 * step, {}, step,
                             start + step);
  test_permutations_roundoff({{.end = end}}, start, 2.0 * step, {}, end - start,
                             end);

  test_permutations_roundoff({{.size_goal = 0.5 * step}}, start, step,
                             {0.5 * step}, 0.5 * step, start + 0.5 * step);
  test_permutations_roundoff({{.size = 0.5 * step}}, start, step, {},
                             0.5 * step, start + 0.5 * step);
  test_permutations_roundoff({{.size_goal = 2.0 * step}}, start, step,
                             {2.0 * step}, 2.0 * step, start + 2.0 * step);
  test_permutations_roundoff({{.size = 2.0 * step}}, start, step, {}, step,
                             start + step);

  const double far_end = end + 0.5 * step;
  const double close_end = start + 0.5 * step;
  test_permutations_roundoff({{.end = close_end}}, start, step, {},
                             close_end - start, close_end);
  test_permutations_roundoff({{.end = far_end}}, start, step, {}, step,
                             start + step);

  // Test using the looser limit when two are given.
  test_permutations_roundoff({{.size = step, .end = start + 0.5 * step}}, start,
                             2.0 * step, {}, step, start + step);
  test_permutations_roundoff({{.size = 0.5 * step, .end = end}}, start,
                             2.0 * step, {}, end - start, end);
}

void test_processor(const bool time_runs_forward) {
  const double time_sign = time_runs_forward ? 1.0 : -1.0;

  {
    const double start = 2.0 * time_sign;
    const double end = 5.0 * time_sign;
    const double step = end - start;
    // Ignore roundoff subtleties for this part of the test.
    CHECK(start + step == end);

    {
      TimeStepRequestProcessor processor(time_runs_forward);
      CHECK(processor == TimeStepRequestProcessor(time_runs_forward));
      CHECK_FALSE(processor != TimeStepRequestProcessor(time_runs_forward));
      CHECK(processor != TimeStepRequestProcessor(not time_runs_forward));
      CHECK_FALSE(processor == TimeStepRequestProcessor(not time_runs_forward));
      processor.process({.end = end});
      CHECK(processor != TimeStepRequestProcessor(time_runs_forward));
      CHECK_FALSE(processor == TimeStepRequestProcessor(time_runs_forward));
    }

    // No requests.
    test_permutations({}, start, step, {}, step);
    test_permutations({{}}, start, step, {}, step);
    test_permutations({{}, {}}, start, step, {}, step);

    // Simple goal setting requests.
    test_permutations({{.size_goal = 0.5 * step}}, start, step, {0.5 * step},
                      0.5 * step);
    test_permutations({{.size_goal = 2.0 * step}}, start, step, {2.0 * step},
                      2.0 * step);

    // Simple step size requests.
    test_permutations({{.size = 0.5 * step}}, start, step, {}, 0.5 * step);
    test_permutations({{.size = 2.0 * step}}, start, step, {}, step);

    // Simple step end requests.
    test_permutations({{.end = start + 0.5 * step}}, start, step, {},
                      0.5 * step);
    test_permutations({{.end = start + 2.0 * step}}, start, step, {}, step);

    // Multiple requests of the same type.
    test_permutations({{.size_goal = 0.5 * step}, {.size_goal = 0.25 * step}},
                      start, step, {0.25 * step}, 0.25 * step);
    test_permutations({{.size = 0.5 * step}, {.size = 0.25 * step}}, start,
                      step, {}, 0.25 * step);
    test_permutations(
        {{.end = start + 0.5 * step}, {.end = start + 0.25 * step}}, start,
        step, {}, 0.25 * step);

    // Multiple requests of different types.
    // size_goal wins
    test_permutations({{.size_goal = 0.5 * step}, {.size = 0.6 * step}}, start,
                      step, {0.5 * step}, 0.5 * step);
    test_permutations({{.size_goal = 0.5 * step}, {.end = start + 0.6 * step}},
                      start, step, {0.5 * step}, 0.5 * step);
    test_permutations({{.size_goal = 0.5 * step},
                       {.size = 0.6 * step},
                       {.end = start + 0.6 * step}},
                      start, step, {0.5 * step}, 0.5 * step);
    // size wins
    test_permutations({{.size_goal = 0.6 * step}, {.size = 0.5 * step}}, start,
                      step, {0.6 * step}, 0.5 * step);
    test_permutations({{.size = 0.5 * step}, {.end = start + 0.6 * step}},
                      start, step, {}, 0.5 * step);
    test_permutations({{.size_goal = 0.6 * step},
                       {.size = 0.5 * step},
                       {.end = start + 0.6 * step}},
                      start, step, {0.6 * step}, 0.5 * step);
    // end wins
    test_permutations({{.size_goal = 0.6 * step}, {.end = start + 0.5 * step}},
                      start, step, {0.6 * step}, 0.5 * step);
    test_permutations({{.size = 0.6 * step}, {.end = start + 0.5 * step}},
                      start, step, {}, 0.5 * step);
    test_permutations({{.size_goal = 0.6 * step},
                       {.size = 0.6 * step},
                       {.end = start + 0.5 * step}},
                      start, step, {0.6 * step}, 0.5 * step);

    // Requests with both .size and .end are fulfilled if either is
    // used without adjustment.
    test_permutations({{.size_goal = 0.6 * step,
                        .size = 0.4 * step,
                        .end = start + 0.5 * step}},
                      start, step, {0.6 * step}, 0.5 * step);
    test_permutations({{.size_goal = 0.6 * step,
                        .size = 0.5 * step,
                        .end = start + 0.4 * step}},
                      start, step, {0.6 * step}, 0.5 * step);
    test_permutations({{.size = 0.4 * step, .end = start + 0.5 * step}}, start,
                      step, {}, 0.5 * step);
    test_permutations({{.size = 0.5 * step, .end = start + 0.4 * step}}, start,
                      step, {}, 0.5 * step);
    // .size_goal doesn't participate in that.
    test_permutations({{.size_goal = 0.5 * step, .end = start + 0.6 * step}},
                      start, step, {0.5 * step}, 0.5 * step);
    test_permutations({{.size_goal = 0.6 * step, .end = start + 0.5 * step}},
                      start, step, {0.6 * step}, 0.5 * step);

    // Repeats don't affect the result.
    test_permutations({{.size = 0.4 * step, .end = start + 0.5 * step},
                       {.size = 0.4 * step, .end = start + 0.5 * step}},
                      start, step, {}, 0.5 * step);
    test_permutations({{.size = 0.5 * step, .end = start + 0.4 * step},
                       {.size = 0.5 * step, .end = start + 0.4 * step}},
                      start, step, {}, 0.5 * step);

    // Less stringent requirements don't affect that, either.
    test_permutations(
        {{.size = 0.4 * step, .end = start + 0.5 * step}, {.size = 0.6 * step}},
        start, step, {}, 0.5 * step);
    test_permutations({{.size = 0.4 * step, .end = start + 0.5 * step},
                       {.end = start + 0.6 * step}},
                      start, step, {}, 0.5 * step);
    test_permutations(
        {{.size = 0.5 * step, .end = start + 0.4 * step}, {.size = 0.6 * step}},
        start, step, {}, 0.5 * step);
    test_permutations({{.size = 0.5 * step, .end = start + 0.4 * step},
                       {.end = start + 0.6 * step}},
                      start, step, {}, 0.5 * step);

    // Requests with both .size and .end usually must satisfy both in
    // other cases.
    test_permutations({{.size = 0.5 * step, .end = start + 0.6 * step},
                       {.size = 0.55 * step}},
                      start, step, {}, 0.5 * step);
    test_permutations({{.size = 0.6 * step, .end = start + 0.5 * step},
                       {.size = 0.55 * step}},
                      start, step, {}, 0.5 * step);
    test_permutations({{.size = 0.5 * step, .end = start + 0.6 * step},
                       {.end = start + 0.55 * step}},
                      start, step, {}, 0.5 * step);
    test_permutations({{.size = 0.6 * step, .end = start + 0.5 * step},
                       {.end = start + 0.55 * step}},
                      start, step, {}, 0.5 * step);

    // Including when other requests partially duplicate them.
    test_permutations(
        {{.size = 0.5 * step, .end = start + 0.6 * step}, {.size = 0.5 * step}},
        start, step, {}, 0.5 * step);
    test_permutations(
        {{.size = 0.6 * step, .end = start + 0.5 * step}, {.size = 0.5 * step}},
        start, step, {}, 0.5 * step);
    test_permutations({{.size = 0.5 * step, .end = start + 0.6 * step},
                       {.end = start + 0.5 * step}},
                      start, step, {}, 0.5 * step);
    test_permutations({{.size = 0.6 * step, .end = start + 0.5 * step},
                       {.end = start + 0.5 * step}},
                      start, step, {}, 0.5 * step);
    test_permutations({{.size = 0.5 * step, .end = start + 0.6 * step},
                       {.size = 0.5 * step},
                       {.end = start + 0.6 * step}},
                      start, step, {}, 0.5 * step);
    test_permutations({{.size = 0.6 * step, .end = start + 0.5 * step},
                       {.size = 0.6 * step},
                       {.end = start + 0.5 * step}},
                      start, step, {}, 0.5 * step);

    // And when multiple different ones occur.
    test_permutations({{.size = 0.7 * step, .end = start + 0.6 * step},
                       {.size = 0.65 * step, .end = start + 0.5 * step}},
                      start, step, {}, 0.5 * step);
    test_permutations({{.size = 0.7 * step, .end = start + 0.6 * step},
                       {.size = 0.5 * step, .end = start + 0.65 * step}},
                      start, step, {}, 0.5 * step);
    test_permutations({{.size = 0.6 * step, .end = start + 0.7 * step},
                       {.size = 0.65 * step, .end = start + 0.5 * step}},
                      start, step, {}, 0.5 * step);
    test_permutations({{.size = 0.6 * step, .end = start + 0.7 * step},
                       {.size = 0.5 * step, .end = start + 0.65 * step}},
                      start, step, {}, 0.5 * step);

    // Test with several entries.
    test_permutations({{.size_goal = 0.6 * step, .size = 0.5 * step},
                       {.size_goal = 0.7 * step},
                       {.size = 0.6 * step},
                       {.end = start + 0.6 * step},
                       {.end = start + 0.7 * step}},
                      start, step, {0.6 * step}, 0.5 * step);

    // Hard limits don't affect the suggestion
    test_permutations({{.size_hard_limit = 0.5 * step}}, start, step, {}, step);
    test_permutations({{.end_hard_limit = start + 0.5 * step}}, start, step, {},
                      step);

    {
      TimeStepRequestProcessor processor(step > 0.0);
      processor.process({.size_hard_limit = step, .end_hard_limit = end});
      processor.error_on_hard_limit(0.9 * step, start);
      processor.error_on_hard_limit(step, end);
      CHECK_THROWS_WITH(
          processor.error_on_hard_limit(1.1 * step, end),
          Catch::Matchers::ContainsSubstring("Could not adjust step below ") and
              Catch::Matchers::ContainsSubstring(
                  " to meet maximum step size "));
      CHECK_THROWS_WITH(
          processor.error_on_hard_limit(step, start + 1.1 * step),
          Catch::Matchers::ContainsSubstring(
              "Could not adjust step to before ") and
              Catch::Matchers::ContainsSubstring(" to avoid exceeding time "));
    }
  }

  {
    const double start = -1.0e20 * time_sign;
    const double end = 1.0 * time_sign;
    const double step = 1.0e20 * time_sign;
    CHECK(end - start == step);
    CHECK(start + step != end);
    test_with_roundoff(start, step, end);
  }

  {
    const double start = 1.0 * time_sign;
    const double end = (1.0 + 1.0e-10) * time_sign;
    const double step = (1.0e-10 + 1.0e-20) * time_sign;
    CHECK(end - start != step);
    CHECK(start + step == end);
    test_with_roundoff(start, step, end);
  }
}

SPECTRE_TEST_CASE("Unit.Time.TimeStepRequestProcessor", "[Unit][Time]") {
  test_processor(true);
  test_processor(false);
}
}  // namespace
