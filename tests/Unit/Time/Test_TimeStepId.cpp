// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstdint>
#include <functional>
#include <string>

#include "Framework/TestHelpers.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/GetOutput.hpp"

namespace {
void check(const bool time_runs_forward) {
  using Hash = std::hash<TimeStepId>;

  const Slab slab(1.25, 3.5);
  const Time start = time_runs_forward ? slab.start() : slab.end();
  const Time end = time_runs_forward ? slab.end() : slab.start();
  const TimeDelta step = end - start;

  CHECK(TimeStepId(time_runs_forward, 4, start + step / 3) ==
        TimeStepId(time_runs_forward, 4, start + step / 3, 0, step,
                   (start + step / 3).value()));

  CHECK_FALSE(TimeStepId(time_runs_forward, 4, start + step / 3, 2, step,
                         (start + step / 2).value())
                  .is_at_slab_boundary());
  CHECK_FALSE(TimeStepId(time_runs_forward, 4, start, 2, step, start.value())
                  .is_at_slab_boundary());
  CHECK_FALSE(TimeStepId(time_runs_forward, 4, start, 2, step, end.value())
                  .is_at_slab_boundary());
  CHECK_FALSE(
      TimeStepId(time_runs_forward, 4, start + step / 3).is_at_slab_boundary());
  CHECK_FALSE(TimeStepId(time_runs_forward, 4, start, 1, step, start.value())
                  .is_at_slab_boundary());
  CHECK_FALSE(TimeStepId(time_runs_forward, 4, start, 1, step, end.value())
                  .is_at_slab_boundary());
  CHECK(TimeStepId(time_runs_forward, 4, start).is_at_slab_boundary());
  CHECK(TimeStepId(time_runs_forward, 4, end).is_at_slab_boundary());

  CHECK(TimeStepId(time_runs_forward, 5, start).slab_number() == 5);
  CHECK(TimeStepId(time_runs_forward, 5, start).step_time().slab() == slab);
  CHECK(TimeStepId(time_runs_forward, 5, end).slab_number() == 6);
  CHECK(TimeStepId(time_runs_forward, 5, end).step_time().slab() ==
        slab.advance_towards(step));

  CHECK(
      TimeStepId(time_runs_forward, 4, start + step / 2).next_step(step / 4) ==
      TimeStepId(time_runs_forward, 4, start + step * 3 / 4));
  CHECK(TimeStepId(time_runs_forward, 4, start + step / 2, 1, step / 4,
                   end.value())
            .next_step(step / 4) ==
        TimeStepId(time_runs_forward, 4, start + step * 3 / 4));
  CHECK(TimeStepId(time_runs_forward, 4, start + step / 2, 1, step / 4,
                   end.value())
            .next_substep(step / 4, 1.0 / 8.0) ==
        TimeStepId(time_runs_forward, 4, start + step / 2, 2, step / 4,
                   (start + step * 17 / 32).value()));

  const TimeStepId id(time_runs_forward, 4, start + step / 3, 2, step / 2,
                      (start + step / 2).value());
  CHECK(id.step_size() == step / 2);

  CHECK(id == id);
  CHECK_FALSE(id != id);
  CHECK(id == TimeStepId(id));

  const auto check_comparisons = [&id](const int64_t slab_delta,
                                       const TimeDelta& step_time_delta,
                                       const int64_t substep_delta,
                                       const TimeDelta& substep_time_delta) {
    const TimeStepId id2(id.time_runs_forward(),
                         id.slab_number() + slab_delta,
                         id.step_time() + step_time_delta,
                         id.substep() + static_cast<uint64_t>(substep_delta),
                         id.step_size(),
                         id.substep_time() + substep_time_delta.value());
    check_cmp(id, id2);
    CHECK(Hash{}(id) != Hash{}(id2));
  };

  check_comparisons(1, 0 * step, 0, 0 * step);
  check_comparisons(0, step / 8, 0, 0 * step);
  check_comparisons(0, 0 * step, 1, 0 * step);

  check_comparisons(1, -step / 4, 0, 0 * step);
  check_comparisons(1, 0 * step, -1, 0 * step);
  check_comparisons(1, 0 * step, 0, -step / 8);
  check_comparisons(0, step / 8, -1, 0 * step);
  check_comparisons(0, step / 16, 0, -step / 16);
  check_comparisons(0, 0 * step, 1, -step / 8);

  test_serialization(id);

  {
    const TimeStepId id2(id.time_runs_forward(), id.slab_number() + 1,
                         id.step_time());
    check_cmp(id, id2);
    CHECK(Hash{}(id) != Hash{}(id2));
    test_serialization(id2);
  }

  CHECK(get_output(id) == "4:" + get_output(id.step_time()) +
                              ":2:" + get_output(id.substep_time()));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.TimeStepId", "[Unit][Time]") {
  check(true);
  check(false);
}
