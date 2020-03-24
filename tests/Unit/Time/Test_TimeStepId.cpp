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
void check(const bool time_runs_forward) noexcept {
  using Hash = std::hash<TimeStepId>;

  const Slab slab(1.2, 3.4);
  const Time start = time_runs_forward ? slab.start() : slab.end();
  const Time end = time_runs_forward ? slab.end() : slab.start();
  const TimeDelta step = end - start;

  CHECK(TimeStepId(time_runs_forward, 4, start + step / 3) ==
        TimeStepId(time_runs_forward, 4, start + step / 3, 0,
                   start + step / 3));

  CHECK_FALSE(
      TimeStepId(time_runs_forward, 4, start + step / 3, 2, start + step / 2)
          .is_at_slab_boundary());
  CHECK_FALSE(TimeStepId(time_runs_forward, 4, start + step / 3, 2, start)
                  .is_at_slab_boundary());
  CHECK_FALSE(TimeStepId(time_runs_forward, 4, start + step / 3, 2, end)
                  .is_at_slab_boundary());
  CHECK_FALSE(
      TimeStepId(time_runs_forward, 4, start + step / 3, 0, start + step / 3)
          .is_at_slab_boundary());
  CHECK_FALSE(
      TimeStepId(time_runs_forward, 4, start, 1, start).is_at_slab_boundary());
  CHECK_FALSE(
      TimeStepId(time_runs_forward, 4, start, 1, end).is_at_slab_boundary());
  CHECK(TimeStepId(time_runs_forward, 4, start).is_at_slab_boundary());
  CHECK(TimeStepId(time_runs_forward, 4, end).is_at_slab_boundary());

  CHECK(TimeStepId(time_runs_forward, 5, start).slab_number() == 5);
  CHECK(TimeStepId(time_runs_forward, 5, start).substep_time().slab() == slab);
  CHECK(TimeStepId(time_runs_forward, 5, end).slab_number() == 6);
  CHECK(TimeStepId(time_runs_forward, 5, end).substep_time().slab() ==
        slab.advance_towards(step));

  const TimeStepId id(time_runs_forward, 4, start + step / 3, 2,
                      start + step / 2);

  CHECK(id == id);
  CHECK_FALSE(id != id);
  CHECK(id == TimeStepId(id));

  const auto check_comparisons = [&id](
      const int64_t slab_delta,
      const TimeDelta& step_time_delta,
      const int64_t substep_delta,
      const TimeDelta& substep_time_delta) noexcept {
    const TimeStepId id2(id.time_runs_forward(),
                         id.slab_number() + slab_delta,
                         id.step_time() + step_time_delta,
                         id.substep() + static_cast<uint64_t>(substep_delta),
                         id.substep_time() + substep_time_delta);
    check_cmp(id, id2);
    CHECK(Hash{}(id) != Hash{}(id2));
  };

  check_comparisons(1, 0 * step, 0, 0 * step);
  check_comparisons(0, step / 2, 0, 0 * step);
  check_comparisons(0, 0 * step, 1, 0 * step);

  check_comparisons(1, -step / 4, 0, 0 * step);
  check_comparisons(1, 0 * step, -1, 0 * step);
  check_comparisons(1, 0 * step, 0, -step / 4);
  check_comparisons(0, step / 2, -1, 0 * step);
  check_comparisons(0, step / 2, 0, -step / 4);
  check_comparisons(0, 0 * step, 1, -step / 4);

  test_serialization(id);

  CHECK(get_output(id) == "4:" + get_output(id.step_time()) +
                              ":2:" + get_output(id.substep_time()));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.TimeStepId", "[Unit][Time]") {
  check(true);
  check(false);
}
