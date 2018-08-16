// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstdint>
#include <functional>
#include <string>

#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/GetOutput.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
void check(const bool time_runs_forward) noexcept {
  using Hash = std::hash<TimeId>;

  const Slab slab(1.2, 3.4);
  const Time start = time_runs_forward ? slab.start() : slab.end();
  const Time end = time_runs_forward ? slab.end() : slab.start();
  const TimeDelta step = end - start;

  CHECK(TimeId(time_runs_forward, 4, start + step / 3) ==
        TimeId(time_runs_forward, 4, start + step / 3, 0, start + step / 3));

  CHECK_FALSE(
      TimeId(time_runs_forward, 4, start + step / 3, 2, start + step / 2)
          .is_at_slab_boundary());
  CHECK_FALSE(TimeId(time_runs_forward, 4, start + step / 3, 2, start)
                  .is_at_slab_boundary());
  CHECK_FALSE(TimeId(time_runs_forward, 4, start + step / 3, 2, end)
                  .is_at_slab_boundary());
  CHECK_FALSE(
      TimeId(time_runs_forward, 4, start + step / 3, 0, start + step / 3)
          .is_at_slab_boundary());
  CHECK_FALSE(
      TimeId(time_runs_forward, 4, start, 1, start).is_at_slab_boundary());
  CHECK_FALSE(
      TimeId(time_runs_forward, 4, start, 1, end).is_at_slab_boundary());
  CHECK(TimeId(time_runs_forward, 4, start).is_at_slab_boundary());
  CHECK(TimeId(time_runs_forward, 4, end).is_at_slab_boundary());

  CHECK(TimeId(time_runs_forward, 5, start).slab_number() == 5);
  CHECK(TimeId(time_runs_forward, 5, start).time().slab() == slab);
  CHECK(TimeId(time_runs_forward, 5, end).slab_number() == 6);
  CHECK(TimeId(time_runs_forward, 5, end).time().slab() ==
        slab.advance_towards(step));

  const TimeId id(time_runs_forward, 4, start + step / 3, 2, start + step / 2);

  CHECK(id == id);
  CHECK_FALSE(id != id);
  CHECK(id == TimeId(id));

  const auto check_comparisons =
      [&id](const int64_t slab_delta, const TimeDelta& step_time_delta,
            const int64_t substep_delta, const TimeDelta& time_delta) noexcept {
    const TimeId id2(id.time_runs_forward(),
                     id.slab_number() + slab_delta,
                     id.step_time() + step_time_delta,
                     id.substep() + static_cast<uint64_t>(substep_delta),
                     id.time() + time_delta);
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

  CHECK(get_output(id) ==
        "4:" + get_output(id.step_time()) + ":2:" + get_output(id.time()));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.TimeId", "[Unit][Time]") {
  check(true);
  check(false);
}
