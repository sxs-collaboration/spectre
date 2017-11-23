// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "Time/Slab.hpp"
#include "Time/TimeId.hpp"

#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Time.TimeId", "[Unit][Time]") {
  using Hash = std::hash<TimeId>;

  const Slab slab(1.2, 3.4);

  const TimeId id{4, slab.start() + slab.duration() / 3, 2};

  {
    TimeId id2 = id;
    CHECK_FALSE(id2.is_at_slab_boundary());
    id2.substep = 0;
    CHECK_FALSE(id2.is_at_slab_boundary());
    id2 = id;
    id2.time = slab.start();
    CHECK_FALSE(id2.is_at_slab_boundary());
    id2.time = slab.end();
    CHECK_FALSE(id2.is_at_slab_boundary());
    id2.substep = 0;
    CHECK(id2.is_at_slab_boundary());
    id2.time = slab.start();
    CHECK(id2.is_at_slab_boundary());
  }

  CHECK(id == id);
  CHECK_FALSE(id != id);
  CHECK(id == TimeId(id));
  const size_t hash = Hash{}(id);

  {
    TimeId id2 = id;
    id2.slab_number = 5;
    CHECK(id != id2);
    CHECK_FALSE(id == id2);
    CHECK(hash != Hash{}(id2));
  }

  {
    TimeId id2 = id;
    id2.time += slab.duration() / 2;
    CHECK(id != id2);
    CHECK_FALSE(id == id2);
    CHECK(hash != Hash{}(id2));
  }

  {
    TimeId id2 = id;
    id2.substep = 3;
    CHECK(id != id2);
    CHECK_FALSE(id == id2);
    CHECK(hash != Hash{}(id2));
  }

  test_serialization(id);

  CHECK(get_output(id) == "4:" + get_output(id.time) + ":2");
}
