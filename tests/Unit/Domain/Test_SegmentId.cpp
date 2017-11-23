// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "Domain/SegmentId.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Domain.SegmentId", "[Domain][Unit]") {
  // Test default constructor:
  SegmentId test_id;
  CHECK(test_id.index() == std::numeric_limits<size_t>::max());
  CHECK(test_id.refinement_level() == std::numeric_limits<size_t>::max());
  static_cast<void>(test_id);

  // Test equality operator:
  SegmentId segment_one(4, 3);
  SegmentId segment_two(4, 3);
  SegmentId segment_three(4, 0);
  SegmentId segment_four(5, 4);
  CHECK(segment_one == segment_two);
  CHECK(segment_two != segment_three);
  CHECK(segment_two != segment_four);

  // Test pup operations:
  test_serialization(segment_one);

  // Test parent and child operations:
  for (size_t level = 1; level < 5; ++level) {
    const double segment_length = 2.0 / two_to_the(level);
    double midpoint = -1.0 + 0.5 * segment_length;
    for (size_t segment_index = 0; segment_index < two_to_the(level);
         ++segment_index) {
      SegmentId id(level, segment_index);
      CHECK(id.midpoint() == midpoint);
      CHECK((id.endpoint(Side::Upper) + id.endpoint(Side::Lower)) / 2. ==
            midpoint);
      CHECK(id.endpoint(Side::Upper) - id.endpoint(Side::Lower) ==
            segment_length);
      midpoint += segment_length;
      CHECK(id == id.id_of_child(Side::Lower).id_of_parent());
      CHECK(id == id.id_of_child(Side::Upper).id_of_parent());
      CHECK(id.overlaps(id));
      CHECK(id.overlaps(id.id_of_parent()));
      const Side side_of_parent =
          0 == segment_index % 2 ? Side::Lower : Side::Upper;
      CHECK(id == id.id_of_parent().id_of_child(side_of_parent));
      CHECK_FALSE(
          id.overlaps(id.id_of_parent().id_of_child(opposite(side_of_parent))));
    }
  }

  // Test retrieval functions:
  SegmentId level_2_index_3(2, 3);
  CHECK(level_2_index_3.refinement_level() == 2);
  CHECK(level_2_index_3.index() == 3);

  // Test output operator:
  SegmentId level_3_index_2(3, 2);
  CHECK(get_output(level_3_index_2) == "L3I2");
}
