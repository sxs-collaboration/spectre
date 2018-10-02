// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <string>

#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/SegmentId.hpp"
#include "Domain/Side.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Domain.ElementId", "[Domain][Unit]") {
  // Test default constructor:
  ElementId<3> test_id;
  CHECK(test_id.block_id() == std::numeric_limits<size_t>::max());
  CHECK(test_id.segment_ids() == make_array<3>(SegmentId()));

  // Test retrieval functions:
  auto segment_ids = std::array<SegmentId, 3>(
      {{SegmentId(2, 3), SegmentId(1, 0), SegmentId(1, 1)}});
  ElementId<3> block_2_3d(2, segment_ids);
  CHECK(block_2_3d.block_id() == 2);
  CHECK(block_2_3d.segment_ids() == segment_ids);

  // Test parent and child operations:
  ElementId<3> id = block_2_3d;
  for (size_t dim = 0; dim < 3; dim++) {
    CHECK(id == id.id_of_child(dim, Side::Lower).id_of_parent(dim));
    CHECK(id == id.id_of_child(dim, Side::Upper).id_of_parent(dim));
    if (0 == gsl::at(id.segment_ids(), dim).index() % 2) {
      CHECK(id == id.id_of_parent(dim).id_of_child(dim, Side::Lower));
    } else {
      CHECK(id == id.id_of_parent(dim).id_of_child(dim, Side::Upper));
    }
  }

  // Test equality operator:
  ElementId<3> element_one(1);
  ElementId<3> element_two(1);
  ElementId<3> element_three(2);
  ElementId<3> element_four(4);
  CHECK(element_one == element_two);
  CHECK(element_two != element_three);
  CHECK(element_two != element_four);
  CHECK(element_three != block_2_3d);

  // Test pup operations:
  test_serialization(element_one);

  // Test output operator:
  CHECK(get_output(block_2_3d) == "[B2,(L2I3,L1I0,L1I1)]");

  CHECK(ElementId<3>::external_boundary_id().block_id() ==
        std::numeric_limits<size_t>::max() / 2);
  CHECK(ElementId<3>::external_boundary_id().segment_ids() ==
        make_array<3>(SegmentId(0, 0)));
}

SPECTRE_TEST_CASE("Unit.Domain.ElementId.ElementIndexConversion",
                  "[Domain][Unit]") {
  auto segment_ids = std::array<SegmentId, 3>(
      {{SegmentId(2, 3), SegmentId(1, 0), SegmentId(1, 1)}});
  ElementId<3> block_2_3d(2, segment_ids);
  CHECK(block_2_3d.block_id() == 2);
  CHECK(block_2_3d.segment_ids() == segment_ids);

  ElementIndex<3> block_2_3d_index(block_2_3d);
  CHECK(block_2_3d_index.block_id() == 2);
  CHECK(block_2_3d_index.segments().size() == segment_ids.size());
  for (size_t i = 0; i < segment_ids.size(); ++i) {
    CHECK(gsl::at(block_2_3d_index.segments(), i).block_id() == 2);
    CHECK(gsl::at(block_2_3d_index.segments(), i).index() ==
          gsl::at(segment_ids, i).index());
    CHECK(gsl::at(block_2_3d_index.segments(), i).refinement_level() ==
          gsl::at(segment_ids, i).refinement_level());
  }
  ElementId<3> block_2_3d_from_index(block_2_3d_index);
  CHECK(block_2_3d_from_index.block_id() == 2);
  CHECK(block_2_3d_from_index.segment_ids() == segment_ids);
}
