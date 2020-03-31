// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <string>

#include "Domain/ElementId.hpp"
#include "Domain/SegmentId.hpp"
#include "Domain/Side.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep

namespace {
template <size_t VolumeDim>
void test_placement_new_and_hashing_impl(
    const size_t block1, const std::array<SegmentId, VolumeDim>& segments1,
    const size_t block2, const std::array<SegmentId, VolumeDim>& segments2) {
  using Hash = std::hash<ElementId<VolumeDim>>;

  const ElementId<VolumeDim> id1(block1, segments1);
  const ElementId<VolumeDim> id2(block2, segments2);

  ElementId<VolumeDim> test_id1{}, test_id2{};
  // Check for nondeterminacy due to previous memory state.
  std::memset(&test_id1, 0, sizeof(test_id1));
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ > 7)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif  // defined(__GNUC__) && !defined(__clang__) && (__GNUC__ > 7)
  std::memset(&test_id2, 255, sizeof(test_id2));
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ > 7)
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) && !defined(__clang__) && (__GNUC__ > 7)
  new (&test_id1) ElementId<VolumeDim>(id1);
  new (&test_id2) ElementId<VolumeDim>(id2);

  CHECK((test_id1 == test_id2) == (id1 == id2));
  CHECK((test_id1 != test_id2) == (id1 != id2));
  CHECK((Hash{}(test_id1) == Hash{}(test_id2)) == (id1 == id2));
}

void test_placement_new_and_hashing() {
  const std::array<size_t, 3> blocks{{0, 1, 4}};
  const std::array<SegmentId, 4> segments{{{0, 0}, {1, 0}, {1, 1}, {8, 4}}};

  for (const auto& block1 : blocks) {
    for (const auto& block2 : blocks) {
      for (const auto& segment10 : segments) {
        for (const auto& segment20 : segments) {
          test_placement_new_and_hashing_impl<1>(block1, {{segment10}}, block2,
                                                 {{segment20}});
          for (const auto& segment11 : segments) {
            for (const auto& segment21 : segments) {
              test_placement_new_and_hashing_impl<2>(
                  block1, {{segment10, segment11}}, block2,
                  {{segment20, segment21}});
              for (const auto& segment12 : segments) {
                for (const auto& segment22 : segments) {
                  test_placement_new_and_hashing_impl<3>(
                      block1, {{segment10, segment11, segment12}}, block2,
                      {{segment20, segment21, segment22}});
                }
              }
            }
          }
        }
      }
    }
  }
}

void test_element_id() {
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
        two_to_the(SegmentId::block_id_bits) - 1);
  CHECK(ElementId<3>::external_boundary_id().segment_ids() ==
        make_array<3>(SegmentId(SegmentId::max_refinement_level, 0)));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.ElementId", "[Domain][Unit]") {
  test_element_id();
  test_placement_new_and_hashing();
}
