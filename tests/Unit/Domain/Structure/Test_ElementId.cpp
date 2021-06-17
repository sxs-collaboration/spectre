// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Framework/TestHelpers.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep

namespace {
template <size_t VolumeDim>
void test_placement_new_and_hashing_impl(
    const size_t block1, const std::array<SegmentId, VolumeDim>& segments1,
    const size_t grid1, const size_t block2,
    const std::array<SegmentId, VolumeDim>& segments2, const size_t grid2) {
  using Hash = std::hash<ElementId<VolumeDim>>;

  const ElementId<VolumeDim> id1(block1, segments1, grid1);
  const ElementId<VolumeDim> id2(block2, segments2, grid2);

  ElementId<VolumeDim> test_id1{};
  ElementId<VolumeDim> test_id2{};
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
  const std::array<size_t, 3> grids{{0, 1, 3}};

  for (const auto& block1 : blocks) {
    for (const auto& block2 : blocks) {
      for (const auto& grid1 : grids) {
        for (const auto& grid2 : grids) {
          for (const auto& segment10 : segments) {
            for (const auto& segment20 : segments) {
              test_placement_new_and_hashing_impl<1>(
                  block1, {{segment10}}, grid1, block2, {{segment20}}, grid2);
              for (const auto& segment11 : segments) {
                for (const auto& segment21 : segments) {
                  test_placement_new_and_hashing_impl<2>(
                      block1, {{segment10, segment11}}, grid1, block2,
                      {{segment20, segment21}}, grid2);
                  for (const auto& segment12 : segments) {
                    for (const auto& segment22 : segments) {
                      test_placement_new_and_hashing_impl<3>(
                          block1, {{segment10, segment11, segment12}}, grid1,
                          block2, {{segment20, segment21, segment22}}, grid2);
                    }
                  }
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
  CHECK(block_2_3d.segment_id(0) == segment_ids[0]);
  CHECK(block_2_3d.segment_id(1) == segment_ids[1]);
  CHECK(block_2_3d.segment_id(2) == segment_ids[2]);
  CHECK(block_2_3d.grid_index() == 0);

  // Test parent and child operations:
  const auto check_parent_and_child = [](const ElementId<3>& id) {
    for (size_t dim = 0; dim < 3; dim++) {
      CHECK(id == id.id_of_child(dim, Side::Lower).id_of_parent(dim));
      CHECK(id == id.id_of_child(dim, Side::Upper).id_of_parent(dim));
      if (0 == id.segment_id(dim).index() % 2) {
        CHECK(id == id.id_of_parent(dim).id_of_child(dim, Side::Lower));
      } else {
        CHECK(id == id.id_of_parent(dim).id_of_child(dim, Side::Upper));
      }
    }
  };
  check_parent_and_child(block_2_3d);
  check_parent_and_child({3, segment_ids, 4});

  // Test equality operator:
  ElementId<3> element_one(1);
  ElementId<3> element_two(1);
  ElementId<3> element_three(2);
  ElementId<3> element_four(4);
  ElementId<3> element_five(4, 0);
  ElementId<3> element_six(4, 1);
  CHECK(element_one == element_two);
  CHECK(element_two != element_three);
  CHECK(element_two != element_four);
  CHECK(element_three != block_2_3d);
  CHECK(element_five == element_four);
  CHECK(element_six != element_four);

  // Test pup operations:
  test_serialization(element_one);
  test_serialization(element_six);

  // Test output operator:
  CHECK(get_output(block_2_3d) == "[B2,(L2I3,L1I0,L1I1)]");
  CHECK(get_output(element_six) == "[B4,(L0I0,L0I0,L0I0),G1]");

  CHECK(ElementId<3>::external_boundary_id().block_id() ==
        two_to_the(ElementId<3>::block_id_bits) - 1);
  CHECK(ElementId<3>::external_boundary_id().segment_ids() ==
        make_array<3>(SegmentId(ElementId<3>::max_refinement_level - 1, 0)));
  CHECK(ElementId<3>::external_boundary_id().grid_index() == 0);
}

template <size_t VolumeDim>
void test_serialization() noexcept {
  constexpr size_t volume_dim = VolumeDim;
  const ElementId<volume_dim> unused_id(0);
  const auto initial_ref_levels = make_array<volume_dim>(1_st);
  // We restrict the test to 2^7 blocks and 2^grid_index_bits-2 grid indices so
  // it finishes in a reasonable amount of time.
  for (size_t block_id = 0; block_id < two_to_the(7_st); ++block_id) {
    for (size_t grid_index = 0;
         grid_index < two_to_the(ElementId<VolumeDim>::grid_index_bits - 2);
         ++grid_index) {
      const std::vector<ElementId<volume_dim>> element_ids =
          initial_element_ids(block_id, initial_ref_levels, grid_index);
      for (const auto element_id : element_ids) {
        const auto serialized_id = serialize_and_deserialize(element_id);
        CHECK(serialized_id == element_id);
        // The following checks that ElementId can be used as a Charm array
        // index
        Parallel::ArrayIndex<ElementId<volume_dim>> array_index(element_id);
        CHECK(element_id == array_index.get_index());
        // now check pupping the ArrayIndex works...
        const auto serialized_array_index =
            serialize<Parallel::ArrayIndex<ElementId<volume_dim>>>(array_index);
        PUP::fromMem reader(serialized_array_index.data());
        Parallel::ArrayIndex<ElementId<volume_dim>> deserialized_array_index(
            unused_id);
        reader | deserialized_array_index;
        CHECK(array_index == deserialized_array_index);
        CHECK(element_id == deserialized_array_index.get_index());
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Structure.ElementId", "[Domain][Unit]") {
  test_element_id();
  test_placement_new_and_hashing();
  test_serialization<1>();
  test_serialization<2>();
  test_serialization<3>();
}

// [[OutputRegex, Block id out of bounds]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.Structure.ElementId.BadBlockId",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto failed_element_id =
      ElementId<1>(two_to_the(ElementId<1>::block_id_bits));
  static_cast<void>(failed_element_id);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Grid index out of bounds]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.Structure.ElementId.BadGridIndex",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto failed_element_id =
      ElementId<1>(0, {{{0, 0}}}, two_to_the(ElementId<1>::grid_index_bits));
  static_cast<void>(failed_element_id);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
