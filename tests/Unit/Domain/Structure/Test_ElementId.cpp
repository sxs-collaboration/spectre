// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <random>
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
template <size_t Dim>
struct DirectionTester : public ElementId<Dim> {
  DirectionTester(const Direction<Dim>& direction, const ElementId<Dim>& id)
      : ElementId<Dim>(direction, id) {}
};

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

  for (const auto& dir : Direction<VolumeDim>::all_directions()) {
    const DirectionTester<VolumeDim> dir_test_id1{dir, test_id1};
    CHECK(static_cast<const ElementId<VolumeDim>&>(dir_test_id1) == test_id1);
    CHECK(Hash{}(dir_test_id1) == Hash{}(test_id1));
  }
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
  CHECK(block_2_3d.refinement_levels() == std::array{2_st, 1_st, 1_st});
  CHECK(ElementId<2>{31, {{{6, 3}, {9, 67}}}}.refinement_levels() ==
        std::array{6_st, 9_st});
  CHECK(ElementId<1>{4, {{{4, 7}}}}.refinement_levels() == std::array{4_st});

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

  // Test comparison operator
  check_cmp(ElementId<1>{1, {{{1, 0}}}, 0}, ElementId<1>{1, {{{1, 1}}}, 0});
  check_cmp(ElementId<1>{0, {{{1, 0}}}, 0}, ElementId<1>{1, {{{1, 0}}}, 0});
  check_cmp(ElementId<1>{1, {{{0, 0}}}, 0}, ElementId<1>{1, {{{1, 0}}}, 0});
  check_cmp(ElementId<1>{1, {{{1, 0}}}, 0}, ElementId<1>{1, {{{1, 0}}}, 1});
  check_cmp(ElementId<2>{1, {{{1, 0}, {2, 1}}}, 0},
            ElementId<2>{1, {{{1, 0}, {2, 2}}}, 0});
  check_cmp(ElementId<2>{1, {{{1, 0}, {2, 1}}}, 0},
            ElementId<2>{1, {{{1, 1}, {2, 1}}}, 0});
  check_cmp(ElementId<2>{1, {{{1, 0}, {2, 1}}}, 0},
            ElementId<2>{1, {{{1, 1}, {1, 1}}}, 0});
  check_cmp(ElementId<2>{1, {{{0, 0}, {2, 1}}}, 0},
            ElementId<2>{1, {{{1, 0}, {1, 1}}}, 0});
  check_cmp(ElementId<2>{1, {{{1, 1}, {1, 1}}}, 0},
            ElementId<2>{1, {{{1, 1}, {2, 1}}}, 0});
  check_cmp(ElementId<2>{0, {{{1, 0}, {2, 1}}}, 0},
            ElementId<2>{1, {{{1, 0}, {2, 1}}}, 0});
  check_cmp(ElementId<2>{1, {{{1, 0}, {2, 1}}}, 0},
            ElementId<2>{1, {{{1, 0}, {2, 1}}}, 1});
  check_cmp(ElementId<3>{1, {{{1, 0}, {2, 1}, {1, 0}}}, 0},
            ElementId<3>{1, {{{1, 0}, {2, 2}, {1, 0}}}, 0});
  check_cmp(ElementId<3>{1, {{{1, 0}, {2, 1}, {1, 0}}}, 0},
            ElementId<3>{1, {{{1, 1}, {2, 1}, {1, 0}}}, 0});
  check_cmp(ElementId<3>{1, {{{1, 0}, {2, 1}, {1, 0}}}, 0},
            ElementId<3>{1, {{{1, 0}, {2, 1}, {1, 1}}}, 0});
  check_cmp(ElementId<3>{0, {{{1, 0}, {2, 1}, {1, 0}}}, 0},
            ElementId<3>{1, {{{1, 0}, {2, 1}, {1, 0}}}, 0});
  check_cmp(ElementId<3>{1, {{{1, 0}, {2, 1}, {1, 0}}}, 0},
            ElementId<3>{1, {{{1, 0}, {2, 1}, {1, 0}}}, 1});

  // Test pup operations:
  test_serialization(element_one);
  test_serialization(element_six);

  // Test output operator:
  CHECK(get_output(block_2_3d) == "[B2,(L2I3,L1I0,L1I1)]");
  CHECK(get_output(element_six) == "[B4,(L0I0,L0I0,L0I0),G1]");
  CHECK(ElementId<3>{"[B2,(L2I3,L1I0,L1I1)]"} == block_2_3d);
  CHECK(ElementId<3>{"[B4,(L0I0,L0I0,L0I0),G1]"} == element_six);
  CHECK(ElementId<1>{"[B1,(L2I3)]"} == ElementId<1>{1, {{{2, 3}}}});
  CHECK(ElementId<1>{"[B1,(L2I3),G2]"} == ElementId<1>{1, {{{2, 3}}}, 2});
  CHECK(ElementId<1>{"[B10,(L2I3)]"} == ElementId<1>{10, {{{2, 3}}}});
  CHECK(ElementId<2>{"[B2,(L1I1,L2I0)]"} ==
        ElementId<2>{2, {{{1, 1}, {2, 0}}}});
  CHECK(ElementId<2>{"[B2,(L1I1,L2I0),G12]"} ==
        ElementId<2>{2, {{{1, 1}, {2, 0}}}, 12});
  CHECK(ElementId<2>{"[B52,(L12I133,L6I38)]"} ==
        ElementId<2>{52, {{{12, 133}, {6, 38}}}});
  CHECK_THROWS_WITH(ElementId<1>("somegrid"),
                    Catch::Matchers::ContainsSubstring("Invalid grid name"));
  CHECK_THROWS_WITH(ElementId<2>("[B0,(L1I0)]"),
                    Catch::Matchers::ContainsSubstring("Invalid grid name"));
  CHECK_THROWS_WITH(ElementId<2>("[B0]"),
                    Catch::Matchers::ContainsSubstring("Invalid grid name"));
  CHECK_THROWS_WITH(ElementId<3>("L1I0,L2I1,L2I0"),
                    Catch::Matchers::ContainsSubstring("Invalid grid name"));

  CHECK(ElementId<3>::external_boundary_id().block_id() ==
        two_to_the(ElementId<3>::block_id_bits) - 1);
  CHECK(ElementId<3>::external_boundary_id().segment_ids() ==
        make_array<3>(SegmentId(ElementId<3>::max_refinement_level - 1, 0)));
  CHECK(ElementId<3>::external_boundary_id().grid_index() == 0);

  ElementId<3> element1(0);
  ElementId<3> element2{0, 1};
  ElementId<3> element3(1);
  ElementId<3> element4{1, 1};
  ElementId<3> element5{0, {{{1, 0}, {2, 0}, {1, 0}}}};
  ElementId<3> element6{0, {{{1, 0}, {2, 0}, {1, 0}}}, 1};
  ElementId<3> element7{0, {{{1, 0}, {2, 1}, {1, 0}}}};
  ElementId<3> element8{0, {{{1, 0}, {2, 1}, {1, 0}}}, 1};
  ElementId<3> element9{1, {{{1, 0}, {2, 0}, {1, 0}}}};
  ElementId<3> element10{1, {{{1, 0}, {2, 0}, {1, 0}}}, 1};
  ElementId<3> element11{1, {{{1, 0}, {2, 1}, {1, 0}}}};
  ElementId<3> element12{1, {{{1, 0}, {2, 1}, {1, 0}}}, 1};

  CHECK(is_zeroth_element(element1));
  CHECK_FALSE(is_zeroth_element(element1, {1}));
  CHECK(is_zeroth_element(element2));
  CHECK_FALSE(is_zeroth_element(element2, {0}));
  CHECK(is_zeroth_element(element2, {1}));
  CHECK(is_zeroth_element(element5));
  CHECK_FALSE(is_zeroth_element(element5, {1}));
  CHECK(is_zeroth_element(element6));
  CHECK_FALSE(is_zeroth_element(element6, {0}));
  CHECK(is_zeroth_element(element6, {1}));
  // Do this just so we don't duplicate the same two checks over and over for
  // all elements that aren't the zeroth element
  const std::vector<ElementId<3>> not_zeroth_elements{
      {element3, element4, element7, element8, element9, element10, element11,
       element12}};
  for (const auto& element : not_zeroth_elements) {
    CHECK_FALSE(is_zeroth_element(element));
    CHECK_FALSE(is_zeroth_element(element, {1}));
  }
}

template <size_t VolumeDim>
void test_serialization() {
  constexpr size_t volume_dim = VolumeDim;

  // Generate random element IDs so we test the full range of possible values
  MAKE_GENERATOR(gen);
  std::uniform_int_distribution<size_t> dist_block_id(
      0, two_to_the(ElementId<VolumeDim>::block_id_bits) - 1);
  std::uniform_int_distribution<size_t> dist_grid_index(
      0, two_to_the(ElementId<VolumeDim>::grid_index_bits) - 1);
  std::uniform_int_distribution<size_t> dist_refinement(
      0, two_to_the(ElementId<VolumeDim>::refinement_bits) - 1);
  const auto random_segment_id = [&gen, &dist_refinement]() -> SegmentId {
    const size_t refinement = dist_refinement(gen);
    std::uniform_int_distribution<size_t> dist_index(
        0, two_to_the(refinement) - 1);
    return {refinement, dist_index(gen)};
  };
  const auto random_segment_ids =
      [&random_segment_id]() -> std::array<SegmentId, VolumeDim> {
    if constexpr (VolumeDim == 1) {
      return {{random_segment_id()}};
    } else if constexpr (VolumeDim == 2) {
      return {{random_segment_id(), random_segment_id()}};
    } else {
      return {{random_segment_id(), random_segment_id(), random_segment_id()}};
    }
  };

  const ElementId<volume_dim> unused_id(0);
  for (size_t i = 0; i < 100; ++i) {
    ElementId<volume_dim> element_id{dist_block_id(gen), random_segment_ids(),
                                     dist_grid_index(gen)};
    CAPTURE(element_id);

    // Test serialization
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

    // Check roundtrip to string representation and back
    CHECK(ElementId<VolumeDim>(get_output(element_id)) == element_id);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Structure.ElementId", "[Domain][Unit]") {
  test_element_id();
  test_placement_new_and_hashing();
  test_serialization<1>();
  test_serialization<2>();
  test_serialization<3>();
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      ElementId<1>(two_to_the(ElementId<1>::block_id_bits)),
      Catch::Matchers::ContainsSubstring("Block id out of bounds"));
  CHECK_THROWS_WITH(
      ElementId<1>(0, {{{0, 0}}}, two_to_the(ElementId<1>::grid_index_bits)),
      Catch::Matchers::ContainsSubstring("Grid index out of bounds"));
#endif
}
