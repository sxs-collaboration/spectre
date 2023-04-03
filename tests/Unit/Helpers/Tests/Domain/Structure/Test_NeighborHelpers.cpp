// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>

#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/Structure/NeighborHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"

namespace TestHelpers::domain {
namespace {
void test_valid_sibling_side_segments() {
  for (size_t l = 1; l < 5; ++l) {
    for (size_t i = 0; i < two_to_the(l); ++i) {
      SegmentId my_id{l, i};
      const auto valid_segments = valid_neighbor_segments(my_id);
      if (i % 2 == 0) {
        CHECK(valid_segments ==
              std::vector{SegmentId{l, i + 1}, SegmentId{l + 1, 2 * i + 2}});
      } else {
        CHECK(valid_segments ==
              std::vector{SegmentId{l, i - 1}, SegmentId{l + 1, 2 * i - 1}});
      }
    }
  }
}

void test_valid_nonsibling_side_segments() {
  CHECK(valid_neighbor_segments(SegmentId{1, 0}, FaceType::Periodic) ==
        std::vector{SegmentId{1, 1}, SegmentId{2, 3}});
  CHECK(valid_neighbor_segments(SegmentId{1, 1}, FaceType::Periodic) ==
        std::vector{SegmentId{1, 0}, SegmentId{2, 0}});
  for (size_t l = 1; l < 5; ++l) {
    SegmentId lower_boundary_segment{l, 0};
    SegmentId upper_boundary_segment{l, two_to_the(l) - 1};
    CHECK(valid_neighbor_segments(lower_boundary_segment, FaceType::External) ==
          std::vector<SegmentId>{});
    CHECK(valid_neighbor_segments(upper_boundary_segment, FaceType::External) ==
          std::vector<SegmentId>{});
    CHECK(valid_neighbor_segments(lower_boundary_segment, FaceType::Block) ==
          std::vector{SegmentId{l - 1, two_to_the(l - 1) - 1},
                      SegmentId{l, two_to_the(l) - 1},
                      SegmentId{l + 1, two_to_the(l + 1) - 1}});
    CHECK(
        valid_neighbor_segments(upper_boundary_segment, FaceType::Block) ==
        std::vector{SegmentId{l - 1, 0}, SegmentId{l, 0}, SegmentId{l + 1, 0}});
    if (l > 1) {
      CHECK(
          valid_neighbor_segments(lower_boundary_segment, FaceType::Periodic) ==
          std::vector{SegmentId{l - 1, two_to_the(l - 1) - 1},
                      SegmentId{l, two_to_the(l) - 1},
                      SegmentId{l + 1, two_to_the(l + 1) - 1}});
      CHECK(
          valid_neighbor_segments(upper_boundary_segment, FaceType::Periodic) ==
          std::vector{SegmentId{l - 1, 0}, SegmentId{l, 0},
                      SegmentId{l + 1, 0}});
    }
  }
  for (size_t l = 2; l < 5; ++l) {
    for (size_t i = 1; i < two_to_the(l) - 1; ++i) {
      SegmentId my_id{l, i};
      const auto valid_segments =
          valid_neighbor_segments(my_id, FaceType::Internal);
      if (i % 2 == 0) {
        CHECK(valid_segments == std::vector{SegmentId{l - 1, i / 2 - 1},
                                            SegmentId{l, i - 1},
                                            SegmentId{l + 1, 2 * i - 1}});
      } else {
        CHECK(valid_segments == std::vector{SegmentId{l - 1, i / 2 + 1},
                                            SegmentId{l, i + 1},
                                            SegmentId{l + 1, 2 * i + 2}});
      }
    }
  }
}

template <size_t Dim>
std::vector<ElementId<Dim>> element_ids_to_test() {
  static constexpr size_t max_level = 2;
  std::vector<ElementId<Dim>> result{};
  for (size_t lx = 0; lx <= max_level; ++lx) {
    for (size_t ix = 0; ix < two_to_the(lx); ++ix) {
      if constexpr (Dim == 1) {
        result.emplace_back(0, std::array{SegmentId{lx, ix}});
      } else {
        for (size_t ly = 0; ly <= max_level; ++ly) {
          for (size_t iy = 0; iy < two_to_the(ly); ++iy) {
            if constexpr (Dim == 2) {
              result.emplace_back(
                  0, std::array{SegmentId{lx, ix}, SegmentId{ly, iy}});
            } else {
              for (size_t lz = 0; lz <= max_level; ++lz) {
                for (size_t iz = 0; iz < two_to_the(lz); ++iz) {
                  result.emplace_back(
                      0, std::array{SegmentId{lx, ix}, SegmentId{ly, iy},
                                    SegmentId{lz, iz}});
                }
              }
            }
          }
        }
      }
    }
  }
  return result;
}

template <size_t Dim>
void test_valid_neighbors(const gsl::not_null<std::mt19937*> generator) {
  for (const auto& element_id : element_ids_to_test<Dim>()) {
    for (const auto& direction : Direction<Dim>::all_directions()) {
      const size_t dim = direction.dimension();
      const Side side = direction.side();
      const SegmentId& normal_segment = element_id.segment_id(dim);
      const double endpoint = normal_segment.endpoint(side);
      if (endpoint == 1.0 or endpoint == -1.0) {
        for (const auto face_type :
             std::array{FaceType::Periodic, FaceType::Block}) {
          for (const auto& neighbors :
               valid_neighbors(generator, element_id, direction, face_type)) {
            check_neighbors(neighbors, element_id, direction);
          }
        }
      } else {
        for (const auto& neighbors :
             valid_neighbors(generator, element_id, direction)) {
          check_neighbors(neighbors, element_id, direction);
        }
      }
    }
  }
}

}  // namespace

// [[TimeOut, 10]]
SPECTRE_TEST_CASE("TestHelpers.Domain.Structure.NeighborHelpers",
                  "[Domain][Unit]") {
  MAKE_GENERATOR(generator);
  test_valid_sibling_side_segments();
  test_valid_nonsibling_side_segments();
  test_valid_neighbors<1>(make_not_null(&generator));
  test_valid_neighbors<2>(make_not_null(&generator));
  test_valid_neighbors<3>(make_not_null(&generator));
}
}  // namespace TestHelpers::domain
