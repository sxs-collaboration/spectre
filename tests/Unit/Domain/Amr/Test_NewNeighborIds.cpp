// // Distributed under the MIT License.
// // See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/NewNeighborIds.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/Amr/NeighborFlagHelpers.hpp"
#include "Helpers/Domain/Structure/NeighborHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/StdHelpers.hpp"

namespace {
template <size_t Dim>
std::vector<ElementId<Dim>> element_ids_to_test() {
  static constexpr size_t max_level = 5 - Dim;
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
void test(const gsl::not_null<std::mt19937*> generator) {
  for (const auto& element_id : element_ids_to_test<Dim>()) {
    CAPTURE(element_id);
    for (const auto& direction :
         random_sample(2, Direction<Dim>::all_directions(), generator)) {
      CAPTURE(direction);
      const size_t dim = direction.dimension();
      const Side side = direction.side();
      const SegmentId& normal_segment = element_id.segment_id(dim);
      const double endpoint = normal_segment.endpoint(side);
      if (endpoint == 1.0 or endpoint == -1.0) {
        for (const auto face_type :
             std::array{TestHelpers::domain::FaceType::Periodic,
                        TestHelpers::domain::FaceType::Block}) {
          for (const auto& neighbors :
               random_sample(5,
                             TestHelpers::domain::valid_neighbors(
                                 generator, element_id, direction, face_type),
                             generator)) {
            CAPTURE(neighbors);
            for (const auto& neighbor_flags : random_sample(
                     5,
                     TestHelpers::amr::valid_neighbor_flags(
                         element_id,
                         make_array<Dim>(::amr::Flag::IncreaseResolution),
                         neighbors),
                     generator)) {
              CAPTURE(neighbor_flags);
              const auto new_neighbors =
                  Neighbors<Dim>{new_neighbor_ids(element_id, direction,
                                                  neighbors, neighbor_flags),
                                 neighbors.orientation()};
              CAPTURE(new_neighbors);
              TestHelpers::domain::check_neighbors(new_neighbors, element_id,
                                                   direction);
            }
          }
        }
      } else {
        for (const auto& neighbors :
             random_sample(5,
                           TestHelpers::domain::valid_neighbors(
                               generator, element_id, direction),
                           generator)) {
          CAPTURE(neighbors);
          for (const auto& neighbor_flags : random_sample(
                   5,
                   TestHelpers::amr::valid_neighbor_flags(
                       element_id,
                       make_array<Dim>(::amr::Flag::IncreaseResolution),
                       neighbors),
                   generator)) {
            CAPTURE(neighbor_flags);
            const auto new_neighbors =
                Neighbors<Dim>{new_neighbor_ids(element_id, direction,
                                                neighbors, neighbor_flags),
                               neighbors.orientation()};
            CAPTURE(new_neighbors);
            TestHelpers::domain::check_neighbors(new_neighbors, element_id,
                                                 direction);
          }
        }
      }
    }
  }
}
}  // namespace

// [[TimeOut, 10]]
SPECTRE_TEST_CASE("Unit.Domain.Amr.NewNeighborIds", "[Domain][Unit]") {
  MAKE_GENERATOR(generator);
  test<1>(make_not_null(&generator));
  test<2>(make_not_null(&generator));
  test<3>(make_not_null(&generator));
}
