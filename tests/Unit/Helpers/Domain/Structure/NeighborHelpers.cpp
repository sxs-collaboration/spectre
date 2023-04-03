// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Helpers/Domain/Structure/NeighborHelpers.hpp"

#include <array>
#include <unordered_set>
#include <vector>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Helpers/Domain/Structure/OrientationMapHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <size_t Dim>
std::vector<std::unordered_set<ElementId<Dim>>> valid_aligned_neighbor_ids(
    const ElementId<Dim>& element_id, size_t neighbor_block_id,
    size_t normal_dim,
    const std::vector<SegmentId>& valid_neighbor_normal_segments);

template <>
std::vector<std::unordered_set<ElementId<1>>> valid_aligned_neighbor_ids<1>(
    const ElementId<1>& /*element_id*/, size_t neighbor_block_id,
    size_t /*normal_dim*/,
    const std::vector<SegmentId>& valid_neighbor_normal_segments) {
  std::vector<std::unordered_set<ElementId<1>>> result{};
  for (const auto& normal_segment : valid_neighbor_normal_segments) {
    result.emplace_back(std::unordered_set{
        ElementId<1>{neighbor_block_id, std::array{normal_segment}}});
  }
  return result;
}

std::array<SegmentId, 2> make_segment_ids(const SegmentId& normal_segment,
                                          const SegmentId& transverse_segment,
                                          const size_t normal_dim) {
  return normal_dim == 0 ? std::array{normal_segment, transverse_segment}
                         : std::array{transverse_segment, normal_segment};
}

template <>
std::vector<std::unordered_set<ElementId<2>>> valid_aligned_neighbor_ids<2>(
    const ElementId<2>& element_id, size_t neighbor_block_id,
    const size_t normal_dim,
    const std::vector<SegmentId>& valid_neighbor_normal_segments) {
  const SegmentId transverse_segment =
      element_id.segment_id(normal_dim == 0 ? 1 : 0);
  std::vector<std::unordered_set<ElementId<2>>> result{};
  for (const auto& normal_segment : valid_neighbor_normal_segments) {
    result.emplace_back(std::unordered_set{ElementId<2>{
        neighbor_block_id,
        make_segment_ids(normal_segment, transverse_segment, normal_dim)}});
    if (normal_segment == SegmentId{0, 0} and
        neighbor_block_id == element_id.block_id()) {
      // periodic so transverse extents must match
      continue;
    }
    if (transverse_segment.refinement_level() > 0) {
      result.emplace_back(std::unordered_set{ElementId<2>{
          neighbor_block_id,
          make_segment_ids(normal_segment, transverse_segment.id_of_parent(),
                           normal_dim)}});
    }
    result.emplace_back(std::unordered_set{
        ElementId<2>{
            neighbor_block_id,
            make_segment_ids(normal_segment,
                             transverse_segment.id_of_child(Side::Lower),
                             normal_dim)},
        ElementId<2>{
            neighbor_block_id,
            make_segment_ids(normal_segment,
                             transverse_segment.id_of_child(Side::Upper),
                             normal_dim)}});
    if (normal_segment != valid_neighbor_normal_segments.front()) {
      result.emplace_back(std::unordered_set{
          ElementId<2>{
              neighbor_block_id,
              make_segment_ids(normal_segment.id_of_parent(),
                               transverse_segment.id_of_child(Side::Lower),
                               normal_dim)},
          ElementId<2>{
              neighbor_block_id,
              make_segment_ids(normal_segment,
                               transverse_segment.id_of_child(Side::Upper),
                               normal_dim)}});
      result.emplace_back(std::unordered_set{
          ElementId<2>{
              neighbor_block_id,
              make_segment_ids(normal_segment,
                               transverse_segment.id_of_child(Side::Lower),
                               normal_dim)},
          ElementId<2>{
              neighbor_block_id,
              make_segment_ids(normal_segment.id_of_parent(),
                               transverse_segment.id_of_child(Side::Upper),
                               normal_dim)}});
    }
  }
  return result;
}

std::array<SegmentId, 3> make_segment_ids(
    const SegmentId& normal_segment, const SegmentId& first_transverse_segment,
    const SegmentId& second_transverse_segment, const size_t normal_dim) {
  return normal_dim == 0
             ? std::array{normal_segment, first_transverse_segment,
                          second_transverse_segment}
             : (normal_dim == 1
                    ? std::array{first_transverse_segment, normal_segment,
                                 second_transverse_segment}
                    : std::array{first_transverse_segment,
                                 second_transverse_segment, normal_segment});
}

template <>
std::vector<std::unordered_set<ElementId<3>>> valid_aligned_neighbor_ids<3>(
    const ElementId<3>& element_id, size_t neighbor_block_id, size_t normal_dim,
    const std::vector<SegmentId>& valid_neighbor_normal_segments) {
  const SegmentId first_transverse_segment =
      element_id.segment_id(normal_dim == 0 ? 1 : 0);
  const SegmentId second_transverse_segment =
      element_id.segment_id(normal_dim == 2 ? 1 : 2);
  std::vector<std::unordered_set<ElementId<3>>> result{};
  for (const auto& normal_segment : valid_neighbor_normal_segments) {
    result.emplace_back(std::unordered_set{
        ElementId<3>{neighbor_block_id,
                     make_segment_ids(normal_segment, first_transverse_segment,
                                      second_transverse_segment, normal_dim)}});
    if (normal_segment == SegmentId{0, 0} and
        neighbor_block_id == element_id.block_id()) {
      // periodic so transverse extents must match
      continue;
    }
    if (first_transverse_segment.refinement_level() > 0) {
      result.emplace_back(std::unordered_set{ElementId<3>{
          neighbor_block_id,
          make_segment_ids(normal_segment,
                           first_transverse_segment.id_of_parent(),
                           second_transverse_segment, normal_dim)}});
      if (second_transverse_segment.refinement_level() > 0) {
        result.emplace_back(std::unordered_set{ElementId<3>{
            neighbor_block_id,
            make_segment_ids(
                normal_segment, first_transverse_segment.id_of_parent(),
                second_transverse_segment.id_of_parent(), normal_dim)}});
      }
    }
    if (second_transverse_segment.refinement_level() > 0) {
      result.emplace_back(std::unordered_set{ElementId<3>{
          neighbor_block_id,
          make_segment_ids(normal_segment, first_transverse_segment,
                           second_transverse_segment.id_of_parent(),
                           normal_dim)}});
    }
    result.emplace_back(std::unordered_set{
        ElementId<3>{
            neighbor_block_id,
            make_segment_ids(normal_segment,
                             first_transverse_segment.id_of_child(Side::Lower),
                             second_transverse_segment, normal_dim)},
        ElementId<3>{
            neighbor_block_id,
            make_segment_ids(normal_segment,
                             first_transverse_segment.id_of_child(Side::Upper),
                             second_transverse_segment, normal_dim)}});
    if (second_transverse_segment.refinement_level() > 0) {
      result.emplace_back(std::unordered_set{
          ElementId<3>{
              neighbor_block_id,
              make_segment_ids(
                  normal_segment,
                  first_transverse_segment.id_of_child(Side::Lower),
                  second_transverse_segment.id_of_parent(), normal_dim)},
          ElementId<3>{neighbor_block_id,
                       make_segment_ids(
                           normal_segment,
                           first_transverse_segment.id_of_child(Side::Upper),
                           second_transverse_segment, normal_dim)}});
      result.emplace_back(std::unordered_set{
          ElementId<3>{neighbor_block_id,
                       make_segment_ids(
                           normal_segment,
                           first_transverse_segment.id_of_child(Side::Lower),
                           second_transverse_segment, normal_dim)},
          ElementId<3>{
              neighbor_block_id,
              make_segment_ids(
                  normal_segment,
                  first_transverse_segment.id_of_child(Side::Upper),
                  second_transverse_segment.id_of_parent(), normal_dim)}});
      result.emplace_back(std::unordered_set{
          ElementId<3>{
              neighbor_block_id,
              make_segment_ids(
                  normal_segment,
                  first_transverse_segment.id_of_child(Side::Lower),
                  second_transverse_segment.id_of_parent(), normal_dim)},
          ElementId<3>{
              neighbor_block_id,
              make_segment_ids(
                  normal_segment,
                  first_transverse_segment.id_of_child(Side::Upper),
                  second_transverse_segment.id_of_parent(), normal_dim)}});
    }
    result.emplace_back(std::unordered_set{
        ElementId<3>{
            neighbor_block_id,
            make_segment_ids(normal_segment, first_transverse_segment,
                             second_transverse_segment.id_of_child(Side::Lower),
                             normal_dim)},
        ElementId<3>{
            neighbor_block_id,
            make_segment_ids(normal_segment, first_transverse_segment,
                             second_transverse_segment.id_of_child(Side::Upper),
                             normal_dim)}});
    if (first_transverse_segment.refinement_level() > 0) {
      result.emplace_back(std::unordered_set{
          ElementId<3>{
              neighbor_block_id,
              make_segment_ids(
                  normal_segment, first_transverse_segment.id_of_parent(),
                  second_transverse_segment.id_of_child(Side::Lower),
                  normal_dim)},
          ElementId<3>{neighbor_block_id,
                       make_segment_ids(
                           normal_segment, first_transverse_segment,
                           second_transverse_segment.id_of_child(Side::Upper),
                           normal_dim)}});
      result.emplace_back(std::unordered_set{
          ElementId<3>{neighbor_block_id,
                       make_segment_ids(
                           normal_segment, first_transverse_segment,
                           second_transverse_segment.id_of_child(Side::Lower),
                           normal_dim)},
          ElementId<3>{
              neighbor_block_id,
              make_segment_ids(
                  normal_segment, first_transverse_segment.id_of_parent(),
                  second_transverse_segment.id_of_child(Side::Upper),
                  normal_dim)}});
      result.emplace_back(std::unordered_set{
          ElementId<3>{
              neighbor_block_id,
              make_segment_ids(
                  normal_segment, first_transverse_segment.id_of_parent(),
                  second_transverse_segment.id_of_child(Side::Lower),
                  normal_dim)},
          ElementId<3>{
              neighbor_block_id,
              make_segment_ids(
                  normal_segment, first_transverse_segment.id_of_parent(),
                  second_transverse_segment.id_of_child(Side::Upper),
                  normal_dim)}});
    }
    result.emplace_back(std::unordered_set{
        ElementId<3>{
            neighbor_block_id,
            make_segment_ids(normal_segment,
                             first_transverse_segment.id_of_child(Side::Lower),
                             second_transverse_segment.id_of_child(Side::Lower),
                             normal_dim)},
        ElementId<3>{
            neighbor_block_id,
            make_segment_ids(normal_segment,
                             first_transverse_segment.id_of_child(Side::Upper),
                             second_transverse_segment.id_of_child(Side::Lower),
                             normal_dim)},
        ElementId<3>{
            neighbor_block_id,
            make_segment_ids(normal_segment, first_transverse_segment,
                             second_transverse_segment.id_of_child(Side::Upper),
                             normal_dim)}});
    result.emplace_back(std::unordered_set{
        ElementId<3>{
            neighbor_block_id,
            make_segment_ids(normal_segment, first_transverse_segment,
                             second_transverse_segment.id_of_child(Side::Lower),
                             normal_dim)},
        ElementId<3>{
            neighbor_block_id,
            make_segment_ids(normal_segment,
                             first_transverse_segment.id_of_child(Side::Lower),
                             second_transverse_segment.id_of_child(Side::Upper),
                             normal_dim)},
        ElementId<3>{
            neighbor_block_id,
            make_segment_ids(normal_segment,
                             first_transverse_segment.id_of_child(Side::Upper),
                             second_transverse_segment.id_of_child(Side::Upper),
                             normal_dim)}});
    result.emplace_back(std::unordered_set{
        ElementId<3>{
            neighbor_block_id,
            make_segment_ids(normal_segment,
                             first_transverse_segment.id_of_child(Side::Lower),
                             second_transverse_segment.id_of_child(Side::Lower),
                             normal_dim)},
        ElementId<3>{
            neighbor_block_id,
            make_segment_ids(normal_segment,
                             first_transverse_segment.id_of_child(Side::Upper),
                             second_transverse_segment, normal_dim)},
        ElementId<3>{
            neighbor_block_id,
            make_segment_ids(normal_segment,
                             first_transverse_segment.id_of_child(Side::Lower),
                             second_transverse_segment.id_of_child(Side::Upper),
                             normal_dim)}});
    result.emplace_back(std::unordered_set{
        ElementId<3>{
            neighbor_block_id,
            make_segment_ids(normal_segment,
                             first_transverse_segment.id_of_child(Side::Lower),
                             second_transverse_segment, normal_dim)},
        ElementId<3>{
            neighbor_block_id,
            make_segment_ids(normal_segment,
                             first_transverse_segment.id_of_child(Side::Upper),
                             second_transverse_segment.id_of_child(Side::Lower),
                             normal_dim)},
        ElementId<3>{
            neighbor_block_id,
            make_segment_ids(normal_segment,
                             first_transverse_segment.id_of_child(Side::Upper),
                             second_transverse_segment.id_of_child(Side::Upper),
                             normal_dim)}});
    result.emplace_back(std::unordered_set{
        ElementId<3>{
            neighbor_block_id,
            make_segment_ids(normal_segment,
                             first_transverse_segment.id_of_child(Side::Lower),
                             second_transverse_segment.id_of_child(Side::Lower),
                             normal_dim)},
        ElementId<3>{
            neighbor_block_id,
            make_segment_ids(normal_segment,
                             first_transverse_segment.id_of_child(Side::Upper),
                             second_transverse_segment.id_of_child(Side::Lower),
                             normal_dim)},
        ElementId<3>{
            neighbor_block_id,
            make_segment_ids(normal_segment,
                             first_transverse_segment.id_of_child(Side::Lower),
                             second_transverse_segment.id_of_child(Side::Upper),
                             normal_dim)},
        ElementId<3>{
            neighbor_block_id,
            make_segment_ids(normal_segment,
                             first_transverse_segment.id_of_child(Side::Upper),
                             second_transverse_segment.id_of_child(Side::Upper),
                             normal_dim)}});
  }
  return result;
}

template <size_t Dim>
std::unordered_set<ElementId<Dim>> oriented_neighbor_ids(
    const std::unordered_set<ElementId<Dim>>& aligned_neighbor_ids,
    const OrientationMap<Dim>& orientation) {
  std::unordered_set<ElementId<Dim>> result{};
  for (const auto& neighbor_id : aligned_neighbor_ids) {
    result.emplace(neighbor_id.block_id(),
                   orientation(neighbor_id.segment_ids()));
  }
  return result;
}

template <size_t Dim>
bool elements_overlap(const ElementId<Dim>& element_one,
                      const ElementId<Dim>& element_two) {
  for (size_t d = 0; d < Dim; ++d) {
    if (not element_one.segment_id(d).overlaps(element_two.segment_id(d))) {
      return false;
    }
  }
  return true;
}

template <size_t Dim>
bool are_neighbors_overlapping(const Neighbors<Dim>& neighbors) {
  const auto& neighbor_ids = neighbors.ids();
  for (auto it = std::next(neighbor_ids.begin()); it != neighbor_ids.end();
       ++it) {
    for (auto prev_it = neighbor_ids.begin(); prev_it != it; ++prev_it) {
      if (elements_overlap(*prev_it, *it)) {
        return true;
      }
    }
  }
  return false;
}
}  // namespace

namespace TestHelpers::domain {
std::vector<SegmentId> valid_neighbor_segments(const FaceType face_type,
                                               const Side side) {
  if (face_type == FaceType::External) {
    return std::vector<SegmentId>{};
  }
  if (face_type == FaceType::Periodic) {
    return std::vector{SegmentId{0, 0}};
  }
  ASSERT(face_type != FaceType::Internal,
         "Face type cannot be Internal for the root segment.");
  if (side == Side::Lower) {
    return std::vector{SegmentId{0, 0}, SegmentId{1, 1}};
  }
  return std::vector{SegmentId{0, 0}, SegmentId{1, 0}};
}

std::vector<SegmentId> valid_neighbor_segments(const SegmentId& segment_id) {
  return std::vector{segment_id.id_of_sibling(),
                     segment_id.id_of_abutting_nibling()};
}

std::vector<SegmentId> valid_neighbor_segments(const SegmentId& segment_id,
                                               const FaceType face_type) {
  if (face_type == FaceType::External) {
    return std::vector<SegmentId>{};
  }
  const size_t level = segment_id.refinement_level();
  ASSERT(level != 0,
         "Cannot call this function on the root segment.  Use "
         "valid_neighbor_segments(face_type, side) instead.");

  const Side sibling_side = segment_id.side_of_sibling();
  const SegmentId base_segment =
      (face_type == FaceType::Block or face_type == FaceType::Periodic)
          ? segment_id.id_if_flipped()
          : SegmentId{level, sibling_side == Side::Upper
                                 ? segment_id.index() - 1
                                 : segment_id.index() + 1};

  const bool neighbor_can_be_parent_of_base =
      face_type == FaceType::Block or
      not(segment_id.id_of_parent() == base_segment.id_of_parent());

  if (not neighbor_can_be_parent_of_base) {
    return std::vector{base_segment, base_segment.id_of_child(sibling_side)};
  }

  return std::vector{base_segment.id_of_parent(), base_segment,
                     base_segment.id_of_child(sibling_side)};
}

template <size_t Dim>
std::vector<Neighbors<Dim>> valid_neighbors(
    const gsl::not_null<std::mt19937*> generator,
    const ElementId<Dim>& element_id, const Direction<Dim>& direction,
    const FaceType face_type) {
  const size_t normal_dim = direction.dimension();
  const Side side = direction.side();
  const SegmentId& normal_segment = element_id.segment_id(normal_dim);
  const double endpoint = normal_segment.endpoint(side);
  const OrientationMap<Dim> aligned{};
  const size_t block_id = element_id.block_id();
  const size_t neighbor_block_id =
      (face_type == FaceType::Block
           ? block_id + 2 * normal_dim + (side == Side::Lower ? 1 : 2)
           : block_id);
  std::vector<Neighbors<Dim>> result;
  if (endpoint == 1.0 or endpoint == -1.0) {
    ASSERT(face_type != FaceType::Internal,
           "Element abuts the Block Boundary in the given direction, please "
           "pass a valid FaceType");
    if (face_type == FaceType::External) {
      return std::vector<Neighbors<Dim>>{};
    }
    const auto valid_neighbor_normal_segments =
        (normal_segment.refinement_level() == 0
             ? valid_neighbor_segments(face_type, side)
             : valid_neighbor_segments(normal_segment, face_type));
    const auto valid_orientations =
        (face_type == FaceType::Periodic
             ? std::vector{aligned}
             : random_orientation_maps<Dim>(2, generator));
    for (const auto& aligned_neighbor_ids :
         valid_aligned_neighbor_ids(element_id, neighbor_block_id, normal_dim,
                                    valid_neighbor_normal_segments)) {
      for (const auto& orientation : valid_orientations) {
        if (orientation.is_aligned()) {
          result.emplace_back(aligned_neighbor_ids, orientation);
        } else {
          result.emplace_back(
              oriented_neighbor_ids(aligned_neighbor_ids, orientation),
              orientation);
        }
      }
    }
    return result;
  }
  ASSERT(face_type == FaceType::Internal,
         "Element is in the interior of a Block in the provided direction");
  const auto valid_neighbor_normal_segments =
      (side == normal_segment.side_of_sibling()
           ? valid_neighbor_segments(normal_segment)
           : valid_neighbor_segments(normal_segment, face_type));
  for (const auto& neighbor_ids :
       valid_aligned_neighbor_ids(element_id, neighbor_block_id, normal_dim,
                                  valid_neighbor_normal_segments)) {
    result.emplace_back(neighbor_ids, aligned);
  }
  return result;
}

template <size_t Dim>
void check_neighbors(const Neighbors<Dim>& neighbors,
                     const ElementId<Dim>& element_id,
                     const Direction<Dim>& direction) {
  CAPTURE(element_id);
  CAPTURE(neighbors);
  CAPTURE(direction);
  if (neighbors.size() > 1) {
    CHECK_FALSE(are_neighbors_overlapping(neighbors));
  }
  const size_t dim_normal_to_face = direction.dimension();
  const Side neighbor_side = direction.side();
  const Side element_side = opposite(neighbor_side);
  const OrientationMap<Dim>& orientation_map = neighbors.orientation();
  double surface_area_covered = 0.0;
  for (const auto& neighbor : neighbors) {
    double neighbor_overlap_area = 1.0;
    const std::array<SegmentId, Dim> neighbor_segment_ids =
        orientation_map.is_aligned()
            ? neighbor.segment_ids()
            : orientation_map.inverse_map()(neighbor.segment_ids());
    for (size_t d = 0; d < Dim; ++d) {
      const SegmentId& element_segment = element_id.segment_id(d);
      const SegmentId& neighbor_segment = gsl::at(neighbor_segment_ids, d);
      if (d == dim_normal_to_face) {
        const double endpoint = element_segment.endpoint(neighbor_side);
        if (endpoint == 1.0 or endpoint == -1.0) {
          // point lies on a Block boundary, neighbor point should lie
          // on opposite side of the Block
          CHECK(-endpoint == neighbor_segment.endpoint(element_side));
        } else {
          // interior point, should be shared
          CHECK(endpoint == neighbor_segment.endpoint(element_side));
        }
        if (element_id.block_id() == neighbor.block_id()) {
          CHECK((element_id == neighbor or
                 not element_segment.overlaps(neighbor_segment)));
        }
      } else {  // transverse to face
        if (neighbors.size() == 1) {
          CHECK((element_segment == neighbor_segment or
                 element_segment.id_of_parent() == neighbor_segment));
        } else {
          if (neighbor_segment.refinement_level() > 0 and
              element_segment == neighbor_segment.id_of_parent()) {
            neighbor_overlap_area *= 0.5;
          } else {
            CHECK((element_segment == neighbor_segment or
                   element_segment.id_of_parent() == neighbor_segment));
          }
        }
      }
    }
    surface_area_covered += neighbor_overlap_area;
  }
  CHECK(surface_area_covered == 1.0);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                            \
  template std::vector<Neighbors<DIM(data)>> valid_neighbors(           \
      const gsl::not_null<std::mt19937*> generator,                     \
      const ElementId<DIM(data)>& element_id,                           \
      const Direction<DIM(data)>& direction, const FaceType face_type); \
  template void check_neighbors(const Neighbors<DIM(data)>& neighbors,  \
                                const ElementId<DIM(data)>& element_id, \
                                const Direction<DIM(data)>& direction);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
}  // namespace TestHelpers::domain
