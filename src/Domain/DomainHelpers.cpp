// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/DomainHelpers.hpp"

#include "Domain/Direction.hpp"
#include "Domain/OrientationMap.hpp"

namespace {

// returns the common corners in the global corner numbering scheme.
template <size_t VolumeDim>
std::vector<size_t> get_common_global_corners(
    const std::array<size_t, two_to_the(VolumeDim)>& block_1_global_corners,
    const std::array<size_t, two_to_the(VolumeDim)>& block_2_global_corners) {
  std::vector<size_t> result;

  for (size_t id_i : block_1_global_corners) {
    if (std::find(block_2_global_corners.begin(), block_2_global_corners.end(),
                  id_i) != block_2_global_corners.end()) {
      result.push_back(id_i);
    }
  }

  return result;
}

// Given Directions in block1 and their corresponding Directions
// in block2, this function first repackages them into arrays and
// then makes a std::pair of the arrays.
template <size_t VolumeDim>
std::pair<std::array<Direction<VolumeDim>, VolumeDim>,
          std::array<Direction<VolumeDim>, VolumeDim>>
create_correspondence_between_blocks(
    const Direction<VolumeDim>& direction_to_neighbor_in_self_frame,
    const Direction<VolumeDim>& direction_to_self_in_neighbor_frame,
    const std::vector<Direction<VolumeDim>>&
        alignment_of_shared_face_in_self_frame,
    const std::vector<Direction<VolumeDim>>&
        alignment_of_shared_face_in_neighbor_frame) {
  std::array<Direction<VolumeDim>, VolumeDim> result1;
  result1[0] = direction_to_neighbor_in_self_frame;

  for (size_t i = 1; i < VolumeDim; i++) {
    gsl::at(result1, i) = alignment_of_shared_face_in_self_frame[i - 1];
  }

  std::array<Direction<VolumeDim>, VolumeDim> result2;
  result2[0] = direction_to_self_in_neighbor_frame.opposite();

  for (size_t i = 1; i < VolumeDim; i++) {
    gsl::at(result2, i) = alignment_of_shared_face_in_neighbor_frame[i - 1];
  }
  return std::make_pair(result1, result2);
}

// When setting up periodic-like boundary conditions, the user specifies
// the corners of the two faces to be identified.
// requires that the two faces correspond to the same block.
// returns the corresponding id of the block whose faces are to
// be identified.
template <size_t VolumeDim>
size_t find_block_id(
    const std::vector<size_t>& global_corners_face1,
    const std::vector<size_t>& global_corners_face2,
    const std::vector<std::array<size_t, two_to_the(VolumeDim)>>&
        corners_of_all_blocks) {
  for (size_t j = 0; j < corners_of_all_blocks.size(); j++) {
    const auto& corners_set = corners_of_all_blocks[j];
    auto is_in_corners_set = [&corners_set](const auto element) {
      return std::find(corners_set.begin(), corners_set.end(), element) !=
             corners_set.end();
    };
    const bool face_matches_block_j =
        std::all_of(global_corners_face1.begin(), global_corners_face1.end(),
                    is_in_corners_set)

        and std::all_of(global_corners_face2.begin(),
                        global_corners_face2.end(), is_in_corners_set);
    if (face_matches_block_j) {
      return j;
    }
  }
  ERROR("This pair of faces does not belong to a single block.");
}

template <size_t VolumeDim>
std::array<size_t, two_to_the(VolumeDim - 1)> get_common_local_corners(
    const std::array<size_t, two_to_the(VolumeDim)>& block_global_corners,
    const std::vector<size_t> block_global_common_corners) {
  std::array<size_t, two_to_the(VolumeDim - 1)> result{{0}};
  size_t i = 0;
  for (const auto global_id : block_global_common_corners) {
    ASSERT(std::find(block_global_corners.begin(), block_global_corners.end(),
                     global_id) != block_global_corners.end(),
           "Elements of block_global_common_corners must be in "
           "block_global_corners.");
    gsl::at(result, i) =
        static_cast<size_t>(std::find(block_global_corners.begin(),
                                      block_global_corners.end(), global_id) -
                            block_global_corners.begin());
    i++;
  }
  return result;
}

template <size_t VolumeDim>
std::array<tnsr::I<double, VolumeDim, Frame::Logical>,
           two_to_the(VolumeDim - 1)>
logical_coords_of_corners(
    const std::array<size_t, two_to_the(VolumeDim - 1)>& local_ids) {
  std::array<tnsr::I<double, VolumeDim, Frame::Logical>,
             two_to_the(VolumeDim - 1)>
      result{};
  std::transform(local_ids.begin(), local_ids.end(), result.begin(),
                 [](const size_t id) {
                   tnsr::I<double, VolumeDim, Frame::Logical> point{};
                   for (size_t i = 0; i < VolumeDim; i++) {
                     point[i] = 2.0 * get_nth_bit(id, i) - 1.0;
                   };
                   return point;
                 });
  return result;
}

template <size_t VolumeDim>
Direction<VolumeDim> get_direction_normal_to_face(
    const std::array<tnsr::I<double, VolumeDim, Frame::Logical>,
                     two_to_the(VolumeDim - 1)>& face_pts) {
  const auto summed_point = std::accumulate(
      face_pts.begin(), face_pts.end(),
      tnsr::I<double, VolumeDim, Frame::Logical>(0.0),
      [](tnsr::I<double, VolumeDim, Frame::Logical>& prior,
         const tnsr::I<double, VolumeDim, Frame::Logical>& point) {
        for (size_t i = 0; i < prior.size(); i++) {
          prior[i] += point[i];
        }
        return prior;
      });
  ASSERT(VolumeDim - 1 == static_cast<size_t>(std::count(
                              summed_point.begin(), summed_point.end(), 0.0)),
         "The face_pts passed in do not correspond to a face.");
  const auto index =
      static_cast<size_t>(std::find_if(summed_point.begin(), summed_point.end(),
                                       [](const double x) { return x != 0; }) -
                          summed_point.begin());
  return Direction<VolumeDim>(
      index, summed_point[index] > 0 ? Side::Upper : Side::Lower);
}

template <size_t VolumeDim>
std::vector<Direction<VolumeDim>> get_directions_along_face(
    const std::array<size_t, two_to_the(VolumeDim - 1)>& face_local_corners) {
  std::vector<Direction<VolumeDim>> result;

  // Directions encoded as powers of 2. (xi:1, eta:2, zeta:4)
  std::vector<int> dir_pow2;
  for (size_t i = 0; i < VolumeDim - 1; i++) {
    int direction_pow2 =
        static_cast<int>(gsl::at(face_local_corners, two_to_the(i))) -
        static_cast<int>(gsl::at(face_local_corners, 0));
    ASSERT(
        direction_pow2 ==
            static_cast<int>(
                gsl::at(face_local_corners, face_local_corners.size() - 1)) -
                static_cast<int>(
                    gsl::at(face_local_corners,
                            face_local_corners.size() - 1 - two_to_the(i))),
        "The corners provided do not correspond to a possible face."
        "Help: As the direction of a face can be determined from any pair of "
        "edges, all pairs of edges must agree on their directionality. "
        "Providing '0,1,3,2' as a pair of edges (in the canonical corner "
        "numbering scheme) will trigger this ASSERT as 1-0 != 2-3; they "
        "point in opposite directions. Additionally, 3-0 != 2-1. The proper "
        "numbering for this face is '0,1,2,3', or any other numbering that "
        "results from a rigid manipulation of this face, for example "
        "'1,3,0,2'.");

    ASSERT(abs(direction_pow2) == 1 or abs(direction_pow2) == 2 or
               abs(direction_pow2) == 4,
           "The corners making up an edge always have a difference of either "
           "1, 2, or 4, under the local numbering scheme, depending on whether "
           "the edge is parallel to the xi, eta, or zeta axis, respectively. "
           "This ASSERT is triggered when the two corners passed do not make "
           "up an edge of the n-cube.");
    const size_t dimension = std::round(log2(abs(direction_pow2)));
    const Side side = direction_pow2 > 0 ? Side::Upper : Side::Lower;
    result.push_back(Direction<VolumeDim>(dimension, side));
  }
  return result;
}

template <size_t VolumeDim>
Direction<VolumeDim> mapped(
    size_t block_id1, size_t block_id2,
    const Direction<VolumeDim>& dir_in_block1,
    const std::vector<std::array<size_t, two_to_the(VolumeDim)>>&
        corners_of_all_blocks) {
  const auto correspondence = obtain_correspondence_between_blocks(
      block_id1, block_id2, corners_of_all_blocks);
  for (size_t i = 0; i < VolumeDim; i++) {
    if (dir_in_block1.dimension() == (correspondence.first)[i].dimension()) {
      return dir_in_block1.side() == (correspondence.first)[i].side()
                 ? (correspondence.second)[i]
                 : (correspondence.second)[i].opposite();
    }
  }
  ERROR("Improper Direction was passed to function.");
}

template <size_t VolumeDim>
std::pair<std::array<Direction<VolumeDim>, VolumeDim>,
          std::array<Direction<VolumeDim>, VolumeDim>>
obtain_correspondence_between_blocks(
    size_t block_id1, size_t block_id2,
    const std::vector<std::array<size_t, two_to_the(VolumeDim)>>&
        corners_of_all_blocks) {
  const auto& self = corners_of_all_blocks[block_id1];
  const auto& nhbr = corners_of_all_blocks[block_id2];
  const auto common = get_common_global_corners<VolumeDim>(self, nhbr);
  const auto self_common = get_common_local_corners<VolumeDim>(self, common);
  const auto nhbr_common = get_common_local_corners<VolumeDim>(nhbr, common);
  const auto self_dir = get_direction_normal_to_face<VolumeDim>(
      logical_coords_of_corners<VolumeDim>(self_common));
  const auto nhbr_dir = get_direction_normal_to_face<VolumeDim>(
      logical_coords_of_corners<VolumeDim>(nhbr_common));
  const auto self_align = get_directions_along_face<VolumeDim>(self_common);
  const auto nhbr_align = get_directions_along_face<VolumeDim>(nhbr_common);

  return create_correspondence_between_blocks(self_dir, nhbr_dir, self_align,
                                              nhbr_align);
}
}  // namespace

template <size_t VolumeDim>
void set_internal_boundaries(
    const std::vector<std::array<size_t, two_to_the(VolumeDim)>>&
        corners_of_all_blocks,
    gsl::not_null<std::vector<
        std::unordered_map<Direction<VolumeDim>, BlockNeighbor<VolumeDim>>>*>
        neighbors_of_all_blocks) {
  for (size_t block1_index = 0; block1_index < corners_of_all_blocks.size();
       block1_index++) {
    std::unordered_map<Direction<VolumeDim>, BlockNeighbor<VolumeDim>> neighbor;
    for (size_t block2_index = 0; block2_index < corners_of_all_blocks.size();
         block2_index++) {
      if (block1_index != block2_index and
          get_common_global_corners<VolumeDim>(
              corners_of_all_blocks[block1_index],
              corners_of_all_blocks[block2_index])
                  .size() == two_to_the(VolumeDim - 1)) {
        const auto orientation_helper =
            obtain_correspondence_between_blocks<VolumeDim>(
                block1_index, block2_index, corners_of_all_blocks);
        neighbor.emplace(std::move((orientation_helper.first)[0]),
                         BlockNeighbor<VolumeDim>(
                             block2_index, OrientationMap<VolumeDim>(
                                               orientation_helper.first,
                                               orientation_helper.second)));
      }
    }
    neighbors_of_all_blocks->push_back(neighbor);
  }
}

template <size_t VolumeDim>
void set_periodic_boundaries(
    const std::vector<PairOfFaces>& identifications,
    const std::vector<std::array<size_t, two_to_the(VolumeDim)>>&
        corners_of_all_blocks,
    gsl::not_null<std::vector<
        std::unordered_map<Direction<VolumeDim>, BlockNeighbor<VolumeDim>>>*>
        neighbors_of_all_blocks) {
  for (const auto& pair : identifications) {
    const auto& face1 = pair.first;
    const auto& face2 = pair.second;
    ASSERT(face1.size() == face2.size(),
           "Each set must have the same number of corners.");
    size_t id = ::find_block_id<VolumeDim>(face1, face2, corners_of_all_blocks);
    const auto face1_canon =
        get_common_local_corners<VolumeDim>(corners_of_all_blocks[id], face1);
    const auto face2_canon =
        get_common_local_corners<VolumeDim>(corners_of_all_blocks[id], face2);
    const auto face1_normal_dir = get_direction_normal_to_face(
        logical_coords_of_corners<VolumeDim>(face1_canon));
    const auto face2_normal_dir = get_direction_normal_to_face(
        logical_coords_of_corners<VolumeDim>(face2_canon));
    const auto face1_align_dirs =
        get_directions_along_face<VolumeDim>(face1_canon);
    const auto face2_align_dirs =
        get_directions_along_face<VolumeDim>(face2_canon);
    const auto obtain_correspondence_between_blocks1 =
        create_correspondence_between_blocks<VolumeDim>(
            face1_normal_dir, face2_normal_dir, face1_align_dirs,
            face2_align_dirs);
    const auto obtain_correspondence_between_blocks2 =
        create_correspondence_between_blocks<VolumeDim>(
            face2_normal_dir, face1_normal_dir, face2_align_dirs,
            face1_align_dirs);
    const OrientationMap<VolumeDim> connect1(
        obtain_correspondence_between_blocks1.first,
        obtain_correspondence_between_blocks1.second);
    const OrientationMap<VolumeDim> connect2(
        obtain_correspondence_between_blocks2.first,
        obtain_correspondence_between_blocks2.second);
    (*neighbors_of_all_blocks)[id].emplace(
        face1_normal_dir, BlockNeighbor<VolumeDim>(id, connect1));
    (*neighbors_of_all_blocks)[id].emplace(
        face2_normal_dir, BlockNeighbor<VolumeDim>(id, connect2));
  }
}

template void set_internal_boundaries(
    const std::vector<std::array<size_t, 2>>& corners_of_all_blocks,
    gsl::not_null<
        std::vector<std::unordered_map<Direction<1>, BlockNeighbor<1>>>*>
        neighbors_of_all_blocks);
template void set_internal_boundaries(
    const std::vector<std::array<size_t, 4>>& corners_of_all_blocks,
    gsl::not_null<
        std::vector<std::unordered_map<Direction<2>, BlockNeighbor<2>>>*>
        neighbors_of_all_blocks);
template void set_internal_boundaries(
    const std::vector<std::array<size_t, 8>>& corners_of_all_blocks,
    gsl::not_null<
        std::vector<std::unordered_map<Direction<3>, BlockNeighbor<3>>>*>
        neighbors_of_all_blocks);

template void set_periodic_boundaries(
    const std::vector<PairOfFaces>& identifications,
    const std::vector<std::array<size_t, 2>>& corners_of_all_blocks,
    gsl::not_null<
        std::vector<std::unordered_map<Direction<1>, BlockNeighbor<1>>>*>
        neighbors_of_all_blocks);
template void set_periodic_boundaries(
    const std::vector<PairOfFaces>& identifications,
    const std::vector<std::array<size_t, 4>>& corners_of_all_blocks,
    gsl::not_null<
        std::vector<std::unordered_map<Direction<2>, BlockNeighbor<2>>>*>
        neighbors_of_all_blocks);
template void set_periodic_boundaries(
    const std::vector<PairOfFaces>& identifications,
    const std::vector<std::array<size_t, 8>>& corners_of_all_blocks,
    gsl::not_null<
        std::vector<std::unordered_map<Direction<3>, BlockNeighbor<3>>>*>
        neighbors_of_all_blocks);
