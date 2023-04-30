// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/DomainHelpers.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>
#include <string>
#include <utility>

#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordinateMaps/EquatorialCompression.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/Frustum.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/MapInstantiationMacros.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Side.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace {

// returns the common corners in the global corner numbering scheme.
template <size_t VolumeDim>
std::vector<size_t> get_common_global_corners(
    const std::array<size_t, two_to_the(VolumeDim)>& block_1_global_corners,
    const std::array<size_t, two_to_the(VolumeDim)>& block_2_global_corners) {
  std::vector<size_t> result;

  for (size_t id_i : block_1_global_corners) {
    if (alg::find(block_2_global_corners, id_i) !=
        block_2_global_corners.end()) {
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

// requires that the face is an external boundary of the domain.
// returns the id of the block with that face.
template <size_t VolumeDim>
size_t find_block_id_of_external_face(
    const std::vector<size_t>& global_corners_external_face,
    const std::vector<std::array<size_t, two_to_the(VolumeDim)>>&
        corners_of_all_blocks) {
  size_t number_of_blocks_found = 0;
  size_t id_of_found_block = std::numeric_limits<size_t>::max();
  for (size_t j = 0; j < corners_of_all_blocks.size(); j++) {
    const auto& corners_set = corners_of_all_blocks[j];
    auto is_in_corners_set = [&corners_set](const auto element) {
      return alg::find(corners_set, element) != corners_set.end();
    };
    if (alg::all_of(global_corners_external_face, is_in_corners_set)) {
      number_of_blocks_found++;
      id_of_found_block = j;
    }
  }
  ASSERT(number_of_blocks_found > 0, "This face does not belong to a block.");
  ASSERT(number_of_blocks_found == 1, "This face is not an external boundary.");
  return id_of_found_block;
}

template <size_t VolumeDim>
std::array<size_t, two_to_the(VolumeDim - 1)> get_common_local_corners(
    const std::array<size_t, two_to_the(VolumeDim)>& block_global_corners,
    const std::vector<size_t>& block_global_common_corners) {
  std::array<size_t, two_to_the(VolumeDim - 1)> result{{0}};
  size_t i = 0;
  for (const auto global_id : block_global_common_corners) {
    ASSERT(alg::find(block_global_corners, global_id) !=
               block_global_corners.end(),
           "Elements of block_global_common_corners must be in "
           "block_global_corners.");
    gsl::at(result, i) =
        static_cast<size_t>(alg::find(block_global_corners, global_id) -
                            block_global_corners.begin());
    i++;
  }
  return result;
}

template <size_t VolumeDim>
std::array<tnsr::I<double, VolumeDim, Frame::BlockLogical>,
           two_to_the(VolumeDim - 1)>
logical_coords_of_corners(
    const std::array<size_t, two_to_the(VolumeDim - 1)>& local_ids) {
  std::array<tnsr::I<double, VolumeDim, Frame::BlockLogical>,
             two_to_the(VolumeDim - 1)>
      result{};
  std::transform(local_ids.begin(), local_ids.end(), result.begin(),
                 [](const size_t id) {
                   tnsr::I<double, VolumeDim, Frame::BlockLogical> point{};
                   for (size_t i = 0; i < VolumeDim; i++) {
                     point[i] = 2.0 * get_nth_bit(id, i) - 1.0;
                   };
                   return point;
                 });
  return result;
}

template <size_t VolumeDim>
Direction<VolumeDim> get_direction_normal_to_face(
    const std::array<tnsr::I<double, VolumeDim, Frame::BlockLogical>,
                     two_to_the(VolumeDim - 1)>& face_pts) {
  const auto summed_point = alg::accumulate(
      face_pts, tnsr::I<double, VolumeDim, Frame::BlockLogical>(0.0),
      [](tnsr::I<double, VolumeDim, Frame::BlockLogical>& prior,
         const tnsr::I<double, VolumeDim, Frame::BlockLogical>& point) {
        for (size_t i = 0; i < prior.size(); i++) {
          prior[i] += point[i];
        }
        return prior;
      });
  ASSERT(VolumeDim - 1 == static_cast<size_t>(std::count(
                              summed_point.begin(), summed_point.end(), 0.0)),
         "The face_pts passed in do not correspond to a face.");
  const auto index = static_cast<size_t>(
      alg::find_if(summed_point, [](const double x) { return x != 0; }) -
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

template <size_t VolumeDim>
std::vector<std::array<size_t, two_to_the(VolumeDim)>> corners_from_two_maps(
    const std::unique_ptr<domain::CoordinateMapBase<
        Frame::BlockLogical, Frame::Inertial, VolumeDim>>& map1,
    const std::unique_ptr<domain::CoordinateMapBase<
        Frame::BlockLogical, Frame::Inertial, VolumeDim>>& map2) {
  std::array<size_t, two_to_the(VolumeDim)> corners_for_block1{};
  std::array<size_t, two_to_the(VolumeDim)> corners_for_block2{};
  for (VolumeCornerIterator<VolumeDim> vci{}; vci; ++vci) {
    gsl::at(corners_for_block1, vci.local_corner_number()) =
        vci.local_corner_number();
  }
  // Fill corners_for_block2 based off which corners are shared with map1.
  for (VolumeCornerIterator<VolumeDim> vci_map2{}; vci_map2; ++vci_map2) {
    const auto mapped_coords2 =
        (*map2)(tnsr::I<double, VolumeDim, Frame::BlockLogical>{
            vci_map2.coords_of_corner()});

    // Set num_unshared_corners to zero because we only need a local corner
    // numbering scheme. This is because our algorithm treats each pair of
    // blocks independently to create the orientation (if any) between them.
    size_t num_unshared_corners = 0;
    // Check each corner of map1 to see if it shares any corners with map2.
    for (VolumeCornerIterator<VolumeDim> vci_map1{}; vci_map1; ++vci_map1) {
      const auto mapped_coords1 =
          (*map1)(tnsr::I<double, VolumeDim, Frame::BlockLogical>{
              vci_map1.coords_of_corner()});
      double max_separation = 0.0;
      for (size_t j = 0; j < VolumeDim; j++) {
        max_separation =
            std::max(max_separation,
                     std::abs(mapped_coords2.get(j) - mapped_coords1.get(j)));
      }
      // If true, then the mapped_coords lie on top of one another.
      // This corner of map2 is assigned the same number as that of map1.
      if (max_separation < 1.0e-6) {
        // Note: Ideally the threshold would depend on and be proportional
        // to the map size, but for simplicity we assume that maps have sizes
        // that are of order unity. If you are using maps that have sizes
        // that are much smaller, this threshold (1e-6) will create situations
        // where adjacent corners are incorrectly determined. If you are using
        // very small maps please watch out for this.
        gsl::at(corners_for_block2, vci_map2.local_corner_number()) =
            vci_map1.local_corner_number();
        break;
      } else {
        // Otherwise, a new number is assigned to this corner.
        gsl::at(corners_for_block2, vci_map2.local_corner_number()) =
            two_to_the(VolumeDim) + num_unshared_corners;
        num_unshared_corners++;
      }
    }
  }
  return {corners_for_block1, corners_for_block2};
}

template <size_t VolumeDim>
std::pair<std::array<Direction<VolumeDim>, VolumeDim>,
          std::array<Direction<VolumeDim>, VolumeDim>>
obtain_correspondence_between_maps(
    const size_t map_id1, const size_t map_id2,
    const std::vector<std::unique_ptr<domain::CoordinateMapBase<
        Frame::BlockLogical, Frame::Inertial, VolumeDim>>>& maps) {
  return obtain_correspondence_between_blocks<VolumeDim>(
      0, 1, corners_from_two_maps(maps[map_id1], maps[map_id2]));
}

// Sets periodic boundary conditions for rectilinear multi-cube domains.
//
// For each block with an external boundary in the +i^th direction, traverses
// the block neighbors in the -i^th direction until finding a neighbor block
// with an external boundary in the -i^th direction, and identifies these two
// external boundaries with each other, if `dimension_is_periodic[i]` is
// `true`. This is useful when setting the boundaries of a rectilinear domain.
//
// The argument passed to `orientations_of_all_blocks` is the relative
// orientation of the edifice relative to each block. This is used when
// constructing non-trivial domains for testing such as RotatedBricks.
template <size_t VolumeDim>
void set_cartesian_periodic_boundaries(
    const std::array<bool, VolumeDim>& dimension_is_periodic,
    const std::vector<std::array<size_t, two_to_the(VolumeDim)>>&
        corners_of_all_blocks,
    const std::vector<OrientationMap<VolumeDim>>& orientations_of_all_blocks,
    const gsl::not_null<
        std::vector<DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>>*>
        neighbors_of_all_blocks) {
  ASSERT(orientations_of_all_blocks.size() == corners_of_all_blocks.size(),
         "Each block must have an OrientationMap relative to an edifice.");
  size_t block_id = 0;
  for (auto& block : *neighbors_of_all_blocks) {
    const auto& corners_of_block = corners_of_all_blocks[block_id];
    const auto& orientation_of_block = orientations_of_all_blocks[block_id];
    for (const auto& direction : Direction<VolumeDim>::all_directions()) {
      // Loop over Directions to search which Directions were not set to
      // neighbors, these Directions are external_boundaries, so we proceed
      // in the opposite direction to find the block we wish to identify
      // boundaries with.
      if (not gsl::at(dimension_is_periodic, direction.dimension())) {
        continue;  // Don't identify neighbors in dimensions not specified as
                   // periodic.
      }
      if (block.find(orientation_of_block(direction)) == block.end()) {
        // follow to neighbor in the opposite direction:
        const auto local_direction_in_block = orientation_of_block(direction);
        auto opposite_block_id = block_id;
        auto opposite_block = block;
        auto corners_of_opposite_block = corners_of_block;
        // The local direction that points to the opposite block:
        auto opposite_block_direction = local_direction_in_block.opposite();
        for (size_t neighbor_counter = 0;
             neighbor_counter < corners_of_all_blocks.size();
             neighbor_counter++) {
          if (opposite_block.find(opposite_block_direction) ==
              opposite_block.end()) {
            continue;
          } else {
            opposite_block_id = opposite_block[opposite_block_direction].id();
            opposite_block = (*neighbors_of_all_blocks)[opposite_block_id];
            opposite_block_direction =
                orientations_of_all_blocks[opposite_block_id](direction)
                    .opposite();
            corners_of_opposite_block =
                corners_of_all_blocks[opposite_block_id];
          }
          if (neighbor_counter == corners_of_all_blocks.size() - 1) {
            // There is no domain that can be constructed using
            // `rectilinear_domain`, that triggers this ERROR, so the ERROR is
            // expected to be unreachable. We keep this check in the case that
            // this assumption is wrong.
            // LCOV_EXCL_START
            ERROR(
                "A closed cycle of neighbors exists in this domain. Has "
                "`orientations_of_all_blocks` been set correctly?");
            // LCOV_EXCL_STOP
          }
        }
        //`opposite_block` is now the block on the opposite cartesian side
        // of `block`, so we identify the faces of those blocks that were
        // external boundaries as now being periodic. Note that `opposite_block`
        // and `block`  may be the same block, if the domain only has an extent
        // of one in that dimension.
        std::vector<size_t> face_corners_of_block{};
        std::vector<size_t> face_corners_of_opposite_block{};
        for (FaceCornerIterator<VolumeDim> fci(local_direction_in_block); fci;
             ++fci) {
          face_corners_of_block.push_back(
              gsl::at(corners_of_block, fci.volume_index()));
        }
        for (FaceCornerIterator<VolumeDim> fci(
                 orientations_of_all_blocks[opposite_block_id](direction)
                     .opposite());
             fci; ++fci) {
          face_corners_of_opposite_block.push_back(
              gsl::at(corners_of_opposite_block, fci.volume_index()));
        }
        // The face corners in both blocks should be in ascending order
        // for periodic boundary conditions. Note that this shortcut only
        // works in the special case where the global corners are
        // increasing in each cartesian direction of the domain.
        std::sort(face_corners_of_block.begin(), face_corners_of_block.end());
        std::sort(face_corners_of_opposite_block.begin(),
                  face_corners_of_opposite_block.end());
        // This modifies the BlockNeighbors of both the current block
        // and the block it found to be its neighbor.
        set_identified_boundaries(
            std::vector<PairOfFaces>{PairOfFaces{
                face_corners_of_block, face_corners_of_opposite_block}},
            corners_of_all_blocks, neighbors_of_all_blocks);
      }
    }
    block_id++;
  }
}
}  // namespace

template <size_t VolumeDim>
void set_internal_boundaries(
    const gsl::not_null<
        std::vector<DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>>*>
        neighbors_of_all_blocks,
    const std::vector<std::array<size_t, two_to_the(VolumeDim)>>&
        corners_of_all_blocks) {
  for (size_t block1_index = 0; block1_index < corners_of_all_blocks.size();
       block1_index++) {
    DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>> neighbor;
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
void set_internal_boundaries(
    const gsl::not_null<
        std::vector<DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>>*>
        neighbors_of_all_blocks,
    const std::vector<std::unique_ptr<domain::CoordinateMapBase<
        Frame::BlockLogical, Frame::Inertial, VolumeDim>>>& maps) {
  for (size_t block1_index = 0; block1_index < maps.size(); block1_index++) {
    DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>> neighbor;
    for (size_t block2_index = 0; block2_index < maps.size(); block2_index++) {
      const auto corners_for_blocks1_and_2 =
          corners_from_two_maps(maps[block1_index], maps[block2_index]);
      if (block1_index != block2_index and
          get_common_global_corners<VolumeDim>(corners_for_blocks1_and_2[0],
                                               corners_for_blocks1_and_2[1])
                  .size() == two_to_the(VolumeDim - 1)) {
        const auto orientation_helper =
            obtain_correspondence_between_blocks<VolumeDim>(
                0, 1, corners_for_blocks1_and_2);

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
void set_identified_boundaries(
    const std::vector<PairOfFaces>& identifications,
    const std::vector<std::array<size_t, two_to_the(VolumeDim)>>&
        corners_of_all_blocks,
    const gsl::not_null<
        std::vector<DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>>*>
        neighbors_of_all_blocks) {
  for (const auto& pair : identifications) {
    const auto& face1 = pair.first;
    const auto& face2 = pair.second;
    ASSERT(face1.size() == face2.size(),
           "Each set must have the same number of corners.");
    size_t id1 = ::find_block_id_of_external_face<VolumeDim>(
        face1, corners_of_all_blocks);
    size_t id2 = ::find_block_id_of_external_face<VolumeDim>(
        face2, corners_of_all_blocks);
    const auto face1_canon =
        get_common_local_corners<VolumeDim>(corners_of_all_blocks[id1], face1);
    const auto face2_canon =
        get_common_local_corners<VolumeDim>(corners_of_all_blocks[id2], face2);
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
    (*neighbors_of_all_blocks)[id1].emplace(
        face1_normal_dir, BlockNeighbor<VolumeDim>(id2, connect1));
    (*neighbors_of_all_blocks)[id2].emplace(
        face2_normal_dir, BlockNeighbor<VolumeDim>(id1, connect2));
  }
}

namespace {
// A Block or Blocks can be wrapped in an outer layer of Blocks surrounding
// the original Block(s). In the BBH Domain, this occurs several times, using
// both Wedges and Frustums. The simplest example in which wrapping is used is
// Sphere, where the central Block is wrapped with six Wedge<3>s.
std::array<OrientationMap<3>, 6> orientations_for_wrappings() {
  return {{
      // Upper Z
      OrientationMap<3>{},
      // Lower Z
      OrientationMap<3>{std::array<Direction<3>, 3>{
          {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
           Direction<3>::lower_zeta()}}},
      // Upper Y
      OrientationMap<3>(std::array<Direction<3>, 3>{
          {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
           Direction<3>::lower_eta()}}),
      // Lower Y
      OrientationMap<3>(std::array<Direction<3>, 3>{
          {Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
           Direction<3>::upper_eta()}}),
      // Upper X
      OrientationMap<3>(std::array<Direction<3>, 3>{
          {Direction<3>::upper_zeta(), Direction<3>::upper_xi(),
           Direction<3>::upper_eta()}}),
      // Lower X
      OrientationMap<3>(std::array<Direction<3>, 3>{
          {Direction<3>::lower_zeta(), Direction<3>::lower_xi(),
           Direction<3>::upper_eta()}}),
  }};
}
}  // namespace

size_t which_wedge_index(const ShellWedges& which_wedges) {
  switch (which_wedges) {
    case ShellWedges::All:
      return 0;
    case ShellWedges::FourOnEquator:
      return 2;
    case ShellWedges::OneAlongMinusX:
      return 5;
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR("Unknown ShellWedges type");
      // LCOV_EXCL_STOP
  }
}

std::vector<domain::CoordinateMaps::Wedge<3>> sph_wedge_coordinate_maps(
    const double inner_radius, const double outer_radius,
    const double inner_sphericity, const double outer_sphericity,
    const bool use_equiangular_map, const bool use_half_wedges,
    const std::vector<double>& radial_partitioning,
    const std::vector<domain::CoordinateMaps::Distribution>&
        radial_distribution,
    const ShellWedges which_wedges, const double opening_angle) {
  ASSERT(not use_half_wedges or which_wedges == ShellWedges::All,
         "If we are using half wedges we must also be using ShellWedges::All.");
  ASSERT(radial_distribution.size() == 1 + radial_partitioning.size(),
         "Specify a radial distribution for every spherical shell. You "
         "specified "
             << radial_distribution.size() << " items, but the domain has "
             << 1 + radial_partitioning.size() << " shells.");

  const auto wedge_orientations = orientations_for_wrappings();

  using Wedge3DMap = domain::CoordinateMaps::Wedge<3>;
  using Halves = Wedge3DMap::WedgeHalves;
  std::vector<Wedge3DMap> wedges_for_all_layers{};

  const size_t number_of_layers = 1 + radial_partitioning.size();
  double temp_inner_radius = inner_radius;
  double temp_outer_radius{};
  double temp_inner_sphericity = inner_sphericity;
  for (size_t layer_i = 0; layer_i < number_of_layers; layer_i++) {
    const auto& radial_distribution_this_layer =
        radial_distribution.at(layer_i);
    if (layer_i != radial_partitioning.size()) {
      temp_outer_radius = radial_partitioning.at(layer_i);
    } else {
      temp_outer_radius = outer_radius;
    }
    // Generate wedges/half-wedges a layer at a time.
    std::vector<Wedge3DMap> wedges_for_this_layer{};
    if (not use_half_wedges) {
      for (size_t face_j = which_wedge_index(which_wedges); face_j < 6;
           face_j++) {
        wedges_for_this_layer.emplace_back(
            temp_inner_radius, temp_outer_radius, temp_inner_sphericity,
            outer_sphericity, gsl::at(wedge_orientations, face_j),
            use_equiangular_map, Halves::Both, radial_distribution_this_layer,
            std::array<double, 2>({{M_PI_2, M_PI_2}}));
      }
    } else {
      for (size_t i = 0; i < 4; i++) {
        wedges_for_this_layer.emplace_back(
            temp_inner_radius, temp_outer_radius, temp_inner_sphericity,
            outer_sphericity, gsl::at(wedge_orientations, i),
            use_equiangular_map, Halves::LowerOnly,
            radial_distribution_this_layer,
            std::array<double, 2>({{opening_angle, M_PI_2}}));
        wedges_for_this_layer.emplace_back(
            temp_inner_radius, temp_outer_radius, temp_inner_sphericity,
            outer_sphericity, gsl::at(wedge_orientations, i),
            use_equiangular_map, Halves::UpperOnly,
            radial_distribution_this_layer,
            std::array<double, 2>({{opening_angle, M_PI_2}}));
      }
      const double endcap_opening_angle = M_PI - opening_angle;
      const std::array<double, 2> endcap_opening_angles = {
          {endcap_opening_angle, endcap_opening_angle}};
      wedges_for_this_layer.emplace_back(
          temp_inner_radius, temp_outer_radius, temp_inner_sphericity,
          outer_sphericity, gsl::at(wedge_orientations, 4), use_equiangular_map,
          Halves::Both, radial_distribution_this_layer, endcap_opening_angles,
          false);
      wedges_for_this_layer.emplace_back(
          temp_inner_radius, temp_outer_radius, temp_inner_sphericity,
          outer_sphericity, gsl::at(wedge_orientations, 5), use_equiangular_map,
          Halves::Both, radial_distribution_this_layer, endcap_opening_angles,
          false);
    }
    for (const auto& wedge : wedges_for_this_layer) {
      wedges_for_all_layers.push_back(wedge);
    }

    if (layer_i != radial_partitioning.size()) {
      temp_inner_radius = radial_partitioning.at(layer_i);
      temp_inner_sphericity = outer_sphericity;
    }
  }
  return wedges_for_all_layers;
}

std::vector<domain::CoordinateMaps::Frustum> frustum_coordinate_maps(
    const double length_inner_cube, const double length_outer_cube,
    const bool use_equiangular_map,
    const std::array<double, 3>& origin_preimage,
    const domain::CoordinateMaps::Distribution radial_distribution,
    const std::optional<double> distribution_value, const double sphericity,
    const double opening_angle) {
  ASSERT(length_inner_cube < 0.5 * length_outer_cube,
         "The outer cube is too small! The inner cubes will pierce the surface "
         "of the outer cube.");
  ASSERT(
      abs(origin_preimage[0]) + length_inner_cube < 0.5 * length_outer_cube and
          abs(origin_preimage[1]) + length_inner_cube <
              0.5 * length_outer_cube and
          abs(origin_preimage[2]) + 0.5 * length_inner_cube <
              0.5 * length_outer_cube,
      "The current choice for `origin_preimage` results in the inner cubes "
      "piercing the surface of the outer cube.");
  const auto frustum_orientations = orientations_for_wrappings();
  const double lower = 0.5 * length_inner_cube;
  const double top = 0.5 * length_outer_cube;

  using FrustumMap = domain::CoordinateMaps::Frustum;
  // Binary compact object Domains use 10 Frustums, 4 around each Shell in the
  // +z,-z,+y,-y directions, and 1 Frustum capping each end in +x, -x.
  // The frustums on the left are given a -1.0 indicating a lower half
  // equiangular map while the frustums on the right are given a 1.0 indicating
  // an upper half equiangular map
  // If we wish to stretch this enveloping cube of frustums along its x-axis,
  // the four left frustums and four right frustums have their upper bases
  // stretched, and the end caps have their upper bases shifted upwards.
  const double stretch = tan(0.5 * opening_angle);
  std::vector<FrustumMap> frustums{};
  for (size_t i = 0; i < 4; i++) {
    // frustums on the left
    std::array<double, 3> displacement_from_origin = discrete_rotation(
        gsl::at(frustum_orientations, i).inverse_map(), origin_preimage);
    frustums.push_back(FrustumMap{
        {{{{-2.0 * lower - displacement_from_origin[0],
            -lower - displacement_from_origin[1]}},
          {{-displacement_from_origin[0], lower - displacement_from_origin[1]}},
          {{stretch * -top, -top}},
          {{0.0, top}}}},
        lower - displacement_from_origin[2],
        top,
        gsl::at(frustum_orientations, i),
        use_equiangular_map,
        radial_distribution,
        distribution_value,
        sphericity,
        -1.0,
        opening_angle});
    // frustums on the right
    frustums.push_back(FrustumMap{{{{{-displacement_from_origin[0],
                                      -lower - displacement_from_origin[1]}},
                                    {{2.0 * lower - displacement_from_origin[0],
                                      lower - displacement_from_origin[1]}},
                                    {{0.0, -top}},
                                    {{stretch * top, top}}}},
                                  lower - displacement_from_origin[2],
                                  top,
                                  gsl::at(frustum_orientations, i),
                                  use_equiangular_map,
                                  radial_distribution,
                                  distribution_value,
                                  sphericity,
                                  1.0,
                                  opening_angle});
  }
  // end cap frustum on the right
  std::array<double, 3> displacement_from_origin =
      discrete_rotation(frustum_orientations[4].inverse_map(), origin_preimage);
  frustums.push_back(FrustumMap{{{{{-lower - displacement_from_origin[0],
                                    -lower - displacement_from_origin[1]}},
                                  {{lower - displacement_from_origin[0],
                                    lower - displacement_from_origin[1]}},
                                  {{-top, -top}},
                                  {{top, top}}}},
                                2.0 * lower - displacement_from_origin[2],
                                stretch * top,
                                frustum_orientations[4],
                                use_equiangular_map,
                                radial_distribution,
                                distribution_value,
                                sphericity,
                                0.0,
                                M_PI_2});
  // end cap frustum on the left
  displacement_from_origin =
      discrete_rotation(frustum_orientations[5].inverse_map(), origin_preimage);
  frustums.push_back(FrustumMap{{{{{-lower - displacement_from_origin[0],
                                    -lower - displacement_from_origin[1]}},
                                  {{lower - displacement_from_origin[0],
                                    lower - displacement_from_origin[1]}},
                                  {{-top, -top}},
                                  {{top, top}}}},
                                2.0 * lower - displacement_from_origin[2],
                                stretch * top,
                                frustum_orientations[5],
                                use_equiangular_map,
                                radial_distribution,
                                distribution_value,
                                sphericity,
                                0.0,
                                M_PI_2});
  return frustums;
}

namespace {
std::array<size_t, 8> concatenate_faces(const std::array<size_t, 4>& face1,
                                        const std::array<size_t, 4>& face2) {
  return {{face1[0], face1[1], face1[2], face1[3], face2[0], face2[1], face2[2],
           face2[3]}};
}

std::array<size_t, 4> next_face_in_radial_direction(
    const std::array<size_t, 4>& face) {
  return {{face[0] + 8, face[1] + 8, face[2] + 8, face[3] + 8}};
}

std::array<std::array<size_t, 4>, 6> next_layer_in_radial_direction(
    const std::array<std::array<size_t, 4>, 6>& layer) {
  return {{next_face_in_radial_direction(layer[0]),
           next_face_in_radial_direction(layer[1]),
           next_face_in_radial_direction(layer[2]),
           next_face_in_radial_direction(layer[3]),
           next_face_in_radial_direction(layer[4]),
           next_face_in_radial_direction(layer[5])}};
}

void identify_corners_with_each_other(
    const gsl::not_null<std::vector<std::array<size_t, 8>>*> corners,
    const size_t corner_1, const size_t corner_2) {
  for (auto& block_corners : *corners) {
    for (auto& corner : block_corners) {
      if (corner == corner_2) {
        corner = corner_1;
      }
    }
  }
}
}  // namespace

std::vector<std::array<size_t, 8>> corners_for_radially_layered_domains(
    const size_t number_of_layers, const bool include_central_block,
    const std::array<size_t, 8>& central_block_corners,
    ShellWedges which_wedges) {
  const std::array<size_t, 8>& c = central_block_corners;
  std::array<std::array<size_t, 4>, 6> faces_layer_zero{
      {{{c[4], c[5], c[6], c[7]}},    // Upper z
       {{c[2], c[3], c[0], c[1]}},    // Lower z
       {{c[6], c[7], c[2], c[3]}},    // Upper y
       {{c[0], c[1], c[4], c[5]}},    // Lower y
       {{c[1], c[3], c[5], c[7]}},    // Upper x
       {{c[2], c[0], c[6], c[4]}}}};  // Lower x
  std::vector<std::array<std::array<size_t, 4>, 6>> faces_in_each_layer{
      faces_layer_zero};
  for (size_t layer_j = 0; layer_j < number_of_layers; layer_j++) {
    faces_in_each_layer.push_back(
        next_layer_in_radial_direction(faces_in_each_layer[layer_j]));
  }
  std::vector<std::array<size_t, 8>> corners{};
  for (size_t layer_i = 0; layer_i < number_of_layers; layer_i++) {
    for (size_t face_j = which_wedge_index(which_wedges); face_j < 6;
         face_j++) {
      corners.push_back(concatenate_faces(
          gsl::at(gsl::at(faces_in_each_layer, layer_i), face_j),
          gsl::at(gsl::at(faces_in_each_layer, layer_i + 1), face_j)));
    }
  }
  if (include_central_block) {
    corners.push_back(central_block_corners);
  }
  return corners;
}

std::vector<std::array<size_t, 8>> corners_for_biradially_layered_domains(
    const size_t number_of_radial_layers,
    const size_t number_of_biradial_layers,
    const bool include_central_block_lhs, const bool include_central_block_rhs,
    const std::array<size_t, 8>& central_block_corners_lhs) {
  const size_t total_number_of_layers =
      number_of_radial_layers + number_of_biradial_layers;
  const std::array<size_t, 8> central_block_corners_rhs = [&]() {
    std::array<size_t, 8> local_central_block_corners_rhs{};
    std::transform(
        central_block_corners_lhs.begin(), central_block_corners_lhs.end(),
        local_central_block_corners_rhs.begin(), [&](const size_t corner) {
          return corner + 8 * (total_number_of_layers + 1);
        });
    return local_central_block_corners_rhs;
  }();

  const auto layers_lhs = corners_for_radially_layered_domains(
      total_number_of_layers, include_central_block_lhs,
      central_block_corners_lhs);
  const auto layers_rhs = corners_for_radially_layered_domains(
      total_number_of_layers, include_central_block_rhs,
      central_block_corners_rhs);
  std::vector<std::array<size_t, 8>> corners{};
  // Fill six-block layers on the left-hand side first,
  // then fill six-block layers on the right-hand side.
  for (size_t layer_i = 0; layer_i < number_of_radial_layers; layer_i++) {
    for (size_t face_j = 0; face_j < 6; face_j++) {
      corners.push_back(layers_lhs[face_j + 6 * layer_i]);
    }
  }
  for (size_t layer_i = 0; layer_i < number_of_radial_layers; layer_i++) {
    for (size_t face_j = 0; face_j < 6; face_j++) {
      corners.push_back(layers_rhs[face_j + 6 * layer_i]);
    }
  }
  // Then fill ten-block layers which wrap both sides.
  for (size_t layer_i = number_of_radial_layers;
       layer_i < total_number_of_layers; layer_i++) {
    for (size_t face_j = 0; face_j < 6; face_j++) {
      // Face number 4 is the face in the +x direction,
      // which is the block to be skipped on the left-hand side.
      if (face_j != 4) {
        corners.push_back(layers_lhs[face_j + 6 * layer_i]);
      }
      // Face number 5 is the face in the -x direction,
      // which is the block to be skipped on the right-hand side.
      if (face_j != 5) {
        corners.push_back(layers_rhs[face_j + 6 * layer_i]);
      }
    }
  }
  if (include_central_block_lhs) {
    corners.push_back(central_block_corners_lhs);
  }
  if (include_central_block_rhs) {
    corners.push_back(central_block_corners_rhs);
  }
  // The first corner to identify is on the last full layer.
  const size_t first_corner_to_identify = 2 + number_of_radial_layers * 8;
  const size_t corner_to_identify_with =
      first_corner_to_identify - 1 + (8 * (total_number_of_layers + 1));
  for (size_t j = 0; j < number_of_biradial_layers + 1; j++) {
    for (size_t i = 0; i < 4; i++) {
      const size_t offset = 2 * i + 8 * j;
      identify_corners_with_each_other(&corners,
                                       first_corner_to_identify + offset,
                                       corner_to_identify_with + offset);
    }
  }
  return corners;
}

std::vector<domain::CoordinateMaps::ProductOf3Maps<
    domain::CoordinateMaps::Interval, domain::CoordinateMaps::Interval,
    domain::CoordinateMaps::Interval>>
cyl_wedge_coord_map_center_blocks(
    const double inner_radius, const double lower_z_bound,
    const double upper_z_bound, const bool use_equiangular_map,
    const std::vector<double>& partitioning_in_z,
    const std::vector<domain::CoordinateMaps::Distribution>& distribution_in_z,
    const CylindricalDomainParityFlip parity_flip) {
  using Interval = domain::CoordinateMaps::Interval;
  using Interval3D =
      domain::CoordinateMaps::ProductOf3Maps<Interval, Interval, Interval>;
  std::vector<Interval3D> maps{};
  const double lower_logical_zeta =
      parity_flip == CylindricalDomainParityFlip::z_direction ? 1.0 : -1.0;
  const double upper_logical_zeta =
      parity_flip == CylindricalDomainParityFlip::z_direction ? -1.0 : 1.0;
  double temp_lower_z_bound = lower_z_bound;
  double temp_upper_z_bound{};
  const auto angular_distribution =
      use_equiangular_map ? domain::CoordinateMaps::Distribution::Equiangular
                          : domain::CoordinateMaps::Distribution::Linear;
  for (size_t layer = 0; layer < 1 + partitioning_in_z.size(); layer++) {
    if (layer != partitioning_in_z.size()) {
      temp_upper_z_bound = partitioning_in_z.at(layer);
    } else {
      temp_upper_z_bound = upper_z_bound;
    }
    maps.emplace_back(
        Interval3D{Interval(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                            inner_radius / sqrt(2.0), angular_distribution),
                   Interval(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                            inner_radius / sqrt(2.0), angular_distribution),
                   Interval{lower_logical_zeta, upper_logical_zeta,
                            temp_lower_z_bound, temp_upper_z_bound,
                            distribution_in_z.at(layer), lower_z_bound}});

    if (layer != partitioning_in_z.size()) {
      temp_lower_z_bound = partitioning_in_z.at(layer);
    }
  }
  return maps;
}

std::vector<domain::CoordinateMaps::ProductOf2Maps<
    domain::CoordinateMaps::Wedge<2>, domain::CoordinateMaps::Interval>>
cyl_wedge_coord_map_surrounding_blocks(
    const double inner_radius, const double outer_radius,
    const double lower_z_bound, const double upper_z_bound,
    const bool use_equiangular_map, const double inner_circularity,
    const std::vector<double>& radial_partitioning,
    const std::vector<double>& partitioning_in_z,
    const std::vector<domain::CoordinateMaps::Distribution>&
        radial_distribution,
    const std::vector<domain::CoordinateMaps::Distribution>& distribution_in_z,
    const CylindricalDomainParityFlip parity_flip) {
  using Interval = domain::CoordinateMaps::Interval;
  using Wedge2D = domain::CoordinateMaps::Wedge<2>;
  using Wedge3DPrism =
      domain::CoordinateMaps::ProductOf2Maps<Wedge2D, Interval>;
  ASSERT(radial_distribution.size() == 1 + radial_partitioning.size(),
         "Specify a radial distribution for every cylindrical shell. You "
         "specified "
             << radial_distribution.size() << " items, but the domain has "
             << 1 + radial_partitioning.size() << " shells.");
  ASSERT(radial_distribution.front() ==
             domain::CoordinateMaps::Distribution::Linear,
         "The innermost shell must have a 'Linear' radial distribution because "
         "it changes in circularity.");
  std::vector<Wedge3DPrism> maps{};
  double temp_inner_circularity{};
  double temp_inner_radius{};
  double temp_outer_radius{};
  double temp_lower_z_bound = lower_z_bound;
  double temp_upper_z_bound{};
  const auto use_both_halves = Wedge2D::WedgeHalves::Both;
  const std::array<OrientationMap<2>, 4> wedge_orientations = {
      OrientationMap<2>{std::array<Direction<2>, 2>{
          {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},  //+x wedge
      OrientationMap<2>{std::array<Direction<2>, 2>{
          {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},  //+y wedge
      OrientationMap<2>{std::array<Direction<2>, 2>{
          {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}},  //-x wedge
      OrientationMap<2>{std::array<Direction<2>, 2>{
          {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}}  //-y wedge
  };
  for (size_t layer = 0; layer < 1 + partitioning_in_z.size(); layer++) {
    temp_inner_radius = inner_radius;
    if (layer != partitioning_in_z.size()) {
      temp_upper_z_bound = partitioning_in_z.at(layer);
    } else {
      temp_upper_z_bound = upper_z_bound;
    }
    temp_inner_circularity = inner_circularity;
    for (size_t shell = 0; shell < 1 + radial_partitioning.size(); shell++) {
      if (shell != radial_partitioning.size()) {
        temp_outer_radius = radial_partitioning.at(shell);
      } else {
        temp_outer_radius = outer_radius;
      }
      auto z_map = Interval{
          parity_flip == CylindricalDomainParityFlip::z_direction ? 1.0 : -1.0,
          parity_flip == CylindricalDomainParityFlip::z_direction ? -1.0 : 1.0,
          temp_lower_z_bound,
          temp_upper_z_bound,
          distribution_in_z.at(layer),
          lower_z_bound};
      for (const auto& cardinal_direction : wedge_orientations) {
        maps.emplace_back(Wedge3DPrism{
            Wedge2D{temp_inner_radius, temp_outer_radius,
                    temp_inner_circularity, 1.0, cardinal_direction,
                    use_equiangular_map, use_both_halves,
                    radial_distribution.at(shell)},
            z_map});
      }
      temp_inner_circularity = 1.;
      if (shell != radial_partitioning.size()) {
        temp_inner_radius = radial_partitioning.at(shell);
      }
    }
    if (layer != partitioning_in_z.size()) {
      temp_lower_z_bound = partitioning_in_z.at(layer);
    }
  }
  return maps;
}

template <typename TargetFrame>
std::vector<std::unique_ptr<
    domain::CoordinateMapBase<Frame::BlockLogical, TargetFrame, 3>>>
cyl_wedge_coordinate_maps(
    const double inner_radius, const double outer_radius,
    const double lower_z_bound, const double upper_z_bound,
    const bool use_equiangular_map,
    const std::vector<double>& radial_partitioning,
    const std::vector<double>& partitioning_in_z,
    const std::vector<domain::CoordinateMaps::Distribution>&
        radial_distribution,
    const std::vector<domain::CoordinateMaps::Distribution>&
        distribution_in_z) {
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::BlockLogical, TargetFrame, 3>>>
      cylinder_mapping;
  const auto maps_surrounding = cyl_wedge_coord_map_surrounding_blocks(
      inner_radius, outer_radius, lower_z_bound, upper_z_bound,
      use_equiangular_map, 0.0, radial_partitioning, partitioning_in_z,
      radial_distribution, distribution_in_z,
      CylindricalDomainParityFlip::none);

  // add_to_cylinder_mapping adds the maps for individual blocks in
  // the correct order.  This order is a center block at each height
  // partition and then the surrounding blocks at that height
  // partition.
  auto add_to_cylinder_mapping = [&cylinder_mapping, &partitioning_in_z,
                                  &radial_partitioning,
                                  &maps_surrounding](const auto& maps_center) {
    size_t surrounding_block_index = 0;
    for (size_t height = 0; height < 1 + partitioning_in_z.size(); ++height) {
      cylinder_mapping.emplace_back(
          domain::make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
              maps_center.at(height)));
      for (size_t radius = 0; radius < 1 + radial_partitioning.size();
           ++radius) {
        // There are 4 surrounding blocks at each radius and height
        for (size_t block_at_this_radius = 0; block_at_this_radius < 4;
             ++block_at_this_radius, ++surrounding_block_index) {
          cylinder_mapping.emplace_back(
              domain::make_coordinate_map_base<Frame::BlockLogical,
                                               TargetFrame>(
                  maps_surrounding.at(surrounding_block_index)));
        }
      }
    }
  };

  add_to_cylinder_mapping(cyl_wedge_coord_map_center_blocks(
      inner_radius, lower_z_bound, upper_z_bound, use_equiangular_map,
      partitioning_in_z, distribution_in_z, CylindricalDomainParityFlip::none));
  return cylinder_mapping;
}

std::vector<std::array<size_t, 8>> corners_for_cylindrical_layered_domains(
    const size_t number_of_shells, const size_t number_of_discs) {
  using BlockCorners = std::array<size_t, 8>;
  std::vector<BlockCorners> corners;
  const size_t n_L = 4 * (number_of_shells + 1);  // number of corners per layer
  const BlockCorners center{{0, 1, 2, 3, n_L + 0, n_L + 1, n_L + 2, n_L + 3}};
  const BlockCorners east{{1, 5, 3, 7, n_L + 1, n_L + 5, n_L + 3, n_L + 7}};
  const BlockCorners north{{3, 7, 2, 6, n_L + 3, n_L + 7, n_L + 2, n_L + 6}};
  const BlockCorners west{{2, 6, 0, 4, n_L + 2, n_L + 6, n_L + 0, n_L + 4}};
  const BlockCorners south{{0, 4, 1, 5, n_L + 0, n_L + 4, n_L + 1, n_L + 5}};
  for (size_t i = 0; i < number_of_discs; i++) {
    corners.push_back(center + make_array<8>(i * n_L));
    for (size_t j = 0; j < number_of_shells; j++) {
      const auto offset = make_array<8>(i * n_L + 4 * j);
      corners.push_back(east + offset);   //+x wedge
      corners.push_back(north + offset);  //+y wedge
      corners.push_back(west + offset);   //-x wedge
      corners.push_back(south + offset);  //-y wedge
    }
  }
  return corners;
}

template <size_t VolumeDim>
auto indices_for_rectilinear_domains(
    const Index<VolumeDim>& domain_extents,
    const std::vector<Index<VolumeDim>>& block_indices_to_exclude)
    -> std::vector<Index<VolumeDim>> {
  std::vector<Index<VolumeDim>> block_indices;
  for (IndexIterator<VolumeDim> ii(domain_extents); ii; ++ii) {
    if (std::find(block_indices_to_exclude.begin(),
                  block_indices_to_exclude.end(),
                  *ii) == block_indices_to_exclude.end()) {
      block_indices.push_back(*ii);
    }
  }
  return block_indices;
}

template <size_t VolumeDim>
std::vector<std::array<size_t, two_to_the(VolumeDim)>>
corners_for_rectilinear_domains(
    const Index<VolumeDim>& domain_extents,
    const std::vector<Index<VolumeDim>>& block_indices_to_exclude) {
  std::vector<std::array<size_t, two_to_the(VolumeDim)>> corners{};

  // The number of blocks is one less than
  // the number of corners in each dimension:
  std::array<size_t, VolumeDim> number_of_corners{};
  for (size_t d = 0; d < VolumeDim; d++) {
    gsl::at(number_of_corners, d) = domain_extents[d] + 1;
  }

  Index<VolumeDim> global_corner_extents{number_of_corners};
  for (const auto& block_index : indices_for_rectilinear_domains(
           domain_extents, block_indices_to_exclude)) {
    std::array<size_t, two_to_the(VolumeDim)> corners_for_single_block{};
    for (VolumeCornerIterator<VolumeDim> vci(block_index,
                                             global_corner_extents);
         vci; ++vci) {
      gsl::at(corners_for_single_block, vci.local_corner_number()) =
          vci.global_corner_number();
    }
    corners.push_back(corners_for_single_block);
  }
  return corners;
}

template <typename TargetFrame, typename Map>
std::unique_ptr<domain::CoordinateMapBase<Frame::BlockLogical, TargetFrame, 1>>
product_of_1d_maps(std::array<Map, 1>& maps,
                   const OrientationMap<1>& rotation = {}) {
  if (rotation == OrientationMap<1>{}) {
    return domain::make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
        maps[0]);
  }
  return domain::make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
      domain::CoordinateMaps::DiscreteRotation<1>{rotation}, maps[0]);
}

template <typename TargetFrame, typename Map>
std::unique_ptr<domain::CoordinateMapBase<Frame::BlockLogical, TargetFrame, 2>>
product_of_1d_maps(std::array<Map, 2>& maps,
                   const OrientationMap<2>& rotation = {}) {
  if (rotation == OrientationMap<2>{}) {
    return domain::make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
        domain::CoordinateMaps::ProductOf2Maps<Map, Map>(maps[0], maps[1]));
  }
  return domain::make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
      domain::CoordinateMaps::DiscreteRotation<2>{rotation},
      domain::CoordinateMaps::ProductOf2Maps<Map, Map>(maps[0], maps[1]));
}

template <typename TargetFrame, typename Map>
std::unique_ptr<domain::CoordinateMapBase<Frame::BlockLogical, TargetFrame, 3>>
product_of_1d_maps(std::array<Map, 3>& maps,
                   const OrientationMap<3>& rotation = {}) {
  if (rotation == OrientationMap<3>{}) {
    return domain::make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
        domain::CoordinateMaps::ProductOf3Maps<Map, Map, Map>(maps[0], maps[1],
                                                              maps[2]));
  }
  return domain::make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
      domain::CoordinateMaps::DiscreteRotation<3>{rotation},
      domain::CoordinateMaps::ProductOf3Maps<Map, Map, Map>(maps[0], maps[1],
                                                            maps[2]));
}

template <typename TargetFrame, size_t VolumeDim>
std::vector<std::unique_ptr<
    domain::CoordinateMapBase<Frame::BlockLogical, TargetFrame, VolumeDim>>>
maps_for_rectilinear_domains(
    const Index<VolumeDim>& domain_extents,
    const std::array<std::vector<double>, VolumeDim>& block_demarcations,
    const std::vector<Index<VolumeDim>>& block_indices_to_exclude,
    const std::vector<OrientationMap<VolumeDim>>& orientations_of_all_blocks,
    const bool use_equiangular_map) {
  for (size_t d = 0; d < VolumeDim; d++) {
    ASSERT(gsl::at(block_demarcations, d).size() == domain_extents[d] + 1,
           "If there are N blocks, there must be N+1 demarcations.");
  }
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::BlockLogical, TargetFrame, VolumeDim>>>
      maps{};
  size_t block_orientation_index = 0;
  for (const auto& block_index : indices_for_rectilinear_domains(
           domain_extents, block_indices_to_exclude)) {
    std::array<double, VolumeDim> lower_bounds{};
    std::array<double, VolumeDim> upper_bounds{};
    for (size_t d = 0; d < VolumeDim; d++) {
      gsl::at(lower_bounds, d) =
          gsl::at(gsl::at(block_demarcations, d), block_index[d]);
      gsl::at(upper_bounds, d) =
          gsl::at(gsl::at(block_demarcations, d), block_index[d] + 1);
      ASSERT(gsl::at(upper_bounds, d) > gsl::at(lower_bounds, d),
             "The block demarcations must be strictly increasing.");
    }
    if (not use_equiangular_map) {
      using Affine = domain::CoordinateMaps::Affine;
      std::array<Affine, VolumeDim> affine_maps{};
      for (size_t d = 0; d < VolumeDim; d++) {
        gsl::at(affine_maps, d) = Affine{-1.0, 1.0, gsl::at(lower_bounds, d),
                                         gsl::at(upper_bounds, d)};
      }
      if (not orientations_of_all_blocks.empty()) {
        maps.push_back(product_of_1d_maps<TargetFrame>(
            affine_maps, orientations_of_all_blocks[block_orientation_index]));
      } else {
        maps.push_back(product_of_1d_maps<TargetFrame>(affine_maps));
      }
    } else {
      using Equiangular = domain::CoordinateMaps::Equiangular;
      std::array<Equiangular, VolumeDim> equiangular_maps{};
      for (size_t d = 0; d < VolumeDim; d++) {
        gsl::at(equiangular_maps, d) = Equiangular{
            -1.0, 1.0, gsl::at(lower_bounds, d), gsl::at(upper_bounds, d)};
      }
      if (not orientations_of_all_blocks.empty()) {
        maps.push_back(product_of_1d_maps<TargetFrame>(
            equiangular_maps,
            orientations_of_all_blocks[block_orientation_index]));
      } else {
        maps.push_back(product_of_1d_maps<TargetFrame>(equiangular_maps));
      }
    }
    block_orientation_index++;
  }
  return maps;
}

template <size_t VolumeDim>
std::array<size_t, two_to_the(VolumeDim)> discrete_rotation(
    const OrientationMap<VolumeDim>& orientation,
    const std::array<size_t, two_to_the(VolumeDim)>& corners_of_aligned) {
  // compute the mapped logical corners, as
  // they are the indices into the global corners.
  const std::array<size_t, two_to_the(VolumeDim)> mapped_logical_corners =
      [&orientation]() {
        std::array<size_t, two_to_the(VolumeDim)> result{};
        for (VolumeCornerIterator<VolumeDim> vci{}; vci; ++vci) {
          const std::array<Direction<VolumeDim>, VolumeDim>
              directions_of_logical_corner = vci.directions_of_corner();
          std::array<Direction<VolumeDim>, VolumeDim>
              directions_of_mapped_corner{};
          for (size_t i = 0; i < VolumeDim; i++) {
            gsl::at(directions_of_mapped_corner, i) =
                // The inverse_map is used here to match the sense of rotation
                // used in OrientationMap's discrete_rotation.
                orientation.inverse_map()(
                    gsl::at(directions_of_logical_corner, i));
          }
          size_t mapped_corner = 0;
          for (size_t i = 0; i < VolumeDim; i++) {
            const auto& direction = gsl::at(directions_of_mapped_corner, i);
            if (direction.side() == Side::Upper) {
              mapped_corner += two_to_the(direction.dimension());
            }
          }
          gsl::at(result, vci.local_corner_number()) = mapped_corner;
        }
        return result;
      }();

  std::array<size_t, two_to_the(VolumeDim)> result{};
  for (size_t i = 0; i < two_to_the(VolumeDim); i++) {
    gsl::at(result, i) =
        gsl::at(corners_of_aligned, gsl::at(mapped_logical_corners, i));
  }
  return result;
}

template <size_t VolumeDim>
Domain<VolumeDim> rectilinear_domain(
    const Index<VolumeDim>& domain_extents,
    const std::array<std::vector<double>, VolumeDim>& block_demarcations,
    const std::vector<Index<VolumeDim>>& block_indices_to_exclude,
    const std::vector<OrientationMap<VolumeDim>>& orientations_of_all_blocks,
    const std::array<bool, VolumeDim>& dimension_is_periodic,
    const std::vector<PairOfFaces>& identifications,
    const bool use_equiangular_map) {
  std::vector<Block<VolumeDim>> blocks{};
  auto corners_of_all_blocks =
      corners_for_rectilinear_domains(domain_extents, block_indices_to_exclude);
  auto rotations_of_all_blocks = orientations_of_all_blocks;
  // If no OrientationMaps are provided - default to all aligned:
  if (orientations_of_all_blocks.empty()) {
    rotations_of_all_blocks = std::vector<OrientationMap<VolumeDim>>{
        corners_of_all_blocks.size(), OrientationMap<VolumeDim>{}};
  }
  auto maps = maps_for_rectilinear_domains<Frame::Inertial>(
      domain_extents, block_demarcations, block_indices_to_exclude,
      rotations_of_all_blocks, use_equiangular_map);
  for (size_t i = 0; i < corners_of_all_blocks.size(); i++) {
    corners_of_all_blocks[i] =
        discrete_rotation(rotations_of_all_blocks[i], corners_of_all_blocks[i]);
  }
  std::vector<DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>>
      neighbors_of_all_blocks;
  set_internal_boundaries<VolumeDim>(&neighbors_of_all_blocks,
                                     corners_of_all_blocks);
  set_identified_boundaries<VolumeDim>(identifications, corners_of_all_blocks,
                                       &neighbors_of_all_blocks);
  set_cartesian_periodic_boundaries<VolumeDim>(
      dimension_is_periodic, corners_of_all_blocks, rotations_of_all_blocks,
      &neighbors_of_all_blocks);
  for (size_t i = 0; i < corners_of_all_blocks.size(); i++) {
    blocks.emplace_back(std::move(maps[i]), i,
                        std::move(neighbors_of_all_blocks[i]));
  }
  return Domain<VolumeDim>(std::move(blocks));
}

std::ostream& operator<<(std::ostream& os, const ShellWedges& which_wedges) {
  switch (which_wedges) {
    case ShellWedges::All:
      return os << "All";
    case ShellWedges::FourOnEquator:
      return os << "FourOnEquator";
    case ShellWedges::OneAlongMinusX:
      return os << "OneAlongMinusX";
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR("Unknown ShellWedges type");
      // LCOV_EXCL_STOP
  }
}

template <>
ShellWedges Options::create_from_yaml<ShellWedges>::create<void>(
    const Options::Option& options) {
  const auto which_wedges = options.parse_as<std::string>();
  if (which_wedges == "All") {
    return ShellWedges::All;
  } else if (which_wedges == "FourOnEquator") {
    return ShellWedges::FourOnEquator;
  } else if (which_wedges == "OneAlongMinusX") {
    return ShellWedges::OneAlongMinusX;
  }
  PARSE_ERROR(options.context(),
              "WhichWedges must be 'All', 'FourOnEquator' or 'OneAlongMinusX'");
}

template std::vector<std::unique_ptr<
    domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>>
cyl_wedge_coordinate_maps(
    const double inner_radius, const double outer_radius,
    const double lower_z_bound, const double upper_z_bound,
    const bool use_equiangular_map,
    const std::vector<double>& radial_partitioning,
    const std::vector<double>& partitioning_in_z,
    const std::vector<domain::CoordinateMaps::Distribution>&
        radial_distribution,
    const std::vector<domain::CoordinateMaps::Distribution>& distribution_in_z);
template std::vector<std::unique_ptr<
    domain::CoordinateMapBase<Frame::BlockLogical, Frame::Grid, 3>>>
cyl_wedge_coordinate_maps(
    const double inner_radius, const double outer_radius,
    const double lower_z_bound, const double upper_z_bound,
    const bool use_equiangular_map,
    const std::vector<double>& radial_partitioning,
    const std::vector<double>& partitioning_in_z,
    const std::vector<domain::CoordinateMaps::Distribution>&
        radial_distribution,
    const std::vector<domain::CoordinateMaps::Distribution>& distribution_in_z);
// Explicit instantiations
using Affine2d =
    domain::CoordinateMaps::ProductOf2Maps<domain::CoordinateMaps::Affine,
                                           domain::CoordinateMaps::Affine>;
using Affine3d =
    domain::CoordinateMaps::ProductOf3Maps<domain::CoordinateMaps::Affine,
                                           domain::CoordinateMaps::Affine,
                                           domain::CoordinateMaps::Affine>;
using Equiangular3d =
    domain::CoordinateMaps::ProductOf3Maps<domain::CoordinateMaps::Equiangular,
                                           domain::CoordinateMaps::Equiangular,
                                           domain::CoordinateMaps::Equiangular>;
using AffineIdentity =
    domain::CoordinateMaps::ProductOf2Maps<domain::CoordinateMaps::Affine,
                                           domain::CoordinateMaps::Identity<2>>;

template class domain::CoordinateMaps::ProductOf2Maps<
    domain::CoordinateMaps::Affine, domain::CoordinateMaps::Affine>;
template class domain::CoordinateMaps::ProductOf3Maps<
    domain::CoordinateMaps::Affine, domain::CoordinateMaps::Affine,
    domain::CoordinateMaps::Affine>;
template class domain::CoordinateMaps::ProductOf2Maps<
    domain::CoordinateMaps::Equiangular, domain::CoordinateMaps::Equiangular>;
template class domain::CoordinateMaps::ProductOf3Maps<
    domain::CoordinateMaps::Equiangular, domain::CoordinateMaps::Equiangular,
    domain::CoordinateMaps::Equiangular>;
template class domain::CoordinateMaps::ProductOf2Maps<
    domain::CoordinateMaps::Affine, domain::CoordinateMaps::Identity<2>>;

INSTANTIATE_MAPS_FUNCTIONS(((Affine2d), (Affine3d), (Equiangular3d),
                            (domain::CoordinateMaps::Wedge<3>,
                             domain::CoordinateMaps::EquatorialCompression,
                             AffineIdentity)),
                           (Frame::BlockLogical, Frame::ElementLogical),
                           (Frame::Grid, Frame::Inertial), (double, DataVector))

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                \
  template void set_internal_boundaries(                                    \
      const gsl::not_null<                                                  \
          std::vector<DirectionMap<DIM(data), BlockNeighbor<DIM(data)>>>*>  \
          neighbors_of_all_blocks,                                          \
      const std::vector<std::array<size_t, two_to_the(DIM(data))>>&         \
          corners_of_all_blocks);                                           \
  template void set_internal_boundaries(                                    \
      const gsl::not_null<                                                  \
          std::vector<DirectionMap<DIM(data), BlockNeighbor<DIM(data)>>>*>  \
          neighbors_of_all_blocks,                                          \
      const std::vector<std::unique_ptr<domain::CoordinateMapBase<          \
          Frame::BlockLogical, Frame::Inertial, DIM(data)>>>& maps);        \
  template void set_identified_boundaries(                                  \
      const std::vector<PairOfFaces>& identifications,                      \
      const std::vector<std::array<size_t, two_to_the(DIM(data))>>&         \
          corners_of_all_blocks,                                            \
      const gsl::not_null<                                                  \
          std::vector<DirectionMap<DIM(data), BlockNeighbor<DIM(data)>>>*>  \
          neighbors_of_all_blocks);                                         \
  template auto indices_for_rectilinear_domains(                            \
      const Index<DIM(data)>& domain_extents,                               \
      const std::vector<Index<DIM(data)>>& block_indices_to_exclude)        \
      ->std::vector<Index<DIM(data)>>;                                      \
  template std::vector<std::array<size_t, two_to_the(DIM(data))>>           \
  corners_for_rectilinear_domains(                                          \
      const Index<DIM(data)>& domain_extents,                               \
      const std::vector<Index<DIM(data)>>& block_indices_to_exclude);       \
  template std::vector<std::unique_ptr<domain::CoordinateMapBase<           \
      Frame::BlockLogical, Frame::Inertial, DIM(data)>>>                    \
  maps_for_rectilinear_domains(                                             \
      const Index<DIM(data)>& domain_extents,                               \
      const std::array<std::vector<double>, DIM(data)>& block_demarcations, \
      const std::vector<Index<DIM(data)>>& block_indices_to_exclude,        \
      const std::vector<OrientationMap<DIM(data)>>&                         \
          orientations_of_all_blocks,                                       \
      const bool use_equiangular_map);                                      \
  template std::array<size_t, two_to_the(DIM(data))> discrete_rotation(     \
      const OrientationMap<DIM(data)>& orientation,                         \
      const std::array<size_t, two_to_the(DIM(data))>& corners_of_aligned); \
  template Domain<DIM(data)> rectilinear_domain(                            \
      const Index<DIM(data)>& domain_extents,                               \
      const std::array<std::vector<double>, DIM(data)>& block_demarcations, \
      const std::vector<Index<DIM(data)>>& block_indices_to_exclude,        \
      const std::vector<OrientationMap<DIM(data)>>&                         \
          orientations_of_all_blocks,                                       \
      const std::array<bool, DIM(data)>& dimension_is_periodic,             \
      const std::vector<PairOfFaces>& identifications,                      \
      const bool use_equiangular_map);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1_st, 2_st, 3_st))

#undef DIM
#undef DTYPE
#undef INSTANTIATE
