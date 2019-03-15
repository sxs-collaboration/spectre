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
#include "Domain/BlockNeighbor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/EquatorialCompression.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/Frustum.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/Wedge3D.hpp"
#include "Domain/Direction.hpp"
#include "Domain/DirectionMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/Side.hpp"
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Numeric.hpp"

namespace {

// returns the common corners in the global corner numbering scheme.
template <size_t VolumeDim>
std::vector<size_t> get_common_global_corners(
    const std::array<size_t, two_to_the(VolumeDim)>& block_1_global_corners,
    const std::array<size_t, two_to_the(VolumeDim)>&
        block_2_global_corners) noexcept {
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
        alignment_of_shared_face_in_neighbor_frame) noexcept {
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
        corners_of_all_blocks) noexcept {
  size_t number_of_blocks_found = 0;
  size_t id_of_found_block = std::numeric_limits<size_t>::max();
  for (size_t j = 0; j < corners_of_all_blocks.size(); j++) {
    const auto& corners_set = corners_of_all_blocks[j];
    auto is_in_corners_set = [&corners_set](const auto element) noexcept {
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
    const std::vector<size_t> block_global_common_corners) noexcept {
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
std::array<tnsr::I<double, VolumeDim, Frame::Logical>,
           two_to_the(VolumeDim - 1)>
logical_coords_of_corners(
    const std::array<size_t, two_to_the(VolumeDim - 1)>& local_ids) noexcept {
  std::array<tnsr::I<double, VolumeDim, Frame::Logical>,
             two_to_the(VolumeDim - 1)>
      result{};
  std::transform(local_ids.begin(), local_ids.end(),
                 result.begin(), [](const size_t id) noexcept {
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
                     two_to_the(VolumeDim - 1)>& face_pts) noexcept {
  const auto summed_point = alg::accumulate(
      face_pts, tnsr::I<double, VolumeDim, Frame::Logical>(0.0),
      [](tnsr::I<double, VolumeDim, Frame::Logical> & prior,
         const tnsr::I<double, VolumeDim, Frame::Logical>& point) noexcept {
        for (size_t i = 0; i < prior.size(); i++) {
          prior[i] += point[i];
        }
        return prior;
      });
  ASSERT(VolumeDim - 1 == static_cast<size_t>(std::count(
                              summed_point.begin(), summed_point.end(), 0.0)),
         "The face_pts passed in do not correspond to a face.");
  const auto index = static_cast<size_t>(
      alg::find_if(
          summed_point, [](const double x) noexcept { return x != 0; }) -
      summed_point.begin());
  return Direction<VolumeDim>(
      index, summed_point[index] > 0 ? Side::Upper : Side::Lower);
}

template <size_t VolumeDim>
std::vector<Direction<VolumeDim>> get_directions_along_face(
    const std::array<size_t, two_to_the(VolumeDim - 1)>&
        face_local_corners) noexcept {
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
        corners_of_all_blocks) noexcept {
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
        corners_of_all_blocks) noexcept {
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
    gsl::not_null<
        std::vector<DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>>*>
        neighbors_of_all_blocks) noexcept {
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
    const std::vector<std::array<size_t, two_to_the(VolumeDim)>>&
        corners_of_all_blocks,
    gsl::not_null<
        std::vector<DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>>*>
        neighbors_of_all_blocks) noexcept {
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
void set_identified_boundaries(
    const std::vector<PairOfFaces>& identifications,
    const std::vector<std::array<size_t, two_to_the(VolumeDim)>>&
        corners_of_all_blocks,
    gsl::not_null<
        std::vector<DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>>*>
        neighbors_of_all_blocks) noexcept {
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

// A Block or Blocks can be wrapped in an outer layer of Blocks surrounding
// the original Block(s). In the BBH Domain, this occurs several times, using
// both Wedges and Frustums. The simplest example in which wrapping is used is
// Sphere, where the central Block is wrapped with six Wedge3Ds.
std::array<OrientationMap<3>, 6> orientations_for_wrappings() noexcept {
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

namespace {
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
}  // namespace

template <typename TargetFrame>
std::vector<
    std::unique_ptr<domain::CoordinateMapBase<Frame::Logical, TargetFrame, 3>>>
wedge_coordinate_maps(const double inner_radius, const double outer_radius,
                      const double inner_sphericity,
                      const double outer_sphericity,
                      const bool use_equiangular_map,
                      const double x_coord_of_shell_center,
                      const bool use_half_wedges, const double aspect_ratio,
                      const bool use_logarithmic_map,
                      const ShellWedges which_wedges,
                      const size_t number_of_layers) noexcept {
  ASSERT(not use_half_wedges or which_wedges == ShellWedges::All,
         "If we are using half wedges we must also be using ShellWedges::All.");
  ASSERT(number_of_layers == 1 or inner_sphericity == outer_sphericity,
         "If we are using more than one layer the inner and outer sphericities "
         "must match.");

  const auto wedge_orientations = orientations_for_wrappings();

  using Wedge3DMap = domain::CoordinateMaps::Wedge3D;
  using Halves = Wedge3DMap::WedgeHalves;
  std::vector<Wedge3DMap> wedges{};

  // Set up layers:
  const double delta_zeta = 2.0 / number_of_layers;
  for (size_t layer_i = 0; layer_i < number_of_layers; layer_i++) {
    // Linear interpolation variables in the radial direction from -1 to 1
    const double global_zeta_inner = -1.0 + layer_i * delta_zeta;
    const double global_zeta_outer = -1.0 + (layer_i + 1.0) * delta_zeta;
    const double inner_radius_layer_i =
        use_logarithmic_map
            ? pow(inner_radius, 0.5 * (1.0 - global_zeta_inner)) *
                  pow(outer_radius, 0.5 * (1.0 + global_zeta_inner))
            : inner_radius * 0.5 * (1.0 - global_zeta_inner) +
                  outer_radius * 0.5 * (1.0 + global_zeta_inner);
    const double outer_radius_layer_i =
        use_logarithmic_map
            ? pow(inner_radius, 0.5 * (1.0 - global_zeta_outer)) *
                  pow(outer_radius, 0.5 * (1.0 + global_zeta_outer))
            : inner_radius * 0.5 * (1.0 - global_zeta_outer) +
                  outer_radius * 0.5 * (1.0 + global_zeta_outer);
    for (size_t face_j = which_wedge_index(which_wedges); face_j < 6;
         face_j++) {
      wedges.emplace_back(inner_radius_layer_i, outer_radius_layer_i,
                          gsl::at(wedge_orientations, face_j), inner_sphericity,
                          outer_sphericity, use_equiangular_map, Halves::Both,
                          use_logarithmic_map);
    }
  }

  // Set up translation map:
  using Identity2D = domain::CoordinateMaps::Identity<2>;
  using Affine = domain::CoordinateMaps::Affine;
  const auto translation =
      domain::CoordinateMaps::ProductOf2Maps<Affine, Identity2D>(
          Affine{-1.0, 1.0, -1.0 + x_coord_of_shell_center,
                 1.0 + x_coord_of_shell_center},
          Identity2D{});

  // Set up compression map:
  const auto compression =
      domain::CoordinateMaps::EquatorialCompression{aspect_ratio};

  if (use_half_wedges) {
    std::vector<Wedge3DMap> wedges_and_half_wedges{};
    for (size_t i = 0; i < 4; i++) {
      wedges_and_half_wedges.emplace_back(
          inner_radius, outer_radius, gsl::at(wedge_orientations, i),
          inner_sphericity, outer_sphericity, use_equiangular_map,
          Halves::LowerOnly, use_logarithmic_map);
      wedges_and_half_wedges.emplace_back(
          inner_radius, outer_radius, gsl::at(wedge_orientations, i),
          inner_sphericity, outer_sphericity, use_equiangular_map,
          Halves::UpperOnly, use_logarithmic_map);
    }
    wedges_and_half_wedges.push_back(wedges[4]);
    wedges_and_half_wedges.push_back(wedges[5]);

    return domain::make_vector_coordinate_map_base<Frame::Logical, TargetFrame,
                                                   3>(
        std::move(wedges_and_half_wedges), compression, translation);
  }

  return domain::make_vector_coordinate_map_base<Frame::Logical, TargetFrame,
                                                 3>(std::move(wedges),
                                                    compression, translation);
}

template <typename TargetFrame>
std::vector<
    std::unique_ptr<domain::CoordinateMapBase<Frame::Logical, TargetFrame, 3>>>
frustum_coordinate_maps(const double length_inner_cube,
                        const double length_outer_cube,
                        const bool use_equiangular_map,
                        const std::array<double, 3>& origin_preimage,
                        const double projective_scale_factor) noexcept {
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
  //+z,-z,+y,-y directions, and 1 Frustum capping each end in +x, -x.
  std::vector<FrustumMap> frustums{};
  for (size_t i = 0; i < 4; i++) {
    // frustums on the left
    std::array<double, 3> displacement_from_origin = discrete_rotation(
        gsl::at(frustum_orientations, i).inverse_map(), origin_preimage);
    frustums.push_back(FrustumMap{
        {{{{-2.0 * lower - displacement_from_origin[0],
            -lower - displacement_from_origin[1]}},
          {{-displacement_from_origin[0], lower - displacement_from_origin[1]}},
          {{-top, -top}},
          {{0.0, top}}}},
        lower - displacement_from_origin[2],
        top,
        gsl::at(frustum_orientations, i),
        use_equiangular_map,
        projective_scale_factor});
    // frustums on the right
    frustums.push_back(FrustumMap{{{{{-displacement_from_origin[0],
                                      -lower - displacement_from_origin[1]}},
                                    {{2.0 * lower - displacement_from_origin[0],
                                      lower - displacement_from_origin[1]}},
                                    {{0.0, -top}},
                                    {{top, top}}}},
                                  lower - displacement_from_origin[2],
                                  top,
                                  gsl::at(frustum_orientations, i),
                                  use_equiangular_map,
                                  projective_scale_factor});
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
                                top,
                                frustum_orientations[4],
                                use_equiangular_map,
                                projective_scale_factor});
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
                                top,
                                frustum_orientations[5],
                                use_equiangular_map,
                                projective_scale_factor});

  // clang-tidy: trivially copyable
  return domain::make_vector_coordinate_map_base<Frame::Logical, TargetFrame,
                                                 3>(
      std::move(frustums));  // NOLINT
}

namespace {
std::array<size_t, 8> concatenate_faces(
    const std::array<size_t, 4>& face1,
    const std::array<size_t, 4>& face2) noexcept {
  return {{face1[0], face1[1], face1[2], face1[3], face2[0], face2[1], face2[2],
           face2[3]}};
}

std::array<size_t, 4> next_face_in_radial_direction(
    const std::array<size_t, 4>& face) noexcept {
  return {{face[0] + 8, face[1] + 8, face[2] + 8, face[3] + 8}};
}

std::array<std::array<size_t, 4>, 6> next_layer_in_radial_direction(
    const std::array<std::array<size_t, 4>, 6>& layer) noexcept {
  return {{next_face_in_radial_direction(layer[0]),
           next_face_in_radial_direction(layer[1]),
           next_face_in_radial_direction(layer[2]),
           next_face_in_radial_direction(layer[3]),
           next_face_in_radial_direction(layer[4]),
           next_face_in_radial_direction(layer[5])}};
}

void identify_corners_with_each_other(
    const gsl::not_null<std::vector<std::array<size_t, 8>>*> corners,
    const size_t corner_1, const size_t corner_2) noexcept {
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
    ShellWedges which_wedges) noexcept {
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
    const std::array<size_t, 8>& central_block_corners_lhs) noexcept {
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

template <size_t VolumeDim>
std::vector<std::array<size_t, two_to_the(VolumeDim)>>
corners_for_rectilinear_domains(
    const Index<VolumeDim>& domain_extents,
    const std::vector<Index<VolumeDim>>& block_indices_to_exclude) noexcept {
  std::vector<std::array<size_t, two_to_the(VolumeDim)>> corners{};

  // The number of blocks is one less than
  // the number of corners in each dimension:
  std::array<size_t, VolumeDim> number_of_corners{};
  for (size_t d = 0; d < VolumeDim; d++) {
    gsl::at(number_of_corners, d) = domain_extents[d] + 1;
  }

  Index<VolumeDim> global_corner_extents{number_of_corners};
  for (IndexIterator<VolumeDim> ii(domain_extents); ii; ++ii) {
    std::array<size_t, two_to_the(VolumeDim)> corners_for_single_block{};
    for (VolumeCornerIterator<VolumeDim> vci(*ii, global_corner_extents); vci;
         ++vci) {
      gsl::at(corners_for_single_block, vci.local_corner_number()) =
          vci.global_corner_number();
    }
    if (std::find(block_indices_to_exclude.begin(),
                  block_indices_to_exclude.end(),
                  *ii) == block_indices_to_exclude.end()) {
      corners.push_back(corners_for_single_block);
    }
  }
  return corners;
}

template <typename TargetFrame, typename Map>
std::unique_ptr<domain::CoordinateMapBase<Frame::Logical, TargetFrame, 1>>
product_of_1d_maps(std::array<Map, 1>& maps,
                   const OrientationMap<1>& rotation = {}) noexcept {
  if (rotation == OrientationMap<1>{}) {
    return domain::make_coordinate_map_base<Frame::Logical, TargetFrame>(
        maps[0]);
  }
  return domain::make_coordinate_map_base<Frame::Logical, TargetFrame>(
      domain::CoordinateMaps::DiscreteRotation<1>{rotation}, maps[0]);
}

template <typename TargetFrame, typename Map>
std::unique_ptr<domain::CoordinateMapBase<Frame::Logical, TargetFrame, 2>>
product_of_1d_maps(std::array<Map, 2>& maps,
                   const OrientationMap<2>& rotation = {}) noexcept {
  if (rotation == OrientationMap<2>{}) {
    return domain::make_coordinate_map_base<Frame::Logical, TargetFrame>(
        domain::CoordinateMaps::ProductOf2Maps<Map, Map>(maps[0], maps[1]));
  }
  return domain::make_coordinate_map_base<Frame::Logical, TargetFrame>(
      domain::CoordinateMaps::DiscreteRotation<2>{rotation},
      domain::CoordinateMaps::ProductOf2Maps<Map, Map>(maps[0], maps[1]));
}

template <typename TargetFrame, typename Map>
std::unique_ptr<domain::CoordinateMapBase<Frame::Logical, TargetFrame, 3>>
product_of_1d_maps(std::array<Map, 3>& maps,
                   const OrientationMap<3>& rotation = {}) noexcept {
  if (rotation == OrientationMap<3>{}) {
    return domain::make_coordinate_map_base<Frame::Logical, TargetFrame>(
        domain::CoordinateMaps::ProductOf3Maps<Map, Map, Map>(maps[0], maps[1],
                                                              maps[2]));
  }
  return domain::make_coordinate_map_base<Frame::Logical, TargetFrame>(
      domain::CoordinateMaps::DiscreteRotation<3>{rotation},
      domain::CoordinateMaps::ProductOf3Maps<Map, Map, Map>(maps[0], maps[1],
                                                            maps[2]));
}

template <typename TargetFrame, size_t VolumeDim>
std::vector<std::unique_ptr<
    domain::CoordinateMapBase<Frame::Logical, TargetFrame, VolumeDim>>>
maps_for_rectilinear_domains(
    const Index<VolumeDim>& domain_extents,
    const std::array<std::vector<double>, VolumeDim>& block_demarcations,
    const std::vector<Index<VolumeDim>>& block_indices_to_exclude,
    const std::vector<OrientationMap<VolumeDim>>& orientations_of_all_blocks,
    const bool use_equiangular_map) noexcept {
  for (size_t d = 0; d < VolumeDim; d++) {
    ASSERT(gsl::at(block_demarcations, d).size() == domain_extents[d] + 1,
           "If there are N blocks, there must be N+1 demarcations.");
  }
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Logical, TargetFrame, VolumeDim>>>
      maps{};
  // block_orientation_index is the index into orientation_of_all_blocks,
  // and is equal to IndexIterator.collapsed_index()
  // except in the case where block_indices_to_exclude.size() != 0. That is,
  // orientation_of_all_blocks does not get incremented if a block is excluded.
  // (ii.collapsed_index() is incremented in both case, and if used as the
  // index into orientation_of_all_blocks would lead to
  // out-of-bounds indexing into the orientations_of_all_blocks array.
  size_t block_orientation_index = 0;
  for (IndexIterator<VolumeDim> ii(domain_extents); ii; ++ii) {
    if (std::find(block_indices_to_exclude.begin(),
                  block_indices_to_exclude.end(),
                  *ii) == block_indices_to_exclude.end()) {
      std::array<double, VolumeDim> lower_bounds{};
      std::array<double, VolumeDim> upper_bounds{};
      for (size_t d = 0; d < VolumeDim; d++) {
        gsl::at(lower_bounds, d) =
            gsl::at(gsl::at(block_demarcations, d), (*ii)[d]);
        gsl::at(upper_bounds, d) =
            gsl::at(gsl::at(block_demarcations, d), (*ii)[d] + 1);
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
              affine_maps,
              orientations_of_all_blocks[block_orientation_index]));
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
  }
  return maps;
}

template <size_t VolumeDim>
std::array<size_t, two_to_the(VolumeDim)> discrete_rotation(
    const OrientationMap<VolumeDim>& orientation,
    const std::array<size_t, two_to_the(VolumeDim)>&
        corners_of_aligned) noexcept {
  // compute the mapped logical corners, as
  // they are the indices into the global corners.
  const std::array<size_t, two_to_the(VolumeDim)>
      mapped_logical_corners = [&orientation]() noexcept {
    std::array<size_t, two_to_the(VolumeDim)> result{};
    for (VolumeCornerIterator<VolumeDim> vci{}; vci; ++vci) {
      const std::array<Direction<VolumeDim>, VolumeDim>
          directions_of_logical_corner = vci.directions_of_corner();
      std::array<Direction<VolumeDim>, VolumeDim> directions_of_mapped_corner{};
      for (size_t i = 0; i < VolumeDim; i++) {
        gsl::at(directions_of_mapped_corner, i) =
            // The inverse_map is used here to match the sense of rotation
            // used in OrientationMap's discrete_rotation.
            orientation.inverse_map()(gsl::at(directions_of_logical_corner, i));
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
  }
  ();

  std::array<size_t, two_to_the(VolumeDim)> result{};
  for (size_t i = 0; i < two_to_the(VolumeDim); i++) {
    gsl::at(result, i) =
        gsl::at(corners_of_aligned, gsl::at(mapped_logical_corners, i));
  }
  return result;
}

template <size_t VolumeDim, typename TargetFrame>
Domain<VolumeDim, TargetFrame> rectilinear_domain(
    const Index<VolumeDim>& domain_extents,
    const std::array<std::vector<double>, VolumeDim>& block_demarcations,
    const std::vector<Index<VolumeDim>>& block_indices_to_exclude,
    const std::vector<OrientationMap<VolumeDim>>& orientations_of_all_blocks,
    const std::array<bool, VolumeDim>& dimension_is_periodic,
    const std::vector<PairOfFaces>& identifications,
    bool use_equiangular_map) noexcept {
  std::vector<Block<VolumeDim, TargetFrame>> blocks{};
  auto corners_of_all_blocks =
      corners_for_rectilinear_domains(domain_extents, block_indices_to_exclude);
  auto rotations_of_all_blocks = orientations_of_all_blocks;
  // If no OrientationMaps are provided - default to all aligned:
  if (orientations_of_all_blocks.empty()) {
    rotations_of_all_blocks = std::vector<OrientationMap<VolumeDim>>{
        corners_of_all_blocks.size(), OrientationMap<VolumeDim>{}};
  }
  auto maps = maps_for_rectilinear_domains<TargetFrame>(
      domain_extents, block_demarcations, block_indices_to_exclude,
      rotations_of_all_blocks, use_equiangular_map);
  for (size_t i = 0; i < corners_of_all_blocks.size(); i++) {
    corners_of_all_blocks[i] =
        discrete_rotation(rotations_of_all_blocks[i], corners_of_all_blocks[i]);
  }
  std::vector<DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>>
      neighbors_of_all_blocks;
  set_internal_boundaries<VolumeDim>(corners_of_all_blocks,
                                     &neighbors_of_all_blocks);
  set_identified_boundaries<VolumeDim>(identifications, corners_of_all_blocks,
                                       &neighbors_of_all_blocks);
  set_cartesian_periodic_boundaries<VolumeDim>(
      dimension_is_periodic, corners_of_all_blocks, rotations_of_all_blocks,
      &neighbors_of_all_blocks);
  for (size_t i = 0; i < corners_of_all_blocks.size(); i++) {
    blocks.emplace_back(std::move(maps[i]), i,
                        std::move(neighbors_of_all_blocks[i]));
  }
  return Domain<VolumeDim, TargetFrame>(std::move(blocks));
}

std::ostream& operator<<(std::ostream& os,
                         const ShellWedges& which_wedges) noexcept {
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
ShellWedges create_from_yaml<ShellWedges>::create<void>(const Option& options) {
  const std::string which_wedges = options.parse_as<std::string>();
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

template void set_internal_boundaries(
    const std::vector<std::array<size_t, 2>>& corners_of_all_blocks,
    gsl::not_null<std::vector<DirectionMap<1, BlockNeighbor<1>>>*>
        neighbors_of_all_blocks) noexcept;
template void set_internal_boundaries(
    const std::vector<std::array<size_t, 4>>& corners_of_all_blocks,
    gsl::not_null<std::vector<DirectionMap<2, BlockNeighbor<2>>>*>
        neighbors_of_all_blocks) noexcept;
template void set_internal_boundaries(
    const std::vector<std::array<size_t, 8>>& corners_of_all_blocks,
    gsl::not_null<std::vector<DirectionMap<3, BlockNeighbor<3>>>*>
        neighbors_of_all_blocks) noexcept;

template void set_identified_boundaries(
    const std::vector<PairOfFaces>& identifications,
    const std::vector<std::array<size_t, 2>>& corners_of_all_blocks,
    gsl::not_null<std::vector<DirectionMap<1, BlockNeighbor<1>>>*>
        neighbors_of_all_blocks) noexcept;
template void set_identified_boundaries(
    const std::vector<PairOfFaces>& identifications,
    const std::vector<std::array<size_t, 4>>& corners_of_all_blocks,
    gsl::not_null<std::vector<DirectionMap<2, BlockNeighbor<2>>>*>
        neighbors_of_all_blocks) noexcept;
template void set_identified_boundaries(
    const std::vector<PairOfFaces>& identifications,
    const std::vector<std::array<size_t, 8>>& corners_of_all_blocks,
    gsl::not_null<std::vector<DirectionMap<3, BlockNeighbor<3>>>*>
        neighbors_of_all_blocks) noexcept;
template std::vector<std::array<size_t, 2>> corners_for_rectilinear_domains(
    const Index<1>& domain_extents,
    const std::vector<Index<1>>& block_indices_to_exclude) noexcept;
template std::vector<std::array<size_t, 4>> corners_for_rectilinear_domains(
    const Index<2>& domain_extents,
    const std::vector<Index<2>>& block_indices_to_exclude) noexcept;
template std::vector<std::array<size_t, 8>> corners_for_rectilinear_domains(
    const Index<3>& domain_extents,
    const std::vector<Index<3>>& block_indices_to_exclude) noexcept;
template std::array<size_t, 2> discrete_rotation(
    const OrientationMap<1>& orientation,
    const std::array<size_t, 2>& corners_of_aligned) noexcept;
template std::array<size_t, 4> discrete_rotation(
    const OrientationMap<2>& orientation,
    const std::array<size_t, 4>& corners_of_aligned) noexcept;
template std::array<size_t, 8> discrete_rotation(
    const OrientationMap<3>& orientation,
    const std::array<size_t, 8>& corners_of_aligned) noexcept;
template std::vector<std::unique_ptr<
    domain::CoordinateMapBase<Frame::Logical, Frame::Inertial, 3>>>
wedge_coordinate_maps(const double inner_radius, const double outer_radius,
                      const double inner_sphericity,
                      const double outer_sphericity,
                      const bool use_equiangular_map,
                      const double x_coord_of_shell_center,
                      const bool use_wedge_halves, const double aspect_ratio,
                      const bool use_logarithmic_map,
                      const ShellWedges which_wedges,
                      const size_t number_of_layers) noexcept;
template std::vector<
    std::unique_ptr<domain::CoordinateMapBase<Frame::Logical, Frame::Grid, 3>>>
wedge_coordinate_maps(const double inner_radius, const double outer_radius,
                      const double inner_sphericity,
                      const double outer_sphericity,
                      const bool use_equiangular_map,
                      const double x_coord_of_shell_center,
                      const bool use_wedge_halves, const double aspect_ratio,
                      const bool use_logarithmic_map,
                      const ShellWedges which_wedges,
                      const size_t number_of_layers) noexcept;
template std::vector<std::unique_ptr<
    domain::CoordinateMapBase<Frame::Logical, Frame::Inertial, 3>>>
frustum_coordinate_maps(const double length_inner_cube,
                        const double length_outer_cube,
                        const bool use_equiangular_map,
                        const std::array<double, 3>& origin_preimage,
                        const double projective_scale_factor) noexcept;
template std::vector<
    std::unique_ptr<domain::CoordinateMapBase<Frame::Logical, Frame::Grid, 3>>>
frustum_coordinate_maps(const double length_inner_cube,
                        const double length_outer_cube,
                        const bool use_equiangular_map,
                        const std::array<double, 3>& origin_preimage,
                        const double projective_scale_factor) noexcept;
// Explicit instantiations
/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                \
  template std::vector<std::unique_ptr<                                     \
      domain::CoordinateMapBase<Frame::Logical, FRAME(data), DIM(data)>>>   \
  maps_for_rectilinear_domains(                                             \
      const Index<DIM(data)>& domain_extents,                               \
      const std::array<std::vector<double>, DIM(data)>& block_demarcations, \
      const std::vector<Index<DIM(data)>>& block_indices_to_exclude,        \
      const std::vector<OrientationMap<DIM(data)>>&                         \
          orientations_of_all_blocks,                                       \
      const bool use_equiangular_map) noexcept;                             \
  template Domain<DIM(data), FRAME(data)> rectilinear_domain(               \
      const Index<DIM(data)>& domain_extents,                               \
      const std::array<std::vector<double>, DIM(data)>& block_demarcations, \
      const std::vector<Index<DIM(data)>>& block_indices_to_exclude,        \
      const std::vector<OrientationMap<DIM(data)>>&                         \
          orientations_of_all_blocks,                                       \
      const std::array<bool, DIM(data)>& dimension_is_periodic,             \
      const std::vector<PairOfFaces>& identifications,                      \
      bool use_equiangular_map) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef INSTANTIATE
/// \endcond
