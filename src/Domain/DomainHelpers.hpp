// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DomainHelper functions

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

#include "DataStructures/Index.hpp"
// Can be forward declaration in C++17
#include "Domain/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/Direction.hpp"
// Can be forward declaration in C++17
#include "Domain/DirectionMap.hpp"  // IWYU pragma: keep
#include "Domain/Side.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

/// \cond
template <typename SourceFrame, typename TargetFrame, size_t Dim>
class CoordinateMapBase;
template <size_t VolumeDim>
class OrientationMap;
namespace Frame {
struct Logical;
}  // namespace Frame
/// \endcond

/// \ingroup ComputationalDomainGroup
/// Each member in `PairOfFaces` holds the global corner ids of a block face.
/// `PairOfFaces` is used in setting up periodic boundary conditions by
/// identifying the two faces with each other.
/// \requires The pair of faces must belong to a single block.
struct PairOfFaces {
  std::vector<size_t> first;
  std::vector<size_t> second;
};

/// \ingroup ComputationalDomainGroup
/// Sets up the BlockNeighbors using the corner numbering scheme
/// to deduce the correct neighbors and orientations. Does not set
/// up periodic boundary conditions.
template <size_t VolumeDim>
void set_internal_boundaries(
    const std::vector<std::array<size_t, two_to_the(VolumeDim)>>&
        corners_of_all_blocks,
    gsl::not_null<
        std::vector<DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>>*>
        neighbors_of_all_blocks);

/// \ingroup ComputationalDomainGroup
/// Sets up additional BlockNeighbors corresponding to any
/// periodic boundary condtions provided by the user. These are
/// stored in identifications.
template <size_t VolumeDim>
void set_periodic_boundaries(
    const std::vector<PairOfFaces>& identifications,
    const std::vector<std::array<size_t, two_to_the(VolumeDim)>>&
        corners_of_all_blocks,
    gsl::not_null<
        std::vector<DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>>*>
        neighbors_of_all_blocks);

/// \ingroup ComputationalDomainGroup
/// \brief The corners for a rectilinear domain made of n-cubes.
///
/// The `domain_extents` argument holds the number of blocks to have
/// in each dimension. The blocks all have aligned orientations by
/// construction. The `block_indices_to_exclude` argument allows the user
/// to selectively exclude blocks from the resulting domain. This allows
/// for the creation of non-trivial shapes such as the net for a tesseract.
template <size_t VolumeDim>
std::vector<std::array<size_t, two_to_the(VolumeDim)>>
corners_for_rectilinear_domains(const Index<VolumeDim>& domain_extents,
                                const std::vector<Index<VolumeDim>>&
                                    block_indices_to_exclude = {}) noexcept;

/// \ingroup ComputationalDomainGroup
/// These are the CoordinateMaps of the Wedge3Ds used in the Sphere, Shell, and
/// binary compact object DomainCreators. This function can also be used to
/// wrap the Sphere or Shell in a cube made of six Wedge3Ds.
/// The argument `x_coord_of_shell_center` specifies a translation of the Shell
/// in the x-direction in the TargetFrame. For example, the BBH DomainCreator
/// uses this to set the position of each BH.
/// When the argument `use_half_wedges` is set to `true`, the wedges in the
/// +z,-z,+y,-y directions are cut in half along their xi-axes. The resulting
/// ten CoordinateMaps are used for the outermost Blocks of the BBH Domain.
template <typename TargetFrame>
std::vector<std::unique_ptr<CoordinateMapBase<Frame::Logical, TargetFrame, 3>>>
wedge_coordinate_maps(double inner_radius, double outer_radius,
                      double inner_sphericity, double outer_sphericity,
                      bool use_equiangular_map,
                      double x_coord_of_shell_center = 0.0,
                      bool use_half_wedges = false) noexcept;

/// \ingroup ComputationalDomainGroup
/// These are the ten Frustums used in the DomainCreators for binary compact
/// objects. The cubes enveloping the two Shells each have a side length of
/// `length_inner_cube`. The ten frustums also make up a cube of their own,
/// of side length `length_outer_cube`.
template <typename TargetFrame>
std::vector<std::unique_ptr<CoordinateMapBase<Frame::Logical, TargetFrame, 3>>>
frustum_coordinate_maps(double length_inner_cube, double length_outer_cube,
                        bool use_equiangular_map) noexcept;

/// \ingroup ComputationalDomainGroup
/// \brief The corners for a domain with radial layers.
///
/// Generates the corners for a Domain which is made of one or more layers
/// of Blocks fully enveloping an interior volume, e.g. Shell or Sphere. The
/// `number_of_layers` specifies how many of these layers of Blocks to have
/// in the final domain.
/// `include_central_block` is set to `true` in Sphere, where the interior
/// volume is filled with a central Block, and `false` in Shell, where the
/// interior volume is left empty.
/// The `central_block_corners` are used as seed values to generate the corners
/// for the surrounding Blocks.
std::vector<std::array<size_t, 8>> corners_for_radially_layered_domains(
    size_t number_of_layers, bool include_central_block,
    const std::array<size_t, 8>& central_block_corners = {
        {1, 2, 3, 4, 5, 6, 7, 8}}) noexcept;

/// \ingroup ComputationalDomainGroup
/// \brief The corners for a domain with biradial layers.
///
/// Generates the corners for a BBH-like Domain which is made of one or more
/// layers of Blocks fully enveloping two interior volumes. The
/// `number_of_radial_layers` gives the number of layers that fully envelop
/// each interior volume with six Blocks each. The `number_of_biradial_layers`
/// gives the number of layers that fully envelop both volumes at once, using
/// ten Blocks per layer as opposed to six. The `central_block_corners_lhs`
/// are used as seed values to generate the corners for the surrounding
/// Blocks.
std::vector<std::array<size_t, 8>> corners_for_biradially_layered_domains(
    size_t number_of_radial_layers, size_t number_of_biradial_layers,
    bool include_central_block_lhs, bool include_central_block_rhs,
    const std::array<size_t, 8>& central_block_corners_lhs = {
        {1, 2, 3, 4, 5, 6, 7, 8}}) noexcept;

/// \ingroup ComputationalDomainGroup
/// \brief Permutes the corner numbers of an n-cube.
///
/// Returns the correct ordering of global corner numbers for a block
/// having this orientation relative to an aligned edifice of blocks,
/// given the corner numbering the block would have if it were aligned.
/// This is useful in creating domains for testing purposes, e.g.
/// RotatedIntervals, RotatedRectangles, and RotatedBricks.
template <size_t VolumeDim>
std::array<size_t, two_to_the(VolumeDim)> discrete_rotation(
    const OrientationMap<VolumeDim>& orientation,
    const std::array<size_t, two_to_the(VolumeDim)>&
        corners_of_aligned) noexcept;

/// \ingroup ComputationalDomainGroup
/// \brief The CoordinateMaps for a rectilinear domain of n-cubes.
///
/// Allows for both Affine and Equiangular maps.
template <typename TargetFrame, size_t VolumeDim>
std::vector<
    std::unique_ptr<CoordinateMapBase<Frame::Logical, TargetFrame, VolumeDim>>>
maps_for_rectilinear_domains(
    const Index<VolumeDim>& domain_extents,
    const std::array<std::vector<double>, VolumeDim>& block_demarcations,
    const std::vector<Index<VolumeDim>>& block_indices_to_exclude,
    bool use_equiangular_map) noexcept;

/// \ingroup ComputationalDomainGroup
/// Iterates over the corners of a VolumeDim-dimensional cube.
template <size_t VolumeDim>
class VolumeCornerIterator {
 public:
  VolumeCornerIterator() noexcept { setup_from_local_corner_number(); }
  explicit VolumeCornerIterator(size_t initial_local_corner_number) noexcept
      : local_corner_number_(initial_local_corner_number) {
    setup_from_local_corner_number();
  }
  VolumeCornerIterator(
      // The block index is also global corner
      // index of the lowest corner of the block.
      Index<VolumeDim> block_index,
      Index<VolumeDim> global_corner_extents) noexcept
      : global_corner_number_(
            collapsed_index(block_index, global_corner_extents)),
        global_corner_index_(block_index),
        global_corner_extents_(global_corner_extents) {}
  void operator++() noexcept {
    ++local_corner_number_;
    setup_from_local_corner_number();
  }
  explicit operator bool() const noexcept {
    return local_corner_number_ < two_to_the(VolumeDim);
  }
  const size_t& local_corner_number() const noexcept {
    return local_corner_number_;
  }
  size_t global_corner_number() const noexcept {
    std::array<size_t, VolumeDim> new_indices{};
    for (size_t i = 0; i < VolumeDim; i++) {
      gsl::at(new_indices, i) =
          global_corner_index_[i] +
          (gsl::at(array_sides_, i) == Side::Upper ? 1 : 0);
    }
    const Index<VolumeDim> interior_multi_index(new_indices);
    return collapsed_index(interior_multi_index, global_corner_extents_);
  }
  const std::array<Side, VolumeDim>& operator()() const noexcept {
    return array_sides_;
  }
  const std::array<Side, VolumeDim>& operator*() const noexcept {
    return array_sides_;
  }
  const std::array<double, VolumeDim>& coords_of_corner() const noexcept {
    return coords_of_corner_;
  }
  const std::array<Direction<VolumeDim>, VolumeDim>& directions_of_corner()
      const noexcept {
    return array_directions_;
  }
  void setup_from_local_corner_number() noexcept {
    for (size_t i = 0; i < VolumeDim; i++) {
      gsl::at(coords_of_corner_, i) =
          2.0 * get_nth_bit(local_corner_number_, i) - 1.0;
      gsl::at(array_sides_, i) =
          2 * get_nth_bit(local_corner_number_, i) - 1 == 1 ? Side::Upper
                                                            : Side::Lower;
      gsl::at(array_directions_, i) =
          Direction<VolumeDim>(i, gsl::at(array_sides_, i));
    }
  }

 private:
  size_t local_corner_number_ = 0;
  size_t global_corner_number_{std::numeric_limits<size_t>::max()};
  Index<VolumeDim> global_corner_index_{};
  Index<VolumeDim> global_corner_extents_{};
  std::array<Side, VolumeDim> array_sides_ = make_array<VolumeDim>(Side::Lower);
  std::array<Direction<VolumeDim>, VolumeDim> array_directions_{};
  std::array<double, VolumeDim> coords_of_corner_ = make_array<VolumeDim>(-1.0);
};
