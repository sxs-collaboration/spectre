// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DomainHelper functions

#pragma once

#include <array>
#include <cstddef>
#include <iosfwd>
#include <limits>
#include <memory>
#include <vector>

#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Side.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

/// \cond
template <size_t VolumeDim>
class BlockNeighbor;
namespace domain {
template <typename SourceFrame, typename TargetFrame, size_t Dim>
class CoordinateMapBase;
}  // namespace domain
template <size_t VolumeDim, typename T>
class DirectionMap;
template <size_t VolumeDim, typename TargetFrame>
class Domain;
template <size_t VolumeDim>
class OrientationMap;
class Option;
template <typename T>
struct create_from_yaml;
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
        neighbors_of_all_blocks) noexcept;

/// \ingroup ComputationalDomainGroup
/// Sets up additional BlockNeighbors corresponding to any
/// identifications of faces provided by the user. Can be used
/// for manually setting up periodic boundary conditions.
template <size_t VolumeDim>
void set_identified_boundaries(
    const std::vector<PairOfFaces>& identifications,
    const std::vector<std::array<size_t, two_to_the(VolumeDim)>>&
        corners_of_all_blocks,
    gsl::not_null<
        std::vector<DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>>*>
        neighbors_of_all_blocks) noexcept;

/// \ingroup ComputationalDomainGroup
/// \brief The corners for a rectilinear domain made of n-cubes.
///
/// The `domain_extents` argument holds the number of blocks to have
/// in each dimension. The blocks all have aligned orientations by
/// construction. The `block_indices_to_exclude` argument allows the user
/// to selectively exclude blocks from the resulting domain. This allows
/// for the creation of non-trivial shapes such as the net for a tesseract.
template <size_t VolumeDim>
auto corners_for_rectilinear_domains(
    const Index<VolumeDim>& domain_extents,
    const std::vector<Index<VolumeDim>>& block_indices_to_exclude = {}) noexcept
    -> std::vector<std::array<size_t, two_to_the(VolumeDim)>>;

/// \ingroup ComputationalDomainGroup
/// The number of wedges to include in the Shell domain.
enum class ShellWedges {
  /// Use the entire shell
  All,
  /// Use only the four equatorial wedges
  FourOnEquator,
  /// Use only the single wedge along -x
  OneAlongMinusX
};

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
/// The argument `aspect_ratio` sets the equatorial compression factor,
/// used by the EquatorialCompression maps which get composed with the Wedges.
/// This is done if `aspect_ratio` is set to something other than the default
/// value of one. When the argument `use_logarithmic_map` is set to `true`,
/// the radial gridpoints of the wedge map are set to be spaced logarithmically.
/// The `number_of_layers` is used when the user wants to have multiple layers
/// of Blocks in the radial direction.
template <typename TargetFrame>
auto wedge_coordinate_maps(double inner_radius, double outer_radius,
                           double inner_sphericity, double outer_sphericity,
                           bool use_equiangular_map,
                           double x_coord_of_shell_center = 0.0,
                           bool use_half_wedges = false,
                           double aspect_ratio = 1.0,
                           bool use_logarithmic_map = false,
                           ShellWedges which_wedges = ShellWedges::All,
                           size_t number_of_layers = 1) noexcept
    -> std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Logical, TargetFrame, 3>>>;

/// \ingroup ComputationalDomainGroup
/// These are the ten Frustums used in the DomainCreators for binary compact
/// objects. The Frustums partition the volume defined by two bounding
/// surfaces: The inner surface is the surface of the two joined inner cubes
/// enveloping the two compact objects, while the outer is the surface of the
/// outer cube. The cubes enveloping the two Shells each have a side length of
/// `length_inner_cube`. The outer cube has a side length of
/// `length_outer_cube`. `origin_preimage` is a parameter
/// that moves the center of the two joined inner cubes away from the origin
/// and to `-origin_preimage`. `projective_scale_factor` acts to change the
/// gridpoint distribution in the radial direction. \see Frustum for details.
template <typename TargetFrame>
auto frustum_coordinate_maps(
    double length_inner_cube, double length_outer_cube,
    bool use_equiangular_map,
    const std::array<double, 3>& origin_preimage = {{0.0, 0.0, 0.0}},
    double projective_scale_factor = 1.0) noexcept
    -> std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Logical, TargetFrame, 3>>>;

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
    const std::array<size_t, 8>& central_block_corners = {{1, 2, 3, 4, 5, 6, 7,
                                                           8}},
    ShellWedges which_wedges = ShellWedges::All) noexcept;

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
/// Returns the correct ordering of global corner numbers for a rotated block
/// in an otherwise aligned edifice of blocks, given the OrientationMap a
/// block aligned with the edifice has relative to this one, and given the
/// corner numbering the rotated block would have if it were aligned.
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
auto maps_for_rectilinear_domains(
    const Index<VolumeDim>& domain_extents,
    const std::array<std::vector<double>, VolumeDim>& block_demarcations,
    const std::vector<Index<VolumeDim>>& block_indices_to_exclude = {},
    const std::vector<OrientationMap<VolumeDim>>& orientations_of_all_blocks =
        {},
    bool use_equiangular_map = false) noexcept
    -> std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Logical, TargetFrame, VolumeDim>>>;

/// \ingroup ComputationalDomainGroup
/// \brief Create a rectilinear Domain of multicubes.
///
/// \details Useful for constructing domains for testing non-trivially
/// connected rectilinear domains made up of cubes. We refer to a domain of
/// this type as an edifice. The `domain_extents` provides the size (in the
/// number of blocks) of the initial aligned edifice to construct. The
/// `block_indices_to_exclude` parameter is used in refining the shape of
/// the edifice from a cube to sometime more non-trivial, such as an L-shape
/// or the net of a tesseract. The `block_demarcations` and
/// `use_equiangular_map` parameters determine the CoordinateMaps to be used.
/// `orientations_of_all_blocks` contains the OrientationMap of the edifice
/// relative to each block.
///
/// The `identifications` parameter is used when identifying the faces of
/// blocks in an edifice. This is used to identify the 1D boundaries in the 2D
/// net for a 3D cube to construct a domain with topology S2. Note: If the user
/// wishes to rotate the blocks as well as manually identify their faces, the
/// user must provide the PairOfFaces corresponding to the rotated corners.
template <size_t VolumeDim, typename TargetFrame>
Domain<VolumeDim, TargetFrame> rectilinear_domain(
    const Index<VolumeDim>& domain_extents,
    const std::array<std::vector<double>, VolumeDim>& block_demarcations,
    const std::vector<Index<VolumeDim>>& block_indices_to_exclude = {},
    const std::vector<OrientationMap<VolumeDim>>& orientations_of_all_blocks =
        {},
    const std::array<bool, VolumeDim>& dimension_is_periodic =
        make_array<VolumeDim>(false),
    const std::vector<PairOfFaces>& identifications = {},
    bool use_equiangular_map = false) noexcept;

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

/// \ingroup ComputationalDomainGroup
/// Iterates over the 2^(VolumeDim-1) logical corners of the face of a
/// VolumeDim-dimensional cube in the given direction.
template <size_t VolumeDim>
class FaceCornerIterator {
 public:
  explicit FaceCornerIterator(Direction<VolumeDim> direction) noexcept;

  void operator++() noexcept {
    face_index_++;
    do {
      index_++;
    } while (get_nth_bit(index_, direction_.dimension()) ==
             (direction_.side() == Side::Upper ? 0 : 1));
    for (size_t i = 0; i < VolumeDim; ++i) {
      corner_[i] = 2 * static_cast<int>(get_nth_bit(index_, i)) - 1;
    }
  }

  explicit operator bool() const noexcept {
    return face_index_ < two_to_the(VolumeDim - 1);
  }

  tnsr::I<double, VolumeDim, Frame::Logical> operator()() const noexcept {
    return corner_;
  }

  tnsr::I<double, VolumeDim, Frame::Logical> operator*() const noexcept {
    return corner_;
  }

  // Returns the value used to construct the logical corner.
  size_t volume_index() const noexcept { return index_; }

  // Returns the number of times operator++ has been called.
  size_t face_index() const noexcept { return face_index_; }

 private:
  const Direction<VolumeDim> direction_;
  size_t index_;
  size_t face_index_ = 0;
  tnsr::I<double, VolumeDim, Frame::Logical> corner_;
};

template <size_t VolumeDim>
FaceCornerIterator<VolumeDim>::FaceCornerIterator(
    Direction<VolumeDim> direction) noexcept
    : direction_(std::move(direction)),
      index_(direction.side() == Side::Upper
                 ? two_to_the(direction_.dimension())
                 : 0) {
  for (size_t i = 0; i < VolumeDim; ++i) {
    corner_[i] = 2 * static_cast<int>(get_nth_bit(index_, i)) - 1;
  }
}

std::ostream& operator<<(std::ostream& os,
                         const ShellWedges& which_wedges) noexcept;

template <>
struct create_from_yaml<ShellWedges> {
  template <typename Metavariables>
  static ShellWedges create(const Option& options) {
    return create<void>(options);
  }
};
template <>
ShellWedges create_from_yaml<ShellWedges>::create<void>(const Option& options);
