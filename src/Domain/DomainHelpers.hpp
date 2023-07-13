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
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Side.hpp"
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
template <size_t VolumeDim>
class Domain;
template <size_t VolumeDim>
class OrientationMap;
namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
namespace domain::CoordinateMaps {
template <typename Map1, typename Map2>
class ProductOf2Maps;
template <typename Map1, typename Map2, typename Map3>
class ProductOf3Maps;
class Interval;
template <size_t Dim>
class Wedge;
class Frustum;
}  // namespace domain::CoordinateMaps
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
/// provided by the user to deduce the correct neighbors and
/// orientations. Does not set up periodic boundary conditions.
template <size_t VolumeDim>
void set_internal_boundaries(
    gsl::not_null<
        std::vector<DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>>*>
        neighbors_of_all_blocks,
    const std::vector<std::array<size_t, two_to_the(VolumeDim)>>&
        corners_of_all_blocks);

/// \ingroup ComputationalDomainGroup
/// Sets up the BlockNeighbors using the corner numbering scheme
/// implied by the maps provided by the user to deduce the correct
/// neighbors and orientations.
/// \warning Does not set up periodic boundary conditions.
template <size_t VolumeDim>
void set_internal_boundaries(
    gsl::not_null<
        std::vector<DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>>*>
        neighbors_of_all_blocks,
    const std::vector<std::unique_ptr<domain::CoordinateMapBase<
        Frame::BlockLogical, Frame::Inertial, VolumeDim>>>& maps);

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
        neighbors_of_all_blocks);

/// \ingroup ComputationalDomainGroup
/// \brief The multi-indices that identify the individual Blocks in the lattice
template <size_t VolumeDim>
auto indices_for_rectilinear_domains(
    const Index<VolumeDim>& domain_extents,
    const std::vector<Index<VolumeDim>>& block_indices_to_exclude = {})
    -> std::vector<Index<VolumeDim>>;

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
    const std::vector<Index<VolumeDim>>& block_indices_to_exclude = {})
    -> std::vector<std::array<size_t, two_to_the(VolumeDim)>>;

/// \ingroup ComputationalDomainGroup
/// The number of wedges to include in the Sphere domain.
enum class ShellWedges {
  /// Use the entire shell
  All,
  /// Use only the four equatorial wedges
  FourOnEquator,
  /// Use only the single wedge along -x
  OneAlongMinusX
};

/// \ingroup ComputationalDomainGroup
/// The first index in the list "UpperZ", "LowerZ", "UpperY", "LowerY", "UpperX"
/// "LowerX" that is included in `which_wedges`. It is 0 for `ShellWedges::All`,
/// 2 for `ShellWedges::FourOnEquator`, and 5 for `ShellWedges::OneAlongMinusX`.
size_t which_wedge_index(const ShellWedges& which_wedges);

/*!
 * \ingroup ComputationalDomainGroup
 * These are the CoordinateMaps of the Wedge<3>s used in the Sphere and
 * binary compact object DomainCreators. This function can also be used to
 * wrap the Sphere in a cube made of six Wedge<3>s.
 *
 * \param inner_radius Radius of the inner boundary of the shell, or the
 * radius circumscribing the inner cube of a sphere.
 * \param outer_radius Outer radius of the shell or sphere.
 * \param inner_sphericity Specifies if the wedges form a spherical inner
 * boundary (1.0) or a cubical inner boundary (0.0).
 * \param outer_sphericity Specifies if the wedges form a spherical outer
 * boundary (1.0) or a cubical outer boundary (0.0).
 * \param use_equiangular_map Toggles the equiangular map of the Wedge map.
 * \param use_half_wedges When `true`, the wedges in the +z,-z,+y,-y directions
 * are cut in half along their xi-axes. The resulting ten CoordinateMaps are
 * used for the outermost Blocks of the BBH Domain.
 * \param radial_partitioning Specifies the radial boundaries of sub-shells
 * between `inner_radius` and `outer_radius`. If the inner and outer
 * sphericities are different, the innermost shell does the transition.
 * \param radial_distribution Select the radial distribution of grid points in
 * the spherical shells.
 * \param which_wedges Select a subset of wedges.
 * \param opening_angle sets the combined opening angle of the two half wedges
 * that open up along the y-z plane. The endcap wedges are then given an angle
 * of pi minus this opening angle. This parameter only has an effect if
 * `use_half_wedges` is set to `true`.
 */
std::vector<domain::CoordinateMaps::Wedge<3>> sph_wedge_coordinate_maps(
    double inner_radius, double outer_radius, double inner_sphericity,
    double outer_sphericity, bool use_equiangular_map,
    bool use_half_wedges = false,
    const std::vector<double>& radial_partitioning = {},
    const std::vector<domain::CoordinateMaps::Distribution>&
        radial_distribution = {domain::CoordinateMaps::Distribution::Linear},
    ShellWedges which_wedges = ShellWedges::All, double opening_angle = M_PI_2);

/// \ingroup ComputationalDomainGroup
/// These are the ten Frustums used in the DomainCreators for binary compact
/// objects. The Frustums partition the volume defined by two bounding
/// surfaces: The inner surface is the surface of the two joined inner cubes
/// enveloping the two compact objects, while the outer is the surface of the
/// outer cube.
/// \param length_inner_cube The side length of the cubes enveloping the two
/// shells.
/// \param length_outer_cube The side length of the outer cube.
/// \param use_equiangular_map Whether to apply a tangent map in the angular
/// directions.
/// \param origin_preimage The center of the two joined inner cubes is moved
/// away from the origin and to this point, origin_preimage.
/// \param radial_distribution The gridpoint distribution in the radial
/// direction, possibly dependent on the value passed to `distribution_value`.
/// \param distribution_value Used by `radial_distribution`. \see Frustum for
/// details.
/// \param sphericity Determines whether the outer surface is a cube
/// (value of 0), a sphere (value of 1) or somewhere in between.
/// \param opening_angle determines the gridpoint distribution used
/// in the Frustums such that they conform to the outer sphere of Wedges with
/// the same value for `opening_angle`.
std::vector<domain::CoordinateMaps::Frustum> frustum_coordinate_maps(
    double length_inner_cube, double length_outer_cube,
    bool use_equiangular_map,
    const std::array<double, 3>& origin_preimage = {{0.0, 0.0, 0.0}},
    domain::CoordinateMaps::Distribution radial_distribution =
        domain::CoordinateMaps::Distribution::Linear,
    std::optional<double> distribution_value = std::nullopt,
    double sphericity = 0.0, double opening_angle = M_PI_2);

/// \ingroup ComputationalDomainGroup
/// \brief The corners for a domain with radial layers.
///
/// Generates the corners for a Domain which is made of one or more layers
/// of Blocks fully enveloping an interior volume, e.g. Sphere.
///
/// \param number_of_layers specifies how many layers of Blocks to have
/// in the final domain.
/// \param include_central_block set to `true` where the interior
/// volume is filled with a central Block, and `false` where the
/// interior volume is left empty.
/// \param central_block_corners are used as seed values to generate the corners
/// for the surrounding Blocks.
/// \param which_wedges can be used to exclude a subset of the wedges.
std::vector<std::array<size_t, 8>> corners_for_radially_layered_domains(
    size_t number_of_layers, bool include_central_block,
    const std::array<size_t, 8>& central_block_corners = {{1, 2, 3, 4, 5, 6, 7,
                                                           8}},
    ShellWedges which_wedges = ShellWedges::All);

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
        {1, 2, 3, 4, 5, 6, 7, 8}});

/// \ingroup ComputationalDomainGroup
/// These are the CoordinateMaps used in the Cylinder DomainCreator.
///
/// The `radial_partitioning` specifies the radial boundaries of sub-shells
/// between `inner_radius` and `outer_radius`, while `partitioning_in_z`
/// specifies the z-boundaries, splitting the cylinder into stacked
/// 3-dimensional disks. The circularity of the shell wedges changes from 0 to 1
/// within the innermost sub-shell.
///
/// Set the `radial_distribution` to select the radial distribution of grid
/// points in the cylindrical shells. The innermost shell must have
/// `domain::CoordinateMaps::Distribution::Linear` because it changes the
/// circularity. The distribution along the z-axis for each circular
/// disc is specified through `distribution_in_z`.
template <typename TargetFrame>
auto cyl_wedge_coordinate_maps(
    double inner_radius, double outer_radius, double lower_z_bound,
    double upper_z_bound, bool use_equiangular_map,
    const std::vector<double>& radial_partitioning = {},
    const std::vector<double>& partitioning_in_z = {},
    const std::vector<domain::CoordinateMaps::Distribution>&
        radial_distribution = {domain::CoordinateMaps::Distribution::Linear},
    const std::vector<domain::CoordinateMaps::Distribution>& distribution_in_z =
        {domain::CoordinateMaps::Distribution::Linear})
    -> std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::BlockLogical, TargetFrame, 3>>>;

enum class CylindricalDomainParityFlip { none, z_direction };

/// \ingroup ComputationalDomainGroup
/// Same as `cyl_wedge_coordinate_maps`, but only the center square blocks,
///
/// If `CylindricalDomainParityFlip::z_direction` is specified, then
/// the returned maps describe a cylinder with `lower_z_bound`
/// corresponding to logical coordinate `upper_zeta` and `upper_z_bound`
/// corresponding to logical coordinate `lower_zeta`, and thus the
/// resulting maps are left-handed.
/// `CylindricalDomainParityFlip::z_direction` is therefore useful
/// only when composing with another map that is also left-handed, so
/// that the composed coordinate system is right-handed.
///
/// Returned as a vector of the coordinate maps so that they can
/// be composed with other maps later.
auto cyl_wedge_coord_map_center_blocks(
    double inner_radius, double lower_z_bound, double upper_z_bound,
    bool use_equiangular_map, const std::vector<double>& partitioning_in_z = {},
    const std::vector<domain::CoordinateMaps::Distribution>& distribution_in_z =
        {domain::CoordinateMaps::Distribution::Linear},
    CylindricalDomainParityFlip parity_flip = CylindricalDomainParityFlip::none)
    -> std::vector<domain::CoordinateMaps::ProductOf3Maps<
        domain::CoordinateMaps::Interval, domain::CoordinateMaps::Interval,
        domain::CoordinateMaps::Interval>>;

/// \ingroup ComputationalDomainGroup
/// Same as cyl_wedge_coordinate_maps, but only the surrounding wedge blocks.
///
/// If `CylindricalDomainParityFlip::z_direction` is specified, then
/// the returned maps describe a cylinder with `lower_z_bound`
/// corresponding to logical coordinate `upper_zeta` and `upper_z_bound`
/// corresponding to logical coordinate `lower_zeta`, and thus the
/// resulting maps are left-handed.
/// `CylindricalDomainParityFlip::z_direction` is therefore useful
/// only when composing with another map that is also left-handed, so
/// that the composed coordinate system is right-handed.
///
/// Returned as a vector of the coordinate maps so that they can
/// be composed with other maps later.
auto cyl_wedge_coord_map_surrounding_blocks(
    double inner_radius, double outer_radius, double lower_z_bound,
    double upper_z_bound, bool use_equiangular_map, double inner_circularity,
    const std::vector<double>& radial_partitioning = {},
    const std::vector<double>& partitioning_in_z = {},
    const std::vector<domain::CoordinateMaps::Distribution>&
        radial_distribution = {domain::CoordinateMaps::Distribution::Linear},
    const std::vector<domain::CoordinateMaps::Distribution>& distribution_in_z =
        {domain::CoordinateMaps::Distribution::Linear},
    CylindricalDomainParityFlip parity_flip = CylindricalDomainParityFlip::none)
    -> std::vector<domain::CoordinateMaps::ProductOf2Maps<
        domain::CoordinateMaps::Wedge<2>, domain::CoordinateMaps::Interval>>;

/// \ingroup ComputationalDomainGroup
/// \brief The corners for a cylindrical domain split into discs with radial
/// shells.
///
/// Generates the corners for a Domain which is made of one or more stacked
/// discs consisting of layers of Blocks enveloping an interior square prism.
/// The `number_of_shells` specifies how many of these layers of Blocks to have
/// in each disc.
///
/// The `number_of_discs` specifies how many discs make up the domain.
/// The very basic cylinder with one shell and one layer serves as a base
/// to generate the corners for subsequent shells first and discs second.
std::vector<std::array<size_t, 8>> corners_for_cylindrical_layered_domains(
    size_t number_of_shells, size_t number_of_discs);

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
    const std::array<size_t, two_to_the(VolumeDim)>& corners_of_aligned);

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
    bool use_equiangular_map = false)
    -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
        Frame::BlockLogical, TargetFrame, VolumeDim>>>;

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
template <size_t VolumeDim>
Domain<VolumeDim> rectilinear_domain(
    const Index<VolumeDim>& domain_extents,
    const std::array<std::vector<double>, VolumeDim>& block_demarcations,
    const std::vector<Index<VolumeDim>>& block_indices_to_exclude = {},
    const std::vector<OrientationMap<VolumeDim>>& orientations_of_all_blocks =
        {},
    const std::array<bool, VolumeDim>& dimension_is_periodic =
        make_array<VolumeDim>(false),
    const std::vector<PairOfFaces>& identifications = {},
    bool use_equiangular_map = false);

/// \ingroup ComputationalDomainGroup
/// Iterates over the corners of a VolumeDim-dimensional cube.
template <size_t VolumeDim>
class VolumeCornerIterator {
 public:
  VolumeCornerIterator() { setup_from_local_corner_number(); }

  explicit VolumeCornerIterator(size_t initial_local_corner_number)
      : local_corner_number_(initial_local_corner_number) {
    setup_from_local_corner_number();
  }
  VolumeCornerIterator(
      // The block index is also global corner
      // index of the lowest corner of the block.
      Index<VolumeDim> block_index, Index<VolumeDim> global_corner_extents)
      : global_corner_number_(
            collapsed_index(block_index, global_corner_extents)),
        global_corner_index_(block_index),
        global_corner_extents_(global_corner_extents) {}

  void operator++() {
    ++local_corner_number_;
    setup_from_local_corner_number();
  }

  explicit operator bool() const {
    return local_corner_number_ < two_to_the(VolumeDim);
  }

  size_t local_corner_number() const { return local_corner_number_; }

  size_t global_corner_number() const {
    std::array<size_t, VolumeDim> new_indices{};
    for (size_t i = 0; i < VolumeDim; i++) {
      gsl::at(new_indices, i) =
          global_corner_index_[i] +
          (gsl::at(array_sides_, i) == Side::Upper ? 1 : 0);
    }
    const Index<VolumeDim> interior_multi_index(new_indices);
    return collapsed_index(interior_multi_index, global_corner_extents_);
  }

  const std::array<Side, VolumeDim>& operator()() const { return array_sides_; }

  const std::array<Side, VolumeDim>& operator*() const { return array_sides_; }

  const std::array<double, VolumeDim>& coords_of_corner() const {
    return coords_of_corner_;
  }

  const std::array<Direction<VolumeDim>, VolumeDim>& directions_of_corner()
      const {
    return array_directions_;
  }

  void setup_from_local_corner_number() {
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
  explicit FaceCornerIterator(Direction<VolumeDim> direction);

  void operator++() {
    face_index_++;
    do {
      index_++;
    } while (get_nth_bit(index_, direction_.dimension()) ==
             (direction_.side() == Side::Upper ? 0 : 1));
    for (size_t i = 0; i < VolumeDim; ++i) {
      corner_[i] = 2 * static_cast<int>(get_nth_bit(index_, i)) - 1;
    }
  }

  explicit operator bool() const {
    return face_index_ < two_to_the(VolumeDim - 1);
  }

  tnsr::I<double, VolumeDim, Frame::BlockLogical> operator()() const {
    return corner_;
  }

  tnsr::I<double, VolumeDim, Frame::BlockLogical> operator*() const {
    return corner_;
  }

  // Returns the value used to construct the logical corner.
  size_t volume_index() const { return index_; }

  // Returns the number of times operator++ has been called.
  size_t face_index() const { return face_index_; }

 private:
  const Direction<VolumeDim> direction_;
  size_t index_;
  size_t face_index_ = 0;
  tnsr::I<double, VolumeDim, Frame::BlockLogical> corner_;
};

template <size_t VolumeDim>
FaceCornerIterator<VolumeDim>::FaceCornerIterator(
    Direction<VolumeDim> direction)
    : direction_(std::move(direction)),
      index_(direction_.side() == Side::Upper
                 ? two_to_the(direction_.dimension())
                 : 0) {
  for (size_t i = 0; i < VolumeDim; ++i) {
    corner_[i] = 2 * static_cast<int>(get_nth_bit(index_, i)) - 1;
  }
}

std::ostream& operator<<(std::ostream& os, const ShellWedges& which_wedges);

template <>
struct Options::create_from_yaml<ShellWedges> {
  template <typename Metavariables>
  static ShellWedges create(const Options::Option& options) {
    return create<void>(options);
  }
};
template <>
ShellWedges Options::create_from_yaml<ShellWedges>::create<void>(
    const Options::Option& options);
