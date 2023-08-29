// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <iosfwd>

#include "Domain/Structure/Direction.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/SegmentId.hpp"  // IWYU pragma: keep
#include "Domain/Structure/Side.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

namespace PUP {
class er;
}  // namespace PUP

/*!
 * \ingroup ComputationalDomainGroup
 * \brief A mapping of the logical coordinate axes of a host to the logical
 * coordinate axes of a neighbor of the host.
 *
 * Given a `size_t dimension`, a `Direction`, a `SegmentId`, or a `Mesh` of the
 * host, an `OrientationMap` will give the corresponding value in the neighbor.
 *
 * \tparam VolumeDim the dimension of the blocks.
 *
 * See the [tutorial](@ref tutorial_orientations) for information on how
 * OrientationMaps are used and constructed.
 */
template <size_t VolumeDim>
class OrientationMap {
 public:
  /// The default orientation is the identity map on directions.
  /// The bool `is_aligned_` is correspondingly set to `true`.
  OrientationMap();
  /// Mapped directions relative to the positive (`Side::Upper`) direction in
  /// each logical direction.
  explicit OrientationMap(
      std::array<Direction<VolumeDim>, VolumeDim> mapped_directions);
  OrientationMap(
      const std::array<Direction<VolumeDim>, VolumeDim>& directions_in_host,
      const std::array<Direction<VolumeDim>, VolumeDim>&
          directions_in_neighbor);
  ~OrientationMap() = default;
  OrientationMap(const OrientationMap&) = default;
  OrientationMap& operator=(const OrientationMap&) = default;
  OrientationMap(OrientationMap&& /*rhs*/) = default;
  OrientationMap& operator=(OrientationMap&& /*rhs*/) = default;

  /// True when mapped(Direction) == Direction
  bool is_aligned() const { return is_aligned_; }

  /// The corresponding dimension in the neighbor.
  size_t operator()(const size_t dim) const {
    return gsl::at(mapped_directions_, dim).dimension();
  }

  /// The corresponding direction in the neighbor.
  Direction<VolumeDim> operator()(const Direction<VolumeDim>& direction) const {
    return direction.side() == Side::Upper
               ? gsl::at(mapped_directions_, direction.dimension())
               : gsl::at(mapped_directions_, direction.dimension()).opposite();
  }

  /// The corresponding SegmentIds in the neighbor.
  std::array<SegmentId, VolumeDim> operator()(
      const std::array<SegmentId, VolumeDim>& segmentIds) const;

  /// The corresponding Mesh in the neighbor
  Mesh<VolumeDim> operator()(const Mesh<VolumeDim>& mesh) const;

  /// An array whose elements are permuted such that
  /// `result[this->operator()(d)] = array_to_permute[d]`.
  ///
  /// \note the permutation depends only on how the dimension is mapped
  /// and ignores the side of the mapped direction.
  template <typename T>
  std::array<T, VolumeDim> permute_to_neighbor(
      const std::array<T, VolumeDim>& array_to_permute) const;

  /// An array whose elements are permuted such that
  /// `result[d] = array_in_neighbor[this->operator()(d)]`
  ///
  /// \note the permutation depends only on how the dimension is mapped
  /// and ignores the side of the mapped direction.
  template <typename T>
  std::array<T, VolumeDim> permute_from_neighbor(
      const std::array<T, VolumeDim>& array_in_neighbor) const;

  /// The corresponding Orientation of the host in the frame of the neighbor.
  OrientationMap<VolumeDim> inverse_map() const;

  /// Serialization for Charm++
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  friend bool operator==(const OrientationMap& lhs, const OrientationMap& rhs) {
    return (lhs.mapped_directions_ == rhs.mapped_directions_ and
            lhs.is_aligned_ == rhs.is_aligned_);
  }

  std::array<Direction<VolumeDim>, VolumeDim> mapped_directions_;
  bool is_aligned_ = true;
};

/// Output operator for OrientationMap.
template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const OrientationMap<VolumeDim>& orientation);

template <size_t VolumeDim>
bool operator!=(const OrientationMap<VolumeDim>& lhs,
                const OrientationMap<VolumeDim>& rhs) {
  return not(lhs == rhs);
}

template <size_t VolumeDim>
template <typename T>
std::array<T, VolumeDim> OrientationMap<VolumeDim>::permute_to_neighbor(
    const std::array<T, VolumeDim>& array_to_permute) const {
  std::array<T, VolumeDim> array_in_neighbor = array_to_permute;
  if (is_aligned_ or VolumeDim <= 1) {
    return array_in_neighbor;
  }
  for (size_t i = 0; i < VolumeDim; i++) {
    gsl::at(array_in_neighbor, this->operator()(i)) =
        gsl::at(array_to_permute, i);
  }
  return array_in_neighbor;
}

template <size_t VolumeDim>
template <typename T>
std::array<T, VolumeDim> OrientationMap<VolumeDim>::permute_from_neighbor(
    const std::array<T, VolumeDim>& array_in_neighbor) const {
  std::array<T, VolumeDim> result = array_in_neighbor;
  if (not is_aligned_ and VolumeDim > 1) {
    for (size_t i = 0; i < VolumeDim; i++) {
      gsl::at(result, i) = gsl::at(array_in_neighbor, this->operator()(i));
    }
  }
  return result;
}

/// \ingroup ComputationalDomainGroup
/// `OrientationMap`s define an active rotation of the logical axes that bring
/// the axes of a host block into alignment with the logical axes of the
/// neighbor block. `discrete_rotation` applies this active rotation on the
/// coordinates as opposed to the axes.
/// For a two-dimensional example, consider a host block and a neighbor block,
/// where the OrientationMap between them is \f$\{-\eta,+\xi\}\f$. A quarter-
/// turn counterclockwise of the host block's logical axes would bring them into
/// alignment with those of the neighbor. That is, after this active rotation,
/// the blocks would be Aligned. Now consider a point A with coordinates
/// (+1.0,-0.5). An active quarter-turn rotation counter-clockwise about the
/// origin, keeping the axes fixed, brings point A into the coordinates
/// (+0.5,+1.0). This is how `discrete_rotation` interprets the
/// `OrientationMap` passed to it.
template <size_t VolumeDim, typename T>
std::array<tt::remove_cvref_wrap_t<T>, VolumeDim> discrete_rotation(
    const OrientationMap<VolumeDim>& rotation,
    std::array<T, VolumeDim> source_coords);

/*!
 * \ingroup ComputationalDomainGroup
 * \brief Computes the Jacobian of the transformation that is computed by
 * `discrete_rotation()`
 *
 * \note This always returns a `double` because the Jacobian is spatially
 * constant.
 */
template <size_t VolumeDim>
tnsr::Ij<double, VolumeDim, Frame::NoFrame> discrete_rotation_jacobian(
    const OrientationMap<VolumeDim>& orientation);

/*!
 * \ingroup ComputationalDomainGroup
 * \brief Computes the inverse Jacobian of the transformation that is computed
 * by `discrete_rotation()`
 */
template <size_t VolumeDim>
tnsr::Ij<double, VolumeDim, Frame::NoFrame> discrete_rotation_inverse_jacobian(
    const OrientationMap<VolumeDim>& orientation);
