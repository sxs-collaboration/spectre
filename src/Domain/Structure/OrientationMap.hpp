// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <iosfwd>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
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
  static constexpr uint16_t aligned_mask = 0b1000000000000000;
  static constexpr uint16_t version_mask = 0b0111000000000000;

  /// \brief Creates an OrientationMap in an uninitialized state.
  ///
  /// This can be helpful for debugging code. If you would like the identity
  /// map, please use `create_aligned()`.
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

  /// Creates an OrientationMap that is the identity map on directions.
  /// `is_aligned()` is `true` in this case.
  static OrientationMap<VolumeDim> create_aligned();

  /// True when mapped(Direction) == Direction
  bool is_aligned() const {
    ASSERT(bit_field_ != static_cast<uint16_t>(0b1 << 15),
           "Cannot use a default-constructed OrientationMap");
    return (bit_field_ bitand aligned_mask) == aligned_mask;
  }

  /// The corresponding dimension in the neighbor.
  size_t operator()(const size_t dim) const {
    ASSERT(bit_field_ != static_cast<uint16_t>(0b1 << 15),
           "Cannot use a default-constructed OrientationMap");
    return get_direction(dim).dimension();
  }

  /// The corresponding direction in the neighbor.
  Direction<VolumeDim> operator()(const Direction<VolumeDim>& direction) const {
    ASSERT(bit_field_ != static_cast<uint16_t>(0b1 << 15),
           "Cannot use a default-constructed OrientationMap");
    return direction.side() == Side::Upper
               ? get_direction(direction.dimension())
               : get_direction(direction.dimension()).opposite();
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
    return lhs.bit_field_ == rhs.bit_field_;
  }

  Direction<VolumeDim> get_direction(size_t dim) const;
  void set_direction(size_t dim, const Direction<VolumeDim>& direction);
  void set_aligned(bool is_aligned);
  std::set<size_t> set_of_dimensions() const;

  uint16_t bit_field_{0b1 << 15};
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
  if (is_aligned() or VolumeDim <= 1) {
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
  if (not is_aligned() and VolumeDim > 1) {
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
