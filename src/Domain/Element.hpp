// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Element.

#pragma once

#include <cstddef>
#include <iosfwd>
#include <unordered_set>

#include "Domain/Direction.hpp"  // IWYU pragma: keep
#include "Domain/DirectionMap.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Neighbors.hpp"  // IWYU pragma: keep

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// \ingroup ComputationalDomainGroup
/// A spectral element with knowledge of its neighbors.
///
/// \tparam VolumeDim the volume dimension.
template <size_t VolumeDim>
class Element {
 public:
  using Neighbors_t = DirectionMap<VolumeDim, Neighbors<VolumeDim>>;

  /// Constructor
  ///
  /// \param id a unique identifier for the Element.
  /// \param neighbors info about the Elements that share an interface
  /// with this Element.
  Element(ElementId<VolumeDim> id, Neighbors_t neighbors) noexcept;

  /// Default needed for serialization
  Element() = default;

  ~Element() = default;
  Element(const Element<VolumeDim>& /*rhs*/) = default;
  Element(Element<VolumeDim>&& /*rhs*/) noexcept = default;
  Element<VolumeDim>& operator=(const Element<VolumeDim>& /*rhs*/) = default;
  Element<VolumeDim>& operator=(Element<VolumeDim>&& /*rhs*/) noexcept =
      default;

  /// The directions of the faces of the Element that are external boundaries.
  const std::unordered_set<Direction<VolumeDim>>& external_boundaries() const
      noexcept {
    return external_boundaries_;
  }

  /// A unique ID for the Element.
  const ElementId<VolumeDim>& id() const noexcept { return id_; }

  /// Information about the neighboring Elements.
  const Neighbors_t& neighbors() const noexcept { return neighbors_; }

  /// The number of neighbors this element has
  size_t number_of_neighbors() const noexcept { return number_of_neighbors_; }

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  ElementId<VolumeDim> id_{};
  Neighbors_t neighbors_{};
  size_t number_of_neighbors_{};
  std::unordered_set<Direction<VolumeDim>> external_boundaries_{};
};

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const Element<VolumeDim>& element) noexcept;

template <size_t VolumeDim>
bool operator==(const Element<VolumeDim>& lhs,
                const Element<VolumeDim>& rhs) noexcept;

template <size_t VolumeDim>
bool operator!=(const Element<VolumeDim>& lhs,
                const Element<VolumeDim>& rhs) noexcept;
