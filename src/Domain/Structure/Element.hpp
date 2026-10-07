// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Element.

#pragma once

#include <array>
#include <cstddef>
#include <iosfwd>
#include <unordered_set>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/FaceType.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Utilities/MakeArray.hpp"

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
  /// \param topologies domain::Topology in each dimension (default value is
  /// domain::Topology::I1)
  Element(ElementId<VolumeDim> id, Neighbors_t neighbors,
          std::array<domain::Topology, VolumeDim> topologies =
              make_array<VolumeDim>(domain::Topology::I1));

  /// Default needed for serialization
  Element() = default;

  ~Element() = default;
  Element(const Element<VolumeDim>& /*rhs*/) = default;
  Element(Element<VolumeDim>&& /*rhs*/) = default;
  Element<VolumeDim>& operator=(const Element<VolumeDim>& /*rhs*/) = default;
  Element<VolumeDim>& operator=(Element<VolumeDim>&& /*rhs*/) = default;

  /// The directions of the faces of the Element that are external boundaries.
  const std::unordered_set<Direction<VolumeDim>>& external_boundaries() const {
    return external_boundaries_;
  }

  /// The directions of the faces of the Element that are internal boundaries.
  const std::unordered_set<Direction<VolumeDim>>& internal_boundaries() const {
    return internal_boundaries_;
  }

  /// The directions of the faces of the Element that are boundaries, i.e. that
  /// are either internal boundaries (shared with a neighbor) or external
  /// boundaries. This is the union of `internal_boundaries()` and
  /// `external_boundaries()`. It excludes directions of topological type
  /// (`domain::FaceType::Topological`), such as the angular directions of a
  /// spherical shell, which have no neighbor and are not external boundaries
  /// either. Iterate over this instead of `Direction::all_directions()` to skip
  /// topological directions.
  const std::unordered_set<Direction<VolumeDim>>& all_boundaries() const {
    return all_boundaries_;
  }

  /// A unique ID for the Element.
  const ElementId<VolumeDim>& id() const { return id_; }

  /// Information about the neighboring Elements.
  const Neighbors_t& neighbors() const { return neighbors_; }

  /// The number of neighbors this element has
  size_t number_of_neighbors() const { return number_of_neighbors_; }

  /// The topology in each dimension of this Element
  const std::array<domain::Topology, VolumeDim>& topologies() const {
    return topologies_;
  }

  /// The FaceType in each direction
  const DirectionMap<VolumeDim, domain::FaceType>& face_types() const {
    return face_types_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  ElementId<VolumeDim> id_{};
  Neighbors_t neighbors_{};
  size_t number_of_neighbors_{};
  std::unordered_set<Direction<VolumeDim>> external_boundaries_{};
  std::unordered_set<Direction<VolumeDim>> internal_boundaries_{};
  std::unordered_set<Direction<VolumeDim>> all_boundaries_{};
  std::array<domain::Topology, VolumeDim> topologies_;
  DirectionMap<VolumeDim, domain::FaceType> face_types_{};
};

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os, const Element<VolumeDim>& element);

template <size_t VolumeDim>
bool operator==(const Element<VolumeDim>& lhs, const Element<VolumeDim>& rhs);

template <size_t VolumeDim>
bool operator!=(const Element<VolumeDim>& lhs, const Element<VolumeDim>& rhs);
