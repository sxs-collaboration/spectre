// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iosfwd>
#include <type_traits>
#include <unordered_set>

#include "Domain/Structure/OrientationMap.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
template <size_t VolumeDim>
class ElementId;
/// \endcond

/// \ingroup ComputationalDomainGroup
/// Information about the neighbors of a host Element or Block in a particular
/// direction.
///
/// \tparam VolumeDim the volume dimension.
/// \tparam IdType the type of the Id of the neighbor (either ElementId or
/// size_t for a Block)
template <size_t VolumeDim, typename IdType = ElementId<VolumeDim>>
class Neighbors {
  static_assert(std::is_same_v<IdType, size_t> or
                std::is_same_v<IdType, ElementId<VolumeDim>>);

 public:
  /// Construct with the ids, the orientation of the neighbors relative to the
  /// host, and whether the neighbors are conforming.
  ///
  /// \param ids the ids of the neighbors.
  /// \param orientations An OrientationMap (which takes objects in the logical
  /// coordinate frame of the host and maps them to the logical coordinate frame
  /// of the neighbor) for each neighboring Block (Elements within a Block share
  /// the same orientation).  The key of the unordered map is the Block ID.
  /// \param are_conforming whether or not the block logical coordinates of the
  /// neighbors conform to those of the host (see
  /// domain::neighbor_is_conforming)
  Neighbors(std::unordered_set<IdType> ids,
            std::unordered_map<size_t, OrientationMap<VolumeDim>> orientations,
            bool are_conforming);

  /// Construct with the ids and orientation of the neighbors relative to the
  /// host, assuming the neighbors are conforming.
  ///
  /// \param ids the ids of the neighbors.
  /// \param orientation This OrientationMap takes objects in the logical
  /// coordinate frame of the host and maps them to the logical coordinate frame
  /// of the neighbor.
  Neighbors(std::unordered_set<IdType> ids,
            OrientationMap<VolumeDim> orientation);

  /// Construct with the id and orientation of a single neighbor relative to the
  /// host, assuming the neighbor is conforming.
  ///
  /// \param id the id of the neighbors.
  /// \param orientation This OrientationMap takes objects in the logical
  /// coordinate frame of the host Element and maps them to the logical
  /// coordinate frame of the neighbor Element.
  Neighbors(IdType id, OrientationMap<VolumeDim> orientation);

  /// Default constructor for Charm++ serialization.
  Neighbors() = default;
  ~Neighbors() = default;
  Neighbors(const Neighbors& neighbor) = default;
  Neighbors(Neighbors&&) = default;
  Neighbors& operator=(const Neighbors& rhs) = default;
  Neighbors& operator=(Neighbors&&) = default;

  const std::unordered_set<IdType>& ids() const { return ids_; }

  /// The orientations of the neighbors for each neighboring Block.
  ///
  /// \note All Elements within a Block share the same orientation.
  const std::unordered_map<size_t, OrientationMap<VolumeDim>>& orientations()
      const {
    return orientations_;
  }

  /// Whether or not the block logical coordinates of the neighbors conform to
  /// those of the host (see domain::neighbor_is_conforming)
  bool are_conforming() const { return are_conforming_; }

  /// The orientation of a particular neighbor.
  const OrientationMap<VolumeDim>& orientation(const IdType& id) const;

  /// Reset the ids of the neighbors.
  ///
  /// \note This should only be called to reset Element Neighbors after
  /// h-refinement
  void set_ids_to(std::unordered_set<IdType> new_ids);

  /// Add ids of neighbors.
  ///
  /// \note Adding an existing neighbor is allowed.
  /// \note The additional ids must be from Blocks with the existing
  /// orientations.
  void add_ids(std::unordered_set<IdType> additional_ids);

  /// Serialization for Charm++
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  /// The number of neighbors
  size_t size() const { return ids_.size(); }

  typename std::unordered_set<IdType>::iterator begin() { return ids_.begin(); }

  typename std::unordered_set<IdType>::iterator end() { return ids_.end(); }

  typename std::unordered_set<IdType>::const_iterator begin() const {
    return ids_.begin();
  }

  typename std::unordered_set<IdType>::const_iterator end() const {
    return ids_.end();
  }

  typename std::unordered_set<IdType>::const_iterator cbegin() const {
    return ids_.begin();
  }

  typename std::unordered_set<IdType>::const_iterator cend() const {
    return ids_.end();
  }

 private:
  std::unordered_set<IdType> ids_{};
  std::unordered_map<size_t, OrientationMap<VolumeDim>> orientations_{};
  bool are_conforming_{true};
};

/// Output operator for Neighbors.
template <size_t VolumeDim, typename IdType>
std::ostream& operator<<(std::ostream& os,
                         const Neighbors<VolumeDim, IdType>& neighbors);

template <size_t VolumeDim, typename IdType>
bool operator==(const Neighbors<VolumeDim, IdType>& lhs,
                const Neighbors<VolumeDim, IdType>& rhs);

template <size_t VolumeDim, typename IdType>
bool operator!=(const Neighbors<VolumeDim, IdType>& lhs,
                const Neighbors<VolumeDim, IdType>& rhs);
