// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template Neighbors.

#pragma once

#include <iosfwd>
#include <pup.h>
#include <unordered_set>

#include "Domain/ElementId.hpp"
#include "Domain/OrientationMap.hpp"

/// \ingroup ComputationalDomainGroup
/// Information about the neighbors of an Element in a particular direction.
///
/// \tparam VolumeDim the volume dimension.
template <size_t VolumeDim>
class Neighbors {
 public:
  /// Construct with the ids and orientation of the neighbors.
  ///
  /// \param ids the ids of the neighbors.
  /// \param orientation the OrientationMap of the neighbors.
  Neighbors(std::unordered_set<ElementId<VolumeDim>> ids,
            OrientationMap<VolumeDim> orientation);

  /// Default constructor for Charm++ serialization.
  Neighbors() = default;
  ~Neighbors() = default;
  Neighbors(const Neighbors<VolumeDim>& neighbor)  = default;
  Neighbors(Neighbors<VolumeDim>&&) noexcept = default;
  Neighbors& operator=(const Neighbors& rhs) = default;
  Neighbors& operator=(Neighbors&&) noexcept = default;

  const std::unordered_set<ElementId<VolumeDim>>& ids() const noexcept {
    return ids_;
  }

  const OrientationMap<VolumeDim>& orientation() const noexcept {
    return orientation_;
  }

  /// Reset the ids of the neighbors.
  void set_ids_to(
      const std::unordered_set<ElementId<VolumeDim>> new_ids) noexcept {
    ids_ = std::move(new_ids);
  }

  /// Add ids of neighbors.
  /// Adding an existing neighbor is allowed.
  void add_ids(const std::unordered_set<ElementId<VolumeDim>>& additional_ids);

  /// Serialization for Charm++
  void pup(PUP::er& p);  // NOLINT

  /// The number of neighbors
  size_t size() const noexcept { return ids_.size(); }

  typename std::unordered_set<ElementId<VolumeDim>>::iterator begin() {
    return ids_.begin();
  }

  typename std::unordered_set<ElementId<VolumeDim>>::iterator end() {
    return ids_.end();
  }

  typename std::unordered_set<ElementId<VolumeDim>>::const_iterator begin()
      const {
    return ids_.begin();
  }

  typename std::unordered_set<ElementId<VolumeDim>>::const_iterator end()
      const {
    return ids_.end();
  }

  typename std::unordered_set<ElementId<VolumeDim>>::const_iterator cbegin()
      const {
    return ids_.begin();
  }

  typename std::unordered_set<ElementId<VolumeDim>>::const_iterator cend()
      const {
    return ids_.end();
  }

 private:
  std::unordered_set<ElementId<VolumeDim>> ids_;
  OrientationMap<VolumeDim> orientation_;
};

/// Output operator for Neighborss.
template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os, const Neighbors<VolumeDim>& n);

template <size_t VolumeDim>
bool operator==(const Neighbors<VolumeDim>& lhs,
                const Neighbors<VolumeDim>& rhs) noexcept;

template <size_t VolumeDim>
bool operator!=(const Neighbors<VolumeDim>& lhs,
                const Neighbors<VolumeDim>& rhs) noexcept;
