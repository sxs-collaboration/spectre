// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"

namespace elliptic::dg::subdomain_operator::Tags {

/// The number of points an element-centered subdomain extends into the
/// neighbor, i.e. the "extruding" overlap extents. This tag is used in
/// conjunction with `LinearSolver::Schwarz::Tags::Overlaps` to map describe the
/// extruding extent into each neighbor.
struct ExtrudingExtent : db::SimpleTag {
  using type = size_t;
};

/// Data on the neighbor's side of a mortar.
template <typename Tag, size_t VolumeDim>
struct NeighborMortars : db::PrefixTag, db::SimpleTag {
  using tag = Tag;
  using Key = std::pair<::Direction<VolumeDim>, ::ElementId<VolumeDim>>;
  using type = std::unordered_map<Key, typename Tag::type, boost::hash<Key>>;
};

}  // namespace elliptic::dg::subdomain_operator::Tags
