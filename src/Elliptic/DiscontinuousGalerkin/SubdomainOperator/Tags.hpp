// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <unordered_map>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Structure/DirectionalId.hpp"

namespace elliptic::dg::subdomain_operator::Tags {

/// The number of points an element-centered subdomain extends into the
/// neighbor, i.e. the "extruding" overlap extents. This tag is used in
/// conjunction with `LinearSolver::Schwarz::Tags::Overlaps` to describe the
/// extruding extent into each neighbor.
struct ExtrudingExtent : db::SimpleTag {
  using type = size_t;
};

/// Data on the neighbor's side of a mortar. Used to store data for elements
/// that do not overlap with the element-centered subdomain, but play a role
/// in the DG operator nonetheless.
template <typename Tag, size_t VolumeDim>
struct NeighborMortars : db::PrefixTag, db::SimpleTag {
  using tag = Tag;
  using Key = DirectionalId<VolumeDim>;
  using type = std::unordered_map<Key, typename Tag::type, boost::hash<Key>>;
};

}  // namespace elliptic::dg::subdomain_operator::Tags
