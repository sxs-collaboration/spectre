// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_map>

#include "DataStructures/DataBox/Tag.hpp"

/// \cond
template <size_t Dim>
class ElementId;
/// \endcond

namespace Parallel::Tags {
/// \brief The node (location) where different elements are.
///
/// This should be in the nodegroup's DataBox.
template <size_t Dim>
struct ElementLocations : db::SimpleTag {
  using type = std::unordered_map<ElementId<Dim>, size_t>;
};

/// \brief The node (location) where different elements are.
///
/// This should be in the DgElementArrayMember's DataBox and should point to
/// the one located in the nodegroup's DataBox.
template <size_t Dim>
struct ElementLocationsPointer : db::SimpleTag {
  using type = std::unordered_map<ElementId<Dim>, size_t>*;
};
}  // namespace Parallel::Tags
