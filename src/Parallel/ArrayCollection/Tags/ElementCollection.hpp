// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_map>

#include "DataStructures/DataBox/Tag.hpp"

/// \cond
template <size_t Dim>
class ElementId;
namespace Parallel {
template <size_t Dim, typename Metavariables, typename PhaseDepActionList,
          typename SimpleTagsFromOptions>
class DgElementArrayMember;
}  // namespace Parallel
/// \endcond

namespace Parallel::Tags {
/// \brief Holds all of the `DgElementArrayMember` on a node.
///
/// This tag must be in the nodegroup DataBox.
template <size_t Dim, class Metavariables, class PhaseDepActionList,
          typename SimpleTagsFromOptions>
struct ElementCollection : db::SimpleTag {
  using type = std::unordered_map<
      ElementId<Dim>,
      Parallel::DgElementArrayMember<Dim, Metavariables, PhaseDepActionList,
                                     SimpleTagsFromOptions>>;
};
}  // namespace Parallel::Tags
