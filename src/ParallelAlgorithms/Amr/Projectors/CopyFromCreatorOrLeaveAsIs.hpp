// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iterator>
#include <unordered_map>
#include <utility>

#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
template <size_t Dim>
class Element;
template <size_t Dim>
class ElementId;
template <size_t Dim>
class Mesh;
/// \endcond

namespace amr::projectors {

/// \brief For h-refinement copy the items from the parent/child, while for
/// p-refinement leave the items unchanged
///
/// There is a specialization for
/// `CopyFromCreatorOrLeaveAsIs<tmpl::list<Tags...>>` that can be used if a
/// `tmpl::list` is available.
///
/// \details For each item corresponding to each tag:
/// - When changing resolution (p-refinement), leave unchanged
/// - When splitting, copy the value from the parent to the child
/// - When joining, check that the children have the same value, and copy to
///   the parent
template <typename... Tags>
struct CopyFromCreatorOrLeaveAsIs : tt::ConformsTo<amr::protocols::Projector> {
  using return_tags = tmpl::list<Tags...>;
  using argument_tags = tmpl::list<>;

  template <size_t Dim>
  static void apply(
      const gsl::not_null<typename Tags::type*>... /*items*/,
      const std::pair<Mesh<Dim>, Element<Dim>>& /*old_mesh_and_element*/) {
    // do nothing, i.e. leave the items unchanged
  }

  template <typename... ParentTags>
  static void apply(const gsl::not_null<typename Tags::type*>... items,
                    const tuples::TaggedTuple<ParentTags...>& parent_items) {
    ::expand_pack((*items = tuples::get<Tags>(parent_items))...);
  }

  template <size_t Dim, typename... ChildrenTags>
  static void apply(const gsl::not_null<typename Tags::type*>... items,
                    const std::unordered_map<
                        ElementId<Dim>, tuples::TaggedTuple<ChildrenTags...>>&
                        children_items) {
    const auto& first_child_items = children_items.begin();
#ifdef SPECTRE_DEBUG
    for (auto it = std::next(first_child_items); it != children_items.end();
         ++it) {
      const bool children_agree =
          ((tuples::get<Tags>(first_child_items->second) ==
            tuples::get<Tags>(it->second)) and
           ...);
      ASSERT(children_agree, "Children do not agree on all items!");
    }
#endif  // #ifdef SPECTRE_DEBUG
    ::expand_pack((*items = tuples::get<Tags>(first_child_items->second))...);
  }
};

/// \cond
template <typename... Tags>
struct CopyFromCreatorOrLeaveAsIs<tmpl::list<Tags...>>
    : public CopyFromCreatorOrLeaveAsIs<Tags...> {};
/// \endcond
}  // namespace amr::projectors
