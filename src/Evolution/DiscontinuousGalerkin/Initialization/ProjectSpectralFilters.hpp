// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iterator>
#include <memory>
#include <unordered_map>
#include <utility>

#include "DataStructures/TaggedTuple.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Tag.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::Initialization {

/// \brief AMR projector for `Filters::Tags::SpectralFilter<Dim, TagList>`.
///
/// \details
/// - For p-refinement: leaves the filter unchanged.
/// - For h-refinement (splitting): clones the parent's filter for the child.
/// - For h-coarsening (joining): clones the first child's filter for the
///   parent (all siblings share the same block and mesh basis/quadrature, so
///   they carry identical filters).
template <size_t Dim, typename TagList>
struct ProjectSpectralFilters : tt::ConformsTo<amr::protocols::Projector> {
  using return_tags = tmpl::list<Filters::Tags::SpectralFilter<Dim, TagList>>;
  using argument_tags = tmpl::list<>;

  // p-refinement: leave the filter unchanged.
  static void apply(
      const gsl::not_null<
          std::unique_ptr<Filters::Filter<Dim, TagList>>*> /*filter*/,
      const std::pair<Mesh<Dim>, Element<Dim>>& /*old_mesh_and_element*/) {}

  // Splitting: clone the parent's filter for the new child.
  template <typename... ParentTags>
  static void apply(
      const gsl::not_null<std::unique_ptr<Filters::Filter<Dim, TagList>>*>
          filter,
      const tuples::TaggedTuple<ParentTags...>& parent_items) {
    *filter =
        tuples::get<Filters::Tags::SpectralFilter<Dim, TagList>>(parent_items)
            ->get_clone();
  }

  // Joining: clone the first child's filter for the parent.
  template <typename... ChildrenTags>
  static void apply(
      const gsl::not_null<std::unique_ptr<Filters::Filter<Dim, TagList>>*>
          filter,
      const std::unordered_map<ElementId<Dim>,
                               tuples::TaggedTuple<ChildrenTags...>>&
          children_items) {
    const auto first_child_items = children_items.begin();
    const auto& first_filter =
        tuples::get<Filters::Tags::SpectralFilter<Dim, TagList>>(
            first_child_items->second);
    for (auto it = std::next(first_child_items); it != children_items.end();
         ++it) {
      const auto& other_filter =
          tuples::get<Filters::Tags::SpectralFilter<Dim, TagList>>(it->second);
      if (not first_filter->is_equal(*other_filter)) {
        ERROR("Children do not agree on all items!");
      }
    }
    *filter = first_filter->get_clone();
  }
};

}  // namespace evolution::dg::Initialization
