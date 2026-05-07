// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Tag.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Element;
template <size_t Dim>
class Mesh;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
namespace Filters {
template <size_t Dim, typename TagList>
class Filter;
}  // namespace Filters
/// \endcond

namespace evolution::dg::Initialization {
/*!
 * \brief Initialization mutator that selects the spectral filter for each
 * element and stores it as `Filters::Tags::SpectralFilter` in the DataBox.
 *
 * \details Used with `Initialization::Actions::InitializeItems`. For each
 * element, `apply` scans the GlobalCache list
 * `Filters::Tags::SpectralFilters<Dim, TagList>` and selects the unique filter
 * `f` for which `f->supports_mesh(mesh)` returns `true` AND either
 * `f->blocks_to_filter()` is `std::nullopt` (the filter applies to every
 * block) or the element's block id is contained in
 * `f->blocks_to_filter().value()`. The selected filter is deep-copied via
 * `Filters::Filter::get_clone` and stored in
 * `Filters::Tags::SpectralFilter<Dim, TagList>`.
 *
 * Errors if more than one filter in the list matches an element, or if no
 * filter matches (specify a `None` filter to disable filtering in an element).
 *
 * Uses:
 * - GlobalCache:
 *   - `Filters::Tags::SpectralFilters<Dim, TagList>`
 * - DataBox:
 *   - `domain::Tags::Element<Dim>`
 *   - `domain::Tags::Mesh<Dim>`
 *
 * DataBox changes:
 * - Adds:
 *   - `Filters::Tags::SpectralFilter<Dim, TagList>`
 * - Removes: nothing
 * - Modifies: nothing
 *
 * \tparam Dim Spatial dimension of the element mesh.
 * \tparam TagList `tmpl::list` of tensor tags in the `Variables` to filter.
 */
template <size_t Dim, typename TagList>
struct SpectralFilters {
  using const_global_cache_tags =
      tmpl::list<Filters::Tags::SpectralFilters<Dim, TagList>>;
  using mutable_global_cache_tags = tmpl::list<>;
  using simple_tags_from_options = tmpl::list<>;
  using simple_tags = tmpl::list<Filters::Tags::SpectralFilter<Dim, TagList>>;
  using compute_tags = tmpl::list<>;

  using return_tags = simple_tags;
  using argument_tags =
      tmpl::list<Filters::Tags::SpectralFilters<Dim, TagList>,
                 domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>>;

  /// Sets `spectral_filter` to a clone of the unique filter from
  /// `spectral_filters` that supports `mesh` and applies to `element`'s block.
  static void apply(
      gsl::not_null<std::unique_ptr<Filters::Filter<Dim, TagList>>*>
          spectral_filter,
      const std::vector<std::unique_ptr<Filters::Filter<Dim, TagList>>>&
          spectral_filters,
      const Element<Dim>& element, const Mesh<Dim>& mesh);
};
}  // namespace evolution::dg::Initialization
