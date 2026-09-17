// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/ElementDistribution.hpp"
#include "Domain/Tags/ElementDistribution.hpp"
#include "Evolution/DiscontinuousGalerkin/OnlyDgBlockIds.hpp"
#include "Evolution/DiscontinuousGalerkin/OptionTags.hpp"
#include "Evolution/DiscontinuousGalerkin/UsingSubcell.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg {
/*!
 * \brief Returns the per-block extents to use in place of `initial_extents`
 * when weighting subcell-capable elements by their finite-difference grid
 * instead of their DG grid (see
 * `evolution::dg::Tags::UseSubcellGridPointsForDistribution`).
 *
 * For each block not in `only_dg_block_ids`, the returned extents are the
 * number of finite-difference grid points \f$2N-1\f$ corresponding to the
 * \f$N\f$ DG grid points in `initial_extents`, matching the subcell mesh built
 * by `evolution::dg::subcell::fd::mesh`. Blocks in `only_dg_block_ids` are
 * omitted from the returned map, since they always run on the DG grid.
 *
 * Returns `std::nullopt` if every block is in `only_dg_block_ids`, since then
 * there is nothing to override.
 */
template <size_t Dim>
std::optional<std::unordered_map<size_t, std::array<size_t, Dim>>>
compute_weighting_extents_override(
    const std::vector<size_t>& only_dg_block_ids,
    const std::vector<std::array<size_t, Dim>>& initial_extents);

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// Tag that holds whether to use the number of finite-difference subcell
/// grid points (instead of the number of DG grid points) when weighting
/// subcell-capable elements for the `domain::Tags::ElementDistribution`.
///
/// Errors unless the `ElementDistribution` is
/// `domain::ElementWeight::NumGridPoints`, since that is the only weight
/// whose cost depends on the extents in a way that makes substituting the
/// subcell extents meaningful.
struct UseSubcellGridPointsForDistribution : db::SimpleTag {
  using type = bool;
  using option_tags =
      tmpl::list<OptionTags::UseSubcellGridPointsForDistribution,
                 domain::OptionTags::ElementDistribution>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(
      const bool use_subcell_grid_points,
      const std::optional<domain::ElementWeight>& element_weight) {
    // `ElementWeight::NumGridPoints` is the only weight for which
    // substituting the subcell extents produces the intended cost ratio.
    // `ElementWeight::Uniform` ignores the extents entirely, so the option
    // would silently do nothing. `ElementWeight::NumGridPointsAndGridSpacing`
    // derives a minimum grid spacing from the extents by building a DG mesh,
    // which for the subcell extents describes neither the DG grid nor the
    // (uniform) finite-difference grid, and so mis-weights the cost.
    //
    // This could be improved in the future by telling
    // `domain::get_element_costs()` about the finite-difference grid, so that
    // `ElementWeight::NumGridPointsAndGridSpacing` can be supported too.
    if (use_subcell_grid_points and
        element_weight != std::optional{domain::ElementWeight::NumGridPoints}) {
      ERROR(
          "UseSubcellGridPointsForDistribution is only supported with "
          "ElementDistribution: NumGridPoints. Either set "
          "UseSubcellGridPointsForDistribution to false or set "
          "ElementDistribution to NumGridPoints.");
    }
    return use_subcell_grid_points;
  }
};
}  // namespace Tags

/// \brief The global cache tags needed to distribute elements, for use as the
/// `const_global_cache_tags` of a component that creates the elements.
///
/// The subcell-specific tags are only included for executables with DG-subcell
/// support, since the options they are built from only exist there.
template <typename Metavariables, size_t Dim>
using element_distribution_cache_tags = tmpl::append<
    tmpl::list<domain::Tags::Domain<Dim>, domain::Tags::ElementDistribution>,
    tmpl::conditional_t<using_subcell_v<Metavariables>,
                        tmpl::list<Tags::UseSubcellGridPointsForDistribution,
                                   Tags::OnlyDgBlockIds<Dim>>,
                        tmpl::list<>>>;

/*!
 * \brief Reads `element_distribution_cache_tags` out of the \p cache and
 * returns the `weighting_extents_override` to pass to
 * `Parallel::create_elements_using_distribution`.
 *
 * Always `std::nullopt` for executables without DG-subcell support, and for
 * those with it whenever
 * `evolution::dg::Tags::UseSubcellGridPointsForDistribution` is `false`.
 */
template <size_t Dim, typename Metavariables>
std::optional<std::unordered_map<size_t, std::array<size_t, Dim>>>
weighting_extents_override_from_cache(
    const Parallel::GlobalCache<Metavariables>& cache,
    const std::vector<std::array<size_t, Dim>>& initial_extents) {
  if constexpr (using_subcell_v<Metavariables>) {
    if (Parallel::get<Tags::UseSubcellGridPointsForDistribution>(cache)) {
      return compute_weighting_extents_override(
          Parallel::get<Tags::OnlyDgBlockIds<Dim>>(cache), initial_extents);
    }
  }
  return std::nullopt;
}
}  // namespace evolution::dg
