// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/SubcellElementDistribution.hpp"

#include <array>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg {
template <size_t Dim>
std::unordered_map<ElementId<Dim>, std::array<size_t, Dim>>
compute_weighting_extents_override(
    const std::vector<size_t>& only_dg_block_ids,
    const std::vector<Block<Dim>>& blocks,
    const std::vector<std::array<size_t, Dim>>& initial_refinement_levels,
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const bool subcell_enabled_at_external_boundary) {
  ASSERT(blocks.size() == initial_extents.size() and
             blocks.size() == initial_refinement_levels.size(),
         "The number of blocks ("
             << blocks.size() << "), initial extents ("
             << initial_extents.size() << "), and initial refinement levels ("
             << initial_refinement_levels.size() << ") must be the same.");
  std::unordered_map<ElementId<Dim>, std::array<size_t, Dim>> overrides{};
  for (const auto& block : blocks) {
    if (alg::found(only_dg_block_ids, block.id())) {
      continue;
    }
    std::array<size_t, Dim> weighting_extents = initial_extents[block.id()];
    for (size_t d = 0; d < Dim; ++d) {
      ASSERT(gsl::at(weighting_extents, d) > 0,
             "The initial extents of block "
                 << block.id() << " in dimension " << d
                 << " must be non-zero, but are zero.");
      // This must match the subcell mesh built by
      // `evolution::dg::subcell::fd::mesh`, which we cannot call here
      // because the `DgSubcell` library links this one.
      gsl::at(weighting_extents, d) = 2 * gsl::at(weighting_extents, d) - 1;
    }
    for (const auto& element_id : initial_element_ids(
             block.id(), initial_refinement_levels[block.id()])) {
      const Element<Dim> element = domain::create_initial_element(
          element_id, blocks, initial_refinement_levels);
      // Elements bordering a DG-only block and (unless enabled) elements on
      // an external boundary are not allowed to use subcell. This must match
      // `evolution::dg::subcell::Actions::Initialize`.
      const bool bordering_dg_block = alg::any_of(
          element.neighbors(),
          [&only_dg_block_ids](const auto& direction_and_neighbors) {
            return alg::any_of(
                direction_and_neighbors.second.ids(),
                [&only_dg_block_ids](const ElementId<Dim>& neighbor_id) {
                  return alg::found(only_dg_block_ids, neighbor_id.block_id());
                });
          });
      if (bordering_dg_block or (not subcell_enabled_at_external_boundary and
                                 not element.external_boundaries().empty())) {
        continue;
      }
      overrides.emplace(element_id, weighting_extents);
    }
  }
  return overrides;
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                               \
  template std::unordered_map<ElementId<GET_DIM(data)>,                      \
                              std::array<size_t, GET_DIM(data)>>             \
  compute_weighting_extents_override(                                        \
      const std::vector<size_t>& only_dg_block_ids,                          \
      const std::vector<Block<GET_DIM(data)>>& blocks,                       \
      const std::vector<std::array<size_t, GET_DIM(data)>>&                  \
          initial_refinement_levels,                                         \
      const std::vector<std::array<size_t, GET_DIM(data)>>& initial_extents, \
      bool subcell_enabled_at_external_boundary);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace evolution::dg
