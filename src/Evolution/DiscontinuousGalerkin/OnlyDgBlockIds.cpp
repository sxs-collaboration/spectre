// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/OnlyDgBlockIds.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/BlockGroups.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DiscontinuousGalerkin/BlockSupportsSubcell.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace evolution::dg {
template <size_t Dim>
std::vector<size_t> compute_only_dg_block_ids(
    const std::optional<std::vector<std::string>>&
        only_dg_block_and_group_names,
    const std::vector<std::string>& block_names,
    const std::unordered_map<std::string, std::unordered_set<std::string>>&
        block_groups,
    const std::vector<Block<Dim>>& blocks) {
  std::vector<size_t> only_dg_block_ids = domain::block_ids_from_names(
      only_dg_block_and_group_names.value_or(std::vector<std::string>{}),
      block_names, block_groups);

  // Combine with blocks whose topology does not support subcell (e.g.
  // spherical shells, filled balls) and add them to the DG-only list.
  for (const auto& block : blocks) {
    if (not block_supports_subcell(block) and
        not alg::found(only_dg_block_ids, block.id())) {
      only_dg_block_ids.push_back(block.id());
    }
  }
  return only_dg_block_ids;
}

template <size_t Dim>
std::vector<size_t> compute_only_dg_block_ids(
    const std::optional<std::vector<std::string>>&
        only_dg_block_and_group_names,
    const DomainCreator<Dim>& domain_creator) {
  const Domain<Dim> domain = domain_creator.create_domain();
  return compute_only_dg_block_ids(
      only_dg_block_and_group_names, domain_creator.block_names(),
      domain_creator.block_groups(), domain.blocks());
}

template <size_t Dim>
bool borders_only_dg_block(const Element<Dim>& element,
                           const std::vector<size_t>& only_dg_block_ids) {
  return alg::any_of(
      element.neighbors(),
      [&only_dg_block_ids](const auto& direction_and_neighbors) {
        return alg::any_of(
            direction_and_neighbors.second.ids(),
            [&only_dg_block_ids](const ElementId<Dim>& neighbor_id) {
              return alg::found(only_dg_block_ids, neighbor_id.block_id());
            });
      });
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                \
  template std::vector<size_t> compute_only_dg_block_ids(                     \
      const std::optional<std::vector<std::string>>&                          \
          only_dg_block_and_group_names,                                      \
      const std::vector<std::string>& block_names,                            \
      const std::unordered_map<std::string, std::unordered_set<std::string>>& \
          block_groups,                                                       \
      const std::vector<Block<GET_DIM(data)>>& blocks);                       \
  template std::vector<size_t> compute_only_dg_block_ids(                     \
      const std::optional<std::vector<std::string>>&                          \
          only_dg_block_and_group_names,                                      \
      const DomainCreator<GET_DIM(data)>& domain_creator);                    \
  template bool borders_only_dg_block(                                        \
      const Element<GET_DIM(data)>& element,                                  \
      const std::vector<size_t>& only_dg_block_ids);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace evolution::dg
