// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Tags.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Structure/BlockGroups.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace amr::Tags {

template <size_t Dim>
std::optional<std::unordered_set<size_t>> AmrBlocks<Dim>::create_from_options(
    const std::optional<std::vector<std::string>>& block_names,
    const std::unique_ptr<::DomainCreator<Dim>>& domain_creator) {
  if (not block_names.has_value()) {
    return std::nullopt;
  }
  // Resolve block names and IDs
  const auto all_block_names = domain_creator->block_names();
  std::unordered_set<size_t> result{};
  for (const auto& name : domain::expand_block_groups_to_block_names(
           block_names.value(), all_block_names,
           domain_creator->block_groups())) {
    result.insert(static_cast<size_t>(std::distance(
        all_block_names.begin(),
        std::find(all_block_names.begin(), all_block_names.end(), name))));
  }
  return result;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data)                                    \
  template std::optional<std::unordered_set<size_t>>              \
  AmrBlocks<DIM(data)>::create_from_options(                      \
      const std::optional<std::vector<std::string>>& block_names, \
      const std::unique_ptr<::DomainCreator<DIM(data)>>& domain_creator);
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))
#undef INSTANTIATION
#undef DIM

}  // namespace amr::Tags
