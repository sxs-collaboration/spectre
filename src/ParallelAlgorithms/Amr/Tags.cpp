// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Tags.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Structure/BlockGroups.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace amr::Tags {

template <size_t Dim>
std::optional<std::vector<size_t>> AmrBlocks<Dim>::create_from_options(
    const std::optional<std::vector<std::string>>& block_names,
    const std::unique_ptr<::DomainCreator<Dim>>& domain_creator) {
  if (not block_names.has_value()) {
    return std::nullopt;
  }
  return domain::block_ids_from_names(block_names.value(),
                                      domain_creator->block_names(),
                                      domain_creator->block_groups());
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data)                                    \
  template std::optional<std::vector<size_t>>                     \
  AmrBlocks<DIM(data)>::create_from_options(                      \
      const std::optional<std::vector<std::string>>& block_names, \
      const std::unique_ptr<::DomainCreator<DIM(data)>>& domain_creator);
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))
#undef INSTANTIATION
#undef DIM

}  // namespace amr::Tags
