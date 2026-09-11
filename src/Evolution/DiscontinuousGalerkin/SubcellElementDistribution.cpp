// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/SubcellElementDistribution.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg {
template <size_t Dim>
std::optional<std::unordered_map<size_t, std::array<size_t, Dim>>>
compute_weighting_extents_override(
    const std::vector<size_t>& only_dg_block_ids,
    const std::vector<std::array<size_t, Dim>>& initial_extents) {
  std::unordered_map<size_t, std::array<size_t, Dim>> overrides{};
  for (size_t block_id = 0; block_id < initial_extents.size(); ++block_id) {
    if (not alg::found(only_dg_block_ids, block_id)) {
      std::array<size_t, Dim> weighting_extents = initial_extents[block_id];
      for (size_t d = 0; d < Dim; ++d) {
        ASSERT(gsl::at(weighting_extents, d) > 0,
               "The initial extents of block "
                   << block_id << " in dimension " << d
                   << " must be non-zero, but are zero.");
        // This must match the subcell mesh built by
        // `evolution::dg::subcell::fd::mesh`, which we cannot call here
        // because the `DgSubcell` library links this one.
        gsl::at(weighting_extents, d) = 2 * gsl::at(weighting_extents, d) - 1;
      }
      overrides[block_id] = weighting_extents;
    }
  }
  if (overrides.empty()) {
    return std::nullopt;
  }
  return overrides;
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                       \
  template std::optional<                                            \
      std::unordered_map<size_t, std::array<size_t, GET_DIM(data)>>> \
  compute_weighting_extents_override(                                \
      const std::vector<size_t>& only_dg_block_ids,                  \
      const std::vector<std::array<size_t, GET_DIM(data)>>& initial_extents);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace evolution::dg
