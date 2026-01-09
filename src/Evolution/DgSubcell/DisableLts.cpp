// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/DisableLts.hpp"

#include <cstddef>
#include <optional>

#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarInfo.hpp"
#include "Evolution/DiscontinuousGalerkin/TimeSteppingPolicy.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell {
template <size_t Dim>
void DisableLts<Dim>::apply(
    const gsl::not_null<std::optional<size_t>*> fixed_lts_ratio,
    const gsl::not_null<
        DirectionalIdMap<Dim, ::evolution::dg::MortarInfo<Dim>>*>
        mortar_infos,
    const Element<Dim>& element, const SubcellOptions& subcell_options) {
  if (not alg::found(subcell_options.only_dg_block_ids(),
                     element.id().block_id())) {
    fixed_lts_ratio->emplace(subcell_options.lts_steps_per_slab());
    for (auto& [mortar_id, info] : *mortar_infos) {
      if (not alg::found(subcell_options.only_dg_block_ids(),
                         mortar_id.id().block_id())) {
        info.time_stepping_policy() = TimeSteppingPolicy::EqualRate;
      }
    }
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) template struct DisableLts<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace evolution::dg::subcell
