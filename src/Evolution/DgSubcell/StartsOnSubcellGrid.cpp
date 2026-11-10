// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/StartsOnSubcellGrid.hpp"

#include <cstddef>
#include <vector>

#include "Domain/Structure/Element.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DiscontinuousGalerkin/OnlyDgBlockIds.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace evolution::dg::subcell {
template <size_t Dim>
bool subcell_allowed_in_element(const Element<Dim>& element,
                                const Mesh<Dim>& dg_mesh,
                                const std::vector<size_t>& only_dg_block_ids) {
  return fd::dg_mesh_supports_subcell(dg_mesh) and
         not alg::found(only_dg_block_ids, element.id().block_id()) and
         not evolution::dg::borders_only_dg_block(element, only_dg_block_ids);
}

template <size_t Dim>
bool starts_on_subcell_grid(const Element<Dim>& element,
                            const Mesh<Dim>& dg_mesh,
                            const std::vector<size_t>& only_dg_block_ids,
                            const bool subcell_enabled_at_external_boundary) {
  return (element.external_boundaries().empty() or
          subcell_enabled_at_external_boundary) and
         subcell_allowed_in_element(element, dg_mesh, only_dg_block_ids);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                           \
  template bool subcell_allowed_in_element(                              \
      const Element<DIM(data)>& element, const Mesh<DIM(data)>& dg_mesh, \
      const std::vector<size_t>& only_dg_block_ids);                     \
  template bool starts_on_subcell_grid(                                  \
      const Element<DIM(data)>& element, const Mesh<DIM(data)>& dg_mesh, \
      const std::vector<size_t>& only_dg_block_ids,                      \
      bool subcell_enabled_at_external_boundary);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::subcell
