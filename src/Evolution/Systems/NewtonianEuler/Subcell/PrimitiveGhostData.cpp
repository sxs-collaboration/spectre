// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Subcell/PrimitiveGhostData.hpp"

#include <cstddef>

#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler::subcell {
template <size_t Dim>
auto PrimitiveGhostDataOnSubcells<Dim>::apply(const Variables<prim_tags>& prims)
    -> Variables<prims_to_reconstruct_tags> {
  Variables<prims_to_reconstruct_tags> prims_for_reconstruction{
      prims.number_of_grid_points()};
  tmpl::for_each<prims_to_reconstruct_tags>(
      [&prims, &prims_for_reconstruction](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        get<tag>(prims_for_reconstruction) = get<tag>(prims);
      });
  return prims_for_reconstruction;
}

template <size_t Dim>
auto PrimitiveGhostDataToSlice<Dim>::apply(const Variables<prim_tags>& prims,
                                           const Mesh<Dim>& dg_mesh,
                                           const Mesh<Dim>& subcell_mesh)
    -> Variables<prims_to_reconstruct_tags> {
  // We send the projected prims. There are truncation level errors
  // introduced here, but let's try it!
  Variables<prims_to_reconstruct_tags> prims_for_reconstruction{
      prims.number_of_grid_points()};
  tmpl::for_each<prims_to_reconstruct_tags>(
      [&prims, &prims_for_reconstruction](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        get<tag>(prims_for_reconstruction) = get<tag>(prims);
      });

  return evolution::dg::subcell::fd::project(prims_for_reconstruction, dg_mesh,
                                             subcell_mesh.extents());
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data)                            \
  template class PrimitiveGhostDataOnSubcells<DIM(data)>; \
  template class PrimitiveGhostDataToSlice<DIM(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))
#undef INSTANTIATION
#undef DIM
}  // namespace NewtonianEuler::subcell
