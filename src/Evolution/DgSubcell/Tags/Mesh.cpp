// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/Tags/Mesh.hpp"

#include "Evolution/DgSubcell/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace evolution::dg::subcell::Tags {
template <size_t VolumeDim>
void MeshCompute<VolumeDim>::function(
    const gsl::not_null<return_type*> subcell_mesh,
    const ::Mesh<VolumeDim>& dg_mesh) {
  *subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template struct MeshCompute<GET_DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef GET_DIM
}  // namespace evolution::dg::subcell::Tags
