// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/Tags/ObserverMeshVelocity.hpp"

#include "Evolution/DgSubcell/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell::Tags {
template <size_t Dim>
void ObserverMeshVelocityCompute<Dim>::function(
    const gsl::not_null<return_type*> active_mesh_velocity,
    const subcell::ActiveGrid active_grid,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        dg_mesh_velocity,
    const ::Mesh<Dim>& dg_mesh, const ::Mesh<Dim>& subcell_mesh) {
  if (active_grid == subcell::ActiveGrid::Dg) {
    *active_mesh_velocity = dg_mesh_velocity;
  } else if (dg_mesh_velocity.has_value()) {
    // PROJECT
    *active_mesh_velocity = tnsr::I<DataVector, Dim, Frame::Inertial>{};
    for (size_t i = 0; i < Dim; ++i) {
      active_mesh_velocity->value().get(i) =
          evolution::dg::subcell::fd::project(dg_mesh_velocity.value().get(i),
                                              dg_mesh, subcell_mesh.extents());
    }
  } else {
    *active_mesh_velocity = std::nullopt;
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) \
  template class ObserverMeshVelocityCompute<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace evolution::dg::subcell::Tags
