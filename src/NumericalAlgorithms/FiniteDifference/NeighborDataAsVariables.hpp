// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace fd {
/*!
 * \brief Given the type-erased neighbor data for reconstruction stored in a
 * `DataVector`, have `Variables` point into them.
 *
 * This function is helpful for reconstruction, especially when wanting to apply
 * different reconstruction methods to different tags. This can happen, for
 * example, when doing positivity-preserving reconstruction. The density should
 * remain positive, but negative velocities are fine.
 */
template <size_t Dim, typename ReconstructionTags>
void neighbor_data_as_variables(
    const gsl::not_null<
        FixedHashMap<maximum_number_of_neighbors(Dim),
                     std::pair<Direction<Dim>, ElementId<Dim>>,
                     Variables<ReconstructionTags>,
                     boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>*>
        vars_neighbor_data,
    const FixedHashMap<maximum_number_of_neighbors(Dim),
                       std::pair<Direction<Dim>, ElementId<Dim>>,
                       evolution::dg::subcell::GhostData,
                       boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>&
        all_ghost_data,
    const size_t ghost_zone_size, const Mesh<Dim>& subcell_mesh) {
  const size_t neighbor_num_pts =
      ghost_zone_size * subcell_mesh.extents().slice_away(0).product();
  ASSERT(
      subcell_mesh == Mesh<Dim>(subcell_mesh.extents(0), subcell_mesh.basis(0),
                                subcell_mesh.quadrature(0)),
      "subcell_mesh must be isotropic but got " << subcell_mesh);
  vars_neighbor_data->clear();
  for (const auto& [neighbor_id, ghost_data] : all_ghost_data) {
    const DataVector& data =
        ghost_data.neighbor_ghost_data_for_reconstruction();
    (*vars_neighbor_data)[neighbor_id] = {};
    (*vars_neighbor_data)[neighbor_id].set_data_ref(
        const_cast<double*>(data.data()),
        Variables<ReconstructionTags>::number_of_independent_components *
            neighbor_num_pts);
  }
}
}  // namespace fd
