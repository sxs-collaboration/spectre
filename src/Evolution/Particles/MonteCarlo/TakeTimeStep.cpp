// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Particles/MonteCarlo/TakeTimeStep.tpp"

#include <cstddef>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Side.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

namespace Particles::MonteCarlo::TakeTimeStep_detail {

void combine_ghost_data(
    const gsl::not_null<DataVector*> with_ghost_data, const Mesh<3> local_mesh,
    const size_t num_ghost_zones, const DataVector& local_data,
    const DirectionalIdMap<3, std::optional<DataVector>>& ghost_data) {
  const Index<3> local_extents = local_mesh.extents();
  Index<3> ghost_extents = local_extents;
  for (size_t d = 0; d < 3; d++) {
    ghost_extents[d] += 2 * num_ghost_zones;
  }
  for (size_t i = 0; i < local_mesh.extents(0); ++i) {
    for (size_t j = 0; j < local_mesh.extents(1); ++j) {
      for (size_t k = 0; k < local_mesh.extents(2); ++k) {
        (*with_ghost_data)[collapsed_index(
            Index<3>{i + num_ghost_zones, j + num_ghost_zones,
                     k + num_ghost_zones},
            ghost_extents)] =
            local_data[collapsed_index(Index<3>{i, j, k}, local_extents)];
      }
    }
  }
  // Loop over each direction. We assume at most one neighbor in each
  // direction.
  for (auto& [direction_id, ghost_data_dir] : ghost_data) {
    if (ghost_data_dir) {
      const size_t dimension = direction_id.direction().dimension();
      const Side side = direction_id.direction().side();
      Index<3> ghost_zone_extents = local_extents;
      ghost_zone_extents[dimension] = num_ghost_zones;
      for (size_t i = 0; i < ghost_zone_extents[0]; ++i) {
        for (size_t j = 0; j < ghost_zone_extents[1]; ++j) {
          for (size_t k = 0; k < ghost_zone_extents[2]; ++k) {
            const Index<3> ghost_index_3d{i, j, k};
            const size_t ghost_index =
                collapsed_index(ghost_index_3d, ghost_zone_extents);
            Index<3> extended_index_3d{i + num_ghost_zones, j + num_ghost_zones,
                                       k + num_ghost_zones};
            extended_index_3d[dimension] = (side == Side::Lower)
                                               ? ghost_index_3d[dimension]
                                               : local_extents[dimension] +
                                                     num_ghost_zones +
                                                     ghost_index_3d[dimension];
            const size_t extended_index =
                collapsed_index(extended_index_3d, ghost_extents);
            (*with_ghost_data)[extended_index] =
                ghost_data_dir.value()[ghost_index];
          }
        }
      }
    }
  }
}

}  // namespace Particles::MonteCarlo::TakeTimeStep_detail
