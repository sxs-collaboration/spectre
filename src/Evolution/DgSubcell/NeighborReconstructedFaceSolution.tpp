// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/DgSubcell/NeighborReconstructedFaceSolution.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/Access.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarDataHolder.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell {
template <size_t VolumeDim, typename DgComputeSubcellNeighborPackagedData>
void neighbor_reconstructed_face_solution(
    const gsl::not_null<db::Access*> box) {
  std::vector<DirectionalId<VolumeDim>> mortars_to_reconstruct_to{};
  {
    const auto& mortar_data =
        db::get<evolution::dg::Tags::MortarData<VolumeDim>>(*box);
    for (const auto& [mortar_id, data] : mortar_data) {
      if (not data.neighbor().mortar_data.has_value()) {
        mortars_to_reconstruct_to.push_back(mortar_id);
      }
    }
  }
  DirectionalIdMap<VolumeDim, DataVector> neighbor_reconstructed_evolved_vars =
      DgComputeSubcellNeighborPackagedData::apply(*box,
                                                  mortars_to_reconstruct_to);
  ASSERT(neighbor_reconstructed_evolved_vars.size() ==
             mortars_to_reconstruct_to.size(),
         "Should have reconstructed "
             << mortars_to_reconstruct_to.size() << " sides but reconstructed "
             << neighbor_reconstructed_evolved_vars.size() << " sides.");
  const Mesh<VolumeDim>& dg_mesh = db::get<domain::Tags::Mesh<VolumeDim>>(*box);
  db::mutate<evolution::dg::Tags::MortarData<VolumeDim>>(
      [&](const gsl::not_null<DirectionalIdMap<
              VolumeDim, evolution::dg::MortarDataHolder<VolumeDim>>*>
              mortar_data) {
        for (const auto& mortar_id : mortars_to_reconstruct_to) {
          ASSERT(neighbor_reconstructed_evolved_vars.find(mortar_id) !=
                     neighbor_reconstructed_evolved_vars.end(),
                 "Could not find mortar id " << mortar_id
                                             << " in reconstructed data map.");
          auto& neighbor_data = mortar_data->at(mortar_id).neighbor();
          neighbor_data.mortar_data =
              std::move(neighbor_reconstructed_evolved_vars.at(mortar_id));
          neighbor_data.mortar_mesh =
              dg_mesh.slice_away(mortar_id.direction().dimension());
        }
      },
      box);
}
}  // namespace evolution::dg::subcell
