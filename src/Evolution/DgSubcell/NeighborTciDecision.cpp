// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/NeighborTciDecision.hpp"

#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/Access.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/FaceType.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell {
template <size_t Dim>
void neighbor_tci_decision(
    const gsl::not_null<db::Access*> box,
    const DirectionalId<Dim>& directional_element_id,
    const evolution::dg::BoundaryData<Dim>& neighbor_data) {
  const auto& element = db::get<::domain::Tags::Element<Dim>>(*box);
  db::mutate<subcell::Tags::NeighborTciDecisions<Dim>>(
      [&element, &directional_element_id,
       &neighbor_data](const auto neighbor_tci_decisions_ptr) {
        if (not neighbor_tci_decisions_ptr->contains(directional_element_id)) {
          // Non-hypercube elements (e.g. spherical shells) have an empty
          // NeighborTciDecisions map. Subcell-capable elements skip
          // MultipleNonconforming directions. Either way, no update is needed
          ASSERT(
              neighbor_tci_decisions_ptr->empty() or
                  element.face_types().at(directional_element_id.direction()) ==
                      domain::FaceType::MultipleNonconforming,
              "NeighborTciDecisions does not contain the neighbor "
                  << directional_element_id
                  << " but the map is not empty and the face direction is not "
                     "MultipleNonconforming.");
          return;
        }
        neighbor_tci_decisions_ptr->at(directional_element_id) =
            neighbor_data.tci_status;
      },
      box);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                \
  template void neighbor_tci_decision(                        \
      gsl::not_null<db::Access*> box,                         \
      const DirectionalId<DIM(data)>& directional_element_id, \
      const evolution::dg::BoundaryData<DIM(data)>& neighbor_data);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::subcell
