// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/FiniteDifference/MonotisedCentral.hpp"

#include <array>
#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Evolution/DgSubcell/NeighborData.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/ReconstructWork.tpp"
#include "NumericalAlgorithms/FiniteDifference/MonotisedCentral.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Burgers::fd {
MonotisedCentral::MonotisedCentral(CkMigrateMessage* const msg) noexcept
    : Reconstructor(msg) {}

std::unique_ptr<Reconstructor> MonotisedCentral::get_clone() const noexcept {
  return std::make_unique<MonotisedCentral>(*this);
}

void MonotisedCentral::pup(PUP::er& p) { Reconstructor::pup(p); }

PUPable_def(MonotisedCentral)

void MonotisedCentral::reconstruct(
    const gsl::not_null<std::array<Variables<face_vars_tags>, 1>*>
        vars_on_lower_face,
    const gsl::not_null<std::array<Variables<face_vars_tags>, 1>*>
        vars_on_upper_face,
    const Variables<volume_vars_tags>& volume_vars, const Element<1>& element,
    const FixedHashMap<maximum_number_of_neighbors(1) + 1,
                       std::pair<Direction<1>, ElementId<1>>,
                       evolution::dg::subcell::NeighborData,
                       boost::hash<std::pair<Direction<1>, ElementId<1>>>>&
        neighbor_data,
    const Mesh<1>& subcell_mesh) const noexcept {
  reconstruct_prims_work(
      vars_on_lower_face, vars_on_upper_face,
      [](auto upper_face_vars_ptr, auto lower_face_vars_ptr,
         const auto& the_volume_vars, const auto& ghost_cell_vars,
         const auto& subcell_extents, const size_t number_of_variables) {
        ::fd::reconstruction::monotised_central(
            upper_face_vars_ptr, lower_face_vars_ptr, the_volume_vars,
            ghost_cell_vars, subcell_extents, number_of_variables);
      },
      volume_vars, element, neighbor_data, subcell_mesh, ghost_zone_size());
}

void MonotisedCentral::reconstruct_fd_neighbor(
    const gsl::not_null<Variables<face_vars_tags>*> vars_on_face,
    const Variables<volume_vars_tags>& subcell_volume_vars,
    const Element<1>& element,
    const FixedHashMap<maximum_number_of_neighbors(1) + 1,
                       std::pair<Direction<1>, ElementId<1>>,
                       evolution::dg::subcell::NeighborData,
                       boost::hash<std::pair<Direction<1>, ElementId<1>>>>&
        neighbor_data,
    const Mesh<1>& subcell_mesh,
    const Direction<1> direction_to_reconstruct) const noexcept {
  reconstruct_fd_neighbor_work(
      vars_on_face,
      [](const auto tensor_component_on_face_ptr,
         const auto& tensor_component_volume,
         const auto& tensor_component_neighbor, const Index<1>& subcell_extents,
         const Index<1>& ghost_data_extents,
         const Direction<1>& local_direction_to_reconstruct) noexcept {
        ::fd::reconstruction::reconstruct_neighbor<
            Side::Lower,
            ::fd::reconstruction::detail::MonotisedCentralReconstructor>(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct);
      },
      [](const auto tensor_component_on_face_ptr,
         const auto& tensor_component_volume,
         const auto& tensor_component_neighbor, const Index<1>& subcell_extents,
         const Index<1>& ghost_data_extents,
         const Direction<1>& local_direction_to_reconstruct) noexcept {
        ::fd::reconstruction::reconstruct_neighbor<
            Side::Upper,
            ::fd::reconstruction::detail::MonotisedCentralReconstructor>(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct);
      },
      subcell_volume_vars, element, neighbor_data, subcell_mesh,
      direction_to_reconstruct, ghost_zone_size());
}
}  // namespace Burgers::fd
