// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/FiniteDifference/MonotonisedCentral.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <utility>

// FIXME : review header files
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/ReconstructWork.tpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "NumericalAlgorithms/FiniteDifference/MonotonisedCentral.hpp"
#include "NumericalAlgorithms/FiniteDifference/NeighborDataAsVariables.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree::fd {

MonotonisedCentral::MonotonisedCentral(CkMigrateMessage* const msg)
    : Reconstructor(msg) {}

std::unique_ptr<Reconstructor> MonotonisedCentral::get_clone() const {
  return std::make_unique<MonotonisedCentral>(*this);
}

void MonotonisedCentral::pup(PUP::er& p) { Reconstructor::pup(p); }

// NOLINTNEXTLINE
PUP::able::PUP_ID MonotonisedCentral::my_PUP_ID = 0;

void MonotonisedCentral::reconstruct(
    const gsl::not_null<std::array<Variables<face_vars_tags>, dim>*>
        vars_on_lower_face,
    const gsl::not_null<std::array<Variables<face_vars_tags>, dim>*>
        vars_on_upper_face,
    const Variables<volume_vars_tags>& volume_vars,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_j,
    const Element<dim>& element,
    const FixedHashMap<maximum_number_of_neighbors(dim),
                       std::pair<Direction<dim>, ElementId<dim>>,
                       evolution::dg::subcell::GhostData,
                       boost::hash<std::pair<Direction<dim>, ElementId<dim>>>>&
        ghost_data,
    const Mesh<dim>& subcell_mesh) const {
  FixedHashMap<maximum_number_of_neighbors(dim),
               std::pair<Direction<dim>, ElementId<dim>>,
               Variables<volume_vars_tags>,
               boost::hash<std::pair<Direction<dim>, ElementId<dim>>>>
      neighbor_variables_data{};
  ::fd::neighbor_data_as_variables<dim>(make_not_null(&neighbor_variables_data),
                                        ghost_data, ghost_zone_size(),
                                        subcell_mesh);

  reconstruct_work(
      vars_on_lower_face, vars_on_upper_face,
      [](auto upper_face_vars_ptr, auto lower_face_vars_ptr,
         const auto& volume_variables, const auto& ghost_cell_vars,
         const auto& subcell_extents, const size_t number_of_variables) {
        ::fd::reconstruction::monotonised_central(
            upper_face_vars_ptr, lower_face_vars_ptr, volume_variables,
            ghost_cell_vars, subcell_extents, number_of_variables);
      },
      volume_vars, tilde_j, element, ghost_data, subcell_mesh,
      ghost_zone_size());
}

void MonotonisedCentral::reconstruct_fd_neighbor(
    const gsl::not_null<Variables<face_vars_tags>*> vars_on_face,
    const Variables<volume_vars_tags>& volume_vars,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_j,
    const Element<dim>& element,
    const FixedHashMap<maximum_number_of_neighbors(dim),
                       std::pair<Direction<dim>, ElementId<dim>>,
                       evolution::dg::subcell::GhostData,
                       boost::hash<std::pair<Direction<dim>, ElementId<dim>>>>&
        ghost_data,
    const Mesh<dim>& subcell_mesh,
    const Direction<dim> direction_to_reconstruct) const {
  reconstruct_fd_neighbor_work(
      vars_on_face,
      [](const auto tensor_component_on_face_ptr,
         const auto& tensor_component_volume,
         const auto& tensor_component_neighbor, const auto& subcell_extents,
         const auto& ghost_data_extents,
         const auto& local_direction_to_reconstruct) {
        ::fd::reconstruction::reconstruct_neighbor<
            Side::Lower,
            ::fd::reconstruction::detail::MonotonisedCentralReconstructor>(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct);
      },
      [](const auto tensor_component_on_face_ptr,
         const auto& tensor_component_volume,
         const auto& tensor_component_neighbor, const auto& subcell_extents,
         const auto& ghost_data_extents,
         const auto& local_direction_to_reconstruct) {
        ::fd::reconstruction::reconstruct_neighbor<
            Side::Upper,
            ::fd::reconstruction::detail::MonotonisedCentralReconstructor>(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct);
      },
      volume_vars, tilde_j, element, ghost_data, subcell_mesh,
      direction_to_reconstruct, ghost_zone_size());
}

bool operator==(const MonotonisedCentral& /*lhs*/,
                const MonotonisedCentral& /*rhs*/) {
  return true;
}

bool operator!=(const MonotonisedCentral& lhs, const MonotonisedCentral& rhs) {
  return not(lhs == rhs);
}

}  // namespace ForceFree::fd
