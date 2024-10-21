// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/MonotonisedCentral.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/ReconstructWork.tpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "NumericalAlgorithms/FiniteDifference/MonotonisedCentral.hpp"
#include "NumericalAlgorithms/FiniteDifference/NeighborDataAsVariables.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::fd {
MonotonisedCentralPrim::MonotonisedCentralPrim(CkMigrateMessage* const msg)
    : Reconstructor(msg) {}

std::unique_ptr<Reconstructor> MonotonisedCentralPrim::get_clone() const {
  return std::make_unique<MonotonisedCentralPrim>(*this);
}

void MonotonisedCentralPrim::pup(PUP::er& p) { Reconstructor::pup(p); }

// NOLINTNEXTLINE
PUP::able::PUP_ID MonotonisedCentralPrim::my_PUP_ID = 0;

template <size_t ThermodynamicDim>
void MonotonisedCentralPrim::reconstruct(
    const gsl::not_null<std::array<Variables<tags_list_for_reconstruct>, dim>*>
        vars_on_lower_face,
    const gsl::not_null<std::array<Variables<tags_list_for_reconstruct>, dim>*>
        vars_on_upper_face,
    const Variables<hydro::grmhd_tags<DataVector>>& volume_prims,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
    const Element<dim>& element,
    const DirectionalIdMap<dim, evolution::dg::subcell::GhostData>& ghost_data,
    const Mesh<dim>& subcell_mesh,
    const VariableFixing::FixToAtmosphere<dim>& fix_to_atmosphere) const {
  DirectionalIdMap<dim, Variables<prims_to_reconstruct_tags>>
      neighbor_variables_data{};
  ::fd::neighbor_data_as_variables<dim>(make_not_null(&neighbor_variables_data),
                                        ghost_data, ghost_zone_size(),
                                        subcell_mesh);
  reconstruct_prims_work<prims_to_reconstruct_tags>(
      vars_on_lower_face, vars_on_upper_face,
      [](auto upper_face_vars_ptr, auto lower_face_vars_ptr,
         const auto& volume_vars, const auto& ghost_cell_vars,
         const auto& subcell_extents, const size_t number_of_variables) {
        ::fd::reconstruction::monotonised_central(
            upper_face_vars_ptr, lower_face_vars_ptr, volume_vars,
            ghost_cell_vars, subcell_extents, number_of_variables);
      },
      volume_prims, eos, element, neighbor_variables_data, subcell_mesh,
      ghost_zone_size(), true, fix_to_atmosphere);
}

template <size_t ThermodynamicDim>
void MonotonisedCentralPrim::reconstruct_fd_neighbor(
    const gsl::not_null<Variables<tags_list_for_reconstruct>*> vars_on_face,
    const Variables<hydro::grmhd_tags<DataVector>>& subcell_volume_prims,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
    const Element<dim>& element,
    const DirectionalIdMap<dim, evolution::dg::subcell::GhostData>& ghost_data,
    const Mesh<dim>& subcell_mesh,
    const VariableFixing::FixToAtmosphere<dim>& fix_to_atmosphere,
    const Direction<dim> direction_to_reconstruct) const {
  reconstruct_fd_neighbor_work<prims_to_reconstruct_tags,
                               prims_to_reconstruct_tags>(
      vars_on_face,
      [](const auto tensor_component_on_face_ptr,
         const auto& tensor_component_volume,
         const auto& tensor_component_neighbor,
         const Index<dim>& subcell_extents,
         const Index<dim>& ghost_data_extents,
         const Direction<dim>& local_direction_to_reconstruct) {
        ::fd::reconstruction::reconstruct_neighbor<
            Side::Lower,
            ::fd::reconstruction::detail::MonotonisedCentralReconstructor>(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct);
      },
      [](const auto tensor_component_on_face_ptr,
         const auto& tensor_component_volume,
         const auto& tensor_component_neighbor,
         const Index<dim>& subcell_extents,
         const Index<dim>& ghost_data_extents,
         const Direction<dim>& local_direction_to_reconstruct) {
        ::fd::reconstruction::reconstruct_neighbor<
            Side::Upper,
            ::fd::reconstruction::detail::MonotonisedCentralReconstructor>(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct);
      },
      subcell_volume_prims, eos, element, ghost_data, subcell_mesh,
      direction_to_reconstruct, ghost_zone_size(), true, fix_to_atmosphere);
}

bool operator==(const MonotonisedCentralPrim& /*lhs*/,
                const MonotonisedCentralPrim& /*rhs*/) {
  return true;
}

bool operator!=(const MonotonisedCentralPrim& lhs,
                const MonotonisedCentralPrim& rhs) {
  return not(lhs == rhs);
}

#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                              \
  template void MonotonisedCentralPrim::reconstruct(                        \
      gsl::not_null<std::array<Variables<tags_list_for_reconstruct>, 3>*>   \
          vars_on_lower_face,                                               \
      gsl::not_null<std::array<Variables<tags_list_for_reconstruct>, 3>*>   \
          vars_on_upper_face,                                               \
      const Variables<hydro::grmhd_tags<DataVector>>& volume_prims,         \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>& eos, \
      const Element<3>& element,                                            \
      const DirectionalIdMap<3, evolution::dg::subcell::GhostData>&         \
          ghost_data,                                                       \
      const Mesh<3>& subcell_mesh,                                          \
      const VariableFixing::FixToAtmosphere<dim>& fix_to_atmosphere) const; \
  template void MonotonisedCentralPrim::reconstruct_fd_neighbor(            \
      gsl::not_null<Variables<tags_list_for_reconstruct>*> vars_on_face,    \
      const Variables<hydro::grmhd_tags<DataVector>>& subcell_volume_prims, \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>& eos, \
      const Element<3>& element,                                            \
      const DirectionalIdMap<3, evolution::dg::subcell::GhostData>&         \
          ghost_data,                                                       \
      const Mesh<3>& subcell_mesh,                                          \
      const VariableFixing::FixToAtmosphere<dim>& fix_to_atmosphere,        \
      const Direction<3> direction_to_reconstruct) const;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef THERMO_DIM
}  // namespace grmhd::ValenciaDivClean::fd
