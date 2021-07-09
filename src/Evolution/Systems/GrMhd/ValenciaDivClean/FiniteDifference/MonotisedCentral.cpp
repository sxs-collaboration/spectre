// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/MonotisedCentral.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <memory>
#include <pup.h>
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
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/ReconstructWork.tpp"
#include "NumericalAlgorithms/FiniteDifference/MonotisedCentral.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::fd {
MonotisedCentralPrim::MonotisedCentralPrim(CkMigrateMessage* const msg) noexcept
    : Reconstructor(msg) {}

std::unique_ptr<Reconstructor> MonotisedCentralPrim::get_clone()
    const noexcept {
  return std::make_unique<MonotisedCentralPrim>(*this);
}

void MonotisedCentralPrim::pup(PUP::er& p) { Reconstructor::pup(p); }

// NOLINTNEXTLINE
PUP::able::PUP_ID MonotisedCentralPrim::my_PUP_ID = 0;

template <size_t ThermodynamicDim, typename TagsList>
void MonotisedCentralPrim::reconstruct(
    const gsl::not_null<std::array<Variables<TagsList>, dim>*>
        vars_on_lower_face,
    const gsl::not_null<std::array<Variables<TagsList>, dim>*>
        vars_on_upper_face,
    const Variables<hydro::grmhd_tags<DataVector>>& volume_prims,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
    const Element<dim>& element,
    const FixedHashMap<maximum_number_of_neighbors(dim) + 1,
                       std::pair<Direction<dim>, ElementId<dim>>,
                       evolution::dg::subcell::NeighborData,
                       boost::hash<std::pair<Direction<dim>, ElementId<dim>>>>&
        neighbor_data,
    const Mesh<dim>& subcell_mesh) const noexcept {
  reconstruct_prims_work(
      vars_on_lower_face, vars_on_upper_face,
      [](auto upper_face_vars_ptr, auto lower_face_vars_ptr,
         const auto& volume_vars, const auto& ghost_cell_vars,
         const auto& subcell_extents, const size_t number_of_variables) {
        ::fd::reconstruction::monotised_central(
            upper_face_vars_ptr, lower_face_vars_ptr, volume_vars,
            ghost_cell_vars, subcell_extents, number_of_variables);
      },
      volume_prims, eos, element, neighbor_data, subcell_mesh,
      ghost_zone_size());
}

template <size_t ThermodynamicDim, typename TagsList>
void MonotisedCentralPrim::reconstruct_fd_neighbor(
    const gsl::not_null<Variables<TagsList>*> vars_on_face,
    const Variables<hydro::grmhd_tags<DataVector>>& subcell_volume_prims,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
    const Element<dim>& element,
    const FixedHashMap<maximum_number_of_neighbors(dim) + 1,
                       std::pair<Direction<dim>, ElementId<dim>>,
                       evolution::dg::subcell::NeighborData,
                       boost::hash<std::pair<Direction<dim>, ElementId<dim>>>>&
        neighbor_data,
    const Mesh<dim>& subcell_mesh,
    const Direction<dim> direction_to_reconstruct) const noexcept {
  reconstruct_fd_neighbor_work(
      vars_on_face,
      [](const auto tensor_component_on_face_ptr,
         const auto& tensor_component_volume,
         const auto& tensor_component_neighbor,
         const Index<dim>& subcell_extents,
         const Index<dim>& ghost_data_extents,
         const Direction<dim>& local_direction_to_reconstruct) noexcept {
        ::fd::reconstruction::reconstruct_neighbor<
            Side::Lower,
            ::fd::reconstruction::detail::MonotisedCentralReconstructor>(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct);
      },
      [](const auto tensor_component_on_face_ptr,
         const auto& tensor_component_volume,
         const auto& tensor_component_neighbor,
         const Index<dim>& subcell_extents,
         const Index<dim>& ghost_data_extents,
         const Direction<dim>& local_direction_to_reconstruct) noexcept {
        ::fd::reconstruction::reconstruct_neighbor<
            Side::Upper,
            ::fd::reconstruction::detail::MonotisedCentralReconstructor>(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct);
      },
      subcell_volume_prims, eos, element, neighbor_data, subcell_mesh,
      direction_to_reconstruct, ghost_zone_size());
}

bool operator==(const MonotisedCentralPrim& /*lhs*/,
                const MonotisedCentralPrim& /*rhs*/) noexcept {
  return true;
}

bool operator!=(const MonotisedCentralPrim& lhs,
                const MonotisedCentralPrim& rhs) noexcept {
  return not(lhs == rhs);
}

#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAGS_LIST(data)                                                      \
  tmpl::list<Tags::TildeD, Tags::TildeTau, Tags::TildeS<Frame::Inertial>,    \
             Tags::TildeB<Frame::Inertial>, Tags::TildePhi,                  \
             hydro::Tags::RestMassDensity<DataVector>,                       \
             hydro::Tags::SpecificInternalEnergy<DataVector>,                \
             hydro::Tags::SpatialVelocity<DataVector, 3>,                    \
             hydro::Tags::MagneticField<DataVector, 3>,                      \
             hydro::Tags::DivergenceCleaningField<DataVector>,               \
             hydro::Tags::LorentzFactor<DataVector>,                         \
             hydro::Tags::Pressure<DataVector>,                              \
             hydro::Tags::SpecificEnthalpy<DataVector>,                      \
             hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>,  \
             ::Tags::Flux<Tags::TildeD, tmpl::size_t<3>, Frame::Inertial>,   \
             ::Tags::Flux<Tags::TildeTau, tmpl::size_t<3>, Frame::Inertial>, \
             ::Tags::Flux<Tags::TildeS<Frame::Inertial>, tmpl::size_t<3>,    \
                          Frame::Inertial>,                                  \
             ::Tags::Flux<Tags::TildeB<Frame::Inertial>, tmpl::size_t<3>,    \
                          Frame::Inertial>,                                  \
             ::Tags::Flux<Tags::TildePhi, tmpl::size_t<3>, Frame::Inertial>, \
             gr::Tags::Lapse<DataVector>,                                    \
             gr::Tags::Shift<3, Frame::Inertial, DataVector>,                \
             gr::Tags::SpatialMetric<3>,                                     \
             gr::Tags::SqrtDetSpatialMetric<DataVector>,                     \
             gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>, \
             evolution::dg::Actions::detail::NormalVector<3>>

#define INSTANTIATION(r, data)                                                \
  template void MonotisedCentralPrim::reconstruct(                            \
      gsl::not_null<std::array<Variables<TAGS_LIST(data)>, 3>*>               \
          vars_on_lower_face,                                                 \
      gsl::not_null<std::array<Variables<TAGS_LIST(data)>, 3>*>               \
          vars_on_upper_face,                                                 \
      const Variables<hydro::grmhd_tags<DataVector>>& volume_prims,           \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>& eos,   \
      const Element<3>& element,                                              \
      const FixedHashMap<maximum_number_of_neighbors(3) + 1,                  \
                         std::pair<Direction<3>, ElementId<3>>,               \
                         evolution::dg::subcell::NeighborData,                \
                         boost::hash<std::pair<Direction<3>, ElementId<3>>>>& \
          neighbor_data,                                                      \
      const Mesh<3>& subcell_mesh) const noexcept;                            \
  template void MonotisedCentralPrim::reconstruct_fd_neighbor(                \
      gsl::not_null<Variables<TAGS_LIST(data)>*> vars_on_face,                \
      const Variables<hydro::grmhd_tags<DataVector>>& subcell_volume_prims,   \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>& eos,   \
      const Element<3>& element,                                              \
      const FixedHashMap<maximum_number_of_neighbors(3) + 1,                  \
                         std::pair<Direction<3>, ElementId<3>>,               \
                         evolution::dg::subcell::NeighborData,                \
                         boost::hash<std::pair<Direction<3>, ElementId<3>>>>& \
          neighbor_data,                                                      \
      const Mesh<3>& subcell_mesh,                                            \
      const Direction<3> direction_to_reconstruct) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2))

#undef INSTANTIATION
#undef TAGS_LIST
#undef THERMO_DIM
#undef DIM
}  // namespace grmhd::ValenciaDivClean::fd
