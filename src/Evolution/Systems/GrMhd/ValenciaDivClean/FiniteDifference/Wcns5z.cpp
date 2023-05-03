// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Wcns5z.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/ReconstructWork.tpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"
#include "NumericalAlgorithms/FiniteDifference/NeighborDataAsVariables.hpp"
#include "NumericalAlgorithms/FiniteDifference/Reconstruct.tpp"
#include "NumericalAlgorithms/FiniteDifference/Wcns5z.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean::fd {

Wcns5zPrim::Wcns5zPrim(const size_t nonlinear_weight_exponent,
                       const double epsilon,
                       const ::fd::reconstruction::FallbackReconstructorType
                           fallback_reconstructor,
                       const size_t max_number_of_extrema)
    : nonlinear_weight_exponent_(nonlinear_weight_exponent),
      epsilon_(epsilon),
      fallback_reconstructor_(fallback_reconstructor),
      max_number_of_extrema_(max_number_of_extrema) {
  std::tie(reconstruct_, reconstruct_lower_neighbor_,
           reconstruct_upper_neighbor_) =
      ::fd::reconstruction::wcns5z_function_pointers<3>(
          nonlinear_weight_exponent_, fallback_reconstructor_);
}

Wcns5zPrim::Wcns5zPrim(CkMigrateMessage* const msg) : Reconstructor(msg) {}

std::unique_ptr<Reconstructor> Wcns5zPrim::get_clone() const {
  return std::make_unique<Wcns5zPrim>(*this);
}

void Wcns5zPrim::pup(PUP::er& p) {
  Reconstructor::pup(p);
  p | nonlinear_weight_exponent_;
  p | epsilon_;
  p | fallback_reconstructor_;
  p | max_number_of_extrema_;
  if (p.isUnpacking()) {
    std::tie(reconstruct_, reconstruct_lower_neighbor_,
             reconstruct_upper_neighbor_) =
        ::fd::reconstruction::wcns5z_function_pointers<3>(
            nonlinear_weight_exponent_, fallback_reconstructor_);
  }
}

// NOLINTNEXTLINE
PUP::able::PUP_ID Wcns5zPrim::my_PUP_ID = 0;

template <size_t ThermodynamicDim, typename TagsList>
void Wcns5zPrim::reconstruct(
    const gsl::not_null<std::array<Variables<TagsList>, 3>*> vars_on_lower_face,
    const gsl::not_null<std::array<Variables<TagsList>, 3>*> vars_on_upper_face,
    const Variables<hydro::grmhd_tags<DataVector>>& volume_prims,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
    const Element<3>& element,
    const FixedHashMap<
        maximum_number_of_neighbors(3), std::pair<Direction<3>, ElementId<3>>,
        evolution::dg::subcell::GhostData,
        boost::hash<std::pair<Direction<3>, ElementId<3>>>>& ghost_data,
    const Mesh<3>& subcell_mesh) const {
  FixedHashMap<maximum_number_of_neighbors(dim),
               std::pair<Direction<dim>, ElementId<dim>>,
               Variables<prims_to_reconstruct_tags>,
               boost::hash<std::pair<Direction<dim>, ElementId<dim>>>>
      neighbor_variables_data{};
  ::fd::neighbor_data_as_variables<dim>(make_not_null(&neighbor_variables_data),
                                        ghost_data, ghost_zone_size(),
                                        subcell_mesh);

  reconstruct_prims_work<prims_to_reconstruct_tags>(
      vars_on_lower_face, vars_on_upper_face,
      [this](auto upper_face_vars_ptr, auto lower_face_vars_ptr,
             const auto& volume_vars, const auto& ghost_cell_vars,
             const auto& subcell_extents, const size_t number_of_variables) {
        reconstruct_(upper_face_vars_ptr, lower_face_vars_ptr, volume_vars,
                     ghost_cell_vars, subcell_extents, number_of_variables,
                     epsilon_, max_number_of_extrema_);
      },
      volume_prims, eos, element, neighbor_variables_data, subcell_mesh,
      ghost_zone_size(), true);
}

template <size_t ThermodynamicDim, typename TagsList>
void Wcns5zPrim::reconstruct_fd_neighbor(
    const gsl::not_null<Variables<TagsList>*> vars_on_face,
    const Variables<hydro::grmhd_tags<DataVector>>& subcell_volume_prims,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
    const Element<3>& element,
    const FixedHashMap<
        maximum_number_of_neighbors(3), std::pair<Direction<3>, ElementId<3>>,
        evolution::dg::subcell::GhostData,
        boost::hash<std::pair<Direction<3>, ElementId<3>>>>& ghost_data,
    const Mesh<3>& subcell_mesh,
    const Direction<3> direction_to_reconstruct) const {
  reconstruct_fd_neighbor_work<prims_to_reconstruct_tags,
                               prims_to_reconstruct_tags>(
      vars_on_face,
      [this](const auto tensor_component_on_face_ptr,
             const auto& tensor_component_volume,
             const auto& tensor_component_neighbor,
             const Index<3>& subcell_extents,
             const Index<3>& ghost_data_extents,
             const Direction<3>& local_direction_to_reconstruct) {
        reconstruct_lower_neighbor_(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct, epsilon_, max_number_of_extrema_);
      },
      [this](const auto tensor_component_on_face_ptr,
             const auto& tensor_component_volume,
             const auto& tensor_component_neighbor,
             const Index<3>& subcell_extents,
             const Index<3>& ghost_data_extents,
             const Direction<3>& local_direction_to_reconstruct) {
        reconstruct_upper_neighbor_(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct, epsilon_, max_number_of_extrema_);
      },
      subcell_volume_prims, eos, element, ghost_data, subcell_mesh,
      direction_to_reconstruct, ghost_zone_size(), true);
}

bool operator==(const Wcns5zPrim& lhs, const Wcns5zPrim& rhs) {
  // Don't check function pointers since they are set from
  // nonlinear_weight_exponent_ and fallback_reconstructor_
  return lhs.nonlinear_weight_exponent_ == rhs.nonlinear_weight_exponent_ and
         lhs.epsilon_ == rhs.epsilon_ and
         lhs.fallback_reconstructor_ == rhs.fallback_reconstructor_ and
         lhs.max_number_of_extrema_ == rhs.max_number_of_extrema_;
}

bool operator!=(const Wcns5zPrim& lhs, const Wcns5zPrim& rhs) {
  return not(lhs == rhs);
}

#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAGS_LIST(data)                                                      \
  tmpl::list<Tags::TildeD, Tags::TildeYe, Tags::TildeTau,                    \
             Tags::TildeS<Frame::Inertial>, Tags::TildeB<Frame::Inertial>,   \
             Tags::TildePhi, hydro::Tags::RestMassDensity<DataVector>,       \
             hydro::Tags::ElectronFraction<DataVector>,                      \
             hydro::Tags::SpecificInternalEnergy<DataVector>,                \
             hydro::Tags::SpatialVelocity<DataVector, 3>,                    \
             hydro::Tags::MagneticField<DataVector, 3>,                      \
             hydro::Tags::DivergenceCleaningField<DataVector>,               \
             hydro::Tags::LorentzFactor<DataVector>,                         \
             hydro::Tags::Pressure<DataVector>,                              \
             hydro::Tags::SpecificEnthalpy<DataVector>,                      \
             hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>,  \
             ::Tags::Flux<Tags::TildeD, tmpl::size_t<3>, Frame::Inertial>,   \
             ::Tags::Flux<Tags::TildeYe, tmpl::size_t<3>, Frame::Inertial>,  \
             ::Tags::Flux<Tags::TildeTau, tmpl::size_t<3>, Frame::Inertial>, \
             ::Tags::Flux<Tags::TildeS<Frame::Inertial>, tmpl::size_t<3>,    \
                          Frame::Inertial>,                                  \
             ::Tags::Flux<Tags::TildeB<Frame::Inertial>, tmpl::size_t<3>,    \
                          Frame::Inertial>,                                  \
             ::Tags::Flux<Tags::TildePhi, tmpl::size_t<3>, Frame::Inertial>, \
             gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,    \
             gr::Tags::SpatialMetric<DataVector, 3>,                         \
             gr::Tags::SqrtDetSpatialMetric<DataVector>,                     \
             gr::Tags::InverseSpatialMetric<DataVector, 3>,                  \
             evolution::dg::Actions::detail::NormalVector<3>>

#define INSTANTIATION(r, data)                                                \
  template void Wcns5zPrim::reconstruct(                                      \
      gsl::not_null<std::array<Variables<TAGS_LIST(data)>, 3>*>               \
          vars_on_lower_face,                                                 \
      gsl::not_null<std::array<Variables<TAGS_LIST(data)>, 3>*>               \
          vars_on_upper_face,                                                 \
      const Variables<hydro::grmhd_tags<DataVector>>& volume_prims,           \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>& eos,   \
      const Element<3>& element,                                              \
      const FixedHashMap<maximum_number_of_neighbors(3),                      \
                         std::pair<Direction<3>, ElementId<3>>,               \
                         evolution::dg::subcell::GhostData,                   \
                         boost::hash<std::pair<Direction<3>, ElementId<3>>>>& \
          ghost_data,                                                         \
      const Mesh<3>& subcell_mesh) const;                                     \
  template void Wcns5zPrim::reconstruct_fd_neighbor(                          \
      gsl::not_null<Variables<TAGS_LIST(data)>*> vars_on_face,                \
      const Variables<hydro::grmhd_tags<DataVector>>& subcell_volume_prims,   \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>& eos,   \
      const Element<3>& element,                                              \
      const FixedHashMap<maximum_number_of_neighbors(3),                      \
                         std::pair<Direction<3>, ElementId<3>>,               \
                         evolution::dg::subcell::GhostData,                   \
                         boost::hash<std::pair<Direction<3>, ElementId<3>>>>& \
          ghost_data,                                                         \
      const Mesh<3>& subcell_mesh,                                            \
      const Direction<3> direction_to_reconstruct) const;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2))

#undef INSTANTIATION
#undef TAGS_LIST
#undef THERMO_DIM

}  // namespace grmhd::ValenciaDivClean::fd
