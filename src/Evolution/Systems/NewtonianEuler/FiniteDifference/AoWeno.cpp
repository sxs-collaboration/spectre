// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/FiniteDifference/AoWeno.hpp"

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
#include "Evolution/Systems/NewtonianEuler/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/ReconstructWork.tpp"
#include "NumericalAlgorithms/FiniteDifference/AoWeno.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler::fd {
template <size_t Dim>
AoWeno53Prim<Dim>::AoWeno53Prim(const double gamma_hi, const double gamma_lo,
                                const double epsilon,
                                const size_t nonlinear_weight_exponent)
    : gamma_hi_(gamma_hi),
      gamma_lo_(gamma_lo),
      epsilon_(epsilon),
      nonlinear_weight_exponent_(nonlinear_weight_exponent) {
  std::tie(reconstruct_, reconstruct_lower_neighbor_,
           reconstruct_upper_neighbor_) =
      ::fd::reconstruction::aoweno_53_function_pointers<Dim>(
          nonlinear_weight_exponent_);
}

template <size_t Dim>
AoWeno53Prim<Dim>::AoWeno53Prim(CkMigrateMessage* const msg)
    : Reconstructor<Dim>(msg) {}

template <size_t Dim>
std::unique_ptr<Reconstructor<Dim>> AoWeno53Prim<Dim>::get_clone() const {
  return std::make_unique<AoWeno53Prim>(*this);
}

template <size_t Dim>
void AoWeno53Prim<Dim>::pup(PUP::er& p) {
  Reconstructor<Dim>::pup(p);
  p | gamma_hi_;
  p | gamma_lo_;
  p | epsilon_;
  p | nonlinear_weight_exponent_;
  if (p.isUnpacking()) {
    std::tie(reconstruct_, reconstruct_lower_neighbor_,
             reconstruct_upper_neighbor_) =
        ::fd::reconstruction::aoweno_53_function_pointers<Dim>(
            nonlinear_weight_exponent_);
  }
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID AoWeno53Prim<Dim>::my_PUP_ID = 0;

template <size_t Dim>
template <size_t ThermodynamicDim, typename TagsList>
void AoWeno53Prim<Dim>::reconstruct(
    const gsl::not_null<std::array<Variables<TagsList>, Dim>*>
        vars_on_lower_face,
    const gsl::not_null<std::array<Variables<TagsList>, Dim>*>
        vars_on_upper_face,
    const Variables<prims_tags>& volume_prims,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>& eos,
    const Element<Dim>& element,
    const FixedHashMap<maximum_number_of_neighbors(Dim) + 1,
                       std::pair<Direction<Dim>, ElementId<Dim>>,
                       evolution::dg::subcell::NeighborData,
                       boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>&
        neighbor_data,
    const Mesh<Dim>& subcell_mesh) const {
  reconstruct_prims_work(
      vars_on_lower_face, vars_on_upper_face,
      [this](auto upper_face_vars_ptr, auto lower_face_vars_ptr,
             const auto& volume_vars, const auto& ghost_cell_vars,
             const auto& subcell_extents, const size_t number_of_variables) {
        reconstruct_(upper_face_vars_ptr, lower_face_vars_ptr, volume_vars,
                     ghost_cell_vars, subcell_extents, number_of_variables,
                     gamma_hi_, gamma_lo_, epsilon_);
      },
      volume_prims, eos, element, neighbor_data, subcell_mesh,
      ghost_zone_size());
}

template <size_t Dim>
template <size_t ThermodynamicDim, typename TagsList>
void AoWeno53Prim<Dim>::reconstruct_fd_neighbor(
    const gsl::not_null<Variables<TagsList>*> vars_on_face,
    const Variables<prims_tags>& subcell_volume_prims,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>& eos,
    const Element<Dim>& element,
    const FixedHashMap<maximum_number_of_neighbors(Dim) + 1,
                       std::pair<Direction<Dim>, ElementId<Dim>>,
                       evolution::dg::subcell::NeighborData,
                       boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>&
        neighbor_data,
    const Mesh<Dim>& subcell_mesh,
    const Direction<Dim> direction_to_reconstruct) const {
  reconstruct_fd_neighbor_work(
      vars_on_face,
      [this](const auto tensor_component_on_face_ptr,
             const auto& tensor_component_volume,
             const auto& tensor_component_neighbor,
             const Index<Dim>& subcell_extents,
             const Index<Dim>& ghost_data_extents,
             const Direction<Dim>& local_direction_to_reconstruct) {
        reconstruct_lower_neighbor_(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct, gamma_hi_, gamma_lo_, epsilon_);
      },
      [this](const auto tensor_component_on_face_ptr,
             const auto& tensor_component_volume,
             const auto& tensor_component_neighbor,
             const Index<Dim>& subcell_extents,
             const Index<Dim>& ghost_data_extents,
             const Direction<Dim>& local_direction_to_reconstruct) {
        reconstruct_upper_neighbor_(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct, gamma_hi_, gamma_lo_, epsilon_);
      },
      subcell_volume_prims, eos, element, neighbor_data, subcell_mesh,
      direction_to_reconstruct, ghost_zone_size());
}

template <size_t Dim>
bool operator==(const AoWeno53Prim<Dim>& lhs, const AoWeno53Prim<Dim>& rhs) {
  // Don't check function pointers since they are set from
  // nonlinear_weight_exponent_
  return lhs.gamma_hi_ == rhs.gamma_hi_ and lhs.gamma_lo_ == rhs.gamma_lo_ and
         lhs.epsilon_ == rhs.epsilon_ and
         lhs.nonlinear_weight_exponent_ == rhs.nonlinear_weight_exponent_;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define TAGS_LIST(data)                                                   \
  tmpl::list<Tags::MassDensityCons, Tags::MomentumDensity<DIM(data)>,     \
             Tags::EnergyDensity, Tags::MassDensity<DataVector>,          \
             Tags::Velocity<DataVector, DIM(data)>,                       \
             Tags::SpecificInternalEnergy<DataVector>,                    \
             Tags::Pressure<DataVector>,                                  \
             ::Tags::Flux<Tags::MassDensityCons, tmpl::size_t<DIM(data)>, \
                          Frame::Inertial>,                               \
             ::Tags::Flux<Tags::MomentumDensity<DIM(data)>,               \
                          tmpl::size_t<DIM(data)>, Frame::Inertial>,      \
             ::Tags::Flux<Tags::EnergyDensity, tmpl::size_t<DIM(data)>,   \
                          Frame::Inertial>>

#define INSTANTIATION(r, data)                                 \
  template class AoWeno53Prim<DIM(data)>;                      \
  template bool operator==(const AoWeno53Prim<DIM(data)>& lhs, \
                           const AoWeno53Prim<DIM(data)>& rhs);
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))
#undef INSTANTIATION

#define INSTANTIATION(r, data)                                                 \
  template void AoWeno53Prim<DIM(data)>::reconstruct(                          \
      gsl::not_null<std::array<Variables<TAGS_LIST(data)>, DIM(data)>*>        \
          vars_on_lower_face,                                                  \
      gsl::not_null<std::array<Variables<TAGS_LIST(data)>, DIM(data)>*>        \
          vars_on_upper_face,                                                  \
      const Variables<prims_tags>& volume_prims,                               \
      const EquationsOfState::EquationOfState<false, THERMO_DIM(data)>& eos,   \
      const Element<DIM(data)>& element,                                       \
      const FixedHashMap<                                                      \
          maximum_number_of_neighbors(DIM(data)) + 1,                          \
          std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>,               \
          evolution::dg::subcell::NeighborData,                                \
          boost::hash<std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>>>& \
          neighbor_data,                                                       \
      const Mesh<DIM(data)>& subcell_mesh) const;                              \
  template void AoWeno53Prim<DIM(data)>::reconstruct_fd_neighbor(              \
      gsl::not_null<Variables<TAGS_LIST(data)>*> vars_on_face,                 \
      const Variables<prims_tags>& subcell_volume_prims,                       \
      const EquationsOfState::EquationOfState<false, THERMO_DIM(data)>& eos,   \
      const Element<DIM(data)>& element,                                       \
      const FixedHashMap<                                                      \
          maximum_number_of_neighbors(DIM(data)) + 1,                          \
          std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>,               \
          evolution::dg::subcell::NeighborData,                                \
          boost::hash<std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>>>& \
          neighbor_data,                                                       \
      const Mesh<DIM(data)>& subcell_mesh,                                     \
      const Direction<DIM(data)> direction_to_reconstruct) const;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (1, 2))

#undef INSTANTIATION
#undef TAGS_LIST
#undef THERMO_DIM
#undef DIM
}  // namespace NewtonianEuler::fd
