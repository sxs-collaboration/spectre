// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarAdvection/FiniteDifference/MonotisedCentral.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <utility>

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
#include "Evolution/DgSubcell/NeighborData.hpp"
#include "Evolution/Systems/ScalarAdvection/FiniteDifference/ReconstructWork.tpp"
#include "Evolution/Systems/ScalarAdvection/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "NumericalAlgorithms/FiniteDifference/MonotisedCentral.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarAdvection::fd {
template <size_t Dim>
MonotisedCentral<Dim>::MonotisedCentral(CkMigrateMessage* const msg)
    : Reconstructor<Dim>(msg) {}

template <size_t Dim>
std::unique_ptr<Reconstructor<Dim>> MonotisedCentral<Dim>::get_clone() const {
  return std::make_unique<MonotisedCentral>(*this);
}

template <size_t Dim>
void MonotisedCentral<Dim>::pup(PUP::er& p) {
  Reconstructor<Dim>::pup(p);
}

template <size_t Dim>
PUP::able::PUP_ID MonotisedCentral<Dim>::my_PUP_ID = 0;

template <size_t Dim>
template <typename TagsList>
void MonotisedCentral<Dim>::reconstruct(
    const gsl::not_null<std::array<Variables<TagsList>, Dim>*>
        vars_on_lower_face,
    const gsl::not_null<std::array<Variables<TagsList>, Dim>*>
        vars_on_upper_face,
    const Variables<tmpl::list<Tags::U>>& volume_vars,
    const Element<Dim>& element,
    const FixedHashMap<maximum_number_of_neighbors(Dim) + 1,
                       std::pair<Direction<Dim>, ElementId<Dim>>,
                       evolution::dg::subcell::NeighborData,
                       boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>&
        neighbor_data,
    const Mesh<Dim>& subcell_mesh) const {
  reconstruct_work(
      vars_on_lower_face, vars_on_upper_face,
      [](auto upper_face_vars_ptr, auto lower_face_vars_ptr,
         const auto& volume_variables, const auto& ghost_cell_vars,
         const auto& subcell_extents, const size_t number_of_variables) {
        ::fd::reconstruction::monotised_central(
            upper_face_vars_ptr, lower_face_vars_ptr, volume_variables,
            ghost_cell_vars, subcell_extents, number_of_variables);
      },
      volume_vars, element, neighbor_data, subcell_mesh, ghost_zone_size());
}

template <size_t Dim>
template <typename TagsList>
void MonotisedCentral<Dim>::reconstruct_fd_neighbor(
    const gsl::not_null<Variables<TagsList>*> vars_on_face,
    const Variables<tmpl::list<Tags::U>>& volume_vars,
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
      [](const auto tensor_component_on_face_ptr,
         const auto& tensor_component_volume,
         const auto& tensor_component_neighbor, const auto& subcell_extents,
         const auto& ghost_data_extents,
         const auto& local_direction_to_reconstruct) {
        ::fd::reconstruction::reconstruct_neighbor<
            Side::Lower,
            ::fd::reconstruction::detail::MonotisedCentralReconstructor>(
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
            ::fd::reconstruction::detail::MonotisedCentralReconstructor>(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct);
      },
      volume_vars, element, neighbor_data, subcell_mesh,
      direction_to_reconstruct, ghost_zone_size());
}

template <size_t Dim>
bool operator==(const MonotisedCentral<Dim>& /*lhs*/,
                const MonotisedCentral<Dim>& /*rhs*/) {
  return true;
}

template <size_t Dim>
bool operator!=(const MonotisedCentral<Dim>& lhs,
                const MonotisedCentral<Dim>& rhs) {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAGS_LIST(data)                                                       \
  tmpl::list<Tags::U,                                                         \
             ::Tags::Flux<Tags::U, tmpl::size_t<DIM(data)>, Frame::Inertial>, \
             Tags::VelocityField<DIM(data)>>

#define INSTANTIATION(r, data)                                                \
  template class MonotisedCentral<DIM(data)>;                                 \
  template bool operator==                                                    \
      <DIM(data)>(const MonotisedCentral<DIM(data)>& /*lhs*/,                 \
                  const MonotisedCentral<DIM(data)>& /*rhs*/);                \
  template bool operator!=<DIM(data)>(const MonotisedCentral<DIM(data)>& lhs, \
                                      const MonotisedCentral<DIM(data)>& rhs);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2))
#undef INSTANTIATION

#define INSTANTIATION(r, data)                                                 \
  template void MonotisedCentral<DIM(data)>::reconstruct(                      \
      gsl::not_null<std::array<Variables<TAGS_LIST(data)>, DIM(data)>*>        \
          vars_on_lower_face,                                                  \
      gsl::not_null<std::array<Variables<TAGS_LIST(data)>, DIM(data)>*>        \
          vars_on_upper_face,                                                  \
      const Variables<tmpl::list<Tags::U>>& volume_vars,                       \
      const Element<DIM(data)>& element,                                       \
      const FixedHashMap<                                                      \
          maximum_number_of_neighbors(DIM(data)) + 1,                          \
          std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>,               \
          evolution::dg::subcell::NeighborData,                                \
          boost::hash<std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>>>& \
          neighbor_data,                                                       \
      const Mesh<DIM(data)>& subcell_mesh) const;                              \
  template void MonotisedCentral<DIM(data)>::reconstruct_fd_neighbor(          \
      gsl::not_null<Variables<TAGS_LIST(data)>*> vars_on_face,                 \
      const Variables<tmpl::list<Tags::U>>& volume_vars,                       \
      const Element<DIM(data)>& element,                                       \
      const FixedHashMap<                                                      \
          maximum_number_of_neighbors(DIM(data)) + 1,                          \
          std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>,               \
          evolution::dg::subcell::NeighborData,                                \
          boost::hash<std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>>>& \
          neighbor_data,                                                       \
      const Mesh<DIM(data)>& subcell_mesh,                                     \
      const Direction<DIM(data)> direction_to_reconstruct) const;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2))

#undef INSTANTIATION
#undef TAGS_LIST
#undef DIM
}  // namespace ScalarAdvection::fd
