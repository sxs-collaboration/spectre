// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/FiniteDifference/Wcns5z.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <tuple>
#include <utility>

#include "Evolution/Systems/ForceFree/FiniteDifference/ReconstructWork.tpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"
#include "NumericalAlgorithms/FiniteDifference/NeighborDataAsVariables.hpp"
#include "NumericalAlgorithms/FiniteDifference/Reconstruct.tpp"
#include "NumericalAlgorithms/FiniteDifference/Wcns5z.hpp"

namespace ForceFree::fd {

Wcns5z::Wcns5z(const size_t nonlinear_weight_exponent, const double epsilon,
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

Wcns5z::Wcns5z(CkMigrateMessage* const msg) : Reconstructor(msg) {}

std::unique_ptr<Reconstructor> Wcns5z::get_clone() const {
  return std::make_unique<Wcns5z>(*this);
}

void Wcns5z::pup(PUP::er& p) {
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
PUP::able::PUP_ID Wcns5z::my_PUP_ID = 0;

void Wcns5z::reconstruct(
    const gsl::not_null<std::array<Variables<recons_tags>, dim>*>
        vars_on_lower_face,
    const gsl::not_null<std::array<Variables<recons_tags>, dim>*>
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
      [this](auto upper_face_vars_ptr, auto lower_face_vars_ptr,
             const auto& volume_variables, const auto& ghost_cell_vars,
             const auto& subcell_extents, const size_t number_of_variables) {
        reconstruct_(upper_face_vars_ptr, lower_face_vars_ptr, volume_variables,
                     ghost_cell_vars, subcell_extents, number_of_variables,
                     epsilon_, max_number_of_extrema_);
      },
      volume_vars, tilde_j, element, ghost_data, subcell_mesh,
      ghost_zone_size());
}

void Wcns5z::reconstruct_fd_neighbor(
    const gsl::not_null<Variables<recons_tags>*> vars_on_face,
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
      [this](const auto tensor_component_on_face_ptr,
             const auto& tensor_component_volume,
             const auto& tensor_component_neighbor, const auto& subcell_extents,
             const auto& ghost_data_extents,
             const auto& local_direction_to_reconstruct) {
        reconstruct_lower_neighbor_(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct, epsilon_, max_number_of_extrema_);
      },
      [this](const auto tensor_component_on_face_ptr,
             const auto& tensor_component_volume,
             const auto& tensor_component_neighbor, const auto& subcell_extents,
             const auto& ghost_data_extents,
             const auto& local_direction_to_reconstruct) {
        reconstruct_upper_neighbor_(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct, epsilon_, max_number_of_extrema_);
      },
      volume_vars, tilde_j, element, ghost_data, subcell_mesh,
      direction_to_reconstruct, ghost_zone_size());
}

bool operator==(const Wcns5z& lhs, const Wcns5z& rhs) {
  // Don't check function pointers since they are set from
  // nonlinear_weight_exponent_ and fallback_reconstructor_
  return lhs.nonlinear_weight_exponent_ == rhs.nonlinear_weight_exponent_ and
         lhs.epsilon_ == rhs.epsilon_ and
         lhs.fallback_reconstructor_ == rhs.fallback_reconstructor_ and
         lhs.max_number_of_extrema_ == rhs.max_number_of_extrema_;
}

bool operator!=(const Wcns5z& lhs, const Wcns5z& rhs) {
  return not(lhs == rhs);
}

}  // namespace ForceFree::fd
