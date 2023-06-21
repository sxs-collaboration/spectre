// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/FiniteDifference/AdaptiveOrder.hpp"

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
#include "NumericalAlgorithms/FiniteDifference/PositivityPreservingAdaptiveOrder.hpp"
#include "NumericalAlgorithms/FiniteDifference/Reconstruct.tpp"
#include "Options/ParseError.hpp"

namespace ForceFree::fd {

AdaptiveOrder::AdaptiveOrder(
    const double alpha_5, const std::optional<double> alpha_7,
    const std::optional<double> alpha_9,
    const ::fd::reconstruction::FallbackReconstructorType
        low_order_reconstructor,
    const Options::Context& context)
    : four_to_the_alpha_5_(pow(4.0, alpha_5)),
      low_order_reconstructor_(low_order_reconstructor) {
  if (low_order_reconstructor_ ==
      ::fd::reconstruction::FallbackReconstructorType::None) {
    PARSE_ERROR(context, "None is not an allowed low-order reconstructor.");
  }
  if (alpha_7.has_value()) {
    six_to_the_alpha_7_ = pow(6.0, alpha_7.value());
  }
  if (alpha_9.has_value()) {
    eight_to_the_alpha_9_ = pow(8.0, alpha_9.value());
  }
  set_function_pointers();
}

AdaptiveOrder::AdaptiveOrder(CkMigrateMessage* const msg)
    : Reconstructor(msg) {}

std::unique_ptr<Reconstructor> AdaptiveOrder::get_clone() const {
  return std::make_unique<AdaptiveOrder>(*this);
}

void AdaptiveOrder::set_function_pointers() {
  std::tie(reconstruct_, reconstruct_lower_neighbor_,
           reconstruct_upper_neighbor_) = ::fd::reconstruction::
      positivity_preserving_adaptive_order_function_pointers<3, false>(
          false, eight_to_the_alpha_9_.has_value(),
          six_to_the_alpha_7_.has_value(), low_order_reconstructor_);
  std::tie(pp_reconstruct_, pp_reconstruct_lower_neighbor_,
           pp_reconstruct_upper_neighbor_) = ::fd::reconstruction::
      positivity_preserving_adaptive_order_function_pointers<3, true>(
          true, eight_to_the_alpha_9_.has_value(),
          six_to_the_alpha_7_.has_value(), low_order_reconstructor_);
}

void AdaptiveOrder::pup(PUP::er& p) {
  Reconstructor::pup(p);
  p | four_to_the_alpha_5_;
  p | six_to_the_alpha_7_;
  p | eight_to_the_alpha_9_;
  p | low_order_reconstructor_;
  if (p.isUnpacking()) {
    set_function_pointers();
  }
}

// NOLINTNEXTLINE
PUP::able::PUP_ID AdaptiveOrder::my_PUP_ID = 0;

void AdaptiveOrder::reconstruct(
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
      [this](auto upper_face_vars_ptr, auto lower_face_vars_ptr,
             const auto& volume_variables, const auto& ghost_cell_vars,
             const auto& subcell_extents, const size_t number_of_variables) {
        reconstruct_(upper_face_vars_ptr, lower_face_vars_ptr, volume_variables,
                     ghost_cell_vars, subcell_extents, number_of_variables,
                     four_to_the_alpha_5_,
                     six_to_the_alpha_7_.value_or(
                         std::numeric_limits<double>::signaling_NaN()),
                     eight_to_the_alpha_9_.value_or(
                         std::numeric_limits<double>::signaling_NaN()));
      },
      volume_vars, tilde_j, element, ghost_data, subcell_mesh,
      ghost_zone_size());
}

void AdaptiveOrder::reconstruct_fd_neighbor(
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
      [this](const auto tensor_component_on_face_ptr,
             const auto& tensor_component_volume,
             const auto& tensor_component_neighbor, const auto& subcell_extents,
             const auto& ghost_data_extents,
             const auto& local_direction_to_reconstruct) {
        reconstruct_lower_neighbor_(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct, four_to_the_alpha_5_,
            six_to_the_alpha_7_.value_or(
                std::numeric_limits<double>::signaling_NaN()),
            eight_to_the_alpha_9_.value_or(
                std::numeric_limits<double>::signaling_NaN()));
      },
      [this](const auto tensor_component_on_face_ptr,
             const auto& tensor_component_volume,
             const auto& tensor_component_neighbor, const auto& subcell_extents,
             const auto& ghost_data_extents,
             const auto& local_direction_to_reconstruct) {
        reconstruct_upper_neighbor_(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct, four_to_the_alpha_5_,
            six_to_the_alpha_7_.value_or(
                std::numeric_limits<double>::signaling_NaN()),
            eight_to_the_alpha_9_.value_or(
                std::numeric_limits<double>::signaling_NaN()));
      },
      volume_vars, tilde_j, element, ghost_data, subcell_mesh,
      direction_to_reconstruct, ghost_zone_size());
}

bool operator==(const AdaptiveOrder& lhs, const AdaptiveOrder& rhs) {
  // Don't check function pointers since they are set from
  // low_order_reconstructor_
  return lhs.four_to_the_alpha_5_ == rhs.four_to_the_alpha_5_ and
         lhs.six_to_the_alpha_7_ == rhs.six_to_the_alpha_7_ and
         lhs.eight_to_the_alpha_9_ == rhs.eight_to_the_alpha_9_ and
         lhs.low_order_reconstructor_ == rhs.low_order_reconstructor_;
}

bool operator!=(const AdaptiveOrder& lhs, const AdaptiveOrder& rhs) {
  return not(lhs == rhs);
}

}  // namespace ForceFree::fd
