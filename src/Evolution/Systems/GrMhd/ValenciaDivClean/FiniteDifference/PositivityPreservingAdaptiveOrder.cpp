// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/PositivityPreservingAdaptiveOrder.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/ReconstructWork.tpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"
#include "NumericalAlgorithms/FiniteDifference/NeighborDataAsVariables.hpp"
#include "NumericalAlgorithms/FiniteDifference/PositivityPreservingAdaptiveOrder.hpp"
#include "NumericalAlgorithms/FiniteDifference/Reconstruct.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/ParseError.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean::fd {

PositivityPreservingAdaptiveOrderPrim::PositivityPreservingAdaptiveOrderPrim(
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

PositivityPreservingAdaptiveOrderPrim::PositivityPreservingAdaptiveOrderPrim(
    CkMigrateMessage* const msg)
    : Reconstructor(msg) {}

std::unique_ptr<Reconstructor>
PositivityPreservingAdaptiveOrderPrim::get_clone() const {
  return std::make_unique<PositivityPreservingAdaptiveOrderPrim>(*this);
}

void PositivityPreservingAdaptiveOrderPrim::set_function_pointers() {
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

void PositivityPreservingAdaptiveOrderPrim::pup(PUP::er& p) {
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
PUP::able::PUP_ID PositivityPreservingAdaptiveOrderPrim::my_PUP_ID = 0;

template <size_t ThermodynamicDim>
void PositivityPreservingAdaptiveOrderPrim::reconstruct(
    const gsl::not_null<std::array<Variables<tags_list_for_reconstruct>, 3>*>
        vars_on_lower_face,
    const gsl::not_null<std::array<Variables<tags_list_for_reconstruct>, 3>*>
        vars_on_upper_face,
    const gsl::not_null<std::optional<std::array<gsl::span<std::uint8_t>, 3>>*>
        reconstruction_order,
    const Variables<hydro::grmhd_tags<DataVector>>& volume_prims,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
    const Element<3>& element,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>& ghost_data,
    const Mesh<3>& subcell_mesh,
    const VariableFixing::FixToAtmosphere<dim>& fix_to_atmosphere) const {
  DirectionalIdMap<dim, Variables<prims_to_reconstruct_tags>>
      neighbor_variables_data{};
  ::fd::neighbor_data_as_variables<dim>(make_not_null(&neighbor_variables_data),
                                        ghost_data, ghost_zone_size(),
                                        subcell_mesh);

  reconstruct_prims_work<positivity_preserving_tags>(
      vars_on_lower_face, vars_on_upper_face,
      [this, &reconstruction_order](
          auto upper_face_vars_ptr, auto lower_face_vars_ptr,
          const auto& volume_vars, const auto& ghost_cell_vars,
          const auto& subcell_extents, const size_t number_of_variables) {
        pp_reconstruct_(upper_face_vars_ptr, lower_face_vars_ptr,
                        reconstruction_order, volume_vars, ghost_cell_vars,
                        subcell_extents, number_of_variables,
                        four_to_the_alpha_5_,
                        six_to_the_alpha_7_.value_or(
                            std::numeric_limits<double>::signaling_NaN()),
                        eight_to_the_alpha_9_.value_or(
                            std::numeric_limits<double>::signaling_NaN()));
      },
      volume_prims, eos, element, neighbor_variables_data, subcell_mesh,
      ghost_zone_size(), false, fix_to_atmosphere);
  reconstruct_prims_work<non_positive_tags>(
      vars_on_lower_face, vars_on_upper_face,
      [this](auto upper_face_vars_ptr, auto lower_face_vars_ptr,
             const auto& volume_vars, const auto& ghost_cell_vars,
             const auto& subcell_extents, const size_t number_of_variables) {
        reconstruct_(upper_face_vars_ptr, lower_face_vars_ptr, volume_vars,
                     ghost_cell_vars, subcell_extents, number_of_variables,
                     four_to_the_alpha_5_,
                     six_to_the_alpha_7_.value_or(
                         std::numeric_limits<double>::signaling_NaN()),
                     eight_to_the_alpha_9_.value_or(
                         std::numeric_limits<double>::signaling_NaN()));
      },
      volume_prims, eos, element, neighbor_variables_data, subcell_mesh,
      ghost_zone_size(), true, fix_to_atmosphere);
}

template <size_t ThermodynamicDim>
void PositivityPreservingAdaptiveOrderPrim::reconstruct_fd_neighbor(
    const gsl::not_null<Variables<tags_list_for_reconstruct>*> vars_on_face,
    const Variables<hydro::grmhd_tags<DataVector>>& subcell_volume_prims,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
    const Element<3>& element,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>& ghost_data,
    const Mesh<3>& subcell_mesh,
    const VariableFixing::FixToAtmosphere<dim>& fix_to_atmosphere,
    const Direction<3> direction_to_reconstruct) const {
  reconstruct_fd_neighbor_work<positivity_preserving_tags,
                               prims_to_reconstruct_tags>(
      vars_on_face,
      [this](const auto tensor_component_on_face_ptr,
             const auto& tensor_component_volume,
             const auto& tensor_component_neighbor,
             const Index<3>& subcell_extents,
             const Index<3>& ghost_data_extents,
             const Direction<3>& local_direction_to_reconstruct) {
        pp_reconstruct_lower_neighbor_(
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
             const auto& tensor_component_neighbor,
             const Index<3>& subcell_extents,
             const Index<3>& ghost_data_extents,
             const Direction<3>& local_direction_to_reconstruct) {
        pp_reconstruct_upper_neighbor_(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct, four_to_the_alpha_5_,
            six_to_the_alpha_7_.value_or(
                std::numeric_limits<double>::signaling_NaN()),
            eight_to_the_alpha_9_.value_or(
                std::numeric_limits<double>::signaling_NaN()));
      },
      subcell_volume_prims, eos, element, ghost_data, subcell_mesh,
      direction_to_reconstruct, ghost_zone_size(), false, fix_to_atmosphere);
  reconstruct_fd_neighbor_work<non_positive_tags, prims_to_reconstruct_tags>(
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
            local_direction_to_reconstruct, four_to_the_alpha_5_,
            six_to_the_alpha_7_.value_or(
                std::numeric_limits<double>::signaling_NaN()),
            eight_to_the_alpha_9_.value_or(
                std::numeric_limits<double>::signaling_NaN()));
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
            local_direction_to_reconstruct, four_to_the_alpha_5_,
            six_to_the_alpha_7_.value_or(
                std::numeric_limits<double>::signaling_NaN()),
            eight_to_the_alpha_9_.value_or(
                std::numeric_limits<double>::signaling_NaN()));
      },
      subcell_volume_prims, eos, element, ghost_data, subcell_mesh,
      direction_to_reconstruct, ghost_zone_size(), true, fix_to_atmosphere);
}

bool operator==(const PositivityPreservingAdaptiveOrderPrim& lhs,
                const PositivityPreservingAdaptiveOrderPrim& rhs) {
  // Don't check function pointers since they are set from
  // low_order_reconstructor_
  return lhs.four_to_the_alpha_5_ == rhs.four_to_the_alpha_5_ and
         lhs.six_to_the_alpha_7_ == rhs.six_to_the_alpha_7_ and
         lhs.eight_to_the_alpha_9_ == rhs.eight_to_the_alpha_9_ and
         lhs.low_order_reconstructor_ == rhs.low_order_reconstructor_;
}

bool operator!=(const PositivityPreservingAdaptiveOrderPrim& lhs,
                const PositivityPreservingAdaptiveOrderPrim& rhs) {
  return not(lhs == rhs);
}

#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                              \
  template void PositivityPreservingAdaptiveOrderPrim::reconstruct(         \
      gsl::not_null<std::array<Variables<tags_list_for_reconstruct>, 3>*>   \
          vars_on_lower_face,                                               \
      gsl::not_null<std::array<Variables<tags_list_for_reconstruct>, 3>*>   \
          vars_on_upper_face,                                               \
      const gsl::not_null<                                                  \
          std::optional<std::array<gsl::span<std::uint8_t>, 3>>*>           \
          reconstruction_order,                                             \
      const Variables<hydro::grmhd_tags<DataVector>>& volume_prims,         \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>& eos, \
      const Element<3>& element,                                            \
      const DirectionalIdMap<3, evolution::dg::subcell::GhostData>&         \
          ghost_data,                                                       \
      const Mesh<3>& subcell_mesh,                                          \
      const VariableFixing::FixToAtmosphere<dim>& fix_to_atmosphere) const; \
  template void                                                             \
  PositivityPreservingAdaptiveOrderPrim::reconstruct_fd_neighbor(           \
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
