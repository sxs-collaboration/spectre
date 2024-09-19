// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/PositivityPreservingAdaptiveOrder.hpp"

#include <pup.h>

#include <array>
#include <cstddef>
#include <memory>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/ReconstructWork.tpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"
#include "NumericalAlgorithms/FiniteDifference/MonotonicityPreserving5.hpp"
#include "NumericalAlgorithms/FiniteDifference/NeighborDataAsVariables.hpp"
#include "NumericalAlgorithms/FiniteDifference/PositivityPreservingAdaptiveOrder.hpp"
#include "NumericalAlgorithms/FiniteDifference/Reconstruct.tpp"
#include "NumericalAlgorithms/FiniteDifference/Unlimited.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/ParseError.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::GhValenciaDivClean::fd {
PositivityPreservingAdaptiveOrderPrim::PositivityPreservingAdaptiveOrderPrim(
    CkMigrateMessage* const msg)
    : Reconstructor(msg) {}

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
    PARSE_ERROR(context, "Alpha7 hasn't been tested.");
    six_to_the_alpha_7_ = pow(6.0, alpha_7.value());
  }
  if (alpha_9.has_value()) {
    PARSE_ERROR(context, "Alpha9 hasn't been tested.");
    eight_to_the_alpha_9_ = pow(8.0, alpha_9.value());
  }
  set_function_pointers();
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

std::unique_ptr<Reconstructor>
PositivityPreservingAdaptiveOrderPrim::get_clone() const {
  return std::make_unique<PositivityPreservingAdaptiveOrderPrim>(*this);
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

template <size_t ThermodynamicDim, typename TagsList>
void PositivityPreservingAdaptiveOrderPrim::reconstruct(
    const gsl::not_null<std::array<Variables<TagsList>, dim>*>
        vars_on_lower_face,
    const gsl::not_null<std::array<Variables<TagsList>, dim>*>
        vars_on_upper_face,
    const gsl::not_null<std::optional<std::array<gsl::span<std::uint8_t>, 3>>*>
        reconstruction_order,
    const Variables<hydro::grmhd_tags<DataVector>>& volume_prims,
    const Variables<typename System::variables_tag::type::tags_list>&
        volume_spacetime_and_cons_vars,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
    const Element<dim>& element,
    const DirectionalIdMap<dim, evolution::dg::subcell::GhostData>& ghost_data,
    const Mesh<dim>& subcell_mesh,
    const VariableFixing::FixToAtmosphere<dim>& fix_to_atmosphere) const {
  using all_tags_for_reconstruction = grmhd::GhValenciaDivClean::Tags::
      primitive_grmhd_and_spacetime_reconstruction_tags;

  DirectionalIdMap<dim, Variables<all_tags_for_reconstruction>>
      neighbor_variables_data{};
  ::fd::neighbor_data_as_variables<dim>(make_not_null(&neighbor_variables_data),
                                        ghost_data, ghost_zone_size(),
                                        subcell_mesh);

  reconstruct_prims_work<tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>>,
                         positivity_preserving_tags>(
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
      [](auto upper_face_vars_ptr, auto lower_face_vars_ptr,
         const auto& volume_vars, const auto& ghost_cell_vars,
         const auto& subcell_extents, const size_t number_of_variables) {
        ::fd::reconstruction::unlimited<4>(
            upper_face_vars_ptr, lower_face_vars_ptr, volume_vars,
            ghost_cell_vars, subcell_extents, number_of_variables);
      },
      [](const auto vars_on_face_ptr) {
        const auto& spacetime_metric =
            get<gr::Tags::SpacetimeMetric<DataVector, 3>>(*vars_on_face_ptr);
        auto& spatial_metric =
            get<gr::Tags::SpatialMetric<DataVector, 3>>(*vars_on_face_ptr);
        gr::spatial_metric(make_not_null(&spatial_metric), spacetime_metric);
        auto& inverse_spatial_metric =
            get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(
                *vars_on_face_ptr);
        auto& sqrt_det_spatial_metric =
            get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(*vars_on_face_ptr);

        determinant_and_inverse(make_not_null(&sqrt_det_spatial_metric),
                                make_not_null(&inverse_spatial_metric),
                                spatial_metric);
        get(sqrt_det_spatial_metric) = sqrt(get(sqrt_det_spatial_metric));

        auto& shift = get<gr::Tags::Shift<DataVector, 3>>(*vars_on_face_ptr);
        gr::shift(make_not_null(&shift), spacetime_metric,
                  inverse_spatial_metric);
        gr::lapse(
            make_not_null(&get<gr::Tags::Lapse<DataVector>>(*vars_on_face_ptr)),
            shift, spacetime_metric);
      },
      volume_prims, volume_spacetime_and_cons_vars, eos, element,
      neighbor_variables_data, subcell_mesh, ghost_zone_size(), false,
      fix_to_atmosphere);

  reconstruct_prims_work<tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>>,
                         non_positive_tags>(
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
      [](auto upper_face_vars_ptr, auto lower_face_vars_ptr,
         const auto& volume_vars, const auto& ghost_cell_vars,
         const auto& subcell_extents, const size_t number_of_variables) {
        ::fd::reconstruction::unlimited<4>(
            upper_face_vars_ptr, lower_face_vars_ptr, volume_vars,
            ghost_cell_vars, subcell_extents, number_of_variables);
      },
      [](const auto vars_on_face_ptr) {
        const auto& spacetime_metric =
            get<gr::Tags::SpacetimeMetric<DataVector, 3>>(*vars_on_face_ptr);
        auto& spatial_metric =
            get<gr::Tags::SpatialMetric<DataVector, 3>>(*vars_on_face_ptr);
        gr::spatial_metric(make_not_null(&spatial_metric), spacetime_metric);
        auto& inverse_spatial_metric =
            get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(
                *vars_on_face_ptr);
        auto& sqrt_det_spatial_metric =
            get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(*vars_on_face_ptr);

        determinant_and_inverse(make_not_null(&sqrt_det_spatial_metric),
                                make_not_null(&inverse_spatial_metric),
                                spatial_metric);
        get(sqrt_det_spatial_metric) = sqrt(get(sqrt_det_spatial_metric));

        auto& shift = get<gr::Tags::Shift<DataVector, 3>>(*vars_on_face_ptr);
        gr::shift(make_not_null(&shift), spacetime_metric,
                  inverse_spatial_metric);
        gr::lapse(
            make_not_null(&get<gr::Tags::Lapse<DataVector>>(*vars_on_face_ptr)),
            shift, spacetime_metric);
      },
      volume_prims, volume_spacetime_and_cons_vars, eos, element,
      neighbor_variables_data, subcell_mesh, ghost_zone_size(), true,
      fix_to_atmosphere);
}

// The current implementation does not use positivity-preserving
// reconstruction at Dg/Subcell boundary. PP should only be required
// at shocks / surfaces, which should be within the subcell region
// if the Dg/Subcell code is performing as expected.
template <size_t ThermodynamicDim, typename TagsList>
void PositivityPreservingAdaptiveOrderPrim::reconstruct_fd_neighbor(
    const gsl::not_null<Variables<TagsList>*> vars_on_face,
    const Variables<hydro::grmhd_tags<DataVector>>& subcell_volume_prims,
    const Variables<
        grmhd::GhValenciaDivClean::Tags::spacetime_reconstruction_tags>&
        subcell_volume_spacetime_metric,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
    const Element<dim>& element,
    const DirectionalIdMap<dim, evolution::dg::subcell::GhostData>& ghost_data,
    const Mesh<dim>& subcell_mesh,
    const VariableFixing::FixToAtmosphere<dim>& fix_to_atmosphere,
    const Direction<dim>& direction_to_reconstruct) const {
  using prim_tags_for_reconstruction =
      grmhd::GhValenciaDivClean::Tags::primitive_grmhd_reconstruction_tags;
  using all_tags_for_reconstruction = grmhd::GhValenciaDivClean::Tags::
      primitive_grmhd_and_spacetime_reconstruction_tags;
  reconstruct_fd_neighbor_work<Tags::spacetime_reconstruction_tags,
                               prim_tags_for_reconstruction,
                               all_tags_for_reconstruction>(
      vars_on_face,
      [this](const auto tensor_component_on_face_ptr,
             const auto& tensor_component_volume,
             const auto& tensor_component_neighbor,
             const Index<dim>& subcell_extents,
             const Index<dim>& ghost_data_extents,
             const Direction<dim>& local_direction_to_reconstruct) {
        reconstruct_lower_neighbor_(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct, four_to_the_alpha_5_,
            six_to_the_alpha_7_.value_or(
                std::numeric_limits<double>::signaling_NaN()),
            eight_to_the_alpha_9_.value_or(
                std::numeric_limits<double>::signaling_NaN()));
      },
      [](const auto tensor_component_on_face_ptr,
         const auto& tensor_component_volume,
         const auto& tensor_component_neighbor,
         const Index<dim>& subcell_extents,
         const Index<dim>& ghost_data_extents,
         const Direction<dim>& local_direction_to_reconstruct) {
        ::fd::reconstruction::reconstruct_neighbor<
            Side::Lower,
            ::fd::reconstruction::detail::UnlimitedReconstructor<4>>(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct);
      },
      [this](const auto tensor_component_on_face_ptr,
             const auto& tensor_component_volume,
             const auto& tensor_component_neighbor,
             const Index<dim>& subcell_extents,
             const Index<dim>& ghost_data_extents,
             const Direction<dim>& local_direction_to_reconstruct) {
        reconstruct_upper_neighbor_(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct, four_to_the_alpha_5_,
            six_to_the_alpha_7_.value_or(
                std::numeric_limits<double>::signaling_NaN()),
            eight_to_the_alpha_9_.value_or(
                std::numeric_limits<double>::signaling_NaN()));
      },
      [](const auto tensor_component_on_face_ptr,
         const auto& tensor_component_volume,
         const auto& tensor_component_neighbor,
         const Index<dim>& subcell_extents,
         const Index<dim>& ghost_data_extents,
         const Direction<dim>& local_direction_to_reconstruct) {
        ::fd::reconstruction::reconstruct_neighbor<
            Side::Upper,
            ::fd::reconstruction::detail::UnlimitedReconstructor<4>>(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct);
      },
      [](const auto vars_on_face_ptr) {
        const auto& spacetime_metric =
            get<gr::Tags::SpacetimeMetric<DataVector, 3>>(*vars_on_face_ptr);
        auto& spatial_metric =
            get<gr::Tags::SpatialMetric<DataVector, 3>>(*vars_on_face_ptr);
        gr::spatial_metric(make_not_null(&spatial_metric), spacetime_metric);
        auto& inverse_spatial_metric =
            get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(
                *vars_on_face_ptr);
        auto& sqrt_det_spatial_metric =
            get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(*vars_on_face_ptr);

        determinant_and_inverse(make_not_null(&sqrt_det_spatial_metric),
                                make_not_null(&inverse_spatial_metric),
                                spatial_metric);
        get(sqrt_det_spatial_metric) = sqrt(get(sqrt_det_spatial_metric));

        auto& shift = get<gr::Tags::Shift<DataVector, 3>>(*vars_on_face_ptr);
        gr::shift(make_not_null(&shift), spacetime_metric,
                  inverse_spatial_metric);
        gr::lapse(
            make_not_null(&get<gr::Tags::Lapse<DataVector>>(*vars_on_face_ptr)),
            shift, spacetime_metric);
      },
      subcell_volume_prims, subcell_volume_spacetime_metric, eos, element,
      ghost_data, subcell_mesh, direction_to_reconstruct, ghost_zone_size(),
      true, fix_to_atmosphere);
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
      const Variables<typename System::variables_tag::type::tags_list>&     \
          volume_spacetime_and_cons_vars,                                   \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>& eos, \
      const Element<3>& element,                                            \
      const DirectionalIdMap<3, evolution::dg::subcell::GhostData>&         \
          ghost_data,                                                       \
      const Mesh<3>& subcell_mesh,                                          \
      const VariableFixing::FixToAtmosphere<dim>& fix_to_atmosphere) const; \
  template void                                                             \
  PositivityPreservingAdaptiveOrderPrim::reconstruct_fd_neighbor(           \
      gsl::not_null<Variables<tags_list_for_reconstruct_fd_neighbor>*>      \
          vars_on_face,                                                     \
      const Variables<hydro::grmhd_tags<DataVector>>& subcell_volume_prims, \
      const Variables<                                                      \
          grmhd::GhValenciaDivClean::Tags::spacetime_reconstruction_tags>&  \
          subcell_volume_spacetime_metric,                                  \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>& eos, \
      const Element<3>& element,                                            \
      const DirectionalIdMap<3, evolution::dg::subcell::GhostData>&         \
          ghost_data,                                                       \
      const Mesh<3>& subcell_mesh,                                          \
      const VariableFixing::FixToAtmosphere<dim>& fix_to_atmosphere,        \
      const Direction<3> direction_to_reconstruct) const;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef TAGS_LIST
#undef THERMO_DIM
}  // namespace grmhd::GhValenciaDivClean::fd
