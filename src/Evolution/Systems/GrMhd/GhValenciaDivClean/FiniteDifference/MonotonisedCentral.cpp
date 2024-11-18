// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/MonotonisedCentral.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
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
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "NumericalAlgorithms/FiniteDifference/MonotonisedCentral.hpp"
#include "NumericalAlgorithms/FiniteDifference/NeighborDataAsVariables.hpp"
#include "NumericalAlgorithms/FiniteDifference/Unlimited.hpp"
#include "NumericalAlgorithms/FiniteDifference/Wcns5z.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean::fd {
MonotonisedCentralPrim::MonotonisedCentralPrim(CkMigrateMessage* const msg)
    : Reconstructor(msg) {}

std::unique_ptr<Reconstructor> MonotonisedCentralPrim::get_clone() const {
  return std::make_unique<MonotonisedCentralPrim>(*this);
}

void MonotonisedCentralPrim::pup(PUP::er& p) { Reconstructor::pup(p); }

// NOLINTNEXTLINE
PUP::able::PUP_ID MonotonisedCentralPrim::my_PUP_ID = 0;

template <size_t ThermodynamicDim, typename TagsList>
void MonotonisedCentralPrim::reconstruct(
    const gsl::not_null<std::array<Variables<TagsList>, dim>*>
        vars_on_lower_face,
    const gsl::not_null<std::array<Variables<TagsList>, dim>*>
        vars_on_upper_face,
    const Variables<hydro::grmhd_tags<DataVector>>& volume_prims,
    const Variables<typename System::variables_tag::type::tags_list>&
        volume_spacetime_and_cons_vars,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
    const Element<dim>& element,
    const DirectionalIdMap<dim, evolution::dg::subcell::GhostData>& ghost_data,
    const Mesh<dim>& subcell_mesh,
    const VariableFixing::FixToAtmosphere<dim>& fix_to_atmosphere) const {
  using prim_tags_for_reconstruction =
      grmhd::GhValenciaDivClean::Tags::primitive_grmhd_reconstruction_tags;
  using all_tags_for_reconstruction = grmhd::GhValenciaDivClean::Tags::
      primitive_grmhd_and_spacetime_reconstruction_tags;

  DirectionalIdMap<dim, Variables<all_tags_for_reconstruction>>
      neighbor_variables_data{};
  ::fd::neighbor_data_as_variables<dim>(make_not_null(&neighbor_variables_data),
                                        ghost_data, ghost_zone_size(),
                                        subcell_mesh);

  reconstruct_prims_work<tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>>,
                         prim_tags_for_reconstruction>(
      vars_on_lower_face, vars_on_upper_face,
      [](auto upper_face_vars_ptr, auto lower_face_vars_ptr,
         const auto& volume_vars, const auto& ghost_cell_vars,
         const auto& subcell_extents, const size_t number_of_variables) {
        ::fd::reconstruction::monotonised_central(
            upper_face_vars_ptr, lower_face_vars_ptr, volume_vars,
            ghost_cell_vars, subcell_extents, number_of_variables);
      },
      [](auto upper_face_vars_ptr, auto lower_face_vars_ptr,
         const auto& volume_vars, const auto& ghost_cell_vars,
         const auto& subcell_extents, const size_t number_of_variables) {
        ::fd::reconstruction::unlimited<2>(
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

template <size_t ThermodynamicDim, typename TagsList>
void MonotonisedCentralPrim::reconstruct_fd_neighbor(
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
    const Direction<dim> direction_to_reconstruct) const {
  using prim_tags_for_reconstruction =
      grmhd::GhValenciaDivClean::Tags::primitive_grmhd_reconstruction_tags;
  using all_tags_for_reconstruction = grmhd::GhValenciaDivClean::Tags::
      primitive_grmhd_and_spacetime_reconstruction_tags;
  reconstruct_fd_neighbor_work<Tags::spacetime_reconstruction_tags,
                               prim_tags_for_reconstruction,
                               all_tags_for_reconstruction>(
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
            Side::Lower,
            ::fd::reconstruction::detail::UnlimitedReconstructor<2>>(
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
      [](const auto tensor_component_on_face_ptr,
         const auto& tensor_component_volume,
         const auto& tensor_component_neighbor,
         const Index<dim>& subcell_extents,
         const Index<dim>& ghost_data_extents,
         const Direction<dim>& local_direction_to_reconstruct) {
        ::fd::reconstruction::reconstruct_neighbor<
            Side::Upper,
            ::fd::reconstruction::detail::UnlimitedReconstructor<2>>(
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
      const Variables<typename System::variables_tag::type::tags_list>&     \
          volume_spacetime_and_cons_vars,                                   \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>& eos, \
      const Element<3>& element,                                            \
      const DirectionalIdMap<3, evolution::dg::subcell::GhostData>&         \
          ghost_data,                                                       \
      const Mesh<3>& subcell_mesh,                                          \
      const VariableFixing::FixToAtmosphere<dim>& fix_to_atmosphere) const; \
  template void MonotonisedCentralPrim::reconstruct_fd_neighbor(            \
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
