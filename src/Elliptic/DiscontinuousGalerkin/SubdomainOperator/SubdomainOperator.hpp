// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeBoundaryConditions.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/BoundaryData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/BoundaryFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/NumericalFluxHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementCenteredSubdomainData.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/Tuple.hpp"
#include "Utilities/TupleSlice.hpp"

#include "Domain/SurfaceJacobian.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/LinearOperators/Mass.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/CreateIsCallable.hpp"

// #include "Parallel/Printf.hpp"

namespace elliptic {
namespace dg {

namespace Tags {
struct OverlapExtent : db::SimpleTag {
  using type = size_t;
};
}  // namespace Tags

namespace SubdomainOperator_detail {
// These functions are specific to the strong first-order internal penalty
// scheme
template <typename BoundaryData, size_t Dim,
          typename NumericalFluxesComputerType, typename FluxesComputerType,
          typename NormalDotFluxesTags, typename DivFluxesTags,
          typename... FluxesArgs, typename... AuxiliaryFields>
BoundaryData package_boundary_data(
    const NumericalFluxesComputerType& numerical_fluxes_computer,
    const FluxesComputerType& fluxes_computer, const Mesh<Dim>& volume_mesh,
    const Direction<Dim>& direction, const Mesh<Dim - 1>& face_mesh,
    const tnsr::i<DataVector, Dim>& face_normal,
    const Scalar<DataVector>& face_normal_magnitude,
    const Variables<NormalDotFluxesTags>& n_dot_fluxes,
    const Variables<DivFluxesTags>& div_fluxes,
    const std::tuple<FluxesArgs...>& fluxes_args,
    tmpl::list<AuxiliaryFields...> /*meta*/) noexcept {
  return std::apply(
      [&](const auto&... expanded_fluxes_args) {
        return ::dg::FirstOrderScheme::package_boundary_data(
            numerical_fluxes_computer, face_mesh, n_dot_fluxes, volume_mesh,
            direction, face_normal_magnitude,
            get<::Tags::NormalDotFlux<AuxiliaryFields>>(n_dot_fluxes)...,
            get<::Tags::div<::Tags::Flux<AuxiliaryFields, tmpl::size_t<Dim>,
                                         Frame::Inertial>>>(div_fluxes)...,
            face_normal, fluxes_computer, expanded_fluxes_args...);
      },
      fluxes_args);
}
template <bool MassiveOperator, size_t Dim, typename FieldsTagsList,
          typename NumericalFluxesComputerType, typename BoundaryData>
void apply_boundary_contribution(
    const gsl::not_null<Variables<FieldsTagsList>*> result,
    const NumericalFluxesComputerType& numerical_fluxes_computer,
    const BoundaryData& local_boundary_data,
    const BoundaryData& remote_boundary_data,
    const Scalar<DataVector>& magnitude_of_face_normal,
    const Scalar<DataVector>& surface_jacobian, const Mesh<Dim>& mesh,
    const Direction<Dim>& direction, const Mesh<Dim - 1>& mortar_mesh,
    const ::dg::MortarSize<Dim - 1>& mortar_size) noexcept {
  const size_t dimension = direction.dimension();
  auto boundary_contribution = ::dg::FirstOrderScheme::boundary_flux(
      local_boundary_data, remote_boundary_data, numerical_fluxes_computer,
      mesh.slice_away(dimension), mortar_mesh, mortar_size);
  auto lifted_flux = [&]() noexcept {
    if constexpr (MassiveOperator) {
      return ::dg::lift_flux_massive_no_mass_lumping(
          std::move(boundary_contribution), mesh.slice_away(dimension),
          surface_jacobian);
    } else {
      return ::dg::lift_flux(std::move(boundary_contribution),
                             mesh.extents(dimension), magnitude_of_face_normal);
    }
  }();
  add_slice_to_data(result, std::move(lifted_flux), mesh.extents(), dimension,
                    index_to_slice_at(mesh.extents(), direction));
}

template <typename PrimalFields, typename AuxiliaryFields, size_t Dim,
          typename BoundaryData, typename FluxesComputerType,
          typename NumericalFluxesComputerType, typename... FluxesArgs,
          typename FieldsTags = tmpl::append<PrimalFields, AuxiliaryFields>,
          typename FluxesTags = db::wrap_tags_in<
              ::Tags::Flux, FieldsTags, tmpl::size_t<Dim>, Frame::Inertial>,
          typename DivFluxesTags = db::wrap_tags_in<::Tags::div, FluxesTags>>
void exterior_boundary_data(
    const gsl::not_null<BoundaryData*> boundary_data,
    const Variables<FieldsTags>& vars_on_interior_face,
    const Variables<DivFluxesTags>& div_fluxes_on_interior_face,
    const Mesh<Dim>& volume_mesh, const Direction<Dim>& interior_direction,
    const Mesh<Dim - 1>& face_mesh,
    const tnsr::i<DataVector, Dim>& interior_face_normal,
    const Scalar<DataVector>& interior_face_normal_magnitude,
    const FluxesComputerType& fluxes_computer,
    const NumericalFluxesComputerType& numerical_fluxes_computer,
    const std::tuple<FluxesArgs...>& fluxes_args) noexcept {
  static constexpr size_t volume_dim = Dim;
  // On exterior ("ghost") faces, manufacture boundary data that represent
  // homogeneous Dirichlet boundary conditions
  Variables<FieldsTags> ghost_vars{
      vars_on_interior_face.number_of_grid_points()};
  ::elliptic::dg::homogeneous_dirichlet_boundary_conditions<PrimalFields>(
      make_not_null(&ghost_vars), vars_on_interior_face);
  const auto ghost_fluxes = std::apply(
      [&ghost_vars, &fluxes_computer](const auto&... expanded_fluxes_args) {
        return ::elliptic::first_order_fluxes<volume_dim, PrimalFields,
                                              AuxiliaryFields>(
            ghost_vars, fluxes_computer, expanded_fluxes_args...);
      },
      fluxes_args);
  auto exterior_face_normal = interior_face_normal;
  for (size_t d = 0; d < volume_dim; d++) {
    exterior_face_normal.get(d) *= -1.;
  }
  const auto ghost_normal_dot_fluxes =
      normal_dot_flux<FieldsTags>(exterior_face_normal, ghost_fluxes);
  *boundary_data = package_boundary_data<BoundaryData>(
      numerical_fluxes_computer, fluxes_computer, volume_mesh,
      interior_direction.opposite(), face_mesh, exterior_face_normal,
      interior_face_normal_magnitude, ghost_normal_dot_fluxes,
      // TODO: Is this correct?
      div_fluxes_on_interior_face, fluxes_args, AuxiliaryFields{});
}

template <size_t Dim>
std::pair<tnsr::i<DataVector, Dim>, Scalar<DataVector>>
face_normal_and_magnitude(const Mesh<Dim - 1>& face_mesh,
                          const ElementMap<Dim, Frame::Inertial>& element_map,
                          const Direction<Dim>& direction) noexcept {
  auto face_normal =
      unnormalized_face_normal(face_mesh, element_map, direction);
  // TODO: handle curved backgrounds
  auto magnitude_of_face_normal = magnitude(face_normal);
  for (size_t d = 0; d < Dim; d++) {
    face_normal.get(d) /= get(magnitude_of_face_normal);
  }
  return {std::move(face_normal), std::move(magnitude_of_face_normal)};
}

// By default don't do any slicing
template <typename Arg>
struct slice_arg_to_face_impl {
  template <size_t Dim>
  static Arg apply(const Arg& arg, const Index<Dim>& /*extents*/,
                   const Direction<Dim>& /*direction*/) noexcept {
    return arg;
  }
};
// Slice tensors to the face
template <typename DataType, typename Symm, typename IndexList>
struct slice_arg_to_face_impl<Tensor<DataType, Symm, IndexList>> {
  struct TempTag : db::SimpleTag {
    using type = Tensor<DataType, Symm, IndexList>;
  };
  template <size_t Dim>
  static Tensor<DataType, Symm, IndexList> apply(
      const Tensor<DataType, Symm, IndexList>& arg, const Index<Dim>& extents,
      const Direction<Dim>& direction) noexcept {
    Variables<tmpl::list<TempTag>> temp_vars{arg.begin()->size()};
    get<TempTag>(temp_vars) = arg;
    return get<TempTag>(data_on_slice(temp_vars, extents, direction.dimension(),
                                      index_to_slice_at(extents, direction)));
  }
};

// Slice fluxes args to faces. Used on overlap faces perpendicular to the
// subdomain interface.
// TODO: find a way to avoid this.
template <size_t Dim, typename Arg>
auto slice_arg_to_face(const Arg& arg, const Index<Dim>& extents,
                       const Direction<Dim>& direction) noexcept {
  return slice_arg_to_face_impl<Arg>::apply(arg, extents, direction);
}

CREATE_IS_CALLABLE(at);
CREATE_IS_CALLABLE_V(at);

template <
    size_t Dim, typename Arg,
    Requires<is_at_callable_v<Arg, LinearSolver::Schwarz::OverlapId<Dim>>> =
        nullptr>
decltype(auto) unmap_overlap_arg(
    const Arg& arg,
    const LinearSolver::Schwarz::OverlapId<Dim>& overlap_id) noexcept {
  return arg.at(overlap_id);
}

template <
    size_t Dim, typename Arg,
    Requires<not is_at_callable_v<Arg, LinearSolver::Schwarz::OverlapId<Dim>>> =
        nullptr>
decltype(auto) unmap_overlap_arg(
    const Arg& arg,
    const LinearSolver::Schwarz::OverlapId<Dim>& /*overlap_id*/) noexcept {
  return arg;
}

template <size_t Dim, typename... Args>
decltype(auto) unmap_overlap_args(
    const std::tuple<Args...>& overlap_args,
    const LinearSolver::Schwarz::OverlapId<Dim>& overlap_id) noexcept {
  return std::apply(
      [&](const auto&... expanded_overlap_args) noexcept {
        return std::make_tuple(
            unmap_overlap_arg(expanded_overlap_args, overlap_id)...);
      },
      overlap_args);
};

}  // namespace SubdomainOperator_detail

// Compute bulk contribution in central element
template <typename PrimalFields, typename AuxiliaryFields,
          typename SourcesComputer, bool MassiveOperator,
          typename ResultTagsList, typename ArgTagsList,
          typename FluxesComputer, size_t Dim, typename... FluxesArgs,
          typename... SourcesArgs,
          typename AllFields = tmpl::append<PrimalFields, AuxiliaryFields>,
          typename FluxesTags = db::wrap_tags_in<
              ::Tags::Flux, AllFields, tmpl::size_t<Dim>, Frame::Inertial>>
void apply_operator_volume(
    const gsl::not_null<Variables<ResultTagsList>*> result_element_data,
    const gsl::not_null<Variables<FluxesTags>*> fluxes,
    const gsl::not_null<Variables<db::wrap_tags_in<::Tags::div, FluxesTags>>*>
        div_fluxes,
    const FluxesComputer& fluxes_computer, const Mesh<Dim>& mesh,
    const Scalar<DataVector>& det_jacobian,
    const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
        inv_jacobian,
    const std::tuple<FluxesArgs...>& fluxes_args,
    const std::tuple<SourcesArgs...>& sources_args,
    const Variables<ArgTagsList>& arg_element_data) noexcept {
  // Compute volume fluxes
  std::apply(
      [&](const auto&... expanded_fluxes_args) {
        elliptic::first_order_fluxes<Dim, PrimalFields, AuxiliaryFields>(
            fluxes, Variables<AllFields>(arg_element_data), fluxes_computer,
            expanded_fluxes_args...);
      },
      fluxes_args);
  // Compute divergence of volume fluxes
  *div_fluxes = divergence(*fluxes, mesh, inv_jacobian);
  // Compute volume sources
  auto sources = std::apply(
      [&](const auto&... expanded_sources_args) {
        return elliptic::first_order_sources<PrimalFields, AuxiliaryFields,
                                             SourcesComputer>(
            Variables<AllFields>(arg_element_data), expanded_sources_args...);
      },
      sources_args);
  if constexpr (MassiveOperator) {
    elliptic::first_order_operator_massive(result_element_data, *div_fluxes,
                                           std::move(sources), mesh,
                                           det_jacobian);
  } else {
    elliptic::first_order_operator(result_element_data, *div_fluxes,
                                   std::move(sources));
  }
}

// Add boundary contributions
template <typename Directions, typename PrimalFields, typename AuxiliaryFields,
          typename SourcesComputerType, bool MassiveOperator, size_t Dim,
          typename FluxesComputerType, typename NumericalFluxesComputerType,
          typename... FluxesArgs, typename... OverlapFluxesArgs,
          typename... OverlapSourcesArgs, typename ResultTags, typename ArgTags,
          typename AllFieldsTags = tmpl::append<PrimalFields, AuxiliaryFields>>
static void apply_subdomain_face(
    const gsl::not_null<
        LinearSolver::Schwarz::ElementCenteredSubdomainData<Dim, ResultTags>*>
        result,
    const Element<Dim>& element, const Mesh<Dim>& mesh,
    const FluxesComputerType& fluxes_computer,
    const NumericalFluxesComputerType& numerical_fluxes_computer,
    const Direction<Dim>& direction,
    const tnsr::i<DataVector, Dim>& face_normal,
    const Scalar<DataVector>& magnitude_of_face_normal,
    const Scalar<DataVector>& surface_jacobian,
    const db::const_item_type<
        ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>>& mortar_meshes,
    const db::const_item_type<
        ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>& mortar_sizes,
    const LinearSolver::Schwarz::OverlapMap<Dim, size_t>& all_overlap_extents,
    const LinearSolver::Schwarz::OverlapMap<Dim, Mesh<Dim>>& all_overlap_meshes,
    const LinearSolver::Schwarz::OverlapMap<
        Dim, ElementMap<Dim, Frame::Inertial>>& all_overlap_element_maps,
    const LinearSolver::Schwarz::OverlapMap<
        Dim, ::dg::MortarMap<Dim, Mesh<Dim - 1>>>& all_overlap_mortar_meshes,
    const LinearSolver::Schwarz::OverlapMap<
        Dim, ::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>>&
        all_overlap_mortar_sizes,
    const std::tuple<FluxesArgs...>& fluxes_args,
    const std::tuple<OverlapFluxesArgs...>& all_overlap_fluxes_args,
    const std::tuple<OverlapSourcesArgs...>& all_overlap_sources_args,
    const LinearSolver::Schwarz::ElementCenteredSubdomainData<Dim, ArgTags>&
        arg,
    const Variables<db::wrap_tags_in<
        ::Tags::Flux, tmpl::append<PrimalFields, AuxiliaryFields>,
        tmpl::size_t<Dim>, Frame::Inertial>>& central_fluxes,
    const Variables<db::wrap_tags_in<
        ::Tags::div,
        db::wrap_tags_in<::Tags::Flux,
                         tmpl::append<PrimalFields, AuxiliaryFields>,
                         tmpl::size_t<Dim>, Frame::Inertial>>>&
        central_div_fluxes) noexcept {
  static constexpr size_t volume_dim = Dim;
  using SubdomainDataType =
      LinearSolver::Schwarz::ElementCenteredSubdomainData<Dim, ResultTags>;
  using all_fields_tags = AllFieldsTags;
  using vars_tag = ::Tags::Variables<all_fields_tags>;
  using fluxes_tag =
      db::add_tag_prefix<::Tags::Flux, vars_tag, tmpl::size_t<volume_dim>,
                         Frame::Inertial>;
  using div_fluxes_tag = db::add_tag_prefix<::Tags::div, fluxes_tag>;
  //   using n_dot_fluxes_tag = db::add_tag_prefix<::Tags::NormalDotFlux,
  //   vars_tag>;
  using BoundaryData =
      ::dg::FirstOrderScheme::BoundaryData<NumericalFluxesComputerType>;

  const size_t dimension = direction.dimension();
  const auto face_mesh = mesh.slice_away(dimension);
  const size_t slice_index = index_to_slice_at(mesh.extents(), direction);

  // Compute normal dot fluxes
  const auto central_fluxes_on_face =
      data_on_slice(central_fluxes, mesh.extents(), dimension, slice_index);
  const auto normal_dot_central_fluxes =
      normal_dot_flux<all_fields_tags>(face_normal, central_fluxes_on_face);

  // Slice flux divergences to face
  const auto central_div_fluxes_on_face =
      data_on_slice(central_div_fluxes, mesh.extents(), dimension, slice_index);

  // Assemble boundary data
  const auto center_boundary_data_on_face =
      SubdomainOperator_detail::package_boundary_data<BoundaryData>(
          numerical_fluxes_computer, fluxes_computer, mesh, direction,
          face_mesh, face_normal, magnitude_of_face_normal,
          normal_dot_central_fluxes, central_div_fluxes_on_face, fluxes_args,
          AuxiliaryFields{});

  constexpr bool is_boundary =
      std::is_same_v<Directions, domain::Tags::BoundaryDirectionsInterior<Dim>>;
  if constexpr (is_boundary) {
    const auto central_vars_on_face = Variables<all_fields_tags>(data_on_slice(
        arg.element_data, mesh.extents(), dimension, slice_index));
    BoundaryData remote_boundary_data;
    SubdomainOperator_detail::exterior_boundary_data<PrimalFields,
                                                     AuxiliaryFields>(
        make_not_null(&remote_boundary_data), central_vars_on_face,
        central_div_fluxes_on_face, mesh, direction, face_mesh, face_normal,
        magnitude_of_face_normal, fluxes_computer, numerical_fluxes_computer,
        fluxes_args);
    // No projections necessary since exterior mortars cover the full face
    SubdomainOperator_detail::apply_boundary_contribution<MassiveOperator>(
        make_not_null(&result->element_data), numerical_fluxes_computer,
        center_boundary_data_on_face, std::move(remote_boundary_data),
        magnitude_of_face_normal, surface_jacobian, mesh, direction, face_mesh,
        make_array<Dim - 1>(Spectral::MortarSize::Full));
  } else {
    const auto& neighbors = element.neighbors().at(direction);
    const auto& orientation = neighbors.orientation();
    const auto overlap_direction_in_neighbor =
        orientation(direction.opposite());
    const size_t overlap_dimension_in_neighbor =
        overlap_direction_in_neighbor.dimension();

    // Iterate over all neighbors in this direction, computing their neighbor
    // volume operator and all boundary contributions within the subdomain.
    // Note that data on overlaps is oriented according to the neighbor that
    // it is on, as is all geometric information that we have on the neighbor.
    // Only when data cross element boundaries do we need to re-orient.
    std::unordered_map<
        std::pair<ElementId<Dim>, ::dg::MortarId<Dim>>, BoundaryData,
        boost::hash<std::pair<ElementId<Dim>, ::dg::MortarId<Dim>>>>
        neighbors_boundary_data{};
    std::unordered_map<ElementId<Dim>, typename SubdomainDataType::ElementData>
        neighbor_results_extended{};
    std::unordered_map<std::pair<ElementId<Dim>, Direction<Dim>>,
                       Scalar<DataVector>,
                       boost::hash<std::pair<ElementId<Dim>, Direction<Dim>>>>
        neighbor_face_normal_magnitudes{};
    std::unordered_map<std::pair<ElementId<Dim>, Direction<Dim>>,
                       Scalar<DataVector>,
                       boost::hash<std::pair<ElementId<Dim>, Direction<Dim>>>>
        neighbor_surface_jacobians{};
    for (const auto& neighbor_id : neighbors) {
      const auto overlap_id = std::make_pair(direction, neighbor_id);
      const auto& mortar_id = overlap_id;
      const auto& overlap_data = arg.overlap_data.at(overlap_id);
      const auto& overlap_extents = all_overlap_extents.at(overlap_id);
      const auto& neighbor_mesh = all_overlap_meshes.at(overlap_id);

      // Extend the overlap data to the full neighbor mesh by filling it
      // with zeros and adding the overlapping slices
      const auto neighbor_data = Variables<all_fields_tags>(
          LinearSolver::Schwarz::extended_overlap_data(
              overlap_data, neighbor_mesh.extents(), overlap_extents,
              overlap_direction_in_neighbor));

      // TODO: These could be cached in the databox
      const auto& neighbor_element_map =
          all_overlap_element_maps.at(overlap_id);
      const auto neighbor_logical_coords = logical_coordinates(neighbor_mesh);
      const auto neighbor_inv_jacobian =
          neighbor_element_map.inv_jacobian(neighbor_logical_coords);
      const auto neighbor_jacobian =
          neighbor_element_map.jacobian(neighbor_logical_coords);
      const auto neighbor_det_jacobian = determinant(neighbor_jacobian);

      // TODO: avoid allocations by buffering these
      const size_t neighbor_num_points = neighbor_mesh.number_of_grid_points();
      neighbor_results_extended.emplace(neighbor_id, neighbor_num_points);
      auto& neighbor_result_extended =
          neighbor_results_extended.at(neighbor_id);
      typename fluxes_tag::type neighbor_fluxes{neighbor_num_points};
      typename div_fluxes_tag::type neighbor_div_fluxes{neighbor_num_points};

      // Compute the volume contribution in the neighbor from the extended
      // overlap data
      const auto neighbor_fluxes_args =
          SubdomainOperator_detail::unmap_overlap_args(all_overlap_fluxes_args,
                                                       overlap_id);
      const auto neighbor_sources_args =
          SubdomainOperator_detail::unmap_overlap_args(all_overlap_sources_args,
                                                       overlap_id);
      apply_operator_volume<PrimalFields, AuxiliaryFields, SourcesComputerType,
                            MassiveOperator>(
          make_not_null(&neighbor_result_extended),
          make_not_null(&neighbor_fluxes), make_not_null(&neighbor_div_fluxes),
          fluxes_computer, neighbor_mesh, neighbor_det_jacobian,
          neighbor_inv_jacobian, neighbor_fluxes_args, neighbor_sources_args,
          neighbor_data);

      // Iterate over the neighbor's mortars to compute boundary data. For the
      // mortars to the subdomain center, to external boundaries and to elements
      // that are not part of the subdomain we can handle the boundary
      // contribution to the neighbor and to the potential other element right
      // away. For mortars to other neighbors within the subdomain we cache the
      // boundary data and handle the boundary contribution in the second pass
      // over the same mortar.
      const auto& neighbor_mortar_meshes =
          all_overlap_mortar_meshes.at(overlap_id);
      const auto& neighbor_mortar_sizes =
          all_overlap_mortar_sizes.at(overlap_id);
      // One of the iterations of the loop will go over the mortar with the
      // central element, so we need the projected data
      const auto& mortar_mesh = mortar_meshes.at(mortar_id);
      const auto& mortar_size = mortar_sizes.at(mortar_id);
      auto center_boundary_data =
          ::dg::needs_projection(face_mesh, mortar_mesh, mortar_size)
              ? center_boundary_data_on_face.project_to_mortar(
                    face_mesh, mortar_mesh, mortar_size)
              : center_boundary_data_on_face;
      for (const auto& neighbor_mortar_id_and_mesh : neighbor_mortar_meshes) {
        const auto& neighbor_mortar_id = neighbor_mortar_id_and_mesh.first;
        const auto& neighbor_face_direction = neighbor_mortar_id.first;
        // We can skip neighbor mortars that face away from the subdomain center
        // because they only contribute to points on the face, which are never
        // part of the subdomain (see LinearSolver::Schwarz::overlap_extents).
        if (neighbor_face_direction ==
            overlap_direction_in_neighbor.opposite()) {
          continue;
        }

        // Compute the boundary data on the neighbor's local side of the mortar
        const size_t neighbor_face_dimension =
            neighbor_face_direction.dimension();
        const size_t neighbor_face_slice_index =
            index_to_slice_at(neighbor_mesh.extents(), neighbor_face_direction);
        const auto neighbor_face_mesh =
            neighbor_mesh.slice_away(neighbor_face_dimension);
        const auto& neighbor_mortar_mesh = neighbor_mortar_id_and_mesh.second;
        const auto& neighbor_mortar_size =
            neighbor_mortar_sizes.at(neighbor_mortar_id);
        // TODO: these can be cached in the DataBox
        const auto neighbor_face_normal_and_magnitude =
            SubdomainOperator_detail::face_normal_and_magnitude(
                neighbor_face_mesh, neighbor_element_map,
                neighbor_face_direction);
        const auto neighbor_surface_jacobian = domain::surface_jacobian(
            neighbor_element_map, neighbor_face_mesh, neighbor_face_direction,
            neighbor_face_normal_and_magnitude.second);
        neighbor_face_normal_magnitudes.insert_or_assign(
            std::make_pair(neighbor_id, neighbor_face_direction),
            neighbor_face_normal_and_magnitude.second);
        neighbor_surface_jacobians.insert_or_assign(
            std::make_pair(neighbor_id, neighbor_face_direction),
            neighbor_surface_jacobian);
        // TODO: buffer these to reduce allocations?
        const auto neighbor_fluxes_on_face =
            data_on_slice(neighbor_fluxes, neighbor_mesh.extents(),
                          neighbor_face_dimension, neighbor_face_slice_index);
        const auto neighbor_div_fluxes_on_face =
            data_on_slice(neighbor_div_fluxes, neighbor_mesh.extents(),
                          neighbor_face_dimension, neighbor_face_slice_index);
        const auto neighbor_normal_dot_fluxes =
            normal_dot_flux<all_fields_tags>(
                neighbor_face_normal_and_magnitude.first,
                neighbor_fluxes_on_face);
        const auto neighbor_fluxes_args_on_face = std::apply(
            [&](const auto&... expanded_neighbor_fluxes_args) noexcept {
              return std::make_tuple(
                  SubdomainOperator_detail::slice_arg_to_face(
                      expanded_neighbor_fluxes_args, neighbor_mesh.extents(),
                      neighbor_face_direction)...);
            },
            neighbor_fluxes_args);
        auto neighbor_local_boundary_data =
            SubdomainOperator_detail::package_boundary_data<BoundaryData>(
                numerical_fluxes_computer, fluxes_computer, neighbor_mesh,
                neighbor_face_direction, neighbor_face_mesh,
                neighbor_face_normal_and_magnitude.first,
                neighbor_face_normal_and_magnitude.second,
                neighbor_normal_dot_fluxes, neighbor_div_fluxes_on_face,
                neighbor_fluxes_args_on_face, AuxiliaryFields{});
        if (::dg::needs_projection(neighbor_face_mesh, neighbor_mortar_mesh,
                                   neighbor_mortar_size)) {
          neighbor_local_boundary_data =
              neighbor_local_boundary_data.project_to_mortar(
                  neighbor_face_mesh, neighbor_mortar_mesh,
                  neighbor_mortar_size);
        }

        // Decide what to do based on what's on the other side of the mortar
        const auto& neighbors_neighbor_id = neighbor_mortar_id.second;
        BoundaryData neighbor_remote_boundary_data;
        if (neighbors_neighbor_id == element.id()) {
          // This is the mortar to the subdomain center. We apply the boundary
          // contribution both to the subdomain center and to the neighbor.
          // First, apply the boundary contribution to the central element
          auto reoriented_neighbor_boundary_data = neighbor_local_boundary_data;
          if (not orientation.is_aligned()) {
            reoriented_neighbor_boundary_data.orient_on_slice(
                neighbor_mortar_mesh.extents(), overlap_dimension_in_neighbor,
                orientation.inverse_map());
          }
          SubdomainOperator_detail::apply_boundary_contribution<
              MassiveOperator>(make_not_null(&result->element_data),
                               numerical_fluxes_computer, center_boundary_data,
                               std::move(reoriented_neighbor_boundary_data),
                               magnitude_of_face_normal, surface_jacobian, mesh,
                               direction, mortar_mesh, mortar_size);
          // Second, prepare applying the boundary contribution to the neighbor
          neighbor_remote_boundary_data = center_boundary_data;
          if (not orientation.is_aligned()) {
            neighbor_remote_boundary_data.orient_on_slice(
                mortar_mesh.extents(), dimension, orientation);
          }
        } else if (neighbors_neighbor_id ==
                   ElementId<volume_dim>::external_boundary_id()) {
          // This is an external boundary of the neighbor. We apply the boundary
          // contribution directly to the neighbor.
          const auto neighbor_face_data =
              data_on_slice(neighbor_data, neighbor_mesh.extents(),
                            neighbor_face_dimension, neighbor_face_slice_index);
          SubdomainOperator_detail::exterior_boundary_data<PrimalFields,
                                                           AuxiliaryFields>(
              make_not_null(&neighbor_remote_boundary_data), neighbor_face_data,
              neighbor_div_fluxes_on_face, neighbor_mesh,
              neighbor_face_direction, neighbor_face_mesh,
              neighbor_face_normal_and_magnitude.first,
              neighbor_face_normal_and_magnitude.second, fluxes_computer,
              numerical_fluxes_computer, neighbor_fluxes_args_on_face);
        } else {
          // This is an internal boundary to another element, which may or may
          // not overlap with the subdomain.
          const auto neighbors_neighbor_overlap_id =
              std::make_pair(direction, neighbors_neighbor_id);
          if (arg.overlap_data.count(neighbors_neighbor_overlap_id) > 0) {
            // The neighbor's neighbor overlaps with the subdomain, so we can
            // retrieve data from it. We store the data on one side of the
            // mortar in a cache, so when encountering this mortar the second
            // time (from its other side) we apply the boundary contributions to
            // both sides.
            //
            // Note that the other element is guaranteed be another neighbor of
            // the subdomain center in the same direction, so both neighbors
            // have the same orientation and we don't need to re-orient
            // anything.
            const auto direction_from_neighbors_neighbor =
                neighbor_face_direction.opposite();
            const auto mortar_id_from_neighbors_neighbor =
                std::make_pair(direction_from_neighbors_neighbor, neighbor_id);
            const auto found_neighbors_neighbor_boundary_data =
                neighbors_boundary_data.find(std::make_pair(
                    neighbors_neighbor_id, mortar_id_from_neighbors_neighbor));
            if (found_neighbors_neighbor_boundary_data ==
                neighbors_boundary_data.end()) {
              neighbors_boundary_data[std::make_pair(neighbor_id,
                                                     neighbor_mortar_id)] =
                  neighbor_local_boundary_data;

              // Skip applying the boundary contribution from this mortar
              // because we don't have the remote data available yet. We apply
              // the boundary contribution when re-visiting this mortar from the
              // other side.
              continue;

            } else {
              neighbor_remote_boundary_data =
                  std::move(found_neighbors_neighbor_boundary_data->second);
              neighbors_boundary_data.erase(
                  found_neighbors_neighbor_boundary_data->first);

              // Apply the boundary contribution also to the other side of the
              // mortar that we had previously skipped
              SubdomainOperator_detail::apply_boundary_contribution<
                  MassiveOperator>(
                  make_not_null(
                      &neighbor_results_extended.at(neighbors_neighbor_id)),
                  numerical_fluxes_computer, neighbor_remote_boundary_data,
                  neighbor_local_boundary_data,
                  neighbor_face_normal_magnitudes.at(
                      std::pair(neighbors_neighbor_id,
                                direction_from_neighbors_neighbor)),
                  neighbor_surface_jacobians.at(
                      std::pair(neighbors_neighbor_id,
                                direction_from_neighbors_neighbor)),
                  all_overlap_meshes.at(neighbors_neighbor_overlap_id),
                  direction_from_neighbors_neighbor,
                  all_overlap_mortar_meshes.at(neighbors_neighbor_overlap_id)
                      .at(mortar_id_from_neighbors_neighbor),
                  all_overlap_mortar_sizes.at(neighbors_neighbor_overlap_id)
                      .at(mortar_id_from_neighbors_neighbor));
            }
          } else {
            // The neighbor's neighbor does not overlap with the subdomain, so
            // we assume the data on it is zero. We constructing zero data on
            // the mortar mesh directly so we don't have to project
            //
            // Caveat: When computing the penalty for the numerical flux here we
            // don't take the geometry of the neighbor's neighbor into account.
            // We just use the polynomial degree and the element size
            // perpendicular to the face from the neighbor. This leads to an
            // inconsistency between the subdomain operator and the full
            // operator applied to the domain where non-subdomain points are
            // zeroed.
            //
            // TODO: We might remedy this inconsistency by keeping track of the
            // geometry of all neighbor's neighbors, which wouldn't be too
            // difficult because the geometry only changes with AMR.
            numerical_fluxes_computer.package_zero_data(
                make_not_null(&neighbor_remote_boundary_data),
                neighbor_mortar_mesh.number_of_grid_points());
          }
        }
        SubdomainOperator_detail::apply_boundary_contribution<MassiveOperator>(
            make_not_null(&neighbor_result_extended), numerical_fluxes_computer,
            neighbor_local_boundary_data, neighbor_remote_boundary_data,
            neighbor_face_normal_and_magnitude.second,
            neighbor_surface_jacobian, neighbor_mesh, neighbor_face_direction,
            neighbor_mortar_mesh, neighbor_mortar_size);
      }
    }
    // Take only the part of the neighbor data that lies within the overlap
    //
    // TODO: Better store the data restricted to the overlap directly in the
    // result and apply boundary contributions directly to the restricted data.
    // That removes the need for this loop.
    for (const auto& neighbor_id_and_result : neighbor_results_extended) {
      const auto& neighbor_id = neighbor_id_and_result.first;
      const auto overlap_id = std::make_pair(direction, neighbor_id);
      const auto& neighbor_mesh = all_overlap_meshes.at(overlap_id);
      const auto& overlap_extents = all_overlap_extents.at(overlap_id);
      auto neighbor_result = LinearSolver::Schwarz::data_on_overlap(
          neighbor_id_and_result.second, neighbor_mesh.extents(),
          overlap_extents, overlap_direction_in_neighbor);
      result->overlap_data.insert_or_assign(overlap_id,
                                            std::move(neighbor_result));
    }
  }
}

template <typename Tag, typename Dim, typename OptionsGroup,
          typename CenterTags>
struct make_overlap_tag_impl {
  using type = tmpl::conditional_t<
      tmpl::list_contains_v<CenterTags, Tag>, Tag,
      LinearSolver::Schwarz::Tags::Overlaps<Tag, Dim::value, OptionsGroup>>;
};

template <size_t Dim, typename PrimalFields, typename AuxiliaryFields,
          typename FluxesComputerTag, typename FluxesArgs,
          typename SourcesComputer, typename SourcesArgs,
          typename NumericalFluxesComputerTag, typename OptionsGroup,
          typename FluxesArgsTagsFromCenter, bool MassiveOperator>
struct SubdomainOperator {
 private:
  using all_fields_tags = tmpl::append<PrimalFields, AuxiliaryFields>;
  using vars_tag = ::Tags::Variables<all_fields_tags>;
  using fluxes_tag = db::add_tag_prefix<::Tags::Flux, vars_tag,
                                        tmpl::size_t<Dim>, Frame::Inertial>;
  using div_fluxes_tag = db::add_tag_prefix<::Tags::div, fluxes_tag>;
  using FluxesComputerType = db::const_item_type<FluxesComputerTag>;
  using NumericalFluxesComputerType =
      db::const_item_type<NumericalFluxesComputerTag>;

  using fluxes_args_tags = typename FluxesComputerType::argument_tags;
  static constexpr size_t num_fluxes_args = tmpl::size<fluxes_args_tags>::value;
  using sources_args_tags = typename SourcesComputer::argument_tags;
  static constexpr size_t num_sources_args =
      tmpl::size<sources_args_tags>::value;

  template <typename Tag>
  using overlaps_tag =
      LinearSolver::Schwarz::Tags::Overlaps<Tag, Dim, OptionsGroup>;
  template <typename ValueType>
  using overlaps = LinearSolver::Schwarz::OverlapMap<Dim, ValueType>;

  using Buffer = tuples::TaggedTuple<fluxes_tag, div_fluxes_tag>;

 public:
  static constexpr size_t volume_dim = Dim;

  explicit SubdomainOperator(const size_t element_num_points) noexcept
      : buffer_{db::item_type<fluxes_tag>{element_num_points},
                db::item_type<div_fluxes_tag>{element_num_points}} {}

  struct element_operator {
    using argument_tags = tmpl::append<
        tmpl::list<
            domain::Tags::Mesh<Dim>,
            domain::Tags::DetJacobian<Frame::Logical, Frame::Inertial>,
            domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Inertial>,
            FluxesComputerTag>,
        fluxes_args_tags, sources_args_tags>;

    template <typename... RemainingArgs>
    static void apply(
        const Mesh<Dim>& mesh, const Scalar<DataVector>& det_jacobian,
        const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
            inv_jacobian,
        const FluxesComputerType& fluxes_computer,
        const RemainingArgs&... expanded_remaining_args) noexcept {
      const std::tuple<RemainingArgs...> remaining_args{
          expanded_remaining_args...};
      const auto& arg = get<sizeof...(RemainingArgs) - 3>(remaining_args);
      const auto& result = get<sizeof...(RemainingArgs) - 2>(remaining_args);
      const auto& subdomain_operator =
          get<sizeof...(RemainingArgs) - 1>(remaining_args);
      apply_operator_volume<PrimalFields, AuxiliaryFields, SourcesComputer,
                            MassiveOperator>(
          make_not_null(&(result->element_data)),
          make_not_null(&get<fluxes_tag>(subdomain_operator->buffer_)),
          make_not_null(&get<div_fluxes_tag>(subdomain_operator->buffer_)),
          fluxes_computer, mesh, det_jacobian, inv_jacobian,
          tuple_head<num_fluxes_args>(remaining_args),
          tuple_slice<num_fluxes_args, num_fluxes_args + num_sources_args>(
              remaining_args),
          arg.element_data);
    }
  };

  template <typename Directions>
  struct face_operator {
    using overlap_fluxes_args_tags = tmpl::transform<
        fluxes_args_tags,
        make_overlap_tag_impl<tmpl::_1, tmpl::pin<tmpl::size_t<Dim>>,
                              tmpl::pin<OptionsGroup>,
                              tmpl::pin<FluxesArgsTagsFromCenter>>>;
    using argument_tags = tmpl::append<
        tmpl::list<
            domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>,
            FluxesComputerTag, NumericalFluxesComputerTag,
            domain::Tags::Direction<Dim>,
            ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>,
            ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>,
            domain::Tags::SurfaceJacobian<Frame::Logical, Frame::Inertial>,
            ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
            ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>,
            overlaps_tag<Tags::OverlapExtent>,
            overlaps_tag<domain::Tags::Mesh<Dim>>,
            overlaps_tag<domain::Tags::ElementMap<Dim>>,
            overlaps_tag<::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>>,
            overlaps_tag<::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>>,
        fluxes_args_tags, overlap_fluxes_args_tags,
        db::wrap_tags_in<overlaps_tag, sources_args_tags>>;
    using volume_tags = tmpl::append<
        tmpl::list<
            domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>,
            FluxesComputerTag, NumericalFluxesComputerTag,
            ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
            ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>,
            overlaps_tag<Tags::OverlapExtent>,
            overlaps_tag<domain::Tags::Mesh<Dim>>,
            overlaps_tag<domain::Tags::ElementMap<Dim>>,
            overlaps_tag<::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>>,
            overlaps_tag<::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>>,
        get_volume_tags<FluxesComputerType>, overlap_fluxes_args_tags,
        db::wrap_tags_in<overlaps_tag, sources_args_tags>>;

    // interface_apply doesn't currently support `void` return types
    template <typename... RemainingArgs>
    int operator()(
        const Element<Dim>& element, const Mesh<Dim>& mesh,
        const FluxesComputerType& fluxes_computer,
        const NumericalFluxesComputerType& numerical_fluxes_computer,
        const Direction<Dim>& direction,
        const tnsr::i<DataVector, Dim>& face_normal,
        const Scalar<DataVector>& face_normal_magnitude,
        const Scalar<DataVector>& surface_jacobian,
        const db::const_item_type<
            ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>>& mortar_meshes,
        const db::const_item_type<
            ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>& mortar_sizes,
        const overlaps<size_t>& all_overlap_extents,
        const overlaps<Mesh<Dim>>& all_overlap_meshes,
        const overlaps<ElementMap<Dim, Frame::Inertial>>&
            all_overlap_element_maps,
        const overlaps<::dg::MortarMap<Dim, Mesh<Dim - 1>>>&
            all_overlap_mortar_meshes,
        const overlaps<::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>>&
            all_overlap_mortar_sizes,
        RemainingArgs&&... expanded_remaining_args) const noexcept {
      const auto remaining_args =
          std::forward_as_tuple(expanded_remaining_args...);
      const auto& arg = get<sizeof...(RemainingArgs) - 3>(remaining_args);
      const auto& result = get<sizeof...(RemainingArgs) - 2>(remaining_args);
      const auto& subdomain_operator =
          get<sizeof...(RemainingArgs) - 1>(remaining_args);
      apply_subdomain_face<Directions, PrimalFields, AuxiliaryFields,
                           SourcesComputer, MassiveOperator>(
          result, element, mesh, fluxes_computer, numerical_fluxes_computer,
          direction, face_normal, face_normal_magnitude, surface_jacobian,
          mortar_meshes, mortar_sizes, all_overlap_extents, all_overlap_meshes,
          all_overlap_element_maps, all_overlap_mortar_meshes,
          all_overlap_mortar_sizes, tuple_head<num_fluxes_args>(remaining_args),
          tuple_slice<num_fluxes_args, 2 * num_fluxes_args>(remaining_args),
          tuple_slice<2 * num_fluxes_args,
                      2 * num_fluxes_args + num_sources_args>(remaining_args),
          arg, get<fluxes_tag>(subdomain_operator->buffer_),
          get<div_fluxes_tag>(subdomain_operator->buffer_));
      return 0;
    }
  };

 private:
  Buffer buffer_;
};

}  // namespace dg
}  // namespace elliptic
