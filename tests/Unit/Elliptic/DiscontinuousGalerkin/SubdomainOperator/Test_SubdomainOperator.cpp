// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Creators/RotatedBricks.hpp"
#include "Domain/Creators/RotatedIntervals.hpp"
#include "Domain/Creators/RotatedRectangles.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/SurfaceJacobian.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/NumericalFluxes/FirstOrderInternalPenalty.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/SubdomainOperator.hpp"
#include "Elliptic/Systems/Elasticity/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Geometry.hpp"
#include "Elliptic/Tags.hpp"  // Needed by the numerical flux (for now)
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Elliptic/DiscontinuousGalerkin/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/BoundaryFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementCenteredSubdomainData.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/Tuple.hpp"

namespace helpers = TestHelpers::elliptic::dg;

namespace {

struct DummyOptionsGroup {};

template <typename System, typename FluxesArgsTags,
          typename FluxesArgsVolumeTags, typename FluxesArgsTagsFromCenter,
          typename SourcesArgsTags, typename SourcesArgsVolumeTags,
          bool MassiveOperator, typename PackageFluxesArgs,
          typename PackageSourcesArgs, typename... PrimalFields,
          typename... AuxiliaryFields>
void test_subdomain_operator_impl(
    const DomainCreator<System::volume_dim>& domain_creator,
    const size_t overlap, const double penalty_parameter,
    PackageFluxesArgs&& package_fluxes_args,
    PackageSourcesArgs&& package_sources_args,
    tmpl::list<PrimalFields...> /*meta*/,
    tmpl::list<AuxiliaryFields...> /*meta*/) noexcept {
  CAPTURE(overlap);
  CAPTURE(penalty_parameter);
  CAPTURE(MassiveOperator);

  MAKE_GENERATOR(gen);
  UniformCustomDistribution<double> dist{-1., 1.};

  using system = System;
  static constexpr size_t volume_dim = system::volume_dim;
  const typename system::fluxes fluxes_computer{};

  // Get fluxes and sources arg types from the simple tags that are supplied for
  // the DataBox test below.
  // TODO: Perhaps the fluxes and sources arg types should be part of the system
  using FluxesArgs = tmpl::transform<FluxesArgsTags,
                                     tmpl::bind<db::const_item_type, tmpl::_1>>;
  //   using OverlapFluxesArgs =
  //       tmpl::transform<FluxesArgs,
  //       tmpl::bind<LinearSolver::Schwarz::OverlapMap,
  //                                              volume_dim, tmpl::_1>>;
  using SourcesArgs =
      tmpl::transform<SourcesArgsTags,
                      tmpl::bind<db::const_item_type, tmpl::_1>>;
  //   using OverlapSourcesArgs =
  //       tmpl::transform<SourcesArgs,
  //       tmpl::bind<LinearSolver::Schwarz::OverlapMap,
  //                                               volume_dim, tmpl::_1>>;

  // Shortcuts for tags
  using primal_fields = typename system::primal_fields;
  using auxiliary_fields = typename system::auxiliary_fields;
  using all_fields_tags = typename system::fields_tag::tags_list;
  using fluxes_computer_tag =
      elliptic::Tags::FluxesComputer<typename system::fluxes>;
  using fluxes_tags =
      db::wrap_tags_in<::Tags::Flux, all_fields_tags, tmpl::size_t<volume_dim>,
                       Frame::Inertial>;
  using div_fluxes_tags = db::wrap_tags_in<::Tags::div, fluxes_tags>;
  using n_dot_fluxes_tags =
      db::wrap_tags_in<::Tags::NormalDotFlux, all_fields_tags>;

  // Choose a numerical flux
  using NumericalFlux =
      elliptic::dg::NumericalFluxes::FirstOrderInternalPenalty<
          volume_dim, fluxes_computer_tag, primal_fields, auxiliary_fields>;
  const NumericalFlux numerical_fluxes_computer{penalty_parameter};  // C=1.5

  // Setup the elements in the domain
  const auto elements = helpers::create_elements(domain_creator);

  // Choose a subdomain that has internal and external faces
  const auto& subdomain_center = elements.begin()->first;
  CAPTURE(subdomain_center);
  const auto& central_element = elements.at(subdomain_center);

  // Setup the faces and mortars in the subdomain
  // TODO: Support h-refinement in this test
  const auto central_mortars =
      helpers::create_mortars(subdomain_center, elements);
  std::unordered_map<Direction<volume_dim>, tnsr::i<DataVector, volume_dim>>
      internal_face_normals;
  std::unordered_map<Direction<volume_dim>, tnsr::i<DataVector, volume_dim>>
      boundary_face_normals;
  std::unordered_map<Direction<volume_dim>, Scalar<DataVector>>
      internal_face_normal_magnitudes;
  std::unordered_map<Direction<volume_dim>, Scalar<DataVector>>
      boundary_face_normal_magnitudes;
  std::unordered_map<Direction<volume_dim>, Scalar<DataVector>>
      internal_surface_jacobians;
  std::unordered_map<Direction<volume_dim>, Scalar<DataVector>>
      boundary_surface_jacobians;
  dg::MortarMap<volume_dim, Mesh<volume_dim - 1>> central_mortar_meshes;
  dg::MortarMap<volume_dim, dg::MortarSize<volume_dim - 1>>
      central_mortar_sizes;
  for (const auto& mortar : central_mortars) {
    const auto& mortar_id = mortar.first;
    const auto& mortar_mesh_and_size = mortar.second;
    const auto& direction = mortar_id.first;
    const size_t dimension = direction.dimension();
    const auto face_mesh = central_element.mesh.slice_away(dimension);
    auto face_normal = unnormalized_face_normal(
        face_mesh, central_element.element_map, direction);
    // TODO: Use system's magnitude
    auto normal_magnitude = magnitude(face_normal);
    for (size_t d = 0; d < volume_dim; d++) {
      face_normal.get(d) /= get(normal_magnitude);
    }
    auto surface_jacobian = domain::surface_jacobian(
        central_element.element_map, face_mesh, direction, normal_magnitude);
    if (mortar_id.second == ElementId<volume_dim>::external_boundary_id()) {
      boundary_face_normals[direction] = std::move(face_normal);
      boundary_face_normal_magnitudes[direction] = std::move(normal_magnitude);
      boundary_surface_jacobians[direction] = std::move(surface_jacobian);
    } else {
      internal_face_normals[direction] = std::move(face_normal);
      internal_face_normal_magnitudes[direction] = std::move(normal_magnitude);
      internal_surface_jacobians[direction] = std::move(surface_jacobian);
    }
    central_mortar_meshes[mortar_id] = mortar_mesh_and_size.first;
    central_mortar_sizes[mortar_id] = mortar_mesh_and_size.second;
  }

  // Create workspace vars for each element. Fill the operand with random values
  // within the subdomain and with zeros outside. Also construct a subdomain
  // with the same data.
  using Vars = Variables<all_fields_tags>;
  std::unordered_map<ElementId<volume_dim>, Vars> workspace{};
  using SubdomainDataType =
      LinearSolver::Schwarz::ElementCenteredSubdomainData<volume_dim,
                                                          all_fields_tags>;
  SubdomainDataType subdomain_data{};
  for (const auto& id_and_element : elements) {
    const auto& element_id = id_and_element.first;
    const size_t num_points =
        id_and_element.second.mesh.number_of_grid_points();
    if (element_id == subdomain_center) {
      subdomain_data.element_data = make_with_random_values<Vars>(
          make_not_null(&gen), make_not_null(&dist), DataVector{num_points});
      workspace[element_id] = subdomain_data.element_data;
    } else {
      workspace[element_id] = Vars{num_points, 0.};
    }
  }
  // Above we only filled the central element with random values. Now do the
  // same for the regions where the subdomain overlaps with neighbors. We also
  // construct the geometry of the overlaps. All data on an overlap with a
  // neighbor is oriented according to the neighbor's orientation, so
  // re-orientation needs to happen whenever data cross element boundaries.
  LinearSolver::Schwarz::OverlapMap<volume_dim, size_t> all_overlap_extents{};
  LinearSolver::Schwarz::OverlapMap<volume_dim, Mesh<volume_dim>>
      all_overlap_meshes{};
  LinearSolver::Schwarz::OverlapMap<volume_dim,
                                    ElementMap<volume_dim, Frame::Inertial>>
      all_overlap_element_maps{};
  LinearSolver::Schwarz::OverlapMap<
      volume_dim, ::dg::MortarMap<volume_dim, Mesh<volume_dim - 1>>>
      all_overlap_mortar_meshes;
  LinearSolver::Schwarz::OverlapMap<
      volume_dim, ::dg::MortarMap<volume_dim, ::dg::MortarSize<volume_dim - 1>>>
      all_overlap_mortar_sizes;
  for (const auto& direction_and_neighbors :
       central_element.element.neighbors()) {
    const auto& direction = direction_and_neighbors.first;
    const auto& neighbors = direction_and_neighbors.second;
    const auto& orientation = neighbors.orientation();
    const auto direction_from_neighbor = orientation(direction.opposite());
    const size_t dimension_in_neighbor = direction_from_neighbor.dimension();
    for (const auto& neighbor_id : neighbors) {
      const dg::MortarId<volume_dim> mortar_id{direction, neighbor_id};
      const auto& neighbor = elements.at(neighbor_id);
      all_overlap_meshes.emplace(std::make_pair(mortar_id, neighbor.mesh));
      all_overlap_element_maps.emplace(std::make_pair(
          mortar_id,
          ElementMap<volume_dim, Frame::Inertial>{
              neighbor_id, neighbor.element_map.block_map().get_clone()}));
      const size_t overlap_extent = LinearSolver::Schwarz::overlap_extent(
          neighbor.mesh.extents(dimension_in_neighbor), overlap);
      all_overlap_extents.emplace(std::make_pair(mortar_id, overlap_extent));
      auto overlap_vars = make_with_random_values<Vars>(
          make_not_null(&gen), make_not_null(&dist),
          DataVector{LinearSolver::Schwarz::overlap_num_points(
              neighbor.mesh.extents(), overlap_extent, dimension_in_neighbor)});
      workspace[neighbor_id] = LinearSolver::Schwarz::extended_overlap_data(
          overlap_vars, neighbor.mesh.extents(), overlap_extent,
          direction_from_neighbor);
      subdomain_data.overlap_data.emplace(
          std::make_pair(mortar_id, std::move(overlap_vars)));

      // Setup neighbor mortars
      const auto neighbor_mortars =
          helpers::create_mortars(neighbor_id, elements);
      ::dg::MortarMap<volume_dim, Mesh<volume_dim - 1>> mortar_meshes{};
      ::dg::MortarMap<volume_dim, ::dg::MortarSize<volume_dim - 1>>
          mortar_sizes{};
      std::transform(neighbor_mortars.begin(), neighbor_mortars.end(),
                     std::inserter(mortar_meshes, mortar_meshes.end()),
                     [](auto const& id_and_mortar) {
                       return std::make_pair(id_and_mortar.first,
                                             id_and_mortar.second.first);
                     });
      std::transform(neighbor_mortars.begin(), neighbor_mortars.end(),
                     std::inserter(mortar_sizes, mortar_sizes.end()),
                     [](auto const& id_and_mortar) {
                       return std::make_pair(id_and_mortar.first,
                                             id_and_mortar.second.second);
                     });
      all_overlap_mortar_meshes.emplace(
          std::make_pair(mortar_id, std::move(mortar_meshes)));
      all_overlap_mortar_sizes.emplace(
          std::make_pair(mortar_id, std::move(mortar_sizes)));
    }
  }
  const auto all_overlap_fluxes_args =
      package_fluxes_args(subdomain_center, elements, overlap);
  const auto all_overlap_sources_args =
      package_sources_args(subdomain_center, elements, overlap);

  CAPTURE(subdomain_data);

  // (1) Apply the full DG operator
  // We use the StrongFirstOrder scheme, so we'll need the n.F on the boundaries
  // and the data needed by the numerical flux.
  using BoundaryData = ::dg::FirstOrderScheme::BoundaryData<NumericalFlux>;
  const auto package_boundary_data =
      [&numerical_fluxes_computer, &fluxes_computer](
          const Mesh<volume_dim>& volume_mesh,
          const Direction<volume_dim>& direction,
          const tnsr::i<DataVector, volume_dim>& face_normal,
          const Scalar<DataVector>& face_normal_magnitude,
          const Variables<n_dot_fluxes_tags>& n_dot_fluxes,
          const Variables<div_fluxes_tags>& div_fluxes,
          const auto& fluxes_args) -> BoundaryData {
    return std::apply(
        [&](const auto&... expanded_fluxes_args) {
          const auto face_mesh = volume_mesh.slice_away(direction.dimension());
          return ::dg::FirstOrderScheme::package_boundary_data(
              numerical_fluxes_computer, face_mesh, n_dot_fluxes, volume_mesh,
              direction, face_normal_magnitude,
              get<::Tags::NormalDotFlux<AuxiliaryFields>>(n_dot_fluxes)...,
              get<::Tags::div<::Tags::Flux<
                  AuxiliaryFields, tmpl::size_t<volume_dim>, Frame::Inertial>>>(
                  div_fluxes)...,
              face_normal, fluxes_computer, expanded_fluxes_args...);
        },
        fluxes_args);
  };
  const auto apply_boundary_contribution =
      [&numerical_fluxes_computer](
          const auto result, const BoundaryData& local_boundary_data,
          const BoundaryData& remote_boundary_data,
          const Scalar<DataVector>& magnitude_of_face_normal,
          const Scalar<DataVector>& surface_jacobian,
          const Mesh<volume_dim>& mesh,
          const ::dg::MortarId<volume_dim>& mortar_id,
          const Mesh<volume_dim - 1>& mortar_mesh,
          const ::dg::MortarSize<volume_dim - 1>& mortar_size) {
        const size_t dimension = mortar_id.first.dimension();
        auto boundary_contribution = dg::FirstOrderScheme::boundary_flux(
            local_boundary_data, remote_boundary_data,
            numerical_fluxes_computer, mesh.slice_away(dimension), mortar_mesh,
            mortar_size);
        if constexpr (MassiveOperator) {
          boundary_contribution = ::dg::lift_flux_massive_no_mass_lumping(
              boundary_contribution, mesh.slice_away(dimension),
              surface_jacobian);
        } else {
          boundary_contribution =
              ::dg::lift_flux(boundary_contribution, mesh.extents(dimension),
                              magnitude_of_face_normal);
        }
        add_slice_to_data(result, std::move(boundary_contribution),
                          mesh.extents(), dimension,
                          index_to_slice_at(mesh.extents(), mortar_id.first));
      };
  std::unordered_map<ElementId<volume_dim>, Vars>
      operator_applied_to_workspace{};
  for (const auto& id_and_element : elements) {
    const auto& element_id = id_and_element.first;
    operator_applied_to_workspace[element_id] =
        helpers::apply_first_order_dg_operator<system, MassiveOperator>(
            element_id, elements, workspace, fluxes_computer,
            package_fluxes_args, package_sources_args, package_boundary_data,
            apply_boundary_contribution);
  }

  // (2) Apply the subdomain operator to the restricted data (as opposed to
  // applying the full DG operator to the full data and then restricting)
  const size_t center_num_points = central_element.mesh.number_of_grid_points();
  Variables<fluxes_tags> central_fluxes_buffer{center_num_points};
  Variables<div_fluxes_tags> central_div_fluxes_buffer{center_num_points};
  SubdomainDataType subdomain_result{center_num_points};
  const auto det_jacobian = determinant(central_element.element_map.jacobian(
      logical_coordinates(central_element.mesh)));
  elliptic::dg::apply_operator_volume<
      typename system::primal_fields, typename system::auxiliary_fields,
      typename system::sources, MassiveOperator>(
      make_not_null(&subdomain_result.element_data),
      make_not_null(&central_fluxes_buffer),
      make_not_null(&central_div_fluxes_buffer), fluxes_computer,
      central_element.mesh, det_jacobian, central_element.inv_jacobian,
      package_fluxes_args(subdomain_center, central_element),
      package_sources_args(subdomain_center, central_element),
      subdomain_data.element_data);
  for (const auto& direction_and_face_normal : internal_face_normals) {
    const auto& direction = direction_and_face_normal.first;
    elliptic::dg::apply_subdomain_face<
        domain::Tags::InternalDirections<volume_dim>,
        typename system::primal_fields, typename system::auxiliary_fields,
        typename system::sources, MassiveOperator>(
        make_not_null(&subdomain_result), central_element.element,
        central_element.mesh, fluxes_computer, numerical_fluxes_computer,
        direction, direction_and_face_normal.second,
        internal_face_normal_magnitudes.at(direction),
        internal_surface_jacobians.at(direction), central_mortar_meshes,
        central_mortar_sizes, all_overlap_extents, all_overlap_meshes,
        all_overlap_element_maps, all_overlap_mortar_meshes,
        all_overlap_mortar_sizes,
        package_fluxes_args(subdomain_center, central_element, direction),
        all_overlap_fluxes_args, all_overlap_sources_args, subdomain_data,
        central_fluxes_buffer, central_div_fluxes_buffer);
  }
  for (const auto& direction_and_face_normal : boundary_face_normals) {
    const auto& direction = direction_and_face_normal.first;
    elliptic::dg::apply_subdomain_face<
        domain::Tags::BoundaryDirectionsInterior<volume_dim>,
        typename system::primal_fields, typename system::auxiliary_fields,
        typename system::sources, MassiveOperator>(
        make_not_null(&subdomain_result), central_element.element,
        central_element.mesh, fluxes_computer, numerical_fluxes_computer,
        direction, direction_and_face_normal.second,
        boundary_face_normal_magnitudes.at(direction),
        boundary_surface_jacobians.at(direction), central_mortar_meshes,
        central_mortar_sizes, all_overlap_extents, all_overlap_meshes,
        all_overlap_element_maps, all_overlap_mortar_meshes,
        all_overlap_mortar_sizes,
        package_fluxes_args(subdomain_center, central_element, direction),
        all_overlap_fluxes_args, all_overlap_sources_args, subdomain_data,
        central_fluxes_buffer, central_div_fluxes_buffer);
  }

  // (3) Check the subdomain operator is equivalent to the full DG operator
  // restricted to the subdomain
  CHECK_VARIABLES_APPROX(subdomain_result.element_data,
                         operator_applied_to_workspace.at(subdomain_center));
  CHECK(subdomain_result.overlap_data.size() ==
        subdomain_data.overlap_data.size());
  for (const auto& mortar_id_and_overlap_result :
       subdomain_result.overlap_data) {
    const auto& mortar_id = mortar_id_and_overlap_result.first;
    const auto& overlap_result = mortar_id_and_overlap_result.second;
    const auto& direction = mortar_id.first;
    CAPTURE(direction);
    const auto& neighbor_id = mortar_id.second;
    CAPTURE(neighbor_id);
    const auto& neighbor = elements.at(neighbor_id);
    const auto& orientation =
        central_element.element.neighbors().at(direction).orientation();
    const auto& direction_from_neighbor = orientation(direction.opposite());
    const auto overlap_result_from_workspace =
        LinearSolver::Schwarz::data_on_overlap(
            operator_applied_to_workspace.at(neighbor_id),
            neighbor.mesh.extents(), all_overlap_extents.at(mortar_id),
            direction_from_neighbor);
    CHECK_VARIABLES_APPROX(overlap_result, overlap_result_from_workspace);
  }

  // (4) Check the subdomain operator works with the DataBox
  using numerical_flux_tag = ::Tags::NumericalFlux<NumericalFlux>;
  using SubdomainOperator = elliptic::dg::SubdomainOperator<
      volume_dim, typename system::primal_fields,
      typename system::auxiliary_fields, fluxes_computer_tag, FluxesArgs,
      typename system::sources, SourcesArgs, numerical_flux_tag,
      DummyOptionsGroup, FluxesArgsTagsFromCenter, MassiveOperator>;
  auto initial_box = db::create<
      db::AddSimpleTags<
          domain::Tags::Element<volume_dim>, domain::Tags::Mesh<volume_dim>,
          domain::Tags::DetJacobian<Frame::Logical, Frame::Inertial>,
          domain::Tags::InverseJacobian<volume_dim, Frame::Logical,
                                        Frame::Inertial>,
          fluxes_computer_tag, numerical_flux_tag,
          domain::Tags::Interface<
              domain::Tags::InternalDirections<volume_dim>,
              ::Tags::Normalized<
                  domain::Tags::UnnormalizedFaceNormal<volume_dim>>>,
          domain::Tags::Interface<
              domain::Tags::BoundaryDirectionsInterior<volume_dim>,
              ::Tags::Normalized<
                  domain::Tags::UnnormalizedFaceNormal<volume_dim>>>,
          domain::Tags::Interface<
              domain::Tags::InternalDirections<volume_dim>,
              ::Tags::Magnitude<
                  domain::Tags::UnnormalizedFaceNormal<volume_dim>>>,
          domain::Tags::Interface<
              domain::Tags::BoundaryDirectionsInterior<volume_dim>,
              ::Tags::Magnitude<
                  domain::Tags::UnnormalizedFaceNormal<volume_dim>>>,
          domain::Tags::Interface<
              domain::Tags::InternalDirections<volume_dim>,
              domain::Tags::SurfaceJacobian<Frame::Logical, Frame::Inertial>>,
          domain::Tags::Interface<
              domain::Tags::BoundaryDirectionsInterior<volume_dim>,
              domain::Tags::SurfaceJacobian<Frame::Logical, Frame::Inertial>>,
          ::Tags::Mortars<domain::Tags::Mesh<volume_dim - 1>, volume_dim>,
          ::Tags::Mortars<::Tags::MortarSize<volume_dim - 1>, volume_dim>,
          LinearSolver::Schwarz::Tags::Overlaps<
              elliptic::dg::Tags::OverlapExtent, volume_dim, DummyOptionsGroup>,
          LinearSolver::Schwarz::Tags::Overlaps<domain::Tags::Mesh<volume_dim>,
                                                volume_dim, DummyOptionsGroup>,
          LinearSolver::Schwarz::Tags::Overlaps<
              domain::Tags::ElementMap<volume_dim>, volume_dim,
              DummyOptionsGroup>,
          LinearSolver::Schwarz::Tags::Overlaps<
              ::Tags::Mortars<domain::Tags::Mesh<volume_dim - 1>, volume_dim>,
              volume_dim, DummyOptionsGroup>,
          LinearSolver::Schwarz::Tags::Overlaps<
              ::Tags::Mortars<::Tags::MortarSize<volume_dim - 1>, volume_dim>,
              volume_dim, DummyOptionsGroup>>,
      db::AddComputeTags<
          domain::Tags::InternalDirections<volume_dim>,
          domain::Tags::BoundaryDirectionsInterior<volume_dim>,
          domain::Tags::InterfaceCompute<
              domain::Tags::InternalDirections<volume_dim>,
              domain::Tags::Direction<volume_dim>>,
          domain::Tags::InterfaceCompute<
              domain::Tags::BoundaryDirectionsInterior<volume_dim>,
              domain::Tags::Direction<volume_dim>>>>(
      central_element.element, central_element.mesh, det_jacobian,
      central_element.inv_jacobian, fluxes_computer, numerical_fluxes_computer,
      internal_face_normals, boundary_face_normals,
      internal_face_normal_magnitudes, boundary_face_normal_magnitudes,
      internal_surface_jacobians, boundary_surface_jacobians,
      central_mortar_meshes, central_mortar_sizes,
      std::move(all_overlap_extents), std::move(all_overlap_meshes),
      std::move(all_overlap_element_maps), std::move(all_overlap_mortar_meshes),
      std::move(all_overlap_mortar_sizes));
  auto box_with_fluxes_args = std::apply(
      [&initial_box](const auto&... expanded_fluxes_args) {
        return db::create_from<db::RemoveTags<>, FluxesArgsTags>(
            std::move(initial_box), expanded_fluxes_args...);
      },
      package_fluxes_args(subdomain_center, central_element));
  using fluxes_args_interface_tags =
      tmpl::list_difference<FluxesArgsTags, FluxesArgsVolumeTags>;
  using fluxes_args_internal_faces_tags = tmpl::transform<
      fluxes_args_interface_tags,
      tmpl::bind<domain::Tags::Interface,
                 tmpl::pin<domain::Tags::InternalDirections<volume_dim>>,
                 tmpl::_1>>;
  tmpl::as_tuple<tmpl::transform<fluxes_args_internal_faces_tags,
                                 tmpl::bind<db::item_type, tmpl::_1>>>
      fluxes_internal_faces_args{};
  for (const auto& direction :
       get<domain::Tags::InternalDirections<volume_dim>>(
           box_with_fluxes_args)) {
    auto this_direction_args = std::apply(
        [](const auto&... expanded_fluxes_interface_args) {
          return tuples::tagged_tuple_from_typelist<FluxesArgsTags>(
              expanded_fluxes_interface_args...);
        },
        package_fluxes_args(subdomain_center, central_element, direction));
    tmpl::for_each<fluxes_args_interface_tags>(
        [&fluxes_internal_faces_args, &direction,
         &this_direction_args](auto tag_v) noexcept {
          using tag = tmpl::type_from<decltype(tag_v)>;
          using tag_index = tmpl::index_of<fluxes_args_interface_tags, tag>;
          get<tag_index::value>(fluxes_internal_faces_args)
              .emplace(direction, std::move(get<tag>(this_direction_args)));
        });
  }
  using fluxes_args_boundary_faces_tags = tmpl::transform<
      fluxes_args_interface_tags,
      tmpl::bind<
          domain::Tags::Interface,
          tmpl::pin<domain::Tags::BoundaryDirectionsInterior<volume_dim>>,
          tmpl::_1>>;
  tmpl::as_tuple<tmpl::transform<fluxes_args_boundary_faces_tags,
                                 tmpl::bind<db::item_type, tmpl::_1>>>
      fluxes_boundary_faces_args{};
  for (const auto& direction :
       get<domain::Tags::BoundaryDirectionsInterior<volume_dim>>(
           box_with_fluxes_args)) {
    auto this_direction_args = std::apply(
        [](const auto&... expanded_fluxes_interface_args) {
          return tuples::tagged_tuple_from_typelist<FluxesArgsTags>(
              expanded_fluxes_interface_args...);
        },
        package_fluxes_args(subdomain_center, central_element, direction));
    tmpl::for_each<fluxes_args_interface_tags>(
        [&fluxes_boundary_faces_args, &direction,
         &this_direction_args](auto tag_v) noexcept {
          using tag = tmpl::type_from<decltype(tag_v)>;
          using tag_index = tmpl::index_of<fluxes_args_interface_tags, tag>;
          get<tag_index::value>(fluxes_boundary_faces_args)
              .emplace(direction, std::move(get<tag>(this_direction_args)));
        });
  }
  using fluxes_args_overlap_tags = tmpl::transform<
      FluxesArgsTags,
      elliptic::dg::make_overlap_tag_impl<
          tmpl::_1, tmpl::pin<tmpl::size_t<volume_dim>>,
          tmpl::pin<DummyOptionsGroup>, tmpl::pin<FluxesArgsVolumeTags>>>;
  using sources_args_overlap_tags = tmpl::transform<
      SourcesArgsTags,
      elliptic::dg::make_overlap_tag_impl<
          tmpl::_1, tmpl::pin<tmpl::size_t<volume_dim>>,
          tmpl::pin<DummyOptionsGroup>, tmpl::pin<tmpl::list<>>>>;
  auto box_with_fluxes_interface_args = std::apply(
      [&box_with_fluxes_args](const auto&... expanded_args) {
        return ::Initialization::merge_into_databox<
            DummyOptionsGroup,
            tmpl::append<fluxes_args_internal_faces_tags,
                         fluxes_args_boundary_faces_tags,
                         fluxes_args_overlap_tags, sources_args_overlap_tags>,
            db::AddComputeTags<>, ::Initialization::MergePolicy::Overwrite>(
            std::move(box_with_fluxes_args), expanded_args...);
      },
      std::tuple_cat(fluxes_internal_faces_args, fluxes_boundary_faces_args,
                     all_overlap_fluxes_args, all_overlap_sources_args));
  auto box_for_operator = std::apply(
      [&box_with_fluxes_interface_args](const auto&... expanded_sources_args) {
        return db::create_from<db::RemoveTags<>, SourcesArgsTags>(
            std::move(box_with_fluxes_interface_args),
            expanded_sources_args...);
      },
      package_sources_args(subdomain_center, central_element));

  SubdomainOperator subdomain_operator{center_num_points};
  SubdomainDataType subdomain_result_db{center_num_points};
  db::apply<typename SubdomainOperator::element_operator>(
      box_for_operator, subdomain_data, make_not_null(&subdomain_result_db),
      make_not_null(&subdomain_operator));
  using face_operator_internal =
      typename SubdomainOperator::template face_operator<
          domain::Tags::InternalDirections<volume_dim>>;
  using face_operator_external =
      typename SubdomainOperator::template face_operator<
          domain::Tags::BoundaryDirectionsInterior<volume_dim>>;
  interface_apply<domain::Tags::InternalDirections<volume_dim>,
                  typename face_operator_internal::argument_tags,
                  get_volume_tags<face_operator_internal>>(
      face_operator_internal{}, box_for_operator, subdomain_data,
      make_not_null(&subdomain_result_db), make_not_null(&subdomain_operator));
  interface_apply<domain::Tags::BoundaryDirectionsInterior<volume_dim>,
                  typename face_operator_external::argument_tags,
                  get_volume_tags<face_operator_external>>(
      face_operator_external{}, box_for_operator, subdomain_data,
      make_not_null(&subdomain_result_db), make_not_null(&subdomain_operator));
  CHECK(subdomain_result_db.element_data == subdomain_result.element_data);
  CHECK(subdomain_result_db.overlap_data == subdomain_result.overlap_data);
}

template <typename System, typename FluxesArgsTags,
          typename FluxesArgsVolumeTags, typename FluxesArgsTagsFromCenter,
          typename SourcesArgsTags, typename SourcesArgsVolumeTags,
          bool MassiveOperator, typename... Args>
void test_subdomain_operator(Args&&... args) noexcept {
  test_subdomain_operator_impl<System, FluxesArgsTags, FluxesArgsVolumeTags,
                               FluxesArgsTagsFromCenter, SourcesArgsTags,
                               SourcesArgsVolumeTags, MassiveOperator>(
      std::forward<Args>(args)..., typename System::primal_fields{},
      typename System::auxiliary_fields{});
}

template <size_t Dim>
void test_subdomain_operator_poisson(const DomainCreator<Dim>& domain_creator,
                                     const double penalty_parameter) {
  INFO("Poisson system");
  CAPTURE(Dim);
  using system = Poisson::FirstOrderSystem<Dim, Poisson::Geometry::Euclidean>;
  for (size_t overlap = 1; overlap <= 4; overlap++) {
    test_subdomain_operator<system, tmpl::list<>, tmpl::list<>, tmpl::list<>,
                            tmpl::list<>, tmpl::list<>, true>(
        domain_creator, overlap, penalty_parameter,
        [](const auto&... /*unused*/) { return std::tuple<>{}; },
        [](const auto&... /*unused*/) { return std::tuple<>{}; });
    test_subdomain_operator<system, tmpl::list<>, tmpl::list<>, tmpl::list<>,
                            tmpl::list<>, tmpl::list<>, false>(
        domain_creator, overlap, penalty_parameter,
        [](const auto&... /*unused*/) { return std::tuple<>{}; },
        [](const auto&... /*unused*/) { return std::tuple<>{}; });
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.DG.SubdomainOperator", "[Unit][Elliptic]") {
  {
    INFO("Aligned elements");
    const domain::creators::Interval domain_creator_1d{
        {{-2.}}, {{2.}}, {{false}}, {{1}}, {{3}}};
    test_subdomain_operator_poisson(domain_creator_1d, 1.);
    const domain::creators::Rectangle domain_creator_2d{
        {{-2., 0.}}, {{2., 1.}}, {{false, false}}, {{1, 1}}, {{3, 3}}};
    test_subdomain_operator_poisson(domain_creator_2d, 1.);
    const domain::creators::Brick domain_creator_3d{{{-2., 0., -1.}},
                                                    {{2., 1., 1.}},
                                                    {{false, false, false}},
                                                    {{1, 1, 1}},
                                                    {{3, 3, 3}}};
    test_subdomain_operator_poisson(domain_creator_3d, 1.);
  }
  {
    INFO("Rotated elements");
    const domain::creators::RotatedIntervals domain_creator_1d{
        {{-2.}}, {{0.}}, {{2.}}, {{false}}, {{0}}, {{{{3, 3}}}}};
    test_subdomain_operator_poisson(domain_creator_1d, 1.);
    const domain::creators::RotatedRectangles domain_creator_2d{
        {{-2., 0.}},      {{0., 0.5}}, {{2., 1.}},
        {{false, false}}, {{0, 0}},    {{{{3, 3}}, {{3, 3}}}}};
    test_subdomain_operator_poisson(domain_creator_2d, 1.);
    const domain::creators::RotatedBricks domain_creator_3d{
        {{-2., 0., -1.}}, {{0., 0.5, 0.}},
        {{2., 1., 1.}},   {{false, false, false}},
        {{1, 1, 1}},      {{{{3, 3}}, {{3, 3}}, {{3, 3}}}}};
    test_subdomain_operator_poisson(domain_creator_3d, 1.);
  }
  {
    INFO("Refined elements");
    const domain::creators::AlignedLattice<1> domain_creator_1d{
        {{{-2., 0., 2.}}},       {{false}}, {{0}}, {{3}}, {},
        {{{{0}}, {{1}}, {{4}}}}, {}};
    test_subdomain_operator_poisson(domain_creator_1d, 1.);
    const domain::creators::AlignedLattice<2> domain_creator_2d{
        {{{-2., 0., 2.}, {-2., 0., 2.}}},
        {{false, false}},
        {{0, 0}},
        {{3, 3}},
        {{{{1, 0}}, {{2, 1}}, {{0, 1}}}},
        {{{{0, 0}}, {{1, 1}}, {{4, 3}}}},
        {}};
    test_subdomain_operator_poisson(domain_creator_2d, 1.);
    const domain::creators::AlignedLattice<3> domain_creator_3d{
        {{{-2., 0., 2.}, {-2., 0., 2.}, {-2., 0., 2.}}},
        {{false, false, false}},
        {{0, 0, 0}},
        {{3, 3, 3}},
        {{{{1, 0, 0}}, {{2, 1, 1}}, {{0, 1, 1}}}},
        {{{{0, 0, 0}}, {{1, 1, 1}}, {{4, 3, 2}}}},
        {}};
    test_subdomain_operator_poisson(domain_creator_3d, 1.);
  }
  {
    using system = Elasticity::FirstOrderSystem<3>;
    using ConstitutiveRelationType =
        Elasticity::ConstitutiveRelations::IsotropicHomogeneous<3>;
    ConstitutiveRelationType constitutive_relation{1., 2.};
    const domain::creators::Brick domain_creator{{{-2., 0., -1.}},
                                                 {{2., 1., 1.}},
                                                 {{false, false, false}},
                                                 {{1, 1, 1}},
                                                 {{3, 3, 3}}};
    for (size_t overlap = 1; overlap <= 4; overlap++) {
      test_subdomain_operator<
          system,
          tmpl::list<::Elasticity::Tags::ConstitutiveRelation<
                         ConstitutiveRelationType>,
                     ::domain::Tags::Coordinates<3, Frame::Inertial>>,
          tmpl::list<::Elasticity::Tags::ConstitutiveRelation<
              ConstitutiveRelationType>>,
          tmpl::list<::Elasticity::Tags::ConstitutiveRelationBase>,
          tmpl::list<>, tmpl::list<>, false>(
          domain_creator, overlap, 6.75,
          make_overloader(
              [&constitutive_relation](
                  const ElementId<3>& /*element_id*/,
                  const helpers::DgElement<3>& dg_element) {
                return std::make_tuple(
                    constitutive_relation,
                    dg_element.element_map(
                        logical_coordinates(dg_element.mesh)));
              },
              [&constitutive_relation](const ElementId<3>& /*element_id*/,
                                       const helpers::DgElement<3>& dg_element,
                                       const Direction<3>& direction) {
                return std::make_tuple(
                    constitutive_relation,
                    dg_element.element_map(interface_logical_coordinates(
                        dg_element.mesh.slice_away(direction.dimension()),
                        direction)));
              },
              [&constitutive_relation](
                  const ElementId<3>& element_id,
                  const helpers::DgElementArray<3>& dg_elements,
                  const size_t max_overlap) {
                const auto& dg_element = dg_elements.at(element_id);
                LinearSolver::Schwarz::OverlapMap<3, tnsr::I<DataVector, 3>>
                    all_overlap_inertial_coords{};
                for (const auto& direction_and_neighbors :
                     dg_element.element.neighbors()) {
                  const auto& direction = direction_and_neighbors.first;
                  const auto& neighbors = direction_and_neighbors.second;
                  const auto& orientation = neighbors.orientation();
                  const auto direction_in_neighbor =
                      orientation(direction.opposite());
                  for (const auto& neighbor_id : neighbors) {
                    const auto overlap_id =
                        std::make_pair(direction, neighbor_id);
                    const auto& neighbor = dg_elements.at(neighbor_id);
                    const auto intruding_extent =
                        LinearSolver::Schwarz::overlap_extent(
                            neighbor.mesh.extents(
                                direction_in_neighbor.dimension()),
                            max_overlap);
                    const auto neighbor_inertial_coords = neighbor.element_map(
                        logical_coordinates(neighbor.mesh));
                    all_overlap_inertial_coords.emplace(
                        overlap_id,
                        LinearSolver::Schwarz::data_on_overlap(
                            neighbor_inertial_coords, neighbor.mesh.extents(),
                            intruding_extent, direction_in_neighbor));
                  }
                }
                return std::make_tuple(constitutive_relation,
                                       std::move(all_overlap_inertial_coords));
              }),
          [](const auto&... /*unused*/) { return std::tuple<>{}; });
    }
  }
}
