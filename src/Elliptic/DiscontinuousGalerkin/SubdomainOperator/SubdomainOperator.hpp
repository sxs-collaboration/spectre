// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <optional>
#include <ostream>
#include <pup.h>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Domain/Tags/Faces.hpp"
#include "Domain/Tags/NeighborMesh.hpp"
#include "Domain/Tags/SurfaceJacobian.hpp"
#include "Elliptic/BoundaryConditions/ApplyBoundaryCondition.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/Tags.hpp"
#include "Elliptic/Systems/GetSourcesComputer.hpp"
#include "Elliptic/Utilities/ApplyAt.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementCenteredSubdomainData.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/SubdomainOperator.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/TMPL.hpp"

/// Items related to the restriction of the DG operator to an element-centered
/// subdomain
namespace elliptic::dg::subdomain_operator {

namespace detail {
// Wrap the `Tag` in `LinearSolver::Schwarz::Tags::Overlaps`, except if it is
// included in `TakeFromCenterTags`.
template <typename Tag, typename Dim, typename OptionsGroup,
          typename TakeFromCenterTags>
struct make_overlap_tag_impl {
  using type = tmpl::conditional_t<
      tmpl::list_contains_v<TakeFromCenterTags, Tag>, Tag,
      LinearSolver::Schwarz::Tags::Overlaps<Tag, Dim::value, OptionsGroup>>;
};

// Wrap the `Tag` in `Tags::NeighborMortars`
template <typename Tag, typename Dim>
struct make_neighbor_mortars_tag_impl {
  using type = Tags::NeighborMortars<Tag, Dim::value>;
};
}  // namespace detail

/*!
 * \brief The elliptic DG operator on an element-centered subdomain
 *
 * This operator is a restriction of the full (linearized) DG-operator to an
 * element-centered subdomain with a few points overlap into neighboring
 * elements. It is a `LinearSolver::Schwarz::SubdomainOperator` to be used with
 * the Schwarz linear solver when it solves the elliptic DG operator.
 *
 * This operator requires the following tags are available on overlap regions
 * with neighboring elements:
 *
 * - Geometric quantities provided by
 *   `elliptic::dg::subdomain_operator::InitializeSubdomain`.
 * - All `System::fluxes_computer::argument_tags` and
 *   `System::sources_computer::argument_tags` (or
 *   `System::sources_computer_linearized::argument_tags` for nonlinear
 *   systems), except those listed in `ArgsTagsFromCenter`. The latter will be
 *   taken from the central element's DataBox, so they don't need to be made
 *   available on overlaps.
 * - The `System::fluxes_computer::argument_tags` on internal and external
 *   interfaces, except those listed in `System::fluxes_computer::volume_tags`.
 *
 * Some of these tags may require communication between elements. For example,
 * nonlinear system fields are constant background fields for the linearized DG
 * operator, but are updated in every nonlinear solver iteration. Therefore, the
 * updated nonlinear fields must be communicated across overlaps between
 * nonlinear solver iterations. To perform the communication you can use
 * `LinearSolver::Schwarz::Actions::SendOverlapFields` and
 * `LinearSolver::Schwarz::Actions::ReceiveOverlapFields`, setting
 * `RestrictToOverlap` to `false`. See
 * `LinearSolver::Schwarz::SubdomainOperator` for details.
 *
 * \par Overriding boundary conditions
 * Sometimes the subdomain operator should not use the boundary conditions that
 * have been selected when setting up the domain. For example, when the
 * subdomain operator is cached between non-linear solver iterations but the
 * boundary conditions depend on the non-linear fields, the preconditioning can
 * become ineffective (see
 * `LinearSolver::Schwarz::Actions::ResetSubdomainSolver`). Another example is
 * `elliptic::subdomain_preconditioners::MinusLaplacian`, where an auxiliary
 * Poisson system is used for preconditioning that doesn't have boundary
 * conditions set up in the domain. In these cases, the boundary conditions used
 * for the subdomain operator can be overridden with the optional
 * `override_boundary_conditions` argument to the `operator()`. If the
 * overriding boundary conditions are different from those listed in
 * `Metavariables::factory_creation`, you can supply the list of
 * boundary-condition classes to the `BoundaryConditionClasses` template
 * parameter. Note that the subdomain operator always applies the _linearized_
 * boundary conditions.
 *
 * \warning The subdomain operator hasn't been tested with periodic boundary
 * conditions so far.
 */
template <typename System, typename OptionsGroup,
          typename BoundaryConditionClasses = tmpl::list<>>
struct SubdomainOperator
    : LinearSolver::Schwarz::SubdomainOperator<System::volume_dim> {
 public:
  using system = System;
  using options_group = OptionsGroup;

 private:
  static constexpr size_t Dim = System::volume_dim;

  // Operator applications happen sequentially so we don't have to keep track of
  // the temporal id
  static constexpr size_t temporal_id = 0;

  // The subdomain operator always applies the linearized DG operator
  static constexpr bool linearized = true;

  using BoundaryConditionsBase = typename System::boundary_conditions_base;

  // These are the arguments that we need to retrieve from the DataBox and pass
  // to the functions in `elliptic::dg`, both on the central element and on
  // neighbors
  using prepare_args_tags =
      tmpl::list<domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>,
                 domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                               Frame::Inertial>,
                 domain::Tags::Faces<Dim, domain::Tags::FaceNormal<Dim>>,
                 ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
                 ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>;
  using apply_args_tags = tmpl::list<
      domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>,
      domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                    Frame::Inertial>,
      domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>,
      domain::Tags::DetJacobian<Frame::ElementLogical, Frame::Inertial>,
      domain::Tags::DetTimesInvJacobian<Dim, Frame::ElementLogical,
                                        Frame::Inertial>,
      domain::Tags::Faces<Dim, domain::Tags::FaceNormal<Dim>>,
      domain::Tags::Faces<Dim, domain::Tags::FaceNormalVector<Dim>>,
      domain::Tags::Faces<Dim,
                          domain::Tags::UnnormalizedFaceNormalMagnitude<Dim>>,
      domain::Tags::Faces<Dim, domain::Tags::DetSurfaceJacobian<
                                   Frame::ElementLogical, Frame::Inertial>>,
      domain::Tags::Faces<
          Dim, domain::Tags::DetTimesInvJacobian<Dim, Frame::ElementLogical,
                                                 Frame::Inertial>>,
      ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
      ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>,
      ::Tags::Mortars<domain::Tags::DetSurfaceJacobian<Frame::ElementLogical,
                                                       Frame::Inertial>,
                      Dim>,
      ::Tags::Mortars<elliptic::dg::Tags::PenaltyFactor, Dim>,
      elliptic::dg::Tags::Massive, elliptic::dg::Tags::Formulation>;
  using fluxes_args_tags = typename System::fluxes_computer::argument_tags;
  using sources_args_tags =
      elliptic::get_sources_argument_tags<System, linearized>;

  // We need the fluxes args also on interfaces (internal and external). The
  // volume tags are the subset that don't have to be taken from interfaces.
  using fluxes_args_volume_tags = typename System::fluxes_computer::volume_tags;

  // These tags can be taken directly from the central element's DataBox, even
  // when evaluating neighbors
  using args_tags_from_center = tmpl::remove_duplicates<tmpl::push_back<
      typename System::fluxes_computer::const_global_cache_tags,
      elliptic::dg::Tags::Massive, elliptic::dg::Tags::Formulation>>;

  // Data on neighbors is stored in the central element's DataBox in
  // `LinearSolver::Schwarz::Tags::Overlaps` maps, so we wrap the argument tags
  // with this prefix
  using make_overlap_tag =
      detail::make_overlap_tag_impl<tmpl::_1, tmpl::pin<tmpl::size_t<Dim>>,
                                    tmpl::pin<OptionsGroup>,
                                    tmpl::pin<args_tags_from_center>>;
  using prepare_args_tags_overlap =
      tmpl::transform<prepare_args_tags, make_overlap_tag>;
  using apply_args_tags_overlap =
      tmpl::transform<apply_args_tags, make_overlap_tag>;
  using fluxes_args_tags_overlap =
      tmpl::transform<fluxes_args_tags, make_overlap_tag>;
  using sources_args_tags_overlap =
      tmpl::transform<sources_args_tags, make_overlap_tag>;
  using fluxes_args_tags_overlap_faces = tmpl::transform<
      domain::make_faces_tags<Dim, fluxes_args_tags, fluxes_args_volume_tags>,
      make_overlap_tag>;

  // We also need some data on the remote side of all neighbors' mortars. Such
  // data is stored in the central element's DataBox in `Tags::NeighborMortars`
  // maps
  using make_neighbor_mortars_tag =
      detail::make_neighbor_mortars_tag_impl<tmpl::_1,
                                             tmpl::pin<tmpl::size_t<Dim>>>;

 public:
  /// \warning This function is not thread-safe because it accesses mutable
  /// memory buffers.
  template <typename ResultTags, typename OperandTags, typename DbTagsList>
  void operator()(
      const gsl::not_null<
          LinearSolver::Schwarz::ElementCenteredSubdomainData<Dim, ResultTags>*>
          result,
      const LinearSolver::Schwarz::ElementCenteredSubdomainData<
          Dim, OperandTags>& operand,
      const db::DataBox<DbTagsList>& box,
      const std::unordered_map<std::pair<size_t, Direction<Dim>>,
                               const BoundaryConditionsBase&,
                               boost::hash<std::pair<size_t, Direction<Dim>>>>&
          override_boundary_conditions = {}) const {
    result->destructive_resize(operand);
    central_mortar_data_.clear();
    neighbors_mortar_data_.clear();

    // Used to retrieve items out of the DataBox to forward to functions. This
    // replaces a long series of db::get calls.
    const auto get_items = [](const auto&... args) {
      return std::forward_as_tuple(args...);
    };

    // Retrieve data out of the DataBox
    using tags_to_retrieve = tmpl::flatten<tmpl::list<
        domain::Tags::ExternalBoundaryConditions<Dim>,
        domain::Tags::Element<Dim>,
        ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
        // Data on overlaps with neighbors
        tmpl::transform<
            tmpl::flatten<tmpl::list<
                Tags::ExtrudingExtent, domain::Tags::Element<Dim>,
                domain::Tags::Mesh<Dim>, domain::Tags::NeighborMesh<Dim>,
                domain::Tags::Faces<
                    Dim, domain::Tags::UnnormalizedFaceNormalMagnitude<Dim>>,
                ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
                ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>,
                // Data on the remote side of the neighbor's mortars
                tmpl::transform<
                    tmpl::list<
                        domain::Tags::UnnormalizedFaceNormalMagnitude<Dim>,
                        domain::Tags::Mesh<Dim - 1>,
                        ::Tags::MortarSize<Dim - 1>>,
                    make_neighbor_mortars_tag>>>,
            make_overlap_tag>>>;
    const auto& [external_boundary_conditions, central_element,
                 central_mortar_meshes, all_overlap_extents,
                 all_neighbor_elements, all_neighbor_meshes,
                 all_neighbors_neighbor_meshes,
                 all_neighbor_face_normal_magnitudes,
                 all_neighbor_mortar_meshes, all_neighbor_mortar_sizes,
                 all_neighbors_neighbor_face_normal_magnitudes,
                 all_neighbors_neighbor_mortar_meshes,
                 all_neighbors_neighbor_mortar_sizes] =
        db::apply<tags_to_retrieve>(get_items, box);
    const auto fluxes_args = db::apply<fluxes_args_tags>(get_items, box);
    const auto sources_args = db::apply<sources_args_tags>(get_items, box);
    using FluxesArgs = std::decay_t<decltype(fluxes_args)>;
    DirectionMap<Dim, FluxesArgs> fluxes_args_on_faces{};
    for (const auto& direction : Direction<Dim>::all_directions()) {
      fluxes_args_on_faces.emplace(
          direction, elliptic::util::apply_at<
                         domain::make_faces_tags<Dim, fluxes_args_tags,
                                                 fluxes_args_volume_tags>,
                         fluxes_args_volume_tags>(get_items, box, direction));
    }

    // Setup boundary conditions
    const auto apply_boundary_condition =
        [&box, &all_boundary_conditions = external_boundary_conditions,
         &override_boundary_conditions](const ElementId<Dim>& local_element_id,
                                        const Direction<Dim>& local_direction,
                                        auto is_overlap, const auto& map_keys,
                                        auto&&... fields_and_fluxes) {
          constexpr bool is_overlap_v =
              std::decay_t<decltype(is_overlap)>::value;
          // Get boundary conditions from domain, or use overridden boundary
          // conditions
          const auto& boundary_condition = [&all_boundary_conditions,
                                            &local_element_id, &local_direction,
                                            &override_boundary_conditions]()
              -> const BoundaryConditionsBase& {
            if (not override_boundary_conditions.empty()) {
              const auto found_overridden_boundary_conditions =
                  override_boundary_conditions.find(
                      {local_element_id.block_id(), local_direction});
              ASSERT(found_overridden_boundary_conditions !=
                         override_boundary_conditions.end(),
                     "Overriding boundary conditions in subdomain operator, "
                     "but none is available for block "
                         << local_element_id.block_id() << " in direction "
                         << local_direction
                         << ". Make sure you have considered this external "
                            "boundary of the subdomain. If this is "
                            "intentional, add support to "
                            "elliptic::dg::SubdomainOperator.");
              return found_overridden_boundary_conditions->second;
            }
            const auto& boundary_conditions =
                all_boundary_conditions.at(local_element_id.block_id());
            ASSERT(boundary_conditions.contains(local_direction),
                   "No boundary condition is available in block "
                       << local_element_id.block_id() << " in direction "
                       << local_direction
                       << ". Make sure you are setting up boundary conditions "
                          "when creating the domain.");
            ASSERT(
                dynamic_cast<const BoundaryConditionsBase*>(
                    boundary_conditions.at(local_direction).get()) != nullptr,
                "The boundary condition in block "
                    << local_element_id.block_id() << " in direction "
                    << local_direction
                    << " has an unexpected type. Make sure it derives off the "
                       "'boundary_conditions_base' class set in the system.");
            return dynamic_cast<const BoundaryConditionsBase&>(
                *boundary_conditions.at(local_direction));
          }();
          elliptic::apply_boundary_condition<
              linearized,
              tmpl::conditional_t<is_overlap_v, make_overlap_tag, void>,
              BoundaryConditionClasses>(
              boundary_condition, box, map_keys,
              std::forward<decltype(fields_and_fluxes)>(fields_and_fluxes)...);
        };

    // Check if the subdomain data is sparse, i.e. if some elements have zero
    // data. If they are, the operator is a lot cheaper to apply due to its
    // linearity.
    std::unordered_set<ElementId<Dim>> elements_in_subdomain{
        central_element.id()};
    std::unordered_set<ElementId<Dim>> elements_with_zero_data{};
    if (equal_within_roundoff(operand.element_data, 0.)) {
      elements_with_zero_data.insert(central_element.id());
    }
    for (const auto& [overlap_id, overlap_data] : operand.overlap_data) {
      elements_in_subdomain.insert(overlap_id.id());
      if (equal_within_roundoff(overlap_data, 0.)) {
        elements_with_zero_data.insert(overlap_id.id());
      }
    }
    const auto is_in_subdomain =
        [&elements_in_subdomain](const ElementId<Dim>& element_id) {
          return elements_in_subdomain.find(element_id) !=
                 elements_in_subdomain.end();
        };
    const auto data_is_zero = [&elements_with_zero_data, &is_in_subdomain](
                                  const ElementId<Dim>& element_id) {
      return elements_with_zero_data.find(element_id) !=
                 elements_with_zero_data.end() or
             // Data outside the subdomain is zero by definition
             not is_in_subdomain(element_id);
    };
    const bool central_data_is_zero = data_is_zero(central_element.id());

    // The subdomain operator essentially does two sweeps over all elements in
    // the subdomain: In the first sweep it prepares the mortar data and stores
    // them on both sides of all mortars, and in the second sweep it consumes
    // the mortar data to apply the operator. This implementation is relatively
    // simple because it can re-use the implementation for the parallel DG
    // operator. However, it is also possible to apply the subdomain operator in
    // a single sweep over all elements, incrementally building up the mortar
    // data and applying boundary corrections immediately to both adjacent
    // elements once the data is available. That approach is possibly a
    // performance optimization but requires re-implementing a lot of logic for
    // the DG operator here. It should be considered once the subdomain operator
    // has been identified as the performance bottleneck. An alternative to
    // optimizing the subdomain operator performance is to precondition the
    // subdomain solve with a _much_ simpler subdomain operator, such as a
    // finite-difference Laplacian, so fewer applications of the more expensive
    // DG subdomain operator are necessary.

    // 1. Prepare mortar data on all elements in the subdomain and store them on
    //    mortars, reorienting if needed
    //
    // Prepare central element
    const auto apply_boundary_condition_center =
        [&apply_boundary_condition, &local_central_element = central_element](
            const Direction<Dim>& local_direction,
            auto&&... fields_and_fluxes) {
          apply_boundary_condition(
              local_central_element.id(), local_direction, std::false_type{},
              local_direction,
              std::forward<decltype(fields_and_fluxes)>(fields_and_fluxes)...);
        };
    db::apply<prepare_args_tags>(
        [this, &operand](const auto&... args) {
          elliptic::dg::prepare_mortar_data<System, linearized>(
              make_not_null(&central_deriv_vars_),
              make_not_null(&central_primal_fluxes_),
              make_not_null(&central_mortar_data_), operand.element_data,
              args...);
        },
        box, temporal_id, apply_boundary_condition_center, fluxes_args,
        data_is_zero);
    // Prepare neighbors
    for (const auto& [direction, neighbors] : central_element.neighbors()) {
      const auto& orientation = neighbors.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());
      for (const auto& neighbor_id : neighbors) {
        const LinearSolver::Schwarz::OverlapId<Dim> overlap_id{direction,
                                                               neighbor_id};
        const auto& overlap_extent = all_overlap_extents.at(overlap_id);
        const auto& neighbor = all_neighbor_elements.at(overlap_id);
        const auto& neighbor_mesh = all_neighbor_meshes.at(overlap_id);
        const auto& mortar_id = overlap_id;
        const auto& mortar_mesh = central_mortar_meshes.at(mortar_id);
        const bool neighbor_data_is_zero = data_is_zero(neighbor_id);

        // Intercept empty overlaps. In the unlikely case that overlaps have
        // zero extent, meaning no point of the neighbor is part of the
        // subdomain (which is fairly useless, except for testing), the
        // subdomain is identical to the central element and no communication
        // with neighbors is necessary. We can just handle the mortar between
        // central element and neighbor and continue.
        if (UNLIKELY(overlap_extent == 0)) {
          elliptic::dg::BoundaryData<typename System::primal_fields,
                                     typename System::primal_fluxes>
              remote_boundary_data{};
          remote_boundary_data.field_data.initialize(
              mortar_mesh.number_of_grid_points(), 0.);
          central_mortar_data_.at(mortar_id).remote_insert(
              temporal_id, std::move(remote_boundary_data));
          continue;
        }

        // Copy the central element's mortar data to the neighbor
        if (not(central_data_is_zero and neighbor_data_is_zero)) {
          auto oriented_mortar_data =
              central_mortar_data_.at(mortar_id).local_data(temporal_id);
          if (not orientation.is_aligned()) {
            oriented_mortar_data.orient_on_slice(
                mortar_mesh.extents(), direction.dimension(), orientation);
          }
          neighbors_mortar_data_[overlap_id][::dg::MortarId<Dim>{
                                                 direction_from_neighbor,
                                                 central_element.id()}]
              .remote_insert(temporal_id, std::move(oriented_mortar_data));
        }

        // Now we switch perspective to the neighbor. First, we extend the
        // overlap data to the full neighbor mesh by padding it with zeros. This
        // is necessary because spectral operators such as derivatives require
        // data on the full mesh.
        if (not neighbor_data_is_zero) {
          LinearSolver::Schwarz::extended_overlap_data(
              make_not_null(&extended_operand_vars_[overlap_id]),
              operand.overlap_data.at(overlap_id), neighbor_mesh.extents(),
              overlap_extent, direction_from_neighbor);
        }

        const auto apply_boundary_condition_neighbor =
            [&apply_boundary_condition, &local_neighbor_id = neighbor_id,
             &overlap_id](const Direction<Dim>& local_direction,
                          auto&&... fields_and_fluxes) {
              apply_boundary_condition(
                  local_neighbor_id, local_direction, std::true_type{},
                  std::forward_as_tuple(overlap_id, local_direction),
                  std::forward<decltype(fields_and_fluxes)>(
                      fields_and_fluxes)...);
            };

        const auto fluxes_args_on_overlap =
            elliptic::util::apply_at<fluxes_args_tags_overlap,
                                     args_tags_from_center>(get_items, box,
                                                            overlap_id);

        elliptic::util::apply_at<prepare_args_tags_overlap,
                                 args_tags_from_center>(
            [this, &overlap_id](const auto&... args) {
              elliptic::dg::prepare_mortar_data<System, linearized>(
                  make_not_null(&neighbors_deriv_vars_[overlap_id]),
                  make_not_null(&neighbors_primal_fluxes_[overlap_id]),
                  make_not_null(&neighbors_mortar_data_[overlap_id]),
                  extended_operand_vars_[overlap_id], args...);
            },
            box, overlap_id, temporal_id, apply_boundary_condition_neighbor,
            fluxes_args_on_overlap, data_is_zero);

        // Copy this neighbor's mortar data to the other side of the mortars. On
        // the other side we either have the central element, or another element
        // that may or may not be part of the subdomain.
        const auto& neighbor_mortar_meshes =
            all_neighbor_mortar_meshes.at(overlap_id);
        for (const auto& neighbor_mortar_id_and_data :
             neighbors_mortar_data_.at(overlap_id)) {
          // No structured bindings because capturing these in lambdas doesn't
          // work until C++20
          const auto& neighbor_mortar_id = neighbor_mortar_id_and_data.first;
          const auto& neighbor_mortar_data = neighbor_mortar_id_and_data.second;
          const auto& neighbor_direction = neighbor_mortar_id.direction();
          const auto& neighbors_neighbor_id = neighbor_mortar_id.id();
          // No need to do anything on external boundaries
          if (neighbors_neighbor_id == ElementId<Dim>::external_boundary_id()) {
            continue;
          }
          const auto& neighbor_orientation =
              neighbor.neighbors().at(neighbor_direction).orientation();
          const auto neighbors_neighbor_direction =
              neighbor_orientation(neighbor_direction.opposite());
          const ::dg::MortarId<Dim> mortar_id_from_neighbors_neighbor{
              neighbors_neighbor_direction, neighbor_id};
          const auto send_mortar_data =
              [&neighbor_orientation, &neighbor_mortar_meshes,
               &neighbor_mortar_data, &neighbor_mortar_id, &neighbor_direction,
               &neighbor_data_is_zero,
               &data_is_zero](auto& remote_mortar_data,
                              const ElementId<Dim>& remote_element_id) {
                if (neighbor_data_is_zero and data_is_zero(remote_element_id)) {
                  return;
                }
                const auto& neighbor_mortar_mesh =
                    neighbor_mortar_meshes.at(neighbor_mortar_id);
                auto oriented_neighbor_mortar_data =
                    neighbor_mortar_data.local_data(temporal_id);
                if (not neighbor_orientation.is_aligned()) {
                  oriented_neighbor_mortar_data.orient_on_slice(
                      neighbor_mortar_mesh.extents(),
                      neighbor_direction.dimension(), neighbor_orientation);
                }
                remote_mortar_data.remote_insert(
                    temporal_id, std::move(oriented_neighbor_mortar_data));
              };
          if (neighbors_neighbor_id == central_element.id() and
              mortar_id_from_neighbors_neighbor == mortar_id) {
            send_mortar_data(central_mortar_data_.at(mortar_id),
                             central_element.id());
            continue;
          }
          // Determine whether the neighbor's neighbor overlaps with the
          // subdomain and find its overlap ID if it does.
          const auto neighbors_neighbor_overlap_id =
              [&local_all_neighbor_mortar_meshes = all_neighbor_mortar_meshes,
               &neighbors_neighbor_id, &mortar_id_from_neighbors_neighbor,
               &is_in_subdomain]()
              -> std::optional<LinearSolver::Schwarz::OverlapId<Dim>> {
            if (not is_in_subdomain(neighbors_neighbor_id)) {
              return std::nullopt;
            }
            for (const auto& [local_overlap_id, local_mortar_meshes] :
                 local_all_neighbor_mortar_meshes) {
              if (local_overlap_id.id() != neighbors_neighbor_id) {
                continue;
              }
              for (const auto& local_mortar_id_and_mesh : local_mortar_meshes) {
                if (local_mortar_id_and_mesh.first ==
                    mortar_id_from_neighbors_neighbor) {
                  return local_overlap_id;
                }
              }
            }
            ERROR("The neighbor's neighbor "
                  << neighbors_neighbor_id
                  << " is part of the subdomain, but we didn't find its "
                     "overlap ID. This is a bug, so please file an issue.");
          }();
          if (neighbors_neighbor_overlap_id.has_value()) {
            // The neighbor's neighbor is part of the subdomain so we copy the
            // mortar data over. Once the loop is complete we will also have
            // received mortar data back. At that point, both neighbors have a
            // copy of each other's mortar data, which is the subject of the
            // possible optimizations mentioned above. Note that the data may
            // differ by orientations.
            send_mortar_data(
                neighbors_mortar_data_[*neighbors_neighbor_overlap_id]
                                      [mortar_id_from_neighbors_neighbor],
                neighbors_neighbor_overlap_id->id());
          } else if (not neighbor_data_is_zero) {
            // The neighbor's neighbor does not overlap with the subdomain, so
            // we don't copy mortar data and also don't expect to receive any.
            // Instead, we assume the data on it is zero and manufacture
            // appropriate remote boundary data.
            const auto& neighbors_neighbor_mortar_mesh =
                all_neighbors_neighbor_mortar_meshes.at(overlap_id)
                    .at(neighbor_mortar_id);
            elliptic::dg::BoundaryData<typename System::primal_fields,
                                       typename System::primal_fluxes>
                zero_mortar_data{};
            zero_mortar_data.field_data.initialize(
                neighbors_neighbor_mortar_mesh.number_of_grid_points(), 0.);
            neighbors_mortar_data_.at(overlap_id)
                .at(neighbor_mortar_id)
                .remote_insert(temporal_id, std::move(zero_mortar_data));
          }
        }  // loop over neighbor's mortars
      }    // loop over neighbors
    }      // loop over directions

    // 2. Apply the operator on all elements in the subdomain
    //
    // Apply on central element
    db::apply<apply_args_tags>(
        [this, &result, &operand](const auto&... args) {
          elliptic::dg::apply_operator<System, linearized>(
              make_not_null(&result->element_data),
              make_not_null(&central_mortar_data_), operand.element_data,
              central_primal_fluxes_, args...);
        },
        box, temporal_id, fluxes_args_on_faces, sources_args,
        data_is_zero);
    // Apply on neighbors
    for (const auto& [direction, neighbors] : central_element.neighbors()) {
      const auto& orientation = neighbors.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());
      for (const auto& neighbor_id : neighbors) {
        const LinearSolver::Schwarz::OverlapId<Dim> overlap_id{direction,
                                                               neighbor_id};
        const auto& overlap_extent = all_overlap_extents.at(overlap_id);
        const auto& neighbor_mesh = all_neighbor_meshes.at(overlap_id);

        if (UNLIKELY(overlap_extent == 0)) {
          continue;
        }

        DirectionMap<Dim, FluxesArgs> fluxes_args_on_overlap_faces{};
        for (const auto& neighbor_direction :
             Direction<Dim>::all_directions()) {
          fluxes_args_on_overlap_faces.emplace(
              neighbor_direction,
              elliptic::util::apply_at<fluxes_args_tags_overlap_faces,
                                       args_tags_from_center>(
                  get_items, box,
                  std::forward_as_tuple(overlap_id, neighbor_direction)));
        }

        elliptic::util::apply_at<apply_args_tags_overlap,
                                 args_tags_from_center>(
            [this, &overlap_id](const auto&... args) {
              elliptic::dg::apply_operator<System, linearized>(
                  make_not_null(&extended_results_[overlap_id]),
                  make_not_null(&neighbors_mortar_data_.at(overlap_id)),
                  extended_operand_vars_.at(overlap_id),
                  neighbors_primal_fluxes_.at(overlap_id), args...);
            },
            box, overlap_id, temporal_id, fluxes_args_on_overlap_faces,
            elliptic::util::apply_at<sources_args_tags_overlap,
                                     args_tags_from_center>(get_items, box,
                                                            overlap_id),
            data_is_zero);

        // Restrict the extended operator data back to the subdomain, assuming
        // we can discard any data outside the overlaps. WARNING: This
        // assumption may break with changes to the DG operator that affect its
        // sparsity. For example, multiplying the DG operator with the _full_
        // inverse mass-matrix ("massless" scheme with no "mass-lumping"
        // approximation) means that lifted boundary corrections bleed into the
        // volume.
        if (UNLIKELY(
                result->overlap_data[overlap_id].number_of_grid_points() !=
                operand.overlap_data.at(overlap_id).number_of_grid_points())) {
          result->overlap_data[overlap_id].initialize(
              operand.overlap_data.at(overlap_id).number_of_grid_points());
        }
        LinearSolver::Schwarz::data_on_overlap(
            make_not_null(&result->overlap_data[overlap_id]),
            extended_results_.at(overlap_id), neighbor_mesh.extents(),
            overlap_extent, direction_from_neighbor);
      }  // loop over neighbors
    }    // loop over directions
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}

 private:
  // Memory buffers for repeated operator applications
  // NOLINTNEXTLINE(spectre-mutable)
  mutable Variables<
      db::wrap_tags_in<::Tags::deriv, typename System::primal_fields,
                       tmpl::size_t<Dim>, Frame::Inertial>>
      central_deriv_vars_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable Variables<typename System::primal_fluxes> central_primal_fluxes_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable LinearSolver::Schwarz::OverlapMap<
      Dim,
      Variables<db::wrap_tags_in<::Tags::deriv, typename System::primal_fields,
                                 tmpl::size_t<Dim>, Frame::Inertial>>>
      neighbors_deriv_vars_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable LinearSolver::Schwarz::OverlapMap<
      Dim, Variables<typename System::primal_fluxes>>
      neighbors_primal_fluxes_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable LinearSolver::Schwarz::OverlapMap<
      Dim, Variables<typename System::primal_fields>>
      extended_operand_vars_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable LinearSolver::Schwarz::OverlapMap<
      Dim, Variables<typename System::primal_fields>>
      extended_results_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable ::dg::MortarMap<
      Dim, elliptic::dg::MortarData<size_t, typename System::primal_fields,
                                    typename System::primal_fluxes>>
      central_mortar_data_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable LinearSolver::Schwarz::OverlapMap<
      Dim, ::dg::MortarMap<Dim, elliptic::dg::MortarData<
                                    size_t, typename System::primal_fields,
                                    typename System::primal_fluxes>>>
      neighbors_mortar_data_{};
};

}  // namespace elliptic::dg::subdomain_operator
