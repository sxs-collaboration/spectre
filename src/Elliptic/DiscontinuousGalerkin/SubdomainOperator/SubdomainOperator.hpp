// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <boost/range/join.hpp>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/Tags.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/BoundaryData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementCenteredSubdomainData.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/SubdomainOperator.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TupleSlice.hpp"

namespace elliptic::dg::subdomain_operator {

/// Wrap the `Tag` in `LinearSolver::Schwarz::Tags::Overlaps`, except if it is
/// included in `TakeFromCenterTags`.
template <typename Tag, typename Dim, typename OptionsGroup,
          typename TakeFromCenterTags>
struct make_overlap_tag {
  using type = tmpl::conditional_t<
      tmpl::list_contains_v<TakeFromCenterTags, Tag>, Tag,
      LinearSolver::Schwarz::Tags::Overlaps<Tag, Dim::value, OptionsGroup>>;
};

/// Wrap the `Tag` in `Tags::NeighborMortars`
template <typename Tag, typename Dim>
struct make_neighbor_mortars_tag {
  using type = Tags::NeighborMortars<Tag, Dim::value>;
};

/*!
 * \brief A first-order DG operator on an element-centered subdomain
 *
 * This operator is a restriction of the full DG-operator to an element-centered
 * subdomain with a few points overlap into neighboring elements. It is a
 * `LinearSolver::Schwarz::protocols::SubdomainOperator` to be used with the
 * Schwarz linear solver when it solves a first-order DG operator.
 *
 * This operator requires the following tags are available on overlap regions
 * with neighboring elements:
 *
 * - Geometric quantities provided by
 *   `elliptic::dg::subdomain_operator::InitializeElement`.
 * - All `System::fluxes_computer::argument_tags` and
 *   `System::sources_computer::argument_tags`, except those listed in
 *   `ArgsTagsFromCenter` which will be taken from the central element's
 *   DataBox.
 * - The `System::fluxes_computer::argument_tags` on internal and external
 *   interfaces, except those listed in `System::fluxes_computer::volume_tags`.
 */
template <typename System, typename OptionsGroup,
          typename ArgsTagsFromCenter = tmpl::list<>>
struct SubdomainOperator
    : LinearSolver::Schwarz::SubdomainOperator<System::volume_dim> {
 public:
  static constexpr size_t Dim = System::volume_dim;

 private:
  // Operator applications happen sequentially so we don't have to keep track of
  // the temporal id
  static constexpr size_t temporal_id = 0;

  // These are the arguments that we need to retrieve from the DataBox and pass
  // to the functions in `elliptic::dg`, both on the central element and on
  // neighbors.
  using prepare_args_tags = tmpl::list<
      domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>,
      domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Inertial>,
      domain::Tags::Interface<
          domain::Tags::InternalDirections<Dim>,
          ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
      domain::Tags::Interface<
          domain::Tags::BoundaryDirectionsInterior<Dim>,
          ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
      domain::Tags::Interface<
          domain::Tags::InternalDirections<Dim>,
          ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
      domain::Tags::Interface<
          domain::Tags::BoundaryDirectionsInterior<Dim>,
          ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
      ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
      ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>;
  using apply_args_tags = tmpl::list<
      domain::Tags::Mesh<Dim>,
      domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Inertial>,
      domain::Tags::Interface<
          domain::Tags::InternalDirections<Dim>,
          ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
      domain::Tags::Interface<
          domain::Tags::BoundaryDirectionsInterior<Dim>,
          ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
      ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
      ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>,
      elliptic::dg::Tags::PenaltyParameter>;
  using fluxes_args_tags = typename System::fluxes_computer::argument_tags;
  using sources_args_tags = typename System::sources_computer::argument_tags;

  // We need the fluxes args also on interfaces (internal and external). The
  // volume tags are the subset that don't have to be taken from interfaces.
  using fluxes_args_volume_tags =
      get_volume_tags<typename System::fluxes_computer>;

  // These tags can be taken directly from the central element's DataBox, even
  // when evaluating neighbors
  using args_tags_from_center = tmpl::remove_duplicates<tmpl::push_back<
      ArgsTagsFromCenter, elliptic::dg::Tags::PenaltyParameter>>;

  // Data on neighbors is stored in the central element's DataBox in
  // `LinearSolver::Schwarz::Tags::Overlaps` maps, so we wrap the argument tags
  // with this prefix
  using make_overlap_tag = elliptic::dg::subdomain_operator::make_overlap_tag<
      tmpl::_1, tmpl::pin<tmpl::size_t<Dim>>, tmpl::pin<OptionsGroup>,
      tmpl::pin<args_tags_from_center>>;
  using prepare_args_tags_overlap =
      tmpl::transform<prepare_args_tags, make_overlap_tag>;
  using apply_args_tags_overlap =
      tmpl::transform<apply_args_tags, make_overlap_tag>;
  using fluxes_args_tags_overlap =
      tmpl::transform<fluxes_args_tags, make_overlap_tag>;
  using sources_args_tags_overlap =
      tmpl::transform<sources_args_tags, make_overlap_tag>;
  template <typename Directions>
  using fluxes_args_tags_overlap_interface = tmpl::transform<
      tmpl::transform<fluxes_args_tags,
                      make_interface_tag<tmpl::_1, tmpl::pin<Directions>,
                                         tmpl::pin<fluxes_args_volume_tags>>>,
      make_overlap_tag>;

  // We also need some data on the remote side of all neighbors' mortars. Such
  // data is stored in the central element's DataBox in `Tags::NeighborMortars`
  // maps
  using make_neighbor_mortars_tag =
      make_neighbor_mortars_tag<tmpl::_1, tmpl::pin<tmpl::size_t<Dim>>>;

 public:
  template <typename ResultTags, typename OperandTags, typename DbTagsList>
  void operator()(
      const gsl::not_null<
          LinearSolver::Schwarz::ElementCenteredSubdomainData<Dim, ResultTags>*>
          result,
      const LinearSolver::Schwarz::ElementCenteredSubdomainData<
          Dim, OperandTags>& operand,
      db::DataBox<DbTagsList>& box) noexcept {
    // Shortcut to retrieve items out of the DataBox to forward to functions
    const auto get_items = [](const auto&... args) noexcept {
      return std::forward_as_tuple(args...);
    };

    // Retrieve data out of the DataBox
    const auto& [central_element, central_mortar_meshes, all_overlap_extents,
                 all_neighbor_elements, all_neighbor_meshes,
                 all_neighbor_face_normal_magnitudes_internal,
                 all_neighbor_mortar_meshes, all_neighbor_mortar_sizes,
                 all_neighbors_neighbor_meshes,
                 all_neighbors_neighbor_face_normal_magnitudes,
                 all_neighbors_neighbor_mortar_meshes,
                 all_neighbors_neighbor_mortar_sizes] =
        db::apply<tmpl::flatten<tmpl::list<
            domain::Tags::Element<Dim>,
            ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
            // Data on overlaps with neighbors
            tmpl::transform<
                tmpl::flatten<tmpl::list<
                    Tags::ExtrudingExtent, domain::Tags::Element<Dim>,
                    domain::Tags::Mesh<Dim>,
                    domain::Tags::Interface<
                        domain::Tags::InternalDirections<Dim>,
                        ::Tags::Magnitude<
                            domain::Tags::UnnormalizedFaceNormal<Dim>>>,
                    ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
                    ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>,
                    // Data on the remote side of the neighbor's mortars
                    tmpl::transform<
                        tmpl::list<
                            domain::Tags::Mesh<Dim>,
                            ::Tags::Magnitude<
                                domain::Tags::UnnormalizedFaceNormal<Dim>>,
                            domain::Tags::Mesh<Dim - 1>,
                            ::Tags::MortarSize<Dim - 1>>,
                        make_neighbor_mortars_tag>>>,
                make_overlap_tag>>>>(get_items, box);
    const auto fluxes_args = db::apply<fluxes_args_tags>(get_items, box);
    const auto sources_args = db::apply<sources_args_tags>(get_items, box);
    const auto fluxes_args_on_internal_faces =
        interface_apply<domain::Tags::InternalDirections<Dim>, fluxes_args_tags,
                        fluxes_args_volume_tags>(get_items, box);
    const auto fluxes_args_on_external_faces =
        interface_apply<domain::Tags::BoundaryDirectionsInterior<Dim>,
                        fluxes_args_tags, fluxes_args_volume_tags>(get_items,
                                                                   box);

    // Using an analytic solution to apply Dirichlet boundary conditions for
    // now. This will change to the set of boundary conditions retrieved from
    // blocks.
    using Registrar =
        elliptic::BoundaryConditions::Registrars::AnalyticSolution<System>;
    const std::unique_ptr<elliptic::BoundaryConditions::BoundaryCondition<
        Dim, tmpl::list<Registrar>>>
        boundary_condition = std::make_unique<
            typename Registrar::template f<tmpl::list<Registrar>>>(
            elliptic::BoundaryConditionType::Dirichlet);
    const auto apply_boundary_condition_center =
        [&boundary_condition, &box](const Direction<Dim>& direction,
                                    const auto... fields_and_fluxes) noexcept {
          elliptic::apply_boundary_condition<true, void>(
              *boundary_condition, box, direction, fields_and_fluxes...);
        };

    // The subdomain operator essentially does two sweeps over all elements in
    // the subdomain: In the first sweep it prepares the mortar data and
    // "communicates" them across mortars, and in the second sweep it consumes
    // the mortar data to apply the operator. This implementation is relatively
    // simple because it can re-use the implementation for the parallel DG
    // operator. However, it is also possible to apply the subdomain operator in
    // a single sweep over all elements, incrementally building up the mortar
    // data and applying boundary corrections immediately to both adjacent
    // elements once the data is available. That approach is possibly a
    // performance optimization but requires re-implementing the DG operator
    // here. It should be considered once the subdomain operator has been
    // identified as the performance bottleneck. An alternative to optimizing
    // the subdomain operator performance is to precondition the subdomain solve
    // with a _much_ simpler subdomain operator, such as a finite-difference
    // Laplacian, so fewer applications of the more expensive DG subdomain
    // operator are necessary.

    // TODO: Check if everything works for periodic domains

    // 1. Prepare mortar data on all elements in the subdomain and "communicate"
    //    across mortars, reorienting if needed
    //
    // Prepare central element
    db::apply<prepare_args_tags>(
        [this, &operand](const auto&... args) noexcept {
          elliptic::dg::prepare_mortar_data<System, true>(
              make_not_null(&central_auxiliary_vars_),
              make_not_null(&central_mortar_data_), operand.element_data,
              args...);
        },
        box, temporal_id, apply_boundary_condition_center, fluxes_args,
        sources_args, fluxes_args_on_internal_faces,
        fluxes_args_on_external_faces);
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
        const ::dg::MortarId<Dim> mortar_id_from_neighbor{
            direction_from_neighbor, central_element.id()};

        // Intercept empty overlaps. In the unlikely case that overlaps have
        // zero extent (which is fairly useless, except for testing) the
        // subdomain is identical to the central element and no communication
        // with neighbors is necessary. We can just handle the mortar between
        // central element and neighbor and continue.
        if (UNLIKELY(overlap_extent == 0)) {
          const auto& mortar_mesh_from_neighbor =
              all_neighbor_mortar_meshes.at(overlap_id)
                  .at(mortar_id_from_neighbor);
          const auto& mortar_size_from_neighbor =
              all_neighbor_mortar_sizes.at(overlap_id)
                  .at(mortar_id_from_neighbor);
          auto remote_boundary_data =
              elliptic::dg::zero_boundary_data_on_mortar<
                  typename System::primal_fields,
                  typename System::auxiliary_fields>(
                  direction_from_neighbor, neighbor_mesh,
                  all_neighbor_face_normal_magnitudes_internal.at(overlap_id)
                      .at(direction_from_neighbor),
                  mortar_mesh_from_neighbor, mortar_size_from_neighbor);
          if (not orientation.is_aligned()) {
            remote_boundary_data.orient_on_slice(
                mortar_mesh_from_neighbor.extents(),
                direction_from_neighbor.dimension(), orientation.inverse_map());
          }
          central_mortar_data_.at(mortar_id).remote_insert(
              temporal_id, std::move(remote_boundary_data));
          continue;
        }

        // Send the central element's mortar data to the neighbor
        auto oriented_mortar_data =
            central_mortar_data_.at(mortar_id).local_data(temporal_id);
        if (not orientation.is_aligned()) {
          oriented_mortar_data.orient_on_slice(
              mortar_mesh.extents(), direction.dimension(), orientation);
        }
        neighbors_mortar_data_[overlap_id][std::make_pair(
                                               direction_from_neighbor,
                                               central_element.id())]
            .remote_insert(temporal_id, std::move(oriented_mortar_data));

        // Now we switch perspective to the neighbor. First, we extend the
        // overlap data to the full neighbor mesh by padding it with zeros. This
        // is necessary because spectral operators such as derivatives require
        // data on the full mesh.
        LinearSolver::Schwarz::extended_overlap_data(
            make_not_null(&extended_operand_vars_[overlap_id]),
            operand.overlap_data.at(overlap_id), neighbor_mesh.extents(),
            overlap_extent, direction_from_neighbor);

        const auto apply_boundary_condition_neighbor =
            [&boundary_condition, &box, &overlap_id](
                const Direction<Dim>& direction,
                const auto... fields_and_fluxes) noexcept {
              elliptic::apply_boundary_condition<true, make_overlap_tag>(
                  *boundary_condition, box,
                  std::forward_as_tuple(overlap_id, direction),
                  fields_and_fluxes...);
            };

        using FluxesArgs = std::decay_t<decltype(fluxes_args)>;
        const auto fluxes_args_on_overlap =
            elliptic::util::apply_at<fluxes_args_tags_overlap,
                                     args_tags_from_center>(get_items, box,
                                                            overlap_id);
        const auto sources_args_on_overlap =
            elliptic::util::apply_at<sources_args_tags_overlap,
                                     args_tags_from_center>(get_items, box,
                                                            overlap_id);
        DirectionMap<Dim, FluxesArgs> fluxes_args_on_overlap_faces_internal{};
        DirectionMap<Dim, FluxesArgs> fluxes_args_on_overlap_faces_external{};
        for (const auto& neighbor_direction : neighbor.internal_boundaries()) {
          fluxes_args_on_overlap_faces_internal.emplace(
              neighbor_direction,
              elliptic::util::apply_at<
                  fluxes_args_tags_overlap_interface<
                      domain::Tags::InternalDirections<Dim>>,
                  args_tags_from_center>(
                  get_items, box,
                  std::forward_as_tuple(overlap_id, neighbor_direction)));
        }
        for (const auto& neighbor_direction : neighbor.external_boundaries()) {
          fluxes_args_on_overlap_faces_external.emplace(
              neighbor_direction,
              elliptic::util::apply_at<
                  fluxes_args_tags_overlap_interface<
                      domain::Tags::BoundaryDirectionsInterior<Dim>>,
                  args_tags_from_center>(
                  get_items, box,
                  std::forward_as_tuple(overlap_id, neighbor_direction)));
        }

        elliptic::util::apply_at<prepare_args_tags_overlap,
                                 args_tags_from_center>(
            [this, &overlap_id](const auto&... args) noexcept {
              elliptic::dg::prepare_mortar_data<System, true>(
                  make_not_null(&neighbors_auxiliary_vars_[overlap_id]),
                  make_not_null(&neighbors_mortar_data_[overlap_id]),
                  extended_operand_vars_[overlap_id], args...);
            },
            box, overlap_id, temporal_id, apply_boundary_condition_neighbor,
            fluxes_args_on_overlap, sources_args_on_overlap,
            fluxes_args_on_overlap_faces_internal,
            fluxes_args_on_overlap_faces_external);

        // Send this neighbor's mortar data to the other side of the mortars. On
        // the other side we either have the central element, or another element
        // that may or may not be part of the subdomain.
        const auto& neighbor_mortar_meshes =
            all_neighbor_mortar_meshes.at(overlap_id);
        for (const auto& neighbor_mortar_id_and_data :
             neighbors_mortar_data_.at(overlap_id)) {
          // No structured bindings because these are captured in lambdas and
          // the compilers can't handle that.
          const auto& neighbor_mortar_id = neighbor_mortar_id_and_data.first;
          const auto& neighbor_mortar_data = neighbor_mortar_id_and_data.second;
          const auto& neighbor_direction = neighbor_mortar_id.first;
          const auto& neighbors_neighbor_id = neighbor_mortar_id.second;
          // No need to send anything on external boundaries
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
               &neighbor_mortar_data, &neighbor_mortar_id,
               &neighbor_direction](auto& remote_mortar_data) noexcept {
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
            send_mortar_data(central_mortar_data_.at(mortar_id));
            continue;
          }
          // Determine whether the neighbor's neighbor overlaps with the
          // subdomain and find its overlap ID if it does.
          const auto& local_all_neighbor_mortar_meshes =
              all_neighbor_mortar_meshes;
          const auto neighbors_neighbor_overlap_id =
              [&local_all_neighbor_mortar_meshes, &neighbors_neighbor_id,
               &mortar_id_from_neighbors_neighbor]() noexcept
              -> std::optional<LinearSolver::Schwarz::OverlapId<Dim>> {
            for (const auto& [local_overlap_id, local_mortar_meshes] :
                 local_all_neighbor_mortar_meshes) {
              if (local_overlap_id.second != neighbors_neighbor_id) {
                continue;
              }
              for (const auto& local_mortar_id_and_mesh : local_mortar_meshes) {
                if (local_mortar_id_and_mesh.first ==
                    mortar_id_from_neighbors_neighbor) {
                  return local_overlap_id;
                }
              }
            }
            return std::nullopt;
          }();
          if (neighbors_neighbor_overlap_id.has_value()) {
            // The neighbor's neighbor is part of the subdomain so we send the
            // mortar data over. Once the loop is complete we will also have
            // received mortar data back. At that point, both neighbors have a
            // copy of each other's mortar data, which is the subject of the
            // possible optimizations mentioned above. Note that the data may
            // differ by orientations.
            send_mortar_data(
                neighbors_mortar_data_[*neighbors_neighbor_overlap_id]
                                      [mortar_id_from_neighbors_neighbor]);
          } else {
            // The neighbor's neighbor does not overlap with the subdomain, so
            // we don't send mortar data and also don't expect to receive any.
            // Instead, we assume the data on it is zero and manufacture
            // appropriate remote boundary data.
            neighbors_mortar_data_.at(overlap_id)
                .at(neighbor_mortar_id)
                .remote_insert(
                    temporal_id,
                    elliptic::dg::zero_boundary_data_on_mortar<
                        typename System::primal_fields,
                        typename System::auxiliary_fields>(
                        neighbors_neighbor_direction,
                        all_neighbors_neighbor_meshes.at(overlap_id)
                            .at(neighbor_mortar_id),
                        all_neighbors_neighbor_face_normal_magnitudes
                            .at(overlap_id)
                            .at(neighbor_mortar_id),
                        all_neighbors_neighbor_mortar_meshes.at(overlap_id)
                            .at(neighbor_mortar_id),
                        all_neighbors_neighbor_mortar_sizes.at(overlap_id)
                            .at(neighbor_mortar_id)));
          }
        }  // loop over neighbor's mortars
      }    // loop over neighbors
    }      // loop over directions

    // 2. Apply the operator on all elements in the subdomain
    //
    // Apply on central element
    db::apply<apply_args_tags>(
        [this, &result, &operand](const auto&... args) noexcept {
          elliptic::dg::apply_operator<System, true>(
              make_not_null(&result->element_data),
              make_not_null(&central_auxiliary_vars_),
              make_not_null(&central_mortar_data_), operand.element_data,
              args...);
        },
        box, temporal_id, fluxes_args, sources_args);
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

        elliptic::util::apply_at<apply_args_tags_overlap,
                                 args_tags_from_center>(
            [this, &overlap_id](const auto&... args) noexcept {
              elliptic::dg::apply_operator<System, true>(
                  make_not_null(&extended_results_[overlap_id]),
                  make_not_null(&neighbors_auxiliary_vars_.at(overlap_id)),
                  make_not_null(&neighbors_mortar_data_.at(overlap_id)),
                  extended_operand_vars_.at(overlap_id), args...);
            },
            box, overlap_id, temporal_id,
            elliptic::util::apply_at<fluxes_args_tags_overlap,
                                     args_tags_from_center>(get_items, box,
                                                            overlap_id),
            elliptic::util::apply_at<sources_args_tags_overlap,
                                     args_tags_from_center>(get_items, box,
                                                            overlap_id));

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

 private:
  Variables<typename System::auxiliary_fields> central_auxiliary_vars_{};
  LinearSolver::Schwarz::OverlapMap<
      Dim, Variables<typename System::auxiliary_fields>>
      neighbors_auxiliary_vars_{};
  LinearSolver::Schwarz::OverlapMap<Dim,
                                    Variables<typename System::primal_fields>>
      extended_operand_vars_{};
  LinearSolver::Schwarz::OverlapMap<Dim,
                                    Variables<typename System::primal_fields>>
      extended_results_{};
  ::dg::MortarMap<Dim,
                  elliptic::dg::MortarData<typename System::primal_fields,
                                           typename System::auxiliary_fields>>
      central_mortar_data_{};
  LinearSolver::Schwarz::OverlapMap<
      Dim, ::dg::MortarMap<Dim, elliptic::dg::MortarData<
                                    typename System::primal_fields,
                                    typename System::auxiliary_fields>>>
      neighbors_mortar_data_{};
};

}  // namespace elliptic::dg::subdomain_operator
