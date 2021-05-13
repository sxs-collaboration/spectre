// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceComputeTags.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions/ApplyBoundaryCondition.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/Tags.hpp"
#include "Elliptic/Systems/GetSourcesComputer.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/HasReceivedFromAllMortars.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// Actions related to elliptic discontinuous Galerkin schemes
namespace elliptic::dg::Actions {
// The individual actions in this namespace are not exposed publicly because
// they don't work on their own. Instead, the public interface (defined below)
// exposes them in action lists.
namespace detail {

// This tag is used to communicate mortar data across neighbors
template <size_t Dim, typename TemporalIdTag, typename PrimalFields,
          typename PrimalFluxes>
struct MortarDataInboxTag
    : public Parallel::InboxInserters::Map<
          MortarDataInboxTag<Dim, TemporalIdTag, PrimalFields, PrimalFluxes>> {
  using temporal_id = typename TemporalIdTag::type;
  using type = std::map<
      temporal_id,
      FixedHashMap<maximum_number_of_neighbors(Dim), ::dg::MortarId<Dim>,
                   elliptic::dg::BoundaryData<PrimalFields, PrimalFluxes>,
                   boost::hash<::dg::MortarId<Dim>>>>;
};

// Initializes all quantities the DG operator needs on internal and external
// faces, as well as the mortars between neighboring elements.
template <typename System>
struct InitializeFacesAndMortars {
 private:
  static constexpr size_t Dim = System::volume_dim;

  // The DG operator needs these background fields on faces:
  // - The `inv_metric_tag` on internal and external faces to normalize face
  //   normals (should be included in background fields)
  // - The background fields in the `System::fluxes_computer::argument_tags` on
  //   internal and external faces to compute fluxes
  // - All background fields on external faces to impose boundary conditions
  using slice_tags_to_faces = tmpl::conditional_t<
      std::is_same_v<typename System::background_fields, tmpl::list<>>,
      tmpl::list<>,
      tmpl::list<
          // Possible optimization: Only the background fields in the
          // System::fluxes_computer::argument_tags are needed on internal faces
          ::Tags::Variables<typename System::background_fields>>>;
  using face_compute_tags = tmpl::list<
      // For boundary conditions. Possible optimization: Not all systems need
      // the coordinates on internal faces. Adding the compute item shouldn't
      // incur a runtime overhead though since it's lazily evaluated.
      domain::Tags::BoundaryCoordinates<System::volume_dim>>;

  template <typename TagToSlice, typename DirectionsTag>
  struct make_slice_tag {
    using type = domain::Tags::Slice<DirectionsTag, TagToSlice>;
  };

  template <typename ComputeTag, typename DirectionsTag>
  struct make_face_compute_tag {
    using type = domain::Tags::InterfaceCompute<DirectionsTag, ComputeTag>;
  };

  template <typename DirectionsTag>
  using face_tags = tmpl::flatten<tmpl::list<
      domain::Tags::InterfaceCompute<DirectionsTag,
                                     domain::Tags::Direction<Dim>>,
      domain::Tags::InterfaceCompute<DirectionsTag,
                                     domain::Tags::InterfaceMesh<Dim>>,
      tmpl::transform<slice_tags_to_faces,
                      make_slice_tag<tmpl::_1, tmpl::pin<DirectionsTag>>>,
      domain::Tags::InterfaceCompute<
          DirectionsTag, domain::Tags::UnnormalizedFaceNormalCompute<Dim>>,
      domain::Tags::InterfaceCompute<
          DirectionsTag,
          tmpl::conditional_t<
              std::is_same_v<typename System::inv_metric_tag, void>,
              ::Tags::EuclideanMagnitude<
                  domain::Tags::UnnormalizedFaceNormal<Dim>>,
              ::Tags::NonEuclideanMagnitude<
                  domain::Tags::UnnormalizedFaceNormal<Dim>,
                  typename System::inv_metric_tag>>>,
      domain::Tags::InterfaceCompute<
          DirectionsTag,
          ::Tags::NormalizedCompute<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
      tmpl::transform<
          face_compute_tags,
          make_face_compute_tag<tmpl::_1, tmpl::pin<DirectionsTag>>>>>;

 public:
  using simple_tags =
      tmpl::list<::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
                 ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>;
  using compute_tags = tmpl::append<
      tmpl::list<domain::Tags::InternalDirectionsCompute<Dim>,
                 domain::Tags::BoundaryDirectionsInteriorCompute<Dim>>,
      face_tags<domain::Tags::InternalDirections<Dim>>,
      face_tags<domain::Tags::BoundaryDirectionsInterior<Dim>>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& element = db::get<domain::Tags::Element<Dim>>(box);
    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const auto& initial_extents =
        db::get<domain::Tags::InitialExtents<Dim>>(box);

    // Setup initial mortars at interfaces to neighbors
    ::dg::MortarMap<Dim, Mesh<Dim - 1>> all_mortar_meshes{};
    ::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>> all_mortar_sizes{};
    for (const auto& [direction, neighbors] : element.neighbors()) {
      const auto face_mesh = mesh.slice_away(direction.dimension());
      const auto& orientation = neighbors.orientation();
      for (const auto& neighbor : neighbors) {
        const ::dg::MortarId<Dim> mortar_id{direction, neighbor};
        all_mortar_meshes.emplace(
            mortar_id,
            ::dg::mortar_mesh(
                face_mesh,
                domain::Initialization::create_initial_mesh(
                    initial_extents, neighbor, mesh.quadrature(0), orientation)
                    .slice_away(direction.dimension())));
        all_mortar_sizes.emplace(
            mortar_id, ::dg::mortar_size(element.id(), neighbor,
                                         direction.dimension(), orientation));
      }
    }
    ::Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                                 std::move(all_mortar_meshes),
                                                 std::move(all_mortar_sizes));
    return {std::move(box)};
  }
};

// Compute auxiliary variables and fluxes from the primal variables, prepare the
// local side of all mortars and send the local mortar data to neighbors. Also
// handle boundary conditions by preparing the exterior ("ghost") side of
// external mortars.
template <typename System, bool Linearized, typename TemporalIdTag,
          typename PrimalFieldsTag, typename PrimalFluxesTag,
          typename OperatorAppliedToFieldsTag, typename PrimalMortarFieldsTag,
          typename PrimalMortarFluxesTag,
          typename FluxesArgsTags =
              typename System::fluxes_computer::argument_tags,
          typename SourcesArgsTags = typename elliptic::get_sources_computer<
              System, Linearized>::argument_tags>
struct PrepareAndSendMortarData;

template <typename System, bool Linearized, typename TemporalIdTag,
          typename PrimalFieldsTag, typename PrimalFluxesTag,
          typename OperatorAppliedToFieldsTag, typename PrimalMortarFieldsTag,
          typename PrimalMortarFluxesTag, typename... FluxesArgsTags,
          typename... SourcesArgsTags>
struct PrepareAndSendMortarData<
    System, Linearized, TemporalIdTag, PrimalFieldsTag, PrimalFluxesTag,
    OperatorAppliedToFieldsTag, PrimalMortarFieldsTag, PrimalMortarFluxesTag,
    tmpl::list<FluxesArgsTags...>, tmpl::list<SourcesArgsTags...>> {
 private:
  static constexpr size_t Dim = System::volume_dim;
  using all_mortar_data_tag = ::Tags::Mortars<
      elliptic::dg::Tags::MortarData<typename TemporalIdTag::type,
                                     typename PrimalMortarFieldsTag::tags_list,
                                     typename PrimalMortarFluxesTag::tags_list>,
      Dim>;
  using mortar_data_inbox_tag =
      MortarDataInboxTag<Dim, TemporalIdTag,
                         typename PrimalMortarFieldsTag::tags_list,
                         typename PrimalMortarFluxesTag::tags_list>;
  using BoundaryConditionsBase = typename System::boundary_conditions_base;

 public:
  // Request these tags be added to the DataBox by the `SetupDataBox` action. We
  // don't actually need to initialize them, because the `TemporalIdTag` and the
  // `PrimalFieldsTag` will be set by other actions before applying the operator
  // and the remaining tags hold output of the operator.
  using simple_tags =
      tmpl::list<TemporalIdTag, PrimalFieldsTag, PrimalFluxesTag,
                 OperatorAppliedToFieldsTag, all_mortar_data_tag>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& temporal_id = db::get<TemporalIdTag>(box);
    const auto& element = db::get<domain::Tags::Element<Dim>>(box);
    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const size_t num_points = mesh.number_of_grid_points();
    const auto& mortar_meshes =
        db::get<::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>>(box);
    const auto& domain = db::get<domain::Tags::Domain<Dim>>(box);
    const auto& boundary_conditions = domain.blocks()
                                          .at(element_id.block_id())
                                          .external_boundary_conditions();
    const auto apply_boundary_condition =
        [&box, &boundary_conditions, &element_id](
            const Direction<Dim>& direction,
            const auto... fields_and_fluxes) noexcept {
          ASSERT(
              boundary_conditions.contains(direction),
              "No boundary condition is available in block "
                  << element_id.block_id() << " in direction " << direction
                  << ". Make sure you are setting up boundary conditions when "
                     "creating the domain.");
          ASSERT(dynamic_cast<const BoundaryConditionsBase*>(
                     boundary_conditions.at(direction).get()) != nullptr,
                 "The boundary condition in block "
                     << element_id.block_id() << " in direction " << direction
                     << " has an unexpected type. Make sure it derives off the "
                        "'boundary_conditions_base' class set in the system.");
          const auto& boundary_condition =
              dynamic_cast<const BoundaryConditionsBase&>(
                  *boundary_conditions.at(direction));
          elliptic::apply_boundary_condition<Linearized>(
              boundary_condition, box, direction, fields_and_fluxes...);
        };

    // Can't `db::get` the arguments for the boundary conditions within
    // `db::mutate`, so we extract the data to mutate and move it back in when
    // we're done.
    typename PrimalFluxesTag::type primal_fluxes;
    typename all_mortar_data_tag::type all_mortar_data;
    db::mutate<PrimalFluxesTag, all_mortar_data_tag>(
        make_not_null(&box),
        [&primal_fluxes, &all_mortar_data](const auto local_primal_fluxes,
                                           const auto local_all_mortar_data) {
          primal_fluxes = std::move(*local_primal_fluxes);
          all_mortar_data = std::move(*local_all_mortar_data);
        });

    // Prepare mortar data
    //
    // These memory buffers will be discarded when the action returns so we
    // don't inflate the memory usage of the simulation when the element is
    // inactive.
    Variables<typename System::auxiliary_fields> auxiliary_fields_buffer{
        num_points};
    Variables<typename System::auxiliary_fluxes> auxiliary_fluxes_buffer{
        num_points};
    elliptic::dg::prepare_mortar_data<System, Linearized>(
        make_not_null(&auxiliary_fields_buffer),
        make_not_null(&auxiliary_fluxes_buffer), make_not_null(&primal_fluxes),
        make_not_null(&all_mortar_data), db::get<PrimalFieldsTag>(box), element,
        db::get<domain::Tags::Mesh<Dim>>(box),
        db::get<domain::Tags::InverseJacobian<Dim, Frame::Logical,
                                              Frame::Inertial>>(box),
        db::get<domain::Tags::Interface<
            domain::Tags::InternalDirections<Dim>,
            ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>>(
            box),
        db::get<domain::Tags::Interface<
            domain::Tags::BoundaryDirectionsInterior<Dim>,
            ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>>(
            box),
        db::get<domain::Tags::Interface<
            domain::Tags::InternalDirections<Dim>,
            ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>>(box),
        db::get<domain::Tags::Interface<
            domain::Tags::BoundaryDirectionsInterior<Dim>,
            ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>>(box),
        mortar_meshes,
        db::get<::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>(box),
        temporal_id, apply_boundary_condition,
        std::forward_as_tuple(db::get<FluxesArgsTags>(box)...),
        std::forward_as_tuple(db::get<SourcesArgsTags>(box)...),
        interface_apply<domain::Tags::InternalDirections<Dim>,
                        typename System::fluxes_computer::argument_tags,
                        get_volume_tags<typename System::fluxes_computer>>(
            [](const auto&... fluxes_args_on_face) noexcept {
              return std::forward_as_tuple(fluxes_args_on_face...);
            },
            box),
        interface_apply<domain::Tags::BoundaryDirectionsInterior<Dim>,
                        typename System::fluxes_computer::argument_tags,
                        get_volume_tags<typename System::fluxes_computer>>(
            [](const auto&... fluxes_args_on_face) noexcept {
              return std::forward_as_tuple(fluxes_args_on_face...);
            },
            box));

    // Move the mutated data back into the DataBox
    db::mutate<PrimalFluxesTag, all_mortar_data_tag>(
        make_not_null(&box),
        [&primal_fluxes, &all_mortar_data](const auto local_primal_fluxes,
                                           const auto local_all_mortar_data) {
          *local_primal_fluxes = std::move(primal_fluxes);
          *local_all_mortar_data = std::move(all_mortar_data);
        });

    // Send mortar data to neighbors
    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    for (const auto& [direction, neighbors] : element.neighbors()) {
      const size_t dimension = direction.dimension();
      const auto& orientation = neighbors.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());
      for (const auto& neighbor_id : neighbors) {
        const ::dg::MortarId<Dim> mortar_id{direction, neighbor_id};
        // Make a copy of the local boundary data on the mortar to send to the
        // neighbor
        auto remote_boundary_data_on_mortar =
            get<all_mortar_data_tag>(box).at(mortar_id).local_data(
                temporal_id);
        // Reorient the data to the neighbor orientation if necessary
        if (not orientation.is_aligned()) {
          remote_boundary_data_on_mortar.orient_on_slice(
              mortar_meshes.at(mortar_id).extents(), dimension, orientation);
        }
        // Send remote data to neighbor
        Parallel::receive_data<mortar_data_inbox_tag>(
            receiver_proxy[neighbor_id], temporal_id,
            std::make_pair(
                ::dg::MortarId<Dim>{direction_from_neighbor, element.id()},
                std::move(remote_boundary_data_on_mortar)));
      }  // loop over neighbors in direction
    }    // loop over directions

    return {std::move(box)};
  }
};

// Wait until all mortar data from neighbors is available. Then add boundary
// corrections to the primal fluxes, compute their derivatives (i.e. the second
// derivatives of the primal variables) and add boundary corrections to the
// result.
template <typename System, bool Linearized, typename TemporalIdTag,
          typename PrimalFieldsTag, typename PrimalFluxesTag,
          typename OperatorAppliedToFieldsTag, typename PrimalMortarFieldsTag,
          typename PrimalMortarFluxesTag,
          typename FluxesArgsTags =
              typename System::fluxes_computer::argument_tags,
          typename SourcesArgsTags = typename elliptic::get_sources_computer<
              System, Linearized>::argument_tags>
struct ReceiveMortarDataAndApplyOperator;

template <typename System, bool Linearized, typename TemporalIdTag,
          typename PrimalFieldsTag, typename PrimalFluxesTag,
          typename OperatorAppliedToFieldsTag, typename PrimalMortarFieldsTag,
          typename PrimalMortarFluxesTag, typename... FluxesArgsTags,
          typename... SourcesArgsTags>
struct ReceiveMortarDataAndApplyOperator<
    System, Linearized, TemporalIdTag, PrimalFieldsTag, PrimalFluxesTag,
    OperatorAppliedToFieldsTag, PrimalMortarFieldsTag, PrimalMortarFluxesTag,
    tmpl::list<FluxesArgsTags...>, tmpl::list<SourcesArgsTags...>> {
 private:
  static constexpr size_t Dim = System::volume_dim;
  using all_mortar_data_tag = ::Tags::Mortars<
      elliptic::dg::Tags::MortarData<typename TemporalIdTag::type,
                                     typename PrimalMortarFieldsTag::tags_list,
                                     typename PrimalMortarFluxesTag::tags_list>,
      Dim>;
  using mortar_data_inbox_tag =
      MortarDataInboxTag<Dim, TemporalIdTag,
                         typename PrimalMortarFieldsTag::tags_list,
                         typename PrimalMortarFluxesTag::tags_list>;

 public:
  using const_global_cache_tags =
      tmpl::list<elliptic::dg::Tags::PenaltyParameter>;
  using inbox_tags = tmpl::list<mortar_data_inbox_tag>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&, Parallel::AlgorithmExecution> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& temporal_id = get<TemporalIdTag>(box);

    if (not ::dg::has_received_from_all_mortars<mortar_data_inbox_tag>(
            temporal_id, get<domain::Tags::Element<Dim>>(box), inboxes)) {
      return {std::move(box), Parallel::AlgorithmExecution::Retry};
    }

    // Move received "remote" mortar data into the DataBox
    if (LIKELY(db::get<domain::Tags::Element<Dim>>(box).number_of_neighbors() >
               0)) {
      auto received_mortar_data =
          std::move(tuples::get<mortar_data_inbox_tag>(inboxes)
                        .extract(temporal_id)
                        .mapped());
      db::mutate<all_mortar_data_tag>(
          make_not_null(&box), [&received_mortar_data, &temporal_id](
                                   const auto all_mortar_data) noexcept {
            for (auto& [mortar_id, mortar_data] : received_mortar_data) {
              all_mortar_data->at(mortar_id).remote_insert(
                  temporal_id, std::move(mortar_data));
            }
          });
    }

    // Apply DG operator
    db::mutate<OperatorAppliedToFieldsTag, all_mortar_data_tag>(
        make_not_null(&box),
        [](const auto&... args) noexcept {
          elliptic::dg::apply_operator<System, Linearized>(args...);
        },
        db::get<PrimalFieldsTag>(box), db::get<PrimalFluxesTag>(box),
        db::get<domain::Tags::Mesh<Dim>>(box),
        db::get<domain::Tags::InverseJacobian<Dim, Frame::Logical,
                                              Frame::Inertial>>(box),
        db::get<domain::Tags::Interface<
            domain::Tags::InternalDirections<Dim>,
            ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>>(box),
        db::get<domain::Tags::Interface<
            domain::Tags::BoundaryDirectionsInterior<Dim>,
            ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>>(box),
        db::get<::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>>(box),
        db::get<::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>(box),
        db::get<elliptic::dg::Tags::PenaltyParameter>(box), temporal_id,
        std::forward_as_tuple(db::get<SourcesArgsTags>(box)...));

    return {std::move(box), Parallel::AlgorithmExecution::Continue};
  }
};

}  // namespace detail

/*!
 * \brief Initialize DataBox tags for applying the DG operator
 *
 * \see elliptic::dg::Actions::apply_operator
 */
template <typename System>
using initialize_operator =
    tmpl::list<detail::InitializeFacesAndMortars<System>>;

/*!
 * \brief Apply the DG operator to the `PrimalFieldsTag` and write the result to
 * the `OperatorAppliedToFieldsTag`
 *
 * Add this list to the action list of a parallel component to compute the
 * elliptic DG operator or its linearization. The operator involves a
 * communication between nearest-neighbor elements. See `elliptic::dg` for
 * details on the elliptic DG operator. Make sure to add
 * `elliptic::dg::Actions::initialize_operator` to the initialization phase of
 * your parallel component so the required DataBox tags are set up before
 * applying the operator.
 *
 * The result of the computation is written to the `OperatorAppliedToFieldsTag`.
 * Additionally, the primal fluxes are written to the `PrimalFluxesTag` as an
 * intermediate result. The auxiliary fields and fluxes are discarded to avoid
 * inflating the memory usage.
 *
 * You can specify the `PrimalMortarFieldsTag` and the `PrimalMortarFluxesTag`
 * to re-use mortar-data memory buffers from other operator applications, for
 * example when applying the nonlinear and linearized operator. They default to
 * the `PrimalFieldsTag` and the `PrimalFluxesTag`, meaning memory buffers
 * corresponding to these tags are set up in the DataBox.
 */
template <typename System, bool Linearized, typename TemporalIdTag,
          typename PrimalFieldsTag, typename PrimalFluxesTag,
          typename OperatorAppliedToFieldsTag,
          typename PrimalMortarFieldsTag = PrimalFieldsTag,
          typename PrimalMortarFluxesTag = PrimalFluxesTag>
using apply_operator =
    tmpl::list<detail::PrepareAndSendMortarData<
                   System, Linearized, TemporalIdTag, PrimalFieldsTag,
                   PrimalFluxesTag, OperatorAppliedToFieldsTag,
                   PrimalMortarFieldsTag, PrimalMortarFluxesTag>,
               detail::ReceiveMortarDataAndApplyOperator<
                   System, Linearized, TemporalIdTag, PrimalFieldsTag,
                   PrimalFluxesTag, OperatorAppliedToFieldsTag,
                   PrimalMortarFieldsTag, PrimalMortarFluxesTag>>;

/*!
 * \brief For linear systems, impose inhomogeneous boundary conditions as
 * contributions to the fixed sources (i.e. the RHS of the equations).
 *
 * \see elliptic::dg::impose_inhomogeneous_boundary_conditions_on_source
 */
template <
    typename System, typename FixedSourcesTag,
    typename FluxesArgsTags = typename System::fluxes_computer::argument_tags,
    typename SourcesArgsTags = typename System::sources_computer::argument_tags>
struct ImposeInhomogeneousBoundaryConditionsOnSource;

/// \cond
template <typename System, typename FixedSourcesTag, typename... FluxesArgsTags,
          typename... SourcesArgsTags>
struct ImposeInhomogeneousBoundaryConditionsOnSource<
    System, FixedSourcesTag, tmpl::list<FluxesArgsTags...>,
    tmpl::list<SourcesArgsTags...>> {
 private:
  static constexpr size_t Dim = System::volume_dim;
  using BoundaryConditionsBase = typename System::boundary_conditions_base;

 public:
  using const_global_cache_tags =
      tmpl::list<elliptic::dg::Tags::PenaltyParameter>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& domain = db::get<domain::Tags::Domain<Dim>>(box);
    const auto& boundary_conditions = domain.blocks()
                                          .at(element_id.block_id())
                                          .external_boundary_conditions();
    const auto apply_boundary_condition =
        [&box, &boundary_conditions, &element_id](
            const Direction<Dim>& direction,
            const auto... fields_and_fluxes) noexcept {
          ASSERT(
              boundary_conditions.contains(direction),
              "No boundary condition is available in block "
                  << element_id.block_id() << " in direction " << direction
                  << ". Make sure you are setting up boundary conditions when "
                     "creating the domain.");
          ASSERT(dynamic_cast<const BoundaryConditionsBase*>(
                     boundary_conditions.at(direction).get()) != nullptr,
                 "The boundary condition in block "
                     << element_id.block_id() << " in direction " << direction
                     << " has an unexpected type. Make sure it derives off the "
                        "'boundary_conditions_base' class set in the system.");
          const auto& boundary_condition =
              dynamic_cast<const BoundaryConditionsBase&>(
                  *boundary_conditions.at(direction));
          elliptic::apply_boundary_condition<false>(
              boundary_condition, box, direction, fields_and_fluxes...);
        };

    // Can't `db::get` the arguments for the boundary conditions within
    // `db::mutate`, so we extract the data to mutate and move it back in when
    // we're done.
    typename FixedSourcesTag::type fixed_sources;
    db::mutate<FixedSourcesTag>(
        make_not_null(&box), [&fixed_sources](const auto local_fixed_sources) {
          fixed_sources = std::move(*local_fixed_sources);
        });

    elliptic::dg::impose_inhomogeneous_boundary_conditions_on_source<System>(
        make_not_null(&fixed_sources), db::get<domain::Tags::Element<Dim>>(box),
        db::get<domain::Tags::Mesh<Dim>>(box),
        db::get<domain::Tags::InverseJacobian<Dim, Frame::Logical,
                                              Frame::Inertial>>(box),
        db::get<domain::Tags::Interface<
            domain::Tags::BoundaryDirectionsInterior<Dim>,
            ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>>(
            box),
        db::get<domain::Tags::Interface<
            domain::Tags::BoundaryDirectionsInterior<Dim>,
            ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>>(box),
        db::get<::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>>(box),
        db::get<::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>(box),
        db::get<elliptic::dg::Tags::PenaltyParameter>(box),
        apply_boundary_condition,
        std::forward_as_tuple(db::get<FluxesArgsTags>(box)...),
        std::forward_as_tuple(db::get<SourcesArgsTags>(box)...),
        interface_apply<domain::Tags::InternalDirections<Dim>,
                        typename System::fluxes_computer::argument_tags,
                        get_volume_tags<typename System::fluxes_computer>>(
            [](const auto&... fluxes_args_on_face) noexcept {
              return std::forward_as_tuple(fluxes_args_on_face...);
            },
            box),
        interface_apply<domain::Tags::BoundaryDirectionsInterior<Dim>,
                        typename System::fluxes_computer::argument_tags,
                        get_volume_tags<typename System::fluxes_computer>>(
            [](const auto&... fluxes_args_on_face) noexcept {
              return std::forward_as_tuple(fluxes_args_on_face...);
            },
            box));

    // Move the mutated data back into the DataBox
    db::mutate<FixedSourcesTag>(
        make_not_null(&box), [&fixed_sources](const auto local_fixed_sources) {
          *local_fixed_sources = std::move(fixed_sources);
        });
    return {std::move(box)};
  }
};
/// \endcond

}  // namespace elliptic::dg::Actions
