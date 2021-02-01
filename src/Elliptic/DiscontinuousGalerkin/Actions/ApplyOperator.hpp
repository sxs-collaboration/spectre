// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions/AnalyticSolution.hpp"
#include "Elliptic/BoundaryConditions/ApplyBoundaryCondition.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/HasReceivedFromAllMortars.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeInterfaces.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeMortars.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace elliptic::dg {
namespace Tags {
/*!
 * \brief An auxiliary variable for the `CorrespondingPrimal` variable. The
 * `Tag` must have a type appropriate for the auxiliary variable.
 *
 * For example, consider a scalar field represented by the tag `Field` with a
 * corresponding auxiliary field represented by the tag `GradField`. When
 * applying the operator to `Field` we can just use `GradField` as the auxiliary
 * variable. However, when applying the operator to other quantities such as a
 * prefixed `Var<Field>` (e.g. the internal "operand" of a linear solver) we
 * also need a corresponding auxiliary variable tag. This prefix tag can be used
 * to create such tags in cases where the code that invokes the operator doesn't
 * provide them.
 */
template <typename Tag, typename CorrespondingPrimal>
struct Auxiliary : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
}  // namespace Tags

namespace Actions {
// The individual actions in this namespace are not exposed publicly because
// they don't work on their own. Instead, the public interface (defined below)
// exposes them in action lists.
namespace detail {

template <typename AuxiliaryFields, typename PrimalVars>
struct MakeAuxiliaryTagsImpl;

template <typename... AuxiliaryFields, typename... PrimalVars>
struct MakeAuxiliaryTagsImpl<tmpl::list<AuxiliaryFields...>,
                             tmpl::list<PrimalVars...>> {
  using type = tmpl::list<Tags::Auxiliary<AuxiliaryFields, PrimalVars>...>;
};

template <typename AuxiliaryFields, typename PrimalVars>
using make_auxiliary_tags =
    typename detail::MakeAuxiliaryTagsImpl<AuxiliaryFields, PrimalVars>::type;

// This action can be deleted once initialization tags are picked up from
// actions in other phases.
template <typename System, typename TemporalIdTag, typename VarsTag,
          typename OperatorAppliedToVarsTag, typename AuxiliaryVarsTag>
struct InitializeOperator {
  using simple_tags = tmpl::list<TemporalIdTag, VarsTag,
                                 OperatorAppliedToVarsTag, AuxiliaryVarsTag>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Nothing to do. The `TemporalIdTag` and the `VarsTag` will be set by other
    // actions before applying the operator and the other tags hold output of
    // the operator.
    return {std::move(box)};
  }
};

template <size_t Dim, typename TemporalIdTag, typename PrimalVars,
          typename AuxiliaryVars>
struct MortarDataInboxTag
    : public Parallel::InboxInserters::Map<
          MortarDataInboxTag<Dim, TemporalIdTag, PrimalVars, AuxiliaryVars>> {
  using temporal_id = typename TemporalIdTag::type;
  using type = std::map<
      temporal_id,
      FixedHashMap<maximum_number_of_neighbors(Dim), ::dg::MortarId<Dim>,
                   elliptic::dg::BoundaryData<PrimalVars, AuxiliaryVars>,
                   boost::hash<::dg::MortarId<Dim>>>>;
};

// Compute auxiliary variables from the primal variables, prepare the local side
// of all mortars and send the local mortar data to neighbors. Also handle
// boundary conditions by preparing the exterior ("ghost") side of external
// mortars.
template <
    typename System, bool Linearized, typename TemporalIdTag, typename VarsTag,
    typename OperatorAppliedToVarsTag, typename AuxiliaryVarsTag,
    typename FluxesArgsTags = typename System::fluxes_computer::argument_tags,
    // TODO: make these the linearized sources when supported
    typename SourcesArgsTags = typename System::sources_computer::argument_tags>
struct PrepareAndSendMortarData;

template <typename System, bool Linearized, typename TemporalIdTag,
          typename VarsTag, typename OperatorAppliedToVarsTag,
          typename AuxiliaryVarsTag, typename... FluxesArgsTags,
          typename... SourcesArgsTags>
struct PrepareAndSendMortarData<System, Linearized, TemporalIdTag, VarsTag,
                                OperatorAppliedToVarsTag, AuxiliaryVarsTag,
                                tmpl::list<FluxesArgsTags...>,
                                tmpl::list<SourcesArgsTags...>> {
 public:
  static constexpr size_t Dim = System::volume_dim;

 private:
  using all_mortar_data_tag = ::Tags::Mortars<
      elliptic::dg::Tags::MortarData<typename VarsTag::tags_list,
                                     typename AuxiliaryVarsTag::tags_list>,
      Dim>;
  using mortar_data_inbox_tag =
      MortarDataInboxTag<Dim, TemporalIdTag, typename VarsTag::tags_list,
                         typename AuxiliaryVarsTag::tags_list>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& temporal_id = db::get<TemporalIdTag>(box);
    const auto& element = db::get<domain::Tags::Element<Dim>>(box);
    const auto& mortar_meshes =
        db::get<::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>>(box);

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

    const auto apply_boundary_condition =
        [&boundary_condition, &box](const Direction<Dim>& direction,
                                    const auto... fields_and_fluxes) noexcept {
          elliptic::apply_boundary_condition<Linearized>(
              *boundary_condition, box, direction, fields_and_fluxes...);
        };

    // Can't `db::get` the arguments for the boundary conditions within
    // `db::mutate`, so we retrieve the pointers to the memory buffers in
    // advance.
    typename AuxiliaryVarsTag::type* auxiliary_vars{nullptr};
    typename all_mortar_data_tag::type* all_mortar_data{nullptr};
    db::mutate<AuxiliaryVarsTag, all_mortar_data_tag>(
        make_not_null(&box),
        [&auxiliary_vars, &all_mortar_data](const auto local_auxiliary_vars,
                                            const auto local_all_mortar_data) {
          auxiliary_vars = local_auxiliary_vars;
          all_mortar_data = local_all_mortar_data;
        });

    // Prepare mortar data
    elliptic::dg::prepare_mortar_data<System, Linearized>(
        make_not_null(auxiliary_vars), make_not_null(all_mortar_data),
        db::get<VarsTag>(box), element, db::get<domain::Tags::Mesh<Dim>>(box),
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
            all_mortar_data->at(mortar_id).local_data(temporal_id);
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
// corrections to the auxiliary variables, compute their derivatives (i.e. the
// second derivatives of the primal variables) and add boundary corrections to
// the result.
template <
    typename System, bool Linearized, typename TemporalIdTag, typename VarsTag,
    typename OperatorAppliedToVarsTag, typename AuxiliaryVarsTag,
    typename FluxesArgsTags = typename System::fluxes_computer::argument_tags,
    // TODO: make these the linearized sources when supported
    typename SourcesArgsTags = typename System::sources_computer::argument_tags>
struct ReceiveMortarDataAndApplyOperator;

template <typename System, bool Linearized, typename TemporalIdTag,
          typename VarsTag, typename OperatorAppliedToVarsTag,
          typename AuxiliaryVarsTag, typename... FluxesArgsTags,
          typename... SourcesArgsTags>
struct ReceiveMortarDataAndApplyOperator<
    System, Linearized, TemporalIdTag, VarsTag, OperatorAppliedToVarsTag,
    AuxiliaryVarsTag, tmpl::list<FluxesArgsTags...>,
    tmpl::list<SourcesArgsTags...>> {
 public:
  static constexpr size_t Dim = System::volume_dim;

 private:
  using all_mortar_data_tag = ::Tags::Mortars<
      elliptic::dg::Tags::MortarData<typename VarsTag::tags_list,
                                     typename AuxiliaryVarsTag::tags_list>,
      Dim>;
  using mortar_data_inbox_tag =
      MortarDataInboxTag<Dim, TemporalIdTag, typename VarsTag::tags_list,
                         typename AuxiliaryVarsTag::tags_list>;

 public:
  using const_global_cache_tags =
      tmpl::list<elliptic::dg::Tags::PenaltyParameter>;
  using inbox_tags = tmpl::list<mortar_data_inbox_tag>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(const db::DataBox<DbTags>& box,
                       const tuples::TaggedTuple<InboxTags...>& inboxes,
                       const Parallel::GlobalCache<Metavariables>& /*cache*/,
                       const ArrayIndex& /*array_index*/) noexcept {
    return ::dg::has_received_from_all_mortars<mortar_data_inbox_tag>(
        db::get<TemporalIdTag>(box), get<domain::Tags::Element<Dim>>(box),
        inboxes);
  }

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& temporal_id = get<TemporalIdTag>(box);

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
    db::mutate<OperatorAppliedToVarsTag, AuxiliaryVarsTag, all_mortar_data_tag>(
        make_not_null(&box),
        [](const auto operator_applied_to_vars, const auto auxiliary_vars,
           const auto all_mortar_data, const auto& vars,
           const auto&... args) noexcept {
          elliptic::dg::apply_operator<System, Linearized>(
              operator_applied_to_vars, auxiliary_vars, all_mortar_data, vars,
              args...);
        },
        db::get<VarsTag>(box), db::get<domain::Tags::Mesh<Dim>>(box),
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
        std::forward_as_tuple(db::get<FluxesArgsTags>(box)...),
        std::forward_as_tuple(db::get<SourcesArgsTags>(box)...));

    return {std::move(box)};
  }
};

/// Only needed for compatibility with the `InitializeMortars` action
template <typename System, typename TemporalIdTag, typename VarsTag,
          typename OperatorAppliedToVarsTag, typename AuxiliaryVarsTag>
struct BoundaryScheme {
  static constexpr size_t volume_dim = System::volume_dim;
  using mortar_data_tag =
      elliptic::dg::Tags::MortarData<typename VarsTag::tags_list,
                                     typename AuxiliaryVarsTag::tags_list>;
  using temporal_id_tag = TemporalIdTag;
  using receive_temporal_id_tag = TemporalIdTag;
};

}  // namespace detail

template <
    typename System, typename TemporalIdTag, typename VarsTag,
    typename OperatorAppliedToVarsTag,
    typename AuxiliaryVarsTag = ::Tags::Variables<detail::make_auxiliary_tags<
        typename System::auxiliary_fields, typename VarsTag::tags_list>>>
using initialize_operator = tmpl::list<
    ::dg::Actions::InitializeInterfaces<
        System, ::dg::Initialization::slice_tags_to_face<>,
        ::dg::Initialization::slice_tags_to_exterior<>,
        ::dg::Initialization::face_compute_tags<>,
        ::dg::Initialization::exterior_compute_tags<>, false, false>,
    ::dg::Actions::InitializeMortars<
        detail::BoundaryScheme<System, TemporalIdTag, VarsTag,
                               OperatorAppliedToVarsTag, AuxiliaryVarsTag>,
        true>,
    detail::InitializeOperator<System, TemporalIdTag, VarsTag,
                               OperatorAppliedToVarsTag, AuxiliaryVarsTag>>;

template <
    typename System, bool Linearized, typename TemporalIdTag, typename VarsTag,
    typename OperatorAppliedToVarsTag,
    typename AuxiliaryVarsTag = ::Tags::Variables<detail::make_auxiliary_tags<
        typename System::auxiliary_fields, typename VarsTag::tags_list>>>
using apply_operator =
    tmpl::list<detail::PrepareAndSendMortarData<
                   System, Linearized, TemporalIdTag, VarsTag,
                   OperatorAppliedToVarsTag, AuxiliaryVarsTag>,
               detail::ReceiveMortarDataAndApplyOperator<
                   System, Linearized, TemporalIdTag, VarsTag,
                   OperatorAppliedToVarsTag, AuxiliaryVarsTag>>;

template <
    typename System, typename FixedSourcesTag,
    typename FluxesArgsTags = typename System::fluxes_computer::argument_tags,
    // TODO: make these the linearized sources when supported
    typename SourcesArgsTags = typename System::sources_computer::argument_tags>
struct ImposeInhomogeneousBoundaryConditionsOnSource;

template <typename System, typename FixedSourcesTag, typename... FluxesArgsTags,
          typename... SourcesArgsTags>
struct ImposeInhomogeneousBoundaryConditionsOnSource<
    System, FixedSourcesTag, tmpl::list<FluxesArgsTags...>,
    tmpl::list<SourcesArgsTags...>> {
  static constexpr size_t Dim = System::volume_dim;
  using const_global_cache_tags =
      tmpl::list<elliptic::dg::Tags::PenaltyParameter>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    using Registrar =
        elliptic::BoundaryConditions::Registrars::AnalyticSolution<System>;
    const std::unique_ptr<elliptic::BoundaryConditions::BoundaryCondition<
        Dim, tmpl::list<Registrar>>>
        boundary_condition = std::make_unique<
            typename Registrar::template f<tmpl::list<Registrar>>>(
            elliptic::BoundaryConditionType::Dirichlet);

    const auto apply_boundary_condition =
        [&boundary_condition, &box](const Direction<Dim>& direction,
                                    const auto... fields_and_fluxes) noexcept {
          elliptic::apply_boundary_condition<false>(
              *boundary_condition, box, direction, fields_and_fluxes...);
        };

    // Can't `db::get` the arguments for the boundary conditions within
    // `db::mutate`, so we retrieve the pointers to the memory buffers in
    // advance.
    typename FixedSourcesTag::type* fixed_sources{nullptr};
    db::mutate<FixedSourcesTag>(
        make_not_null(&box), [&fixed_sources](const auto local_fixed_sources) {
          fixed_sources = local_fixed_sources;
        });

    elliptic::dg::impose_inhomogeneous_boundary_conditions_on_source<System>(
        make_not_null(fixed_sources), db::get<domain::Tags::Element<Dim>>(box),
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
    return {std::move(box)};
  }
};

}  // namespace Actions
}  // namespace elliptic::dg
