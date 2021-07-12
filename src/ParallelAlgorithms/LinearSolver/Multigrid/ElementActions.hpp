// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <map>
#include <unordered_set>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/Structure/ChildSize.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Observer/Tags.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Actions/RestrictFields.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Hierarchy.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace LinearSolver::multigrid::detail {

/// \cond
template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
struct SendCorrectionToFinerGrid;
/// \endcond

template <size_t Dim, typename FieldsTag, typename OptionsGroup,
          typename SourceTag>
struct InitializeElement {
  using initialization_tags = tmpl::list<Tags::ChildrenRefinementLevels<Dim>,
                                         Tags::ParentRefinementLevels<Dim>>;
  using simple_tags =
      tmpl::list<Tags::ParentId<Dim>, Tags::ChildIds<Dim>,
                 Tags::ParentMesh<Dim>,
                 observers::Tags::ObservationKey<Tags::MultigridLevel>,
                 observers::Tags::ObservationKey<Tags::IsFinestGrid>,
                 Tags::ObservationId<OptionsGroup>,
                 Tags::VolumeDataForOutput<OptionsGroup, FieldsTag>>;
  using compute_tags = tmpl::list<>;
  using const_global_cache_tags =
      tmpl::list<Tags::MaxLevels<OptionsGroup>,
                 Tags::OutputVolumeData<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Note: The following initialization code assumes that all elements in a
    // block have the same p-refinement. This is true for initial domains, but
    // will be broken by AMR.

    const size_t multigrid_level = element_id.grid_index();
    const bool is_finest_grid = multigrid_level == 0;
    const bool is_coarsest_grid =
        db::get<domain::Tags::InitialRefinementLevels<Dim>>(box) ==
        db::get<Tags::ParentRefinementLevels<Dim>>(box);

    std::optional<ElementId<Dim>> parent_id =
        is_coarsest_grid ? std::nullopt
                         : std::make_optional(multigrid::parent_id(element_id));
    std::unordered_set<ElementId<Dim>> child_ids = multigrid::child_ids(
        element_id, db::get<Tags::ChildrenRefinementLevels<Dim>>(
                        box)[element_id.block_id()]);

    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    std::optional<Mesh<Dim>> parent_mesh =
        is_coarsest_grid ? std::nullopt : std::make_optional(mesh);

    auto observation_key_level = std::make_optional(
        is_finest_grid ? std::string{""}
                       : (std::string{"Level"} + get_output(multigrid_level)));
    auto observation_key_is_finest_grid =
        is_finest_grid ? std::make_optional(std::string{""}) : std::nullopt;

    // Initialize volume data output
    using VolumeDataVars =
        typename Tags::VolumeDataForOutput<OptionsGroup, FieldsTag>::type;
    VolumeDataVars volume_data{};
    if (db::get<Tags::OutputVolumeData<OptionsGroup>>(box)) {
      volume_data.initialize(mesh.number_of_grid_points());
    }

    Initialization::mutate_assign<simple_tags>(
        make_not_null(&box), std::move(parent_id), std::move(child_ids),
        std::move(parent_mesh), std::move(observation_key_level),
        std::move(observation_key_is_finest_grid), size_t{0},
        std::move(volume_data));
    return {std::move(box)};
  }
};

// These two actions communicate and project the residual from the finer grid to
// the coarser grid, storing it in the `SourceTag` on the coarser grid.
template <typename FieldsTag, typename OptionsGroup,
          typename ResidualIsMassiveTag, typename SourceTag>
using SendResidualToCoarserGrid = Actions::SendFieldsToCoarserGrid<
    tmpl::list<db::add_tag_prefix<LinearSolver::Tags::Residual, FieldsTag>>,
    OptionsGroup, ResidualIsMassiveTag, tmpl::list<SourceTag>>;

template <size_t Dim, typename FieldsTag, typename OptionsGroup,
          typename SourceTag>
using ReceiveResidualFromFinerGrid = Actions::ReceiveFieldsFromFinerGrid<
    Dim,
    tmpl::list<db::add_tag_prefix<LinearSolver::Tags::Residual, FieldsTag>>,
    OptionsGroup, tmpl::list<SourceTag>>;

// Once the residual from the finer grid has been received and stored in the
// `SourceTag`, this action prepares the pre-smoothing that will determine
// an approximate solution on this grid. The pre-smoother is a separate
// linear solver that runs independently after this action.
template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
struct PreparePreSmoothing {
 private:
  using fields_tag = FieldsTag;
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;
  using source_tag = SourceTag;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const size_t iteration_id =
        db::get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    if (UNLIKELY(db::get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s %s(%zu): Prepare pre-smoothing\n", element_id,
                       Options::name<OptionsGroup>(), iteration_id);
    }

    // On coarser grids the smoother solves for a correction to the finer-grid
    // fields, so we set its initial guess to zero. On the finest grid we smooth
    // the fields directly, so there's nothing to prepare.
    if (element_id.grid_index() > 0) {
      db::mutate<fields_tag, operator_applied_to_fields_tag>(
          make_not_null(&box),
          [](const auto fields, const auto operator_applied_to_fields,
             const auto& source) noexcept {
            *fields = make_with_value<typename fields_tag::type>(source, 0.);
            // We can set the linear operator applied to the initial fields to
            // zero as well, since it's linear. This may save the smoother an
            // operator application on coarser grids if it's optimized for this.
            *operator_applied_to_fields =
                make_with_value<typename operator_applied_to_fields_tag::type>(
                    source, 0.);
          },
          db::get<source_tag>(box));
    }

    // Record pre-smoothing initial fields and source
    if (db::get<Tags::OutputVolumeData<OptionsGroup>>(box)) {
      db::mutate<Tags::VolumeDataForOutput<OptionsGroup, FieldsTag>>(
          make_not_null(&box),
          [](const auto volume_data, const auto& initial_fields,
             const auto& source) noexcept {
            volume_data->assign_subset(
                Variables<db::wrap_tags_in<Tags::PreSmoothingInitial,
                                           typename fields_tag::tags_list>>(
                    initial_fields));
            volume_data->assign_subset(
                Variables<db::wrap_tags_in<Tags::PreSmoothingSource,
                                           typename fields_tag::tags_list>>(
                    source));
          },
          db::get<fields_tag>(box), db::get<source_tag>(box));
    }

    return {std::move(box)};
  }
};

// Once the pre-smoothing is done, we skip the second smoothing step on the
// coarsest grid, i.e. at the "tip" of the V-cycle.
template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
struct SkipPostsmoothingAtBottom {
 private:
  using fields_tag = FieldsTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, Parallel::AlgorithmExecution,
                    size_t>
  apply(db::DataBox<DbTagsList>& box,
        const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
        const Parallel::GlobalCache<Metavariables>& /*cache*/,
        const ElementId<Dim>& /*element_id*/, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) noexcept {
    const bool is_coarsest_grid =
        not db::get<Tags::ParentId<Dim>>(box).has_value();

    // Record pre-smoothing result fields and residual
    if (db::get<Tags::OutputVolumeData<OptionsGroup>>(box)) {
      db::mutate<Tags::VolumeDataForOutput<OptionsGroup, FieldsTag>>(
          make_not_null(&box),
          [](const auto volume_data, const auto& result_fields,
             const auto& residuals) noexcept {
            volume_data->assign_subset(
                Variables<db::wrap_tags_in<Tags::PreSmoothingResult,
                                           typename fields_tag::tags_list>>(
                    result_fields));
            volume_data->assign_subset(
                Variables<db::wrap_tags_in<Tags::PreSmoothingResidual,
                                           typename fields_tag::tags_list>>(
                    residuals));
          },
          db::get<fields_tag>(box), db::get<residual_tag>(box));
    }

    // Skip the second smoothing step on the coarsest grid
    const size_t first_action_after_post_smoothing_index = tmpl::index_of<
        ActionList,
        SendCorrectionToFinerGrid<FieldsTag, OptionsGroup, SourceTag>>::value;
    const size_t this_action_index =
        tmpl::index_of<ActionList, SkipPostsmoothingAtBottom>::value;
    return {std::move(box), Parallel::AlgorithmExecution::Continue,
            is_coarsest_grid ? first_action_after_post_smoothing_index
                             : (this_action_index + 1)};
  }
};

template <typename FieldsTag>
struct CorrectionInboxTag
    : public Parallel::InboxInserters::Value<CorrectionInboxTag<FieldsTag>> {
  using temporal_id = size_t;
  using type = std::map<temporal_id, typename FieldsTag::type>;
};

// The next two actions communicate and project the coarse-grid correction, i.e.
// the solution of the post-smoother, to the finer grid. The post-smoother on
// finer grids runs after receiving this coarse-grid correction. Since the
// post-smoother is skipped on the coarsest level, it directly sends the
// solution of the pre-smoother to the finer grid, thus kicking off the
// "ascending" branch of the V-cycle.
template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
struct SendCorrectionToFinerGrid {
 private:
  using fields_tag = FieldsTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& child_ids = db::get<Tags::ChildIds<Dim>>(box);

    // Record post-smoothing result fields and residual
    if (db::get<Tags::OutputVolumeData<OptionsGroup>>(box)) {
      db::mutate<Tags::VolumeDataForOutput<OptionsGroup, FieldsTag>>(
          make_not_null(&box),
          [](const auto volume_data, const auto& result_fields,
             const auto& residuals) noexcept {
            volume_data->assign_subset(
                Variables<db::wrap_tags_in<Tags::PostSmoothingResult,
                                           typename fields_tag::tags_list>>(
                    result_fields));
            volume_data->assign_subset(
                Variables<db::wrap_tags_in<Tags::PostSmoothingResidual,
                                           typename fields_tag::tags_list>>(
                    residuals));
          },
          db::get<fields_tag>(box), db::get<residual_tag>(box));
    }

    if (child_ids.empty()) {
      return {std::move(box)};
    }

    const size_t iteration_id =
        db::get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    if (UNLIKELY(db::get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s %s(%zu): Send correction to children\n", element_id,
                       Options::name<OptionsGroup>(), iteration_id);
    }

    // Send a copy of the correction to all children
    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    for (const auto& child_id : child_ids) {
      auto coarse_grid_correction = db::get<fields_tag>(box);
      Parallel::receive_data<CorrectionInboxTag<FieldsTag>>(
          receiver_proxy[child_id], iteration_id,
          std::move(coarse_grid_correction));
    }
    return {std::move(box)};
  }
};

template <size_t Dim, typename FieldsTag, typename OptionsGroup,
          typename SourceTag>
struct ReceiveCorrectionFromCoarserGrid {
 private:
  using fields_tag = FieldsTag;
  using source_tag = SourceTag;

 public:
  using inbox_tags = tmpl::list<CorrectionInboxTag<FieldsTag>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, Parallel::AlgorithmExecution>
  apply(db::DataBox<DbTagsList>& box,
        tuples::TaggedTuple<InboxTags...>& inboxes,
        const Parallel::GlobalCache<Metavariables>& /*cache*/,
        const ElementId<Dim>& element_id, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) noexcept {
    const auto& parent_id = db::get<Tags::ParentId<Dim>>(box);
    // We should always have a `parent_id` at this point because we skip this
    // part of the algorithm on the coarsest grid with the
    // `SkipPostsmoothingAtBottom` action
    ASSERT(parent_id.has_value(),
           "Trying to receive data from parent but no parent is set on element "
               << element_id << ".");
    const size_t iteration_id =
        db::get<Convergence::Tags::IterationId<OptionsGroup>>(box);

    // Wait for data from coarser grid
    auto& inbox = tuples::get<CorrectionInboxTag<FieldsTag>>(inboxes);
    if (inbox.find(iteration_id) == inbox.end()) {
      return {std::move(box), Parallel::AlgorithmExecution::Retry};
    }
    auto parent_correction = std::move(inbox.extract(iteration_id).mapped());

    if (UNLIKELY(db::get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s %s(%zu): Prolongate correction from parent\n",
                       element_id, Options::name<OptionsGroup>(), iteration_id);
    }

    // Apply prolongation operator
    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const auto& parent_mesh = db::get<Tags::ParentMesh<Dim>>(box);
    ASSERT(
        parent_mesh.has_value(),
        "Should have a parent mesh, because a parent ID is set. This element: "
            << element_id << ", parent element: " << *parent_id);
    const auto child_size =
        domain::child_size(element_id.segment_ids(), parent_id->segment_ids());
    const auto prolongated_parent_correction =
        [&parent_correction, &parent_mesh, &mesh, &child_size]() noexcept {
          if (Spectral::needs_projection(*parent_mesh, mesh, child_size)) {
            const auto prolongation_operator =
                Spectral::projection_matrix_parent_to_child(*parent_mesh, mesh,
                                                            child_size);
            return apply_matrices(prolongation_operator, parent_correction,
                                  parent_mesh->extents());
          } else {
            return std::move(parent_correction);
          }
        }();

    // Add correction to the solution on this grid
    db::mutate<fields_tag>(
        make_not_null(&box),
        [&prolongated_parent_correction](const auto fields) noexcept {
          *fields += prolongated_parent_correction;
        });

    // Record post-smoothing initial fields and source
    if (db::get<Tags::OutputVolumeData<OptionsGroup>>(box)) {
      db::mutate<Tags::VolumeDataForOutput<OptionsGroup, FieldsTag>>(
          make_not_null(&box),
          [](const auto volume_data, const auto& initial_fields,
             const auto& source) noexcept {
            volume_data->assign_subset(
                Variables<db::wrap_tags_in<Tags::PostSmoothingInitial,
                                           typename fields_tag::tags_list>>(
                    initial_fields));
            volume_data->assign_subset(
                Variables<db::wrap_tags_in<Tags::PostSmoothingSource,
                                           typename fields_tag::tags_list>>(
                    source));
          },
          db::get<fields_tag>(box), db::get<source_tag>(box));
    }

    return {std::move(box), Parallel::AlgorithmExecution::Continue};
  }
};

}  // namespace LinearSolver::multigrid::detail
