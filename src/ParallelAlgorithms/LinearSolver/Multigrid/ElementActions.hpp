// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <unordered_set>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/Structure/ChildSize.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Observer/Tags.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "ParallelAlgorithms/Actions/Goto.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Actions/RestrictFields.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Hierarchy.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace LinearSolver::multigrid::detail {

/// \cond
template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
struct SendCorrectionToFinerGrid;
template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
struct SkipPostSmoothingAtBottom;
/// \endcond

struct PostSmoothingBeginLabel {};

template <size_t Dim, typename FieldsTag, typename OptionsGroup,
          typename SourceTag>
struct InitializeElement {
  using simple_tags_from_options =
      tmpl::list<Tags::ChildrenRefinementLevels<Dim>,
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
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
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
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
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
  using const_global_cache_tags = tmpl::list<
      LinearSolver::multigrid::Tags::EnablePreSmoothing<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const size_t iteration_id =
        db::get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    if (UNLIKELY(db::get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s %s(%zu): Prepare pre-smoothing\n", element_id,
                       pretty_type::name<OptionsGroup>(), iteration_id);
    }

    // On coarser grids the smoother solves for a correction to the finer-grid
    // fields, so we set its initial guess to zero. On the finest grid we smooth
    // the fields directly, so there's nothing to prepare.
    if (element_id.grid_index() > 0) {
      db::mutate<fields_tag, operator_applied_to_fields_tag>(
          [](const auto fields, const auto operator_applied_to_fields,
             const auto& source) {
            *fields = make_with_value<typename fields_tag::type>(source, 0.);
            // We can set the linear operator applied to the initial fields to
            // zero as well, since it's linear
            *operator_applied_to_fields =
                make_with_value<typename operator_applied_to_fields_tag::type>(
                    source, 0.);
          },
          make_not_null(&box), db::get<source_tag>(box));
    }

    // Record pre-smoothing initial fields and source
    if (db::get<Tags::OutputVolumeData<OptionsGroup>>(box)) {
      db::mutate<Tags::VolumeDataForOutput<OptionsGroup, FieldsTag>>(
          [](const auto volume_data, const auto& initial_fields,
             const auto& source) {
            volume_data->assign_subset(
                Variables<db::wrap_tags_in<Tags::PreSmoothingInitial,
                                           typename fields_tag::tags_list>>(
                    initial_fields));
            volume_data->assign_subset(
                Variables<db::wrap_tags_in<Tags::PreSmoothingSource,
                                           typename fields_tag::tags_list>>(
                    source));
          },
          make_not_null(&box), db::get<fields_tag>(box),
          db::get<source_tag>(box));
    }

    // Skip pre-smoothing, if requested
    const size_t first_action_after_pre_smoothing_index = tmpl::index_of<
        ActionList,
        SkipPostSmoothingAtBottom<FieldsTag, OptionsGroup, SourceTag>>::value;
    const size_t this_action_index =
        tmpl::index_of<ActionList, PreparePreSmoothing>::value;
    return {
        Parallel::AlgorithmExecution::Continue,
        db::get<
            LinearSolver::multigrid::Tags::EnablePreSmoothing<OptionsGroup>>(
            box)
            ? (this_action_index + 1)
            : first_action_after_pre_smoothing_index};
  }
};

// Once the pre-smoothing is done, we skip the second smoothing step on the
// coarsest grid, i.e. at the "tip" of the V-cycle.
template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
struct SkipPostSmoothingAtBottom {
 private:
  using fields_tag = FieldsTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;

 public:
  using const_global_cache_tags = tmpl::list<
      LinearSolver::multigrid::Tags::EnablePostSmoothingAtBottom<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*element_id*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const bool is_coarsest_grid =
        not db::get<Tags::ParentId<Dim>>(box).has_value();

    // Record pre-smoothing result fields and residual
    if (db::get<Tags::OutputVolumeData<OptionsGroup>>(box)) {
      db::mutate<Tags::VolumeDataForOutput<OptionsGroup, FieldsTag>>(
          [](const auto volume_data, const auto& result_fields,
             const auto& residuals) {
            volume_data->assign_subset(
                Variables<db::wrap_tags_in<Tags::PreSmoothingResult,
                                           typename fields_tag::tags_list>>(
                    result_fields));
            volume_data->assign_subset(
                Variables<db::wrap_tags_in<Tags::PreSmoothingResidual,
                                           typename fields_tag::tags_list>>(
                    residuals));
          },
          make_not_null(&box), db::get<fields_tag>(box),
          db::get<residual_tag>(box));
    }

    // Skip post-smoothing on the coarsest grid, if requested
    const size_t first_action_after_post_smoothing_index = tmpl::index_of<
        ActionList,
        SendCorrectionToFinerGrid<FieldsTag, OptionsGroup, SourceTag>>::value;
    const size_t post_smoothing_begin_index = tmpl::index_of<
        ActionList,
        ::Actions::Label<PostSmoothingBeginLabel>>::value + 1;
    const size_t this_action_index =
        tmpl::index_of<ActionList, SkipPostSmoothingAtBottom>::value;
    return {Parallel::AlgorithmExecution::Continue,
            is_coarsest_grid
                ? (db::get<LinearSolver::multigrid::Tags::
                               EnablePostSmoothingAtBottom<OptionsGroup>>(box)
                       ? post_smoothing_begin_index
                       : first_action_after_post_smoothing_index)
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
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& child_ids = db::get<Tags::ChildIds<Dim>>(box);

    // Record post-smoothing result fields and residual
    if (db::get<Tags::OutputVolumeData<OptionsGroup>>(box)) {
      db::mutate<Tags::VolumeDataForOutput<OptionsGroup, FieldsTag>>(
          [](const auto volume_data, const auto& result_fields,
             const auto& residuals) {
            volume_data->assign_subset(
                Variables<db::wrap_tags_in<Tags::PostSmoothingResult,
                                           typename fields_tag::tags_list>>(
                    result_fields));
            volume_data->assign_subset(
                Variables<db::wrap_tags_in<Tags::PostSmoothingResidual,
                                           typename fields_tag::tags_list>>(
                    residuals));
          },
          make_not_null(&box), db::get<fields_tag>(box),
          db::get<residual_tag>(box));
    }

    if (child_ids.empty()) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    const size_t iteration_id =
        db::get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    if (UNLIKELY(db::get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s %s(%zu): Send correction to children\n", element_id,
                       pretty_type::name<OptionsGroup>(), iteration_id);
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
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
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
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& parent_id = db::get<Tags::ParentId<Dim>>(box);
    // We should always have a `parent_id` at this point because we skip this
    // part of the algorithm on the coarsest grid with the
    // `SkipPostSmoothingAtBottom` action
    ASSERT(parent_id.has_value(),
           "Trying to receive data from parent but no parent is set on element "
               << element_id << ".");
    const size_t iteration_id =
        db::get<Convergence::Tags::IterationId<OptionsGroup>>(box);

    // Wait for data from coarser grid
    auto& inbox = tuples::get<CorrectionInboxTag<FieldsTag>>(inboxes);
    if (inbox.find(iteration_id) == inbox.end()) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }
    auto parent_correction = std::move(inbox.extract(iteration_id).mapped());

    if (UNLIKELY(db::get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s %s(%zu): Prolongate correction from parent\n",
                       element_id, pretty_type::name<OptionsGroup>(),
                       iteration_id);
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
        [&parent_correction, &parent_mesh, &mesh, &child_size]() {
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
        [&prolongated_parent_correction](const auto fields) {
          *fields += prolongated_parent_correction;
        },
        make_not_null(&box));

    // Record post-smoothing initial fields and source
    if (db::get<Tags::OutputVolumeData<OptionsGroup>>(box)) {
      db::mutate<Tags::VolumeDataForOutput<OptionsGroup, FieldsTag>>(
          [](const auto volume_data, const auto& initial_fields,
             const auto& source) {
            volume_data->assign_subset(
                Variables<db::wrap_tags_in<Tags::PostSmoothingInitial,
                                           typename fields_tag::tags_list>>(
                    initial_fields));
            volume_data->assign_subset(
                Variables<db::wrap_tags_in<Tags::PostSmoothingSource,
                                           typename fields_tag::tags_list>>(
                    source));
          },
          make_not_null(&box), db::get<fields_tag>(box),
          db::get<source_tag>(box));
    }

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace LinearSolver::multigrid::detail
