// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Functionality to build the explicit matrix representation of the linear
/// operator column-by-column. This is useful for debugging and analysis only,
/// not to actually solve the elliptic problem (that should happen iteratively).

#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "IO/Observer/Actions/RegisterWithObservers.hpp"
#include "IO/Observer/GetSectionObservationKey.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GetSection.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/Section.hpp"
#include "Parallel/Tags/Section.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace LinearSolver {
namespace OptionTags {

struct BuildMatrixOptionsGroup {
  static std::string name() { return "BuildMatrix"; }
  static constexpr Options::String help = {
      "Options for building the explicit matrix representation of the linear "
      "operator. This is done by applying the linear operator to unit "
      "vectors and is useful for debugging and analysis only, not to actually "
      "solve the elliptic problem (that should happen iteratively)."};
};

struct MatrixSubfileName {
  using type = std::string;
  using group = BuildMatrixOptionsGroup;
  static constexpr Options::String help = {
      "Subfile name in the volume data H5 files where the matrix will be "
      "stored. Each observation in the subfile is a column of the matrix. The "
      "row index is the order of elements defined by the ElementId in the "
      "volume data, by the order of tensor components encoded in the name of "
      "the components, and by the contiguous ordering of grid points for each "
      "component."};
};

}  // namespace OptionTags

namespace Tags {

/// Subfile name in the volume data H5 files where the matrix will be stored.
struct MatrixSubfileName : db::SimpleTag {
  using type = std::string;
  using option_tags = tmpl::list<OptionTags::MatrixSubfileName>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& value) { return value; }
};

/// Size of the matrix: number of grid points times number of variables
struct TotalNumPoints : db::SimpleTag {
  using type = size_t;
};

/// Index of the first point in this element
struct LocalFirstIndex : db::SimpleTag {
  using type = size_t;
};

}  // namespace Tags

namespace Actions {

namespace detail {

/// \brief The total number of grid points (size of the matrix) and the index of
/// the first grid point in this element (the offset into the matrix
/// corresponding to this element).
template <size_t Dim>
std::pair<size_t, size_t> total_num_points_and_local_first_index(
    const ElementId<Dim>& element_id,
    const std::map<ElementId<Dim>, size_t>& num_points_per_element,
    size_t num_vars);

/// \brief The index of the '1' of the unit vector in this element, or
/// std::nullopt if the '1' is in another element.
///
/// \param iteration_id enumerates all grid points
/// \param local_first_index the index of the first grid point in this element
/// \param local_num_points the number of grid points in this element
std::optional<size_t> local_unit_vector_index(size_t iteration_id,
                                              size_t local_first_index,
                                              size_t local_num_points);

/// \brief Observe matrix column as volume data.
///
/// This is the format of the volume data:
///
/// - Columns of the matrix are enumerated by the observation ID
/// - Rows are enumerated by the order of elements defined by the ElementId
///   in the volume data, by the order of tensor components encoded in the
///   name of the components, and by the contiguous ordering of grid points
///   for each component.
/// - We also observe coordinates so the data can be plotted as volume data.
///
/// The matrix can be reconstructed from this data in Python with the function
/// `spectre.Elliptic.ReadH5.read_matrix`.
template <typename ParallelComponent, typename OperatorAppliedToOperandTags,
          size_t Dim, typename CoordsFrame, typename Metavariables>
void observe_matrix_column(
    const size_t column,
    const Variables<OperatorAppliedToOperandTags>& operator_applied_to_operand,
    const ElementId<Dim>& element_id, const Mesh<Dim>& mesh,
    const tnsr::I<DataVector, Dim, CoordsFrame>& coords,
    const std::string& subfile_name, const std::string& section_observation_key,
    Parallel::GlobalCache<Metavariables>& cache) {
  std::vector<TensorComponent> observe_components{};
  observe_components.reserve(
      Dim +
      Variables<
          OperatorAppliedToOperandTags>::number_of_independent_components);
  for (size_t i = 0; i < Dim; ++i) {
    observe_components.emplace_back(
        get_output(CoordsFrame{}) + "Coordinates" + coords.component_suffix(i),
        coords.get(i));
  }
  size_t component_i = 0;
  tmpl::for_each<OperatorAppliedToOperandTags>([&operator_applied_to_operand,
                                                &observe_components,
                                                &component_i](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    const auto& tensor = get<tag>(operator_applied_to_operand);
    for (size_t i = 0; i < tensor.size(); ++i) {
      observe_components.emplace_back("Variable_" + std::to_string(component_i),
                                      tensor[i]);
      ++component_i;
    }
  });
  auto& local_observer = *Parallel::local_branch(
      Parallel::get_parallel_component<observers::Observer<Metavariables>>(
          cache));
  Parallel::simple_action<observers::Actions::ContributeVolumeData>(
      local_observer,
      observers::ObservationId(static_cast<double>(column),
                               subfile_name + section_observation_key),
      "/" + subfile_name,
      Parallel::make_array_component_id<ParallelComponent>(element_id),
      ElementVolumeData{element_id, std::move(observe_components), mesh});
}

/// \brief Register with the volume observer
template <typename ArraySectionIdTag>
struct RegisterWithVolumeObserver {
  template <typename ParallelComponent, typename DbTagsList,
            typename ArrayIndex>
  static std::pair<observers::TypeOfObservation, observers::ObservationKey>
  register_info(const db::DataBox<DbTagsList>& box,
                const ArrayIndex& /*array_index*/) {
    // Get the observation key, or "Unused" if the element does not belong
    // to a section with this tag. In the latter case, no observations will
    // ever be contributed.
    const std::optional<std::string> section_observation_key =
        observers::get_section_observation_key<ArraySectionIdTag>(box);
    ASSERT(section_observation_key != "Unused",
           "The identifier 'Unused' is reserved to indicate that no "
           "observations with this key will be contributed. Use a different "
           "key, or change the identifier 'Unused' to something else.");
    return {
        observers::TypeOfObservation::Volume,
        observers::ObservationKey(get<Tags::MatrixSubfileName>(box) +
                                  section_observation_key.value_or("Unused"))};
  }
};

}  // namespace detail

/// \cond
template <typename IterationIdTag, typename OperandTag,
          typename OperatorAppliedToOperandTag, typename CoordsTag,
          typename ArraySectionIdTag>
struct PrepareBuildMatrix;
template <typename IterationIdTag, typename OperandTag,
          typename OperatorAppliedToOperandTag, typename CoordsTag,
          typename ArraySectionIdTag>
struct StoreMatrixColumn;
/// \endcond

/// Dispatch global reduction to get the size of the matrix
template <typename IterationIdTag, typename OperandTag,
          typename OperatorAppliedToOperandTag, typename CoordsTag,
          typename ArraySectionIdTag>
struct CollectTotalNumPoints {
  using simple_tags = tmpl::list<Tags::TotalNumPoints, Tags::LocalFirstIndex,
                                 IterationIdTag, OperandTag>;
  using compute_tags = tmpl::list<>;
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<OptionTags::BuildMatrixOptionsGroup>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // Skip everything on elements that are not part of the section
    if constexpr (not std::is_same_v<ArraySectionIdTag, void>) {
      if (not db::get<Parallel::Tags::Section<ParallelComponent,
                                              ArraySectionIdTag>>(box)
                  .has_value()) {
        constexpr size_t last_action_index = tmpl::index_of<
            ActionList, StoreMatrixColumn<IterationIdTag, OperandTag,
                                          OperatorAppliedToOperandTag,
                                          CoordsTag, ArraySectionIdTag>>::value;
        return {Parallel::AlgorithmExecution::Continue, last_action_index + 1};
      }
    }
    db::mutate<Tags::TotalNumPoints, OperandTag>(
        [](const auto total_num_points, const auto operand,
           const size_t num_points) {
          // Set num points to zero until we know the total number of points
          *total_num_points = 0;
          // Size operand and fill with zeros
          operand->initialize(num_points, 0.);
        },
        make_not_null(&box),
        get<domain::Tags::Mesh<Dim>>(box).number_of_grid_points());
    // Collect the total number of grid points
    auto& section = Parallel::get_section<ParallelComponent, ArraySectionIdTag>(
        make_not_null(&box));
    Parallel::contribute_to_reduction<PrepareBuildMatrix<
        IterationIdTag, OperandTag, OperatorAppliedToOperandTag, CoordsTag,
        ArraySectionIdTag>>(
        Parallel::ReductionData<Parallel::ReductionDatum<
            std::map<ElementId<Dim>, size_t>, funcl::Merge<>>>{
            std::map<ElementId<Dim>, size_t>{
                std::make_pair(array_index, get<OperandTag>(box).size())}},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<ParallelComponent>(cache),
        make_not_null(&section));
    // Pause the algorithm for now. The reduction will be broadcast to the next
    // action which is responsible for restarting the algorithm.
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

/// Receive the reduction and initialize the algorithm
template <typename IterationIdTag, typename OperandTag,
          typename OperatorAppliedToOperandTag, typename CoordsTag,
          typename ArraySectionIdTag>
struct PrepareBuildMatrix {
 public:
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, size_t Dim>
  static void apply(
      db::DataBox<DbTagsList>& box, Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id,
      const std::map<ElementId<Dim>, size_t>& num_points_per_element) {
    const auto [total_num_points, local_first_index] =
        detail::total_num_points_and_local_first_index(
            element_id, num_points_per_element,
            OperandTag::type::number_of_independent_components);
    if (get<logging::Tags::Verbosity<OptionTags::BuildMatrixOptionsGroup>>(
            box) >= Verbosity::Quiet and
        local_first_index == 0) {
      Parallel::printf(
          "Building explicit matrix representation of size %zu x %zu.\n",
          total_num_points, total_num_points);
    }
    db::mutate<Tags::TotalNumPoints, Tags::LocalFirstIndex, IterationIdTag>(
        [captured_total_num_points = total_num_points,
         captured_local_first_index = local_first_index](
            const auto stored_total_num_points,
            const auto stored_local_first_index, const auto iteration_id) {
          *stored_total_num_points = captured_total_num_points;
          *stored_local_first_index = captured_local_first_index;
          *iteration_id = 0;
        },
        make_not_null(&box));
    // Proceed with algorithm
    Parallel::get_parallel_component<ParallelComponent>(cache)[element_id]
        .perform_algorithm(true);
  }
};

/// Set the operand to a unit vector with a '1' at the current grid point.
/// Applying the operator to this operand gives a column of the matrix. We jump
/// back to this until we have iterated over all grid points.
template <typename IterationIdTag, typename OperandTag,
          typename OperatorAppliedToOperandTag, typename CoordsTag,
          typename ArraySectionIdTag>
struct SetUnitVector {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const std::optional<size_t> local_unit_vector_index =
        detail::local_unit_vector_index(get<IterationIdTag>(box),
                                        get<Tags::LocalFirstIndex>(box),
                                        get<OperandTag>(box).size());
    if (local_unit_vector_index.has_value()) {
      db::mutate<OperandTag>(
          [&local_unit_vector_index](const auto operand) {
            operand->data()[*local_unit_vector_index] = 1.;
          },
          make_not_null(&box));
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

// --- Linear operator will be applied to the unit vector here ---

/// Write result out to disk, reset operand back to zero, and keep iterating
template <typename IterationIdTag, typename OperandTag,
          typename OperatorAppliedToOperandTag, typename CoordsTag,
          typename ArraySectionIdTag>
struct StoreMatrixColumn {
  using const_global_cache_tags = tmpl::list<Tags::MatrixSubfileName>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const size_t iteration_id = get<IterationIdTag>(box);
    const size_t local_first_index = get<Tags::LocalFirstIndex>(box);
    if (get<logging::Tags::Verbosity<OptionTags::BuildMatrixOptionsGroup>>(
            box) >= Verbosity::Verbose and
        local_first_index == 0) {
      Parallel::printf("Column %zu / %zu done.\n", iteration_id + 1,
                       get<Tags::TotalNumPoints>(box));
    }
    // This is the result of applying the linear operator to the unit vector. It
    // is a column of the operator matrix.
    const auto& operator_applied_to_operand =
        get<OperatorAppliedToOperandTag>(box);
    // Write it out to disk
    detail::observe_matrix_column<ParallelComponent>(
        iteration_id, operator_applied_to_operand, element_id,
        get<domain::Tags::Mesh<Dim>>(box), get<CoordsTag>(box),
        get<Tags::MatrixSubfileName>(box),
        *observers::get_section_observation_key<ArraySectionIdTag>(box), cache);
    // Reset operand to zero
    const std::optional<size_t> local_unit_vector_index =
        detail::local_unit_vector_index(iteration_id, local_first_index,
                                        get<OperandTag>(box).size());
    if (local_unit_vector_index.has_value()) {
      db::mutate<OperandTag>(
          [&local_unit_vector_index](const auto operand) {
            operand->data()[*local_unit_vector_index] = 0.;
          },
          make_not_null(&box));
    }
    // Keep iterating
    db::mutate<IterationIdTag>(
        [](const auto local_iteration_id) { ++(*local_iteration_id); },
        make_not_null(&box));
    if (get<IterationIdTag>(box) < get<Tags::TotalNumPoints>(box)) {
      constexpr size_t set_unit_vector_index = tmpl::index_of<
          ActionList,
          SetUnitVector<IterationIdTag, OperandTag, OperatorAppliedToOperandTag,
                        CoordsTag, ArraySectionIdTag>>::value;
      return {Parallel::AlgorithmExecution::Continue, set_unit_vector_index};
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <typename IterationIdTag, typename OperandTag,
          typename OperatorAppliedToOperandTag, typename CoordsTag,
          typename ArraySectionIdTag>
struct ProjectBuildMatrix : tt::ConformsTo<::amr::protocols::Projector> {
  using return_tags = tmpl::list<Tags::TotalNumPoints, Tags::LocalFirstIndex,
                                 IterationIdTag, OperandTag>;
  using argument_tags = tmpl::list<>;

  template <typename... AmrData>
  static void apply(const gsl::not_null<size_t*> /*unused*/,
                    const AmrData&... /*amr_data*/) {
    // Nothing to do. Everything gets initialized at the start of the algorithm.
  }
};

/*!
 * \brief Build the explicit matrix representation of the linear operator.
 *
 * This is useful for debugging and analysis only, not to actually solve the
 * elliptic problem (that should happen iteratively).
 *
 * Add the `actions` to the action list to build the matrix. The
 * `ApplyOperatorActions` template parameter are the actions that apply the
 * linear operator to the `OperandTag`. Also add the `amr_projectors` to the
 * list of AMR projectors and the `register_actions`
 *
 * \tparam IterationIdTag Used to keep track of the iteration over all matrix
 * columns. Should be the same that's used to identify iterations in the
 * `ApplyOperatorActions`.
 * \tparam OperandTag Will be set to unit vectors, with the '1' moving through
 * all points over the course of the iteration.
 * \tparam OperatorAppliedToOperandTag Where the `ApplyOperatorActions` store
 * the result of applying the linear operator to the `OperandTag`.
 * \tparam CoordsTag The tag of the coordinates observed alongside the matrix
 * for volume data visualization.
 * \tparam ArraySectionIdTag Can identify a subset of elements that this
 * algorithm should run over, e.g. in a multigrid setting.
 */
template <typename IterationIdTag, typename OperandTag,
          typename OperatorAppliedToOperandTag, typename CoordsTag,
          typename ArraySectionIdTag = void>
struct BuildMatrix {
  template <typename ApplyOperatorActions>
  using actions = tmpl::list<
      CollectTotalNumPoints<IterationIdTag, OperandTag,
                            OperatorAppliedToOperandTag, CoordsTag,
                            ArraySectionIdTag>,
      // PrepareBuildMatrix is called on reduction broadcast
      SetUnitVector<IterationIdTag, OperandTag, OperatorAppliedToOperandTag,
                    CoordsTag, ArraySectionIdTag>,
      ApplyOperatorActions,
      StoreMatrixColumn<IterationIdTag, OperandTag, OperatorAppliedToOperandTag,
                        CoordsTag, ArraySectionIdTag>>;

  using amr_projectors =
      tmpl::list<ProjectBuildMatrix<IterationIdTag, OperandTag,
                                    OperatorAppliedToOperandTag, CoordsTag,
                                    ArraySectionIdTag>>;

  /// Add to the register phase to enable observations
  using register_actions = tmpl::list<observers::Actions::RegisterWithObservers<
      detail::RegisterWithVolumeObserver<ArraySectionIdTag>>>;
};

}  // namespace Actions
}  // namespace LinearSolver
