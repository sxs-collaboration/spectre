// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Functionality to build the explicit matrix representation of the linear
/// operator column-by-column. This is useful for debugging and analysis only,
/// not to actually solve the elliptic problem (that should happen iteratively).

#pragma once

#include <blaze/math/Submatrix.h>
#include <cstddef>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "DataStructures/CompressedMatrix.hpp"
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
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GetSection.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/Section.hpp"
#include "Parallel/Tags/Section.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
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
  using type = Options::Auto<std::string, Options::AutoLabel::None>;
  using group = BuildMatrixOptionsGroup;
  static constexpr Options::String help = {
      "Subfile name in the volume data H5 files where the matrix will be "
      "stored. Each observation in the subfile is a column of the matrix. The "
      "row index is the order of elements defined by the ElementId in the "
      "volume data, by the order of tensor components encoded in the name of "
      "the components, and by the contiguous ordering of grid points for each "
      "component."};
};

struct EnableDirectSolve {
  using type = bool;
  using group = BuildMatrixOptionsGroup;
  static constexpr Options::String help = {
      "Directly invert the assembled matrix and solve the linear problem. "
      "This can be unfeasible if the linear problem is too big."};
};

}  // namespace OptionTags

namespace Tags {

/// Subfile name in the volume data H5 files where the matrix will be stored.
struct MatrixSubfileName : db::SimpleTag {
  using type = std::optional<std::string>;
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

/// Matrix representation of the linear operator. Stored as sparse matrix. The
/// full matrix has size `total_num_points` by `total_num_points`. Each element
/// only stores the rows corresponding to its grid points (starting at
/// `local_first_index`), but all columns.
template <typename ValueType>
struct Matrix : db::SimpleTag {
  using type = blaze::CompressedMatrix<ValueType>;
};

struct EnableDirectSolve : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<OptionTags::EnableDirectSolve>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& value) { return value; }
};

}  // namespace Tags

namespace Actions {

namespace detail {

template <typename FieldsTag, typename FixedSourcesTag, typename OperandTag,
          typename OperatorAppliedToOperandTag, typename CoordsTag,
          typename ArraySectionIdTag>
struct BuildMatrixMetavars {
  using iteration_id_tag = Convergence::Tags::IterationId<
      LinearSolver::OptionTags::BuildMatrixOptionsGroup>;
  using fields_tag = FieldsTag;
  using fixed_sources_tag = FixedSourcesTag;
  using operand_tag = OperandTag;
  using operator_applied_to_operand_tag = OperatorAppliedToOperandTag;
  using coords_tag = CoordsTag;
  using array_section_id_tag = ArraySectionIdTag;

  using value_type = typename OperatorAppliedToOperandTag::type::value_type;
};

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
    using TensorType = std::decay_t<decltype(tensor)>;
    using VectorType = typename TensorType::type;
    using ValueType = typename VectorType::value_type;
    for (size_t i = 0; i < tensor.size(); ++i) {
      if constexpr (std::is_same_v<ValueType, std::complex<double>>) {
        observe_components.emplace_back(
            "Re(Variable_" + std::to_string(component_i) + ")",
            real(tensor[i]));
        observe_components.emplace_back(
            "Im(Variable_" + std::to_string(component_i) + ")",
            imag(tensor[i]));
      } else {
        observe_components.emplace_back(
            "Variable_" + std::to_string(component_i), tensor[i]);
      }
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
    const auto& subfile_name = get<Tags::MatrixSubfileName>(box);
    return {
        observers::TypeOfObservation::Volume,
        observers::ObservationKey(subfile_name.value_or("Unused") +
                                  section_observation_key.value_or("Unused"))};
  }
};

}  // namespace detail

/// \cond
template <typename BuildMatrixMetavars>
struct PrepareBuildMatrix;
template <typename BuildMatrixMetavars>
struct AssembleFullMatrix;
/// \endcond

/// Dispatch global reduction to get the size of the matrix
template <typename BuildMatrixMetavars>
struct CollectTotalNumPoints {
 private:
  using IterationIdTag = typename BuildMatrixMetavars::iteration_id_tag;
  using OperandTag = typename BuildMatrixMetavars::operand_tag;
  using ArraySectionIdTag = typename BuildMatrixMetavars::array_section_id_tag;
  using value_type = typename BuildMatrixMetavars::value_type;

 public:
  using simple_tags =
      tmpl::list<Tags::TotalNumPoints, Tags::LocalFirstIndex, IterationIdTag,
                 OperandTag, Tags::Matrix<value_type>>;
  using compute_tags = tmpl::list<>;
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<OptionTags::BuildMatrixOptionsGroup>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // Skip everything on elements that are not part of the section
    if constexpr (not std::is_same_v<ArraySectionIdTag, void>) {
      if (not db::get<Parallel::Tags::Section<ParallelComponent,
                                              ArraySectionIdTag>>(box)
                  .has_value()) {
        constexpr size_t last_action_index =
            tmpl::index_of<ActionList,
                           AssembleFullMatrix<BuildMatrixMetavars>>::value;
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
    Parallel::contribute_to_reduction<PrepareBuildMatrix<BuildMatrixMetavars>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
            Parallel::ReductionDatum<std::map<ElementId<Dim>, size_t>,
                                     funcl::Merge<>>>{
            element_id.grid_index(),
            std::map<ElementId<Dim>, size_t>{
                std::make_pair(element_id, get<OperandTag>(box).size())}},
        Parallel::get_parallel_component<ParallelComponent>(cache)[element_id],
        Parallel::get_parallel_component<ParallelComponent>(cache),
        make_not_null(&section));
    // Pause the algorithm for now. The reduction will be broadcast to the next
    // action which is responsible for restarting the algorithm.
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

/// Receive the reduction and initialize the algorithm
template <typename BuildMatrixMetavars>
struct PrepareBuildMatrix {
 private:
  using IterationIdTag = typename BuildMatrixMetavars::iteration_id_tag;
  using OperandTag = typename BuildMatrixMetavars::operand_tag;
  using value_type = typename BuildMatrixMetavars::value_type;

 public:
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, size_t Dim>
  static void apply(
      db::DataBox<DbTagsList>& box, Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id, const size_t grid_index,
      const std::map<ElementId<Dim>, size_t>& num_points_per_element) {
    if (grid_index != element_id.grid_index()) {
      return;
    }
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
    db::mutate<Tags::TotalNumPoints, Tags::LocalFirstIndex, IterationIdTag,
               Tags::Matrix<value_type>>(
        [captured_total_num_points = total_num_points,
         captured_local_first_index = local_first_index,
         local_num_points = num_points_per_element.at(element_id)](
            const auto stored_total_num_points,
            const auto stored_local_first_index, const auto iteration_id,
            const gsl::not_null<blaze::CompressedMatrix<value_type>*> matrix) {
          *stored_total_num_points = captured_total_num_points;
          *stored_local_first_index = captured_local_first_index;
          *iteration_id = 0;
          matrix->clear();
          matrix->resize(local_num_points, captured_total_num_points);
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
template <typename BuildMatrixMetavars>
struct SetUnitVector {
 private:
  using IterationIdTag = typename BuildMatrixMetavars::iteration_id_tag;
  using OperandTag = typename BuildMatrixMetavars::operand_tag;

 public:
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
template <typename BuildMatrixMetavars>
struct StoreMatrixColumn {
 private:
  using IterationIdTag = typename BuildMatrixMetavars::iteration_id_tag;
  using OperandTag = typename BuildMatrixMetavars::operand_tag;
  using OperatorAppliedToOperandTag =
      typename BuildMatrixMetavars::operator_applied_to_operand_tag;
  using CoordsTag = typename BuildMatrixMetavars::coords_tag;
  using ArraySectionIdTag = typename BuildMatrixMetavars::array_section_id_tag;
  using value_type = typename BuildMatrixMetavars::value_type;

 public:
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
        local_first_index == 0 and (iteration_id + 1) % 100 == 0) {
      Parallel::printf("Column %zu / %zu done.\n", iteration_id + 1,
                       get<Tags::TotalNumPoints>(box));
    }
    // This is the result of applying the linear operator to the unit vector. It
    // is a column of the operator matrix.
    const auto& operator_applied_to_operand =
        get<OperatorAppliedToOperandTag>(box);
    // Store in sparse matrix
    db::mutate<Tags::Matrix<value_type>>(
        [&operator_applied_to_operand, &iteration_id](
            const gsl::not_null<blaze::CompressedMatrix<value_type>*> matrix) {
          for (size_t i = 0; i < operator_applied_to_operand.size(); ++i) {
            const auto value = operator_applied_to_operand.data()[i];
            if (not equal_within_roundoff(value, 0.)) {
              matrix->insert(i, iteration_id, value);
            }
          }
        },
        make_not_null(&box));
    // Write it out to disk
    if (const auto& subfile_name = get<Tags::MatrixSubfileName>(box);
        subfile_name.has_value()) {
      detail::observe_matrix_column<ParallelComponent>(
          iteration_id, operator_applied_to_operand, element_id,
          get<domain::Tags::Mesh<Dim>>(box), get<CoordsTag>(box), *subfile_name,
          *observers::get_section_observation_key<ArraySectionIdTag>(box),
          cache);
    }
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
      constexpr size_t set_unit_vector_index =
          tmpl::index_of<ActionList, SetUnitVector<BuildMatrixMetavars>>::value;
      return {Parallel::AlgorithmExecution::Continue, set_unit_vector_index};
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <typename Metavariables, typename BuildMatrixMetavars>
struct BuildMatrixSingleton {
  using chare_type = Parallel::Algorithms::Singleton;
  static constexpr bool checkpoint_data = true;
  using const_global_cache_tags = tmpl::list<>;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<BuildMatrixSingleton>(local_cache)
        .start_phase(next_phase);
  }
};

template <typename ElementArrayComponent, typename BuildMatrixMetavars>
struct InvertMatrix;

template <typename BuildMatrixMetavars>
struct AssembleFullMatrix {
 private:
  using value_type = typename BuildMatrixMetavars::value_type;
  using ArraySectionIdTag = typename BuildMatrixMetavars::array_section_id_tag;
  using FixedSourcesTag = typename BuildMatrixMetavars::fixed_sources_tag;
  using ReductionType = std::pair<blaze::CompressedMatrix<value_type>,
                                  typename FixedSourcesTag::type>;

 public:
  using const_global_cache_tags = tmpl::list<Tags::EnableDirectSolve>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    if (not db::get<Tags::EnableDirectSolve>(box)) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
    auto& section = Parallel::get_section<ParallelComponent, ArraySectionIdTag>(
        make_not_null(&box));
    Parallel::contribute_to_reduction<
        InvertMatrix<ParallelComponent, BuildMatrixMetavars>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
            Parallel::ReductionDatum<std::map<ElementId<Dim>, ReductionType>,
                                     funcl::Merge<>>>{
            element_id.grid_index(),
            std::map<ElementId<Dim>, ReductionType>{std::make_pair(
                element_id, ReductionType{get<Tags::Matrix<value_type>>(box),
                                          get<FixedSourcesTag>(box)})}},
        Parallel::get_parallel_component<ParallelComponent>(cache)[element_id],
        Parallel::get_parallel_component<
            BuildMatrixSingleton<Metavariables, BuildMatrixMetavars>>(cache),
        make_not_null(&section));
    // Pause the algorithm for now. The reduction will be received by the next
    // action which is responsible for restarting the algorithm.
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

template <typename BuildMatrixMetavars>
struct StoreSolution;

template <typename ElementArrayComponent, typename BuildMatrixMetavars>
struct InvertMatrix {
 private:
  using value_type = typename BuildMatrixMetavars::value_type;
  using FixedSourcesTag = typename BuildMatrixMetavars::fixed_sources_tag;
  using ReductionType = std::pair<blaze::CompressedMatrix<value_type>,
                                  typename FixedSourcesTag::type>;

 public:
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex, size_t Dim>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, const size_t grid_index,
                    const std::map<ElementId<Dim>, ReductionType>&
                        matrix_and_sources_slices) {
    const size_t total_num_points =
        matrix_and_sources_slices.begin()->second.first.columns();
    blaze::DynamicMatrix<value_type> matrix(total_num_points, total_num_points);
    blaze::DynamicVector<value_type> source(total_num_points);
    // Assemble the full matrix and source
    size_t i = 0;
    for (const auto& [element_id, slice] : matrix_and_sources_slices) {
      const auto& [matrix_slice, source_slice] = slice;
      ASSERT(matrix_slice.rows() == source_slice.size(),
             "Matrix and source slice have different sizes.");
      const size_t slice_length = matrix_slice.rows();
      blaze::submatrix(matrix, i, 0, slice_length, matrix_slice.columns()) =
          matrix_slice;
      std::copy_n(source_slice.data(), slice_length, source.data() + i);
      i += slice_length;
    }
    ASSERT(i == total_num_points, "The matrix is not fully assembled. Expected "
                                      << total_num_points << " rows, got "
                                      << i);
    // Solve the equations Ax = b. Could use a more efficient solver here if
    // needed. But anything iterative could be done with the parallel linear
    // solver, the point here is to assemble the full matrix and invert it
    // directly.
    if (get<logging::Tags::Verbosity<OptionTags::BuildMatrixOptionsGroup>>(
            box) >= Verbosity::Quiet) {
      Parallel::printf("Inverting matrix of size %zu x %zu\n", total_num_points,
                       total_num_points);
    }
    const blaze::DynamicVector<value_type> solution =
        blaze::solve(matrix, source);
    // Broadcast the solution to the elements
    Parallel::simple_action<StoreSolution<BuildMatrixMetavars>>(
        Parallel::get_parallel_component<ElementArrayComponent>(cache),
        grid_index, solution);
  }
};

template <typename BuildMatrixMetavars>
struct StoreSolution {
 private:
  using value_type = typename BuildMatrixMetavars::value_type;
  using FieldsTag = typename BuildMatrixMetavars::fields_tag;

 public:
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, size_t Dim>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ElementId<Dim>& element_id, const size_t grid_index,
                    const blaze::DynamicVector<value_type>& solution) {
    if (grid_index != element_id.grid_index()) {
      return;
    }
    const size_t local_first_index = db::get<Tags::LocalFirstIndex>(box);
    db::mutate<FieldsTag>(
        [&solution, &local_first_index](const auto fields) {
          std::copy_n(
              solution.begin() + static_cast<ptrdiff_t>(local_first_index),
              fields->size(), fields->data());
        },
        make_not_null(&box));
    // Proceed with algorithm
    Parallel::get_parallel_component<ParallelComponent>(cache)[element_id]
        .perform_algorithm(true);
  }
};

template <typename BuildMatrixMetavars>
struct ProjectBuildMatrix : tt::ConformsTo<::amr::protocols::Projector> {
 private:
  using IterationIdTag = typename BuildMatrixMetavars::iteration_id_tag;
  using OperandTag = typename BuildMatrixMetavars::operand_tag;
  using value_type = typename BuildMatrixMetavars::value_type;

 public:
  using return_tags =
      tmpl::list<Tags::TotalNumPoints, Tags::LocalFirstIndex, IterationIdTag,
                 OperandTag, Tags::Matrix<value_type>>;
  using argument_tags = tmpl::list<>;

  template <typename... AmrData>
  static void apply(const gsl::not_null<size_t*> /*unused*/,
                    const AmrData&... /*amr_data*/) {
    // Nothing to do. Everything gets initialized at the start of the algorithm.
  }
};

/*!
 * \brief Build the explicit matrix representation of the linear operator and
 * optionally invert it directly to solve the problem.
 *
 * This is useful for debugging and analysis only, not to actually solve the
 * elliptic problem (that should happen iteratively).
 *
 * Add the `actions` to the action list to build the matrix. The
 * `ApplyOperatorActions` template parameter are the actions that apply the
 * linear operator to the `OperandTag`. Also add the `amr_projectors` to the
 * list of AMR projectors and the `register_actions`
 *
 * \tparam FieldsTag The solution will be stored here (if direct solve is
 * enabled).
 * \tparam FixedSourcesTag The source `b` in the problem `Ax = b`. Only used
 * if direct solve is enabled.
 * \tparam OperandTag Will be set to unit vectors, with the '1' moving through
 * all points over the course of the iteration.
 * \tparam OperatorAppliedToOperandTag Where the `ApplyOperatorActions` store
 * the result of applying the linear operator to the `OperandTag`.
 * \tparam CoordsTag The tag of the coordinates observed alongside the matrix
 * for volume data visualization.
 * \tparam ArraySectionIdTag Can identify a subset of elements that this
 * algorithm should run over, e.g. in a multigrid setting.
 */
template <typename FieldsTag, typename FixedSourcesTag, typename OperandTag,
          typename OperatorAppliedToOperandTag, typename CoordsTag,
          typename ArraySectionIdTag = void>
struct BuildMatrix {
 private:
  using BuildMatrixMetavars =
      detail::BuildMatrixMetavars<FieldsTag, FixedSourcesTag, OperandTag,
                                  OperatorAppliedToOperandTag, CoordsTag,
                                  ArraySectionIdTag>;

 public:
  template <typename Metavariables>
  using component_list =
      tmpl::list<BuildMatrixSingleton<Metavariables, BuildMatrixMetavars>>;

  template <typename ApplyOperatorActions>
  using actions =
      tmpl::list<CollectTotalNumPoints<BuildMatrixMetavars>,
                 // PrepareBuildMatrix is called on reduction broadcast
                 SetUnitVector<BuildMatrixMetavars>, ApplyOperatorActions,
                 StoreMatrixColumn<BuildMatrixMetavars>,
                 // Algorithm iterates until matrix is complete, then proceeds
                 // below
                 AssembleFullMatrix<BuildMatrixMetavars>>;

  using amr_projectors = tmpl::list<ProjectBuildMatrix<BuildMatrixMetavars>>;

  /// Add to the register phase to enable observations
  using register_actions = tmpl::list<observers::Actions::RegisterWithObservers<
      detail::RegisterWithVolumeObserver<ArraySectionIdTag>>>;
};

}  // namespace Actions
}  // namespace LinearSolver
