// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include <cstddef>
#include <vector>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/DistributedLinearSolverAlgorithmTestHelpers.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/LinearSolverAlgorithmTestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementCenteredSubdomainData.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Schwarz.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/SubdomainOperator.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace helpers = LinearSolverAlgorithmTestHelpers;
namespace helpers_distributed = DistributedLinearSolverAlgorithmTestHelpers;

namespace {

struct SchwarzSmoother {
  static constexpr Options::String help =
      "Options for the iterative Schwarz smoother";
};

DenseMatrix<double> combine_matrix_slices(
    const std::vector<DenseMatrix<double>>& matrix_slices) {
  const size_t num_slices = matrix_slices.size();
  const size_t num_cols_per_slice = matrix_slices.begin()->columns();
  const size_t total_num_points = num_slices * num_cols_per_slice;
  DenseMatrix<double> full_matrix(total_num_points, total_num_points);
  for (size_t i = 0; i < num_slices; ++i) {
    blaze::submatrix(full_matrix, 0, i * num_cols_per_slice, total_num_points,
                     num_cols_per_slice) = gsl::at(matrix_slices, i);
  }
  return full_matrix;
}

template <typename Tag>
DenseVector<double> extend_subdomain_data(
    const LinearSolver::Schwarz::ElementCenteredSubdomainData<
        1, tmpl::list<Tag>>& subdomain_data,
    const size_t element_index, const size_t num_elements,
    const size_t num_points_per_element, const size_t overlap_extent) {
  DenseVector<double> extended_subdomain_data(
      num_elements * num_points_per_element, 0.);
  blaze::subvector(
      extended_subdomain_data, element_index * num_points_per_element,
      num_points_per_element) = get(get<Tag>(subdomain_data.element_data));
  for (const auto& [overlap_id, overlap_data] : subdomain_data.overlap_data) {
    const auto& direction = overlap_id.first;
    const auto direction_from_neighbor = direction.opposite();
    const size_t overlapped_element_index = direction.side() == Side::Lower
                                                ? (element_index - 1)
                                                : (element_index + 1);
    blaze::subvector(extended_subdomain_data,
                     overlapped_element_index * num_points_per_element,
                     num_points_per_element) =
        get(get<Tag>(LinearSolver::Schwarz::extended_overlap_data(
            overlap_data, Index<1>{num_points_per_element}, overlap_extent,
            direction_from_neighbor)));
  }
  return extended_subdomain_data;
}

template <typename Tag>
void restrict_to_subdomain(
    const gsl::not_null<LinearSolver::Schwarz::ElementCenteredSubdomainData<
        1, tmpl::list<Tag>>*>
        result,
    const DenseVector<double>& extended_result, const size_t element_index,
    const size_t num_points_per_element, const size_t overlap_extent) {
  const DenseVector<double> restricted_element_data =
      blaze::subvector(extended_result, element_index * num_points_per_element,
                       num_points_per_element);
  std::copy(restricted_element_data.begin(), restricted_element_data.end(),
            get(get<Tag>(result->element_data)).begin());
  for (auto& [overlap_id, overlap_result] : result->overlap_data) {
    const auto& direction = overlap_id.first;
    const auto direction_from_neighbor = direction.opposite();
    const size_t overlapped_element_index = direction.side() == Side::Lower
                                                ? (element_index - 1)
                                                : (element_index + 1);
    Scalar<DataVector> extended_result_on_element{num_points_per_element};
    const DenseVector<double> restricted_overlap_data = blaze::subvector(
        extended_result, overlapped_element_index * num_points_per_element,
        num_points_per_element);
    std::copy(restricted_overlap_data.begin(), restricted_overlap_data.end(),
              get(extended_result_on_element).begin());
    LinearSolver::Schwarz::data_on_overlap(
        make_not_null(&get<Tag>(overlap_result)), extended_result_on_element,
        Index<1>{num_points_per_element}, overlap_extent,
        direction_from_neighbor);
  }
}

// [subdomain_operator]
// Applies the linear operator given explicitly in the input file to data on an
// element-centered subdomain, using standard matrix multiplication.
//
// We assume all elements have the same extents so we can use the element's
// intruding overlap extents instead of initializing extruding overlap extents.
struct SubdomainOperator : LinearSolver::Schwarz::SubdomainOperator<1> {
  // The operator can by non-const and can also mutate the DataBox to modify
  // temporary and persistent memory buffers
  template <typename ResultTags, typename OperandTags, typename DbTagsList>
  void operator()(
      const gsl::not_null<
          LinearSolver::Schwarz::ElementCenteredSubdomainData<1, ResultTags>*>
          result,
      const LinearSolver::Schwarz::ElementCenteredSubdomainData<1, OperandTags>&
          operand,
      db::DataBox<DbTagsList>& box) noexcept {
    // Retrieve arguments from the DataBox
    const auto& matrix_slices =
        db::get<helpers_distributed::LinearOperator>(box);
    const auto& element = db::get<domain::Tags::Element<1>>(box);
    const auto& overlap_extents = db::get<
        LinearSolver::Schwarz::Tags::IntrudingExtents<1, SchwarzSmoother>>(box);

    const size_t element_index = helpers_distributed::get_index(element.id());
    const size_t num_elements = matrix_slices.size();
    const size_t num_points_per_element = matrix_slices.begin()->columns();

    // Re-size the result buffer if necessary
    result->destructive_resize(operand);

    // Assemble full operator matrix
    const auto operator_matrix = combine_matrix_slices(matrix_slices);

    // Extend subdomain data with zeros outside the subdomain
    const auto extended_operand =
        extend_subdomain_data(operand, element_index, num_elements,
                              num_points_per_element, overlap_extents[0]);

    // Apply matrix to extended subdomain data
    const DenseVector<double> extended_result =
        operator_matrix * extended_operand;

    // Restrict the result back to the subdomain
    restrict_to_subdomain(result, extended_result, element_index,
                          num_points_per_element, overlap_extents[0]);
  }
};
// [subdomain_operator]

struct Metavariables {
  static constexpr const char* const help{
      "Test the Schwarz linear solver algorithm"};
  static constexpr size_t volume_dim = 1;
  using system =
      TestHelpers::domain::BoundaryConditions::SystemWithoutBoundaryConditions<
          volume_dim>;

  using linear_solver =
      LinearSolver::Schwarz::Schwarz<helpers_distributed::fields_tag,
                                     SchwarzSmoother, SubdomainOperator>;
  using preconditioner = void;

  using Phase = helpers::Phase;
  using observed_reduction_data_tags =
      helpers::observed_reduction_data_tags<Metavariables>;
  using component_list = helpers_distributed::component_list<Metavariables>;
  static constexpr bool ignore_unrecognized_command_line_options = false;
  static constexpr auto determine_next_phase =
      helpers::determine_next_phase<Metavariables>;
};

}  // namespace

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &domain::creators::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<
        Metavariables::linear_solver::subdomain_solver>,
    &TestHelpers::domain::BoundaryConditions::register_derived_with_charm};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};

using charmxx_main_component = Parallel::Main<Metavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
