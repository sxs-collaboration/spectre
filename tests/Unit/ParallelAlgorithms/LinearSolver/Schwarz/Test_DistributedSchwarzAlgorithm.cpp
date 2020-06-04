// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include <cstddef>
#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/DistributedLinearSolverAlgorithmTestHelpers.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/LinearSolverAlgorithmTestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/LinearSolver/Actions/TerminateIfConverged.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementCenteredSubdomainData.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Schwarz.hpp"
#include "Utilities/TMPL.hpp"

#include "Parallel/Printf.hpp"

namespace helpers = LinearSolverAlgorithmTestHelpers;
namespace helpers_distributed = DistributedLinearSolverAlgorithmTestHelpers;

namespace {

struct SchwarzSmoother {
  static constexpr Options::String help =
      "Options for the iterative Schwarz smoother";
};

// Applies the linear operator given explicitly in the input file to data on an
// element-centered subdomain
//
// We assume all elements have the same extents so we can use the element's
// intruding overlap extents instead of initializing extruding overlap extents.
struct SubdomainOperator;

struct SubdomainElementOperator {
  using argument_tags = tmpl::list<
      helpers_distributed::LinearOperator, domain::Tags::Element<1>,
      LinearSolver::Schwarz::Tags::IntrudingExtents<1, SchwarzSmoother>>;
  template <typename ArgTagsList, typename ResultTagsList>
  static auto apply(
      const db::item_type<helpers_distributed::LinearOperator>& linear_operator,
      const Element<1>& element,
      const std::array<size_t, 1>& all_intruding_extents,
      const LinearSolver::Schwarz::ElementCenteredSubdomainData<1, ArgTagsList>&
          subdomain_data,
      const gsl::not_null<LinearSolver::Schwarz::ElementCenteredSubdomainData<
          1, ResultTagsList>*>
          result,
      const gsl::not_null<SubdomainOperator*> /*subdomain_operator*/) noexcept {
    using ElementData = Variables<ArgTagsList>;
    const size_t element_index = helpers_distributed::get_index(element.id());
    // Parallel::printf("Applying operator on element %zu:\n", element_index);

    // Get the operator matrix slice for the central element
    const auto& operator_slice = gsl::at(linear_operator, element_index);
    const size_t num_points = operator_slice.columns();

    // Apply contribution to the central element
    const DenseMatrix<double, blaze::columnMajor> operator_matrix_element =
        blaze::submatrix(operator_slice, element_index * num_points, 0,
                         num_points, num_points);
    dgemv_('N', num_points, num_points, 1, operator_matrix_element.data(),
           num_points, subdomain_data.element_data.data(), 1, 0,
           result->element_data.data(), 1);
    // Parallel::printf("operator_matrix_element: %s\n",
    //                  operator_matrix_element);

    // Apply contribution to the overlaps
    for (const auto& overlap_id_and_data : subdomain_data.overlap_data) {
      const auto& overlap_id = overlap_id_and_data.first;
      const auto& direction = overlap_id.first;
      const auto direction_from_neighbor = direction.opposite();
      const size_t overlapped_element_index =
          element_index + static_cast<size_t>(direction.sign());
      const auto& overlap_extents =
          gsl::at(all_intruding_extents, direction.dimension());
      const Index<1> neighbor_extents{{{num_points}}};
      // Parallel::printf("contribution to overlap %s\n", direction);
      const DenseMatrix<double, blaze::columnMajor> operator_matrix_overlap =
          blaze::submatrix(operator_slice,
                           overlapped_element_index * num_points, 0, num_points,
                           num_points);
      // Parallel::printf("operator_matrix_overlap: %s\n",
      //                  operator_matrix_overlap);
      ElementData overlap_contribution{num_points};
      dgemv_('N', num_points, num_points, 1, operator_matrix_overlap.data(),
             num_points, subdomain_data.element_data.data(), 1, 0,
             overlap_contribution.data(), 1);
      // Parallel::printf("extended_overlap_contribution: %s\n",
      //                  overlap_contribution);
      overlap_contribution = LinearSolver::Schwarz::data_on_overlap(
          overlap_contribution, neighbor_extents, overlap_extents,
          direction_from_neighbor);
      // Parallel::printf("overlap_contribution: %s\n", overlap_contribution);
      result->overlap_data.insert_or_assign(overlap_id, overlap_contribution);
    }
  }
};

template <typename Directions>
struct SubdomainFaceOperator;

template <>
struct SubdomainFaceOperator<domain::Tags::InternalDirections<1>> {
  using argument_tags = tmpl::list<
      helpers_distributed::LinearOperator, domain::Tags::Element<1>,
      domain::Tags::Direction<1>,
      LinearSolver::Schwarz::Tags::IntrudingExtents<1, SchwarzSmoother>>;
  using volume_tags = tmpl::list<
      helpers_distributed::LinearOperator, domain::Tags::Element<1>,
      LinearSolver::Schwarz::Tags::IntrudingExtents<1, SchwarzSmoother>>;
  // interface_apply doesn't currently support `void` return types
  template <typename ArgTagsList, typename ResultTagsList>
  int operator()(
      const db::item_type<helpers_distributed::LinearOperator>& linear_operator,
      const Element<1>& element, const Direction<1>& direction,
      const std::array<size_t, 1>& all_intruding_extents,
      const LinearSolver::Schwarz::ElementCenteredSubdomainData<1, ArgTagsList>&
          subdomain_data,
      const gsl::not_null<LinearSolver::Schwarz::ElementCenteredSubdomainData<
          1, ResultTagsList>*>
          result,
      const gsl::not_null<SubdomainOperator*> /*subdomain_operator*/) const
      noexcept {
    using OverlapData = Variables<ArgTagsList>;
    const size_t element_index = helpers_distributed::get_index(element.id());
    for (const auto& overlap_id_and_data : subdomain_data.overlap_data) {
      const auto& overlap_id = overlap_id_and_data.first;
      const auto& local_direction = overlap_id.first;
      if (local_direction != direction) {
        continue;
      }
      const auto direction_from_neighbor = direction.opposite();
      const size_t overlapped_element_index =
          element_index + static_cast<size_t>(direction.sign());
      const auto& overlap_extents =
          gsl::at(all_intruding_extents, direction.dimension());

      // Parallel::printf("\nface %s on element %zu overlapping %zu\n",
      //                  direction, element_index, overlapped_element_index);

      // Get the operator matrix slice for the overlapped element
      const auto& operator_slice =
          gsl::at(linear_operator, overlapped_element_index);
      const size_t num_points = operator_slice.columns();
      const Index<1> neighbor_extents{{{num_points}}};

      // Get the overlap data extended by zeros
      const auto& overlap_data = overlap_id_and_data.second;
      const auto extended_overlap_data =
          LinearSolver::Schwarz::extended_overlap_data(
              overlap_data, neighbor_extents, overlap_extents,
              direction_from_neighbor);
      // Parallel::printf("overlap_data: %s\n", overlap_data);
      // Parallel::printf("extended_overlap_data: %s\n",
      // extended_overlap_data);

      // Apply contribution to the central element
      const DenseMatrix<double, blaze::columnMajor> operator_matrix_element =
          blaze::submatrix(operator_slice, element_index * num_points, 0,
                           num_points, num_points);
      // Parallel::printf("operator_matrix_element: %s\n",
      //                  operator_matrix_element);
      OverlapData element_contribution{num_points};
      dgemv_('N', num_points, num_points, 1, operator_matrix_element.data(),
             num_points, extended_overlap_data.data(), 1, 0,
             element_contribution.data(), 1);
      // Parallel::printf("element_contribution: %s\n", element_contribution);
      result->element_data += element_contribution;

      // Apply contribution to the overlap
      const DenseMatrix<double, blaze::columnMajor> operator_matrix_overlap =
          blaze::submatrix(operator_slice,
                           overlapped_element_index * num_points, 0, num_points,
                           num_points);
      // Parallel::printf("operator_matrix_overlap: %s\n",
      //                  operator_matrix_overlap);
      OverlapData overlap_contribution{num_points};
      dgemv_('N', num_points, num_points, 1, operator_matrix_overlap.data(),
             num_points, extended_overlap_data.data(), 1, 0,
             overlap_contribution.data(), 1);
      // Parallel::printf("extended_overlap_contribution: %s\n",
      //                  overlap_contribution);
      overlap_contribution = LinearSolver::Schwarz::data_on_overlap(
          overlap_contribution, neighbor_extents, overlap_extents,
          direction_from_neighbor);
      // Parallel::printf("overlap_contribution: %s\n", overlap_contribution);
      result->overlap_data.at(overlap_id) += overlap_contribution;
    }
    // Parallel::printf("Result on element: %s\n", result->element_data);
    // Parallel::printf("Result on overlaps: %s\n", result->overlap_data);
    return 0;
  }
};

template <>
struct SubdomainFaceOperator<domain::Tags::BoundaryDirectionsInterior<1>> {
  using argument_tags = tmpl::list<>;
  template <typename ArgTagsList, typename ResultTagsList>
  int operator()(
      const LinearSolver::Schwarz::ElementCenteredSubdomainData<
          1, ArgTagsList>& /*subdomain_data*/,
      const gsl::not_null<LinearSolver::Schwarz::ElementCenteredSubdomainData<
          1, ResultTagsList>*>
      /*result*/,
      const gsl::not_null<SubdomainOperator*> /*subdomain_operator*/) const
      noexcept {
    return 0;
  }
};

struct SubdomainOperator {
  static constexpr size_t volume_dim = 1;
  using element_operator = SubdomainElementOperator;
  template <typename Directions>
  using face_operator = SubdomainFaceOperator<Directions>;
  explicit SubdomainOperator(const size_t /*element_num_points*/) noexcept {}
};

struct Metavariables {
  static constexpr const char* const help{
      "Test the Schwarz linear solver algorithm"};
  static constexpr size_t volume_dim = 1;

  using linear_solver =
      LinearSolver::Schwarz::Schwarz<Metavariables,
                                     helpers_distributed::fields_tag,
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
    &setup_error_handling, &domain::creators::register_derived_with_charm};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};

using charmxx_main_component = Parallel::Main<Metavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
