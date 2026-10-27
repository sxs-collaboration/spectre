// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/DistributedLinearSolverAlgorithmTestHelpers.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/LinearSolverAlgorithmTestHelpers.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/CharmMain.tpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "ParallelAlgorithms/LinearSolver/Actions/BuildMatrix.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace PUP {
class er;
}  // namespace PUP

namespace helpers = LinearSolverAlgorithmTestHelpers;
namespace helpers_distributed = DistributedLinearSolverAlgorithmTestHelpers;

namespace {

// Factor by which the linear operator is scaled, so we can change the operator
// between solves
struct OperatorScale : db::SimpleTag {
  using type = double;
};

// Counts the direct solves completed so far
struct NumSolves : db::SimpleTag {
  using type = size_t;
};

struct InitializeScaling {
  using simple_tags = tmpl::list<OperatorScale, NumSolves>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<1>& /*element_id*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    ::Initialization::mutate_assign<simple_tags>(make_not_null(&box), 1., 0_st);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

struct ScaleOperatorAppliedToOperand {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<1>& /*element_id*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate<db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
                                  helpers_distributed::fields_tag>>(
        [](const auto operator_applied_to_operand, const double scale) {
          *operator_applied_to_operand *= scale;
        },
        make_not_null(&box), db::get<OperatorScale>(box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

// Check the solution of the direct solve and prepare the next one. Each solve
// tests a different code path of the direct solver:
// 1. Solve Ax=b. Builds and factorizes the matrix.
// 2. Solve Ax=2b. Reuses the factorization.
// 3. Solve 2Ax=2b. Rebuilds the matrix and factorizes it again. Reusing the
//    factorization would give the wrong solution.
// The matrix can only be observed once, so only the first solve runs if the
// matrix is written to disk.
struct CheckSolutionAndPrepareNextSolve {
  using const_global_cache_tags =
      tmpl::list<helpers_distributed::ExpectedResult>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<1>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate<NumSolves>(
        [](const gsl::not_null<size_t*> num_solves) { ++(*num_solves); },
        make_not_null(&box));
    const size_t num_solves = db::get<NumSolves>(box);
    const double expected_factor = num_solves == 2 ? 2. : 1.;
    const auto& expected_result =
        gsl::at(get<helpers_distributed::ExpectedResult>(box),
                helpers_distributed::get_index(element_id));
    const auto& result = get(db::get<helpers_distributed::ScalarFieldTag>(box));
    for (size_t i = 0; i < expected_result.size(); i++) {
      SPECTRE_PARALLEL_REQUIRE(result[i] ==
                               approx(expected_factor * expected_result[i]));
    }
    if (num_solves == 3 or
        db::get<LinearSolver::Tags::MatrixSubfileName>(box).has_value()) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
    // Double the source for the second solve, and the operator for the third
    db::mutate<db::add_tag_prefix<::Tags::FixedSource,
                                  helpers_distributed::fields_tag>,
               OperatorScale>(
        [&num_solves](const auto fixed_sources,
                      const gsl::not_null<double*> scale) {
          if (num_solves == 1) {
            *fixed_sources *= 2.;
          } else {
            *scale = 2.;
          }
        },
        make_not_null(&box));
    // Jump back to the reset actions at the start of the action list. Skip the
    // reset for the second solve, so the factorization is reused.
    const size_t next_action_index =
        num_solves == 1
            ? tmpl::size<
                  typename Metavariables::build_matrix::reset_actions>::value
            : 0;
    return {Parallel::AlgorithmExecution::Continue, next_action_index};
  }
};

struct TestResult {
  using const_global_cache_tags =
      tmpl::list<helpers_distributed::ExpectedResult>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<1>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // Check the solution
    const size_t element_index = helpers_distributed::get_index(element_id);
    const auto& expected_result =
        gsl::at(get<helpers_distributed::ExpectedResult>(box), element_index);
    const auto& result = get(db::get<helpers_distributed::ScalarFieldTag>(box));
    for (size_t i = 0; i < expected_result.size(); i++) {
      SPECTRE_PARALLEL_REQUIRE(result[i] == approx(expected_result[i]));
    }
    if (not db::get<LinearSolver::Tags::MatrixSubfileName>(box).has_value()) {
      return {Parallel::AlgorithmExecution::Pause, std::nullopt};
    }
    // Read the matrix from the file
    h5::H5File<h5::AccessType::ReadOnly> h5file("Test_BuildMatrix_Volume0.h5");
    const auto matrix_data =
        h5file.get<h5::VolumeData>("/Matrix").get_data_by_element(
            std::nullopt, std::nullopt, {{"Variable_0"}});
    // Check the columns of the matrix corresponding to this element
    const auto& linear_operator =
        gsl::at(get<helpers_distributed::LinearOperator>(box), element_index);
    const size_t num_points = linear_operator.columns();
    for (size_t col = 0; col < num_points; ++col) {
      const auto& col_data =
          get<2>(gsl::at(matrix_data, element_index * num_points + col));
      size_t row = 0;
      for (const auto& element_data : col_data) {
        const auto& row_data =
            get<DataVector>(element_data.tensor_components.front().data);
        for (size_t i = 0; i < row_data.size(); ++i) {
          SPECTRE_PARALLEL_REQUIRE(linear_operator(row, col) == row_data[i]);
          ++row;
        }
      }
      SPECTRE_PARALLEL_REQUIRE(row == linear_operator.rows());
    }
    // Also check the matrix stored in the DataBox
    const auto& matrix_slice = db::get<LinearSolver::Tags::Matrix<double>>(box);
    for (size_t row = 0; row < num_points; ++row) {
      for (const auto& element_data : get<2>(matrix_data[0])) {
        const size_t col_offset = helpers_distributed::get_index(
                                      ElementId<1>(element_data.element_name)) *
                                  num_points;
        for (size_t col = 0; col < num_points; ++col) {
          const auto& row_data = get<DataVector>(
              get<2>(matrix_data[col_offset + col])[element_index]
                  .tensor_components.front()
                  .data);
          SPECTRE_PARALLEL_REQUIRE(row_data[row] ==
                                   matrix_slice(row, col_offset + col));
        }
      }
    }
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

struct Metavariables {
  static constexpr Options::String help{
      "Test building an explicit matrix representation of the linear operator"};
  static constexpr size_t volume_dim = 1;
  using system =
      TestHelpers::domain::BoundaryConditions::SystemWithoutBoundaryConditions<
          volume_dim>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DomainCreator<1>, tmpl::list<domain::creators::Interval>>>;
  };

  using build_matrix = LinearSolver::Actions::BuildMatrix<
      helpers_distributed::fields_tag,
      db::add_tag_prefix<::Tags::FixedSource, helpers_distributed::fields_tag>,
      helpers_distributed::fields_tag,
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
                         helpers_distributed::fields_tag>,
      domain::Tags::Coordinates<volume_dim, Frame::ElementLogical>>;

  using build_matrix_actions = typename build_matrix::template actions<
      tmpl::list<helpers_distributed::ComputeOperatorAction<
                     helpers_distributed::fields_tag>,
                 ScaleOperatorAppliedToOperand>>;

  using element_array = elliptic::DgElementArray<
      Metavariables,
      tmpl::list<
          Parallel::PhaseActions<
              Parallel::Phase::Initialization,
              tmpl::list<helpers_distributed::InitializeElement,
                         InitializeScaling, Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::Register,
              tmpl::push_back<build_matrix::register_actions,
                              Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::BuildMatrix,
              tmpl::flatten<tmpl::list<typename build_matrix::reset_actions,
                                       build_matrix_actions,
                                       CheckSolutionAndPrepareNextSolve,
                                       Parallel::Actions::TerminatePhase>>>,
          Parallel::PhaseActions<Parallel::Phase::Testing,
                                 tmpl::list<TestResult>>>>;

  using component_list = tmpl::flatten<tmpl::list<
      element_array, typename build_matrix::component_list<Metavariables>,
      observers::Observer<Metavariables>,
      observers::ObserverWriter<Metavariables>,
      helpers::OutputCleaner<Metavariables, false, false>>>;
  using observed_reduction_data_tags = tmpl::list<>;
  static constexpr bool ignore_unrecognized_command_line_options = false;
  static constexpr std::array<Parallel::Phase, 6> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Register,
       Parallel::Phase::BuildMatrix, Parallel::Phase::Testing,
       Parallel::Phase::Cleanup, Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

}  // namespace

extern "C" void CkRegisterMainModule() {
  Parallel::charmxx::register_main_module<Metavariables>();
  Parallel::charmxx::register_init_node_and_proc(
      {&domain::creators::register_derived_with_charm}, {});
}
