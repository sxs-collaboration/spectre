// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Interval.hpp"
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
#include "ParallelAlgorithms/LinearSolver/Actions/BuildMatrix.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace PUP {
class er;
}  // namespace PUP

namespace helpers = LinearSolverAlgorithmTestHelpers;
namespace helpers_distributed = DistributedLinearSolverAlgorithmTestHelpers;

namespace {

struct IterationIdLabel {};

struct TestResult {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<1>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // Read the matrix from the file
    h5::H5File<h5::AccessType::ReadOnly> h5file("Test_BuildMatrix_Volume0.h5");
    const auto matrix_data =
        h5file.get<h5::VolumeData>("/Matrix").get_data_by_element(
            std::nullopt, std::nullopt, {{"Variable_0"}});
    // Check the columns of the matrix corresponding to this element
    const size_t element_index = helpers_distributed::get_index(element_id);
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

  using build_matrix_actions = LinearSolver::Actions::build_matrix_actions<
      Convergence::Tags::IterationId<IterationIdLabel>,
      helpers_distributed::fields_tag,
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
                         helpers_distributed::fields_tag>,
      helpers_distributed::ComputeOperatorAction<
          helpers_distributed::fields_tag>,
      domain::Tags::Coordinates<volume_dim, Frame::ElementLogical>>;

  using element_array = elliptic::DgElementArray<
      Metavariables,
      tmpl::list<
          Parallel::PhaseActions<
              Parallel::Phase::Initialization,
              tmpl::list<helpers_distributed::InitializeElement,
                         Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::Register,
              tmpl::push_back<LinearSolver::Actions::build_matrix_register<>,
                              Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::BuildMatrix,
              tmpl::push_back<build_matrix_actions,
                              Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<Parallel::Phase::Testing,
                                 tmpl::list<TestResult>>>>;

  using component_list =
      tmpl::list<element_array, observers::Observer<Metavariables>,
                 observers::ObserverWriter<Metavariables>,
                 helpers::OutputCleaner<Metavariables, false, true>>;
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
