// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DenseVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/IO/Observers/ObserverHelpers.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/Observer/Helpers.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "ParallelAlgorithms/Actions/SetData.hpp"
#include "ParallelAlgorithms/LinearSolver/AsynchronousSolvers/ElementActions.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

struct TestSolver {};

struct VectorTag : db::SimpleTag {
  using type = DenseVector<double>;
};

using fields_tag = VectorTag;
using source_tag = ::Tags::FixedSource<fields_tag>;
using operator_applied_to_fields_tag =
    LinearSolver::Tags::OperatorAppliedTo<fields_tag>;
using residual_tag = LinearSolver::Tags::Residual<fields_tag>;
using residual_magnitude_square_tag =
    LinearSolver::Tags::MagnitudeSquare<residual_tag>;

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
                         tmpl::list<fields_tag, source_tag>>,
                     LinearSolver::async_solvers::InitializeElement<
                         fields_tag, TestSolver, source_tag>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Register,
          tmpl::list<LinearSolver::async_solvers::RegisterElement<
              fields_tag, TestSolver, source_tag>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Test,
          tmpl::list<LinearSolver::async_solvers::PrepareSolve<
                         fields_tag, TestSolver, source_tag, TestSolver>,
                     LinearSolver::async_solvers::CompleteStep<
                         fields_tag, TestSolver, source_tag, TestSolver>>>>;
};

struct Metavariables {
  using component_list =
      tmpl::list<TestObservers_detail::observer_component<Metavariables>,
                 TestObservers_detail::observer_writer_component<Metavariables>,
                 ElementArray<Metavariables>>;
  using observed_reduction_data_tags = observers::make_reduction_data_tags<
      tmpl::list<LinearSolver::async_solvers::reduction_data>>;
  enum class Phase { Initialization, Register, Test, Exit };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelLinearSolver.Asynchronous.ElementActions",
                  "[Unit][ParallelAlgorithms][LinearSolver][Actions]") {
  using element_array = ElementArray<Metavariables>;
  using obs_component = TestObservers_detail::observer_component<Metavariables>;
  using obs_writer =
      TestObservers_detail::observer_writer_component<Metavariables>;

  const std::string reduction_file_name{
      "Test_AsynchronousLinearSolvers_ElementActions_Reductions"};
  const std::string volume_file_name{
      "Test_AsynchronousLinearSolvers_ElementActions_Volume"};
  if (file_system::check_if_file_exists(reduction_file_name + ".h5")) {
    file_system::rm(reduction_file_name + ".h5", true);
  }
  if (file_system::check_if_file_exists(volume_file_name + ".h5")) {
    file_system::rm(volume_file_name + ".h5", true);
  }

  const size_t num_iterations = 1;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {reduction_file_name, volume_file_name, num_iterations,
       Verbosity::Verbose}};

  // Setup mock element array
  const int element_id = 0;
  ActionTesting::emplace_component_and_initialize<element_array>(
      make_not_null(&runner), element_id,
      {DenseVector<double>{}, DenseVector<double>{}});
  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);

  // DataBox shortcuts
  const auto get_tag = [&runner, &element_id](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                              element_id);
  };
  const auto tag_is_retrievable = [&runner, &element_id](auto tag_v) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::tag_is_retrievable<element_array, tag>(runner,
                                                                 element_id);
  };
  const auto set_tag = [&runner, &element_id](auto tag_v, const auto& value) {
    using tag = std::decay_t<decltype(tag_v)>;
    ActionTesting::simple_action<element_array,
                                 ::Actions::SetData<tmpl::list<tag>>>(
        make_not_null(&runner), element_id, value);
  };

  // Setup mock observers
  ActionTesting::emplace_component<obs_component>(&runner, 0);
  ActionTesting::next_action<obs_component>(make_not_null(&runner), 0);
  ActionTesting::emplace_component<obs_writer>(&runner, 0);
  ActionTesting::next_action<obs_writer>(make_not_null(&runner), 0);

  // Register with observers
  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Register);
  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);
  ActionTesting::invoke_queued_simple_action<obs_component>(
      make_not_null(&runner), 0);
  ActionTesting::invoke_queued_simple_action<obs_writer>(make_not_null(&runner),
                                                         0);
  ActionTesting::invoke_queued_simple_action<obs_writer>(make_not_null(&runner),
                                                         0);

  ActionTesting::set_phase(make_not_null(&runner), Metavariables::Phase::Test);

  {
    INFO("InitializeElement");
    CHECK(get_tag(Convergence::Tags::IterationId<TestSolver>{}) ==
          std::numeric_limits<size_t>::max());
    tmpl::for_each<tmpl::list<operator_applied_to_fields_tag, residual_tag,
                              LinearSolver::Tags::HasConverged<TestSolver>>>(
        [&tag_is_retrievable](auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          CAPTURE(db::tag_name<tag>());
          CHECK(tag_is_retrievable(tag{}));
        });
  }
  {
    INFO("PrepareSolve");
    set_tag(fields_tag{}, DenseVector<double>{1., 2., 3.});
    set_tag(source_tag{}, DenseVector<double>{4., 5., 6.});
    set_tag(operator_applied_to_fields_tag{}, DenseVector<double>{7., 8., 9.});
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    CHECK(get_tag(Convergence::Tags::IterationId<TestSolver>{}) == 0);
    CHECK_FALSE(get_tag(LinearSolver::Tags::HasConverged<TestSolver>{}));
    ActionTesting::invoke_queued_simple_action<obs_component>(
        make_not_null(&runner), 0);
    ActionTesting::invoke_queued_threaded_action<obs_writer>(
        make_not_null(&runner), 0);
  }
  {
    INFO("CompleteStep");
    set_tag(fields_tag{}, DenseVector<double>{10., 11., 12.});
    set_tag(operator_applied_to_fields_tag{},
            DenseVector<double>{13., 14., 15});
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    CHECK(get_tag(Convergence::Tags::IterationId<TestSolver>{}) == 1);
    CHECK(get_tag(LinearSolver::Tags::HasConverged<TestSolver>{}));
    ActionTesting::invoke_queued_simple_action<obs_component>(
        make_not_null(&runner), 0);
    ActionTesting::invoke_queued_threaded_action<obs_writer>(
        make_not_null(&runner), 0);
    ActionTesting::invoke_queued_threaded_action<obs_writer>(
        make_not_null(&runner), 0);
    ActionTesting::invoke_queued_threaded_action<obs_writer>(
        make_not_null(&runner), 0);
  }

  {
    INFO("Reduction observations");
    REQUIRE(file_system::check_if_file_exists(reduction_file_name + ".h5"));
    {
      const auto reductions_file =
          h5::H5File<h5::AccessType::ReadOnly>(reduction_file_name + ".h5");
      const auto& reductions_subfile =
          reductions_file.get<h5::Dat>("/TestSolverResiduals");
      const auto written_reductions = reductions_subfile.get_data();
      const auto& written_legend = reductions_subfile.get_legend();
      CHECK(written_legend ==
            std::vector<std::string>{"Iteration", "Residual"});
      const Matrix expected_reductions{{0., sqrt(27.)}, {1., sqrt(243.)}};
      CHECK_MATRIX_APPROX(written_reductions, expected_reductions);
    }
    file_system::rm(reduction_file_name + ".h5", true);
  }
}
