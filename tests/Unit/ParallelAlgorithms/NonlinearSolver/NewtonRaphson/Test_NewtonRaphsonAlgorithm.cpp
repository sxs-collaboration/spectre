// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include <vector>

#include "Helpers/ParallelAlgorithms/NonlinearSolver/Algorithm.hpp"
#include "Options/Options.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/Gmres.hpp"
#include "ParallelAlgorithms/NonlinearSolver/NewtonRaphson/NewtonRaphson.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/TMPL.hpp"

namespace PUP {
class er;
}  // namespace PUP

namespace helpers = TestHelpers::NonlinearSolver;

namespace {

struct LinearSolverGroup {
  static std::string name() noexcept { return "LinearSolver"; }
  static constexpr Options::String help = "Options for the linear solver";
};

struct NonlinearSolverGroup {
  static std::string name() noexcept { return "NewtonRaphson"; }
  static constexpr Options::String help = "Options for the nonlinear solver";
};

template <typename OperandTag>
struct ApplyNonlinearOperator {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const int /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*component*/) noexcept {
    db::mutate<::NonlinearSolver::Tags::OperatorAppliedTo<OperandTag>>(
        make_not_null(&box),
        [](const auto Ax, const auto& x) noexcept { *Ax = cube(x) - x; },
        get<OperandTag>(box));
    return {std::move(box)};
  }
};

template <typename OperandTag, typename FieldTag>
struct ApplyLinearizedOperator {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const int /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*component*/) noexcept {
    db::mutate<LinearSolver::Tags::OperatorAppliedTo<OperandTag>>(
        make_not_null(&box),
        [](const auto Ap, const auto& dx, const auto& x) noexcept {
          *Ap = (3. * square(x) - 1) * dx;
        },
        get<OperandTag>(box), get<FieldTag>(box));
    return {std::move(box)};
  }
};

struct Metavariables {
  static constexpr Options::String help{
      "Test the Newton-Raphson nonlinear solver algorithm"};

  using nonlinear_solver = NonlinearSolver::newton_raphson::NewtonRaphson<
      Metavariables, helpers::fields_tag, NonlinearSolverGroup>;
  using linear_solver = LinearSolver::gmres::Gmres<
      Metavariables, typename nonlinear_solver::linear_solver_fields_tag,
      LinearSolverGroup, false,
      typename nonlinear_solver::linear_solver_source_tag>;

  template <typename OperandTag>
  using apply_nonlinear_operator = ApplyNonlinearOperator<OperandTag>;
  template <typename OperandTag, typename FieldTag>
  using apply_linearized_operator =
      ApplyLinearizedOperator<OperandTag, FieldTag>;

  using component_list = helpers::component_list<Metavariables>;
  using observed_reduction_data_tags =
      helpers::observed_reduction_data_tags<Metavariables>;
  static constexpr bool ignore_unrecognized_command_line_options = false;
  using Phase = helpers::Phase;
  static constexpr auto determine_next_phase =
      helpers::determine_next_phase<Metavariables>;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) noexcept {}
};

}  // namespace

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};

using charmxx_main_component = Parallel::Main<Metavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
