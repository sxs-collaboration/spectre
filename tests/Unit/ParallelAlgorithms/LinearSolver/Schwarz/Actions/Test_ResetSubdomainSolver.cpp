// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/ActionTesting.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Actions/ResetSubdomainSolver.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

struct DummyOptionsGroup {};

struct SubdomainSolver {
  void reset() { is_reset = true; }
  bool is_reset = false;
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) { p | is_reset; }
};

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<1>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
              tmpl::list<LinearSolver::Schwarz::Tags::SubdomainSolver<
                  std::unique_ptr<SubdomainSolver>, DummyOptionsGroup>>>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<LinearSolver::Schwarz::Actions::ResetSubdomainSolver<
              DummyOptionsGroup>>>>;
};

struct Metavariables {
  using element_array = ElementArray<Metavariables>;
  using component_list = tmpl::list<element_array>;
  enum class Phase { Initialization, Testing, Exit };
};

void test_reset_subdomain_solver(const bool skip_resets) {
  CAPTURE(skip_resets);

  using element_array = typename Metavariables::element_array;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{tuples::TaggedTuple<
      LinearSolver::Schwarz::Tags::SkipSubdomainSolverResets<DummyOptionsGroup>,
      logging::Tags::Verbosity<DummyOptionsGroup>>{skip_resets,
                                                   Verbosity::Verbose}};
  const ElementId<1> element_id{0};
  ActionTesting::emplace_component_and_initialize<element_array>(
      make_not_null(&runner), element_id,
      {std::make_unique<SubdomainSolver>()});
  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);
  REQUIRE_FALSE(
      ActionTesting::get_databox_tag<
          element_array,
          LinearSolver::Schwarz::Tags::SubdomainSolverBase<DummyOptionsGroup>>(
          runner, element_id)
          .is_reset);
  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);
  CHECK(
      ActionTesting::get_databox_tag<
          element_array,
          LinearSolver::Schwarz::Tags::SubdomainSolverBase<DummyOptionsGroup>>(
          runner, element_id)
          .is_reset != skip_resets);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelSchwarz.Action.ResetSubdomainSolver",
                  "[Unit][ParallelAlgorithms][LinearSolver][Actions]") {
  test_reset_subdomain_solver(false);
  test_reset_subdomain_solver(true);
}
