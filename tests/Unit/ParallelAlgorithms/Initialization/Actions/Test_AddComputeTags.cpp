// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace {
struct SomeNumber : db::SimpleTag {
  using type = double;
};

struct SquareNumber : db::SimpleTag {
  using type = double;
};

struct SquareNumberCompute : SquareNumber, db::ComputeTag {
  static double function(const double some_number) noexcept {
    return square(some_number);
  }
  using argument_tags = tmpl::list<SomeNumber>;
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<tmpl::list<SomeNumber>>>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<
              Initialization::Actions::AddComputeTags<SquareNumberCompute>>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;

  enum class Phase { Initialization, Testing, Exit };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Initialization.AddComputeTags",
                  "[Unit][ParallelAlgorithms]") {
  using component = Component<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_component_and_initialize<component>(&runner, 0, {2.});

  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);
  runner.template next_action<component>(0);

  CHECK(ActionTesting::tag_is_retrievable<component, SquareNumber>(runner, 0));
}
