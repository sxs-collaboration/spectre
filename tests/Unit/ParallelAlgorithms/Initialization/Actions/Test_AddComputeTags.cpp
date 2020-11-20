// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct SomeNumber : db::SimpleTag {
  using type = double;
};

struct SquareNumber : db::SimpleTag {
  using type = double;
};

struct SquareNumberCompute : SquareNumber, db::ComputeTag {
  static void function(const gsl::not_null<double*> result,
                       const double some_number) noexcept {
    *result = square(some_number);
  }
  using argument_tags = tmpl::list<SomeNumber>;
  using base = SquareNumber;
  using return_type = double;
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
              Actions::SetupDataBox,
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
  for (size_t i = 0; i < 2; ++i) {
    runner.template next_action<component>(0);
  }

  CHECK(ActionTesting::tag_is_retrievable<component, SquareNumber>(runner, 0));
}
