// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddSimpleTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct SomeNumber : db::SimpleTag {
  using type = double;
};

struct SomeOtherNumber : db::SimpleTag {
  using type = double;
};

struct SquareNumber : db::SimpleTag {
  using type = double;
};

struct AddSomeAndOtherNumber {
  using return_tags = tmpl::list<SomeNumber, SomeOtherNumber>;
  using argument_tags = tmpl::list<>;

  static void apply(const gsl::not_null<double*> some_number,
                    const gsl::not_null<double*> some_other_number) {
    *some_number = 2.;
    *some_other_number = 3.;
  }
};

struct AddSquareNumber {
  using return_tags = tmpl::list<SquareNumber>;
  using argument_tags = tmpl::list<SomeNumber>;
  static void apply(const gsl::not_null<double*> square_number,
                    const double some_number) {
    *square_number = some_number * some_number;
  }
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Testing,
      tmpl::list<Actions::SetupDataBox,
                 Initialization::Actions::AddSimpleTags<
                     tmpl::list<AddSomeAndOtherNumber, AddSquareNumber>>>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;

  enum class Phase { Initialization, Testing, Exit };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Initialization.AddSimpleTags",
                  "[Unit][ParallelAlgorithms]") {
  using component = Component<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_array_component<component>(make_not_null(&runner), {0},
                                                    {0}, 0);

  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);
  for (size_t i = 0; i < 2; ++i) {
    runner.template next_action<component>(0);
  }

  CHECK(ActionTesting::tag_is_retrievable<component, SomeNumber>(runner, 0));
  CHECK(
      ActionTesting::tag_is_retrievable<component, SomeOtherNumber>(runner, 0));
  CHECK(ActionTesting::tag_is_retrievable<component, SquareNumber>(runner, 0));
  CHECK(ActionTesting::get_databox_tag<component, SomeNumber>(runner, 0) == 2.);
  CHECK(ActionTesting::get_databox_tag<component, SomeOtherNumber>(runner, 0) ==
        3.);
  CHECK(ActionTesting::get_databox_tag<component, SquareNumber>(runner, 0) ==
        4.);
}
