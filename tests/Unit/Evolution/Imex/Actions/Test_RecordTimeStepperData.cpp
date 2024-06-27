// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Imex/Actions/RecordTimeStepperData.hpp"
#include "Evolution/Imex/GuessResult.hpp"
#include "Evolution/Imex/Protocols/ImplicitSector.hpp"
#include "Evolution/Imex/Tags/ImplicitHistory.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct Var2 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct Var3 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct OtherVar : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct Sector1 : tt::ConformsTo<imex::protocols::ImplicitSector> {
  using tensors = tmpl::list<Var1>;
  using initial_guess = imex::GuessExplicitResult;

  struct SolveAttempt {
    struct source : tt::ConformsTo<imex::protocols::ImplicitSource>,
                    tt::ConformsTo<::protocols::StaticReturnApplyable> {
      using return_tags = tmpl::list<Tags::Source<Var1>>;
      using argument_tags = tmpl::list<OtherVar>;
      static void apply(const gsl::not_null<Scalar<DataVector>*> source,
                        const Scalar<DataVector>& other_var) {
        get(*source) = get(other_var);
      }
    };

    struct jacobian : tt::ConformsTo<imex::protocols::ImplicitSourceJacobian>,
                      tt::ConformsTo<::protocols::StaticReturnApplyable> {
      using return_tags = tmpl::list<>;
      using argument_tags = tmpl::list<>;
      static void apply();
    };

    using tags_from_evolution = tmpl::list<>;
    using simple_tags = tmpl::list<>;
    using compute_tags = tmpl::list<>;
    using source_prep = tmpl::list<>;
    using jacobian_prep = tmpl::list<>;
  };
  using solve_attempts = tmpl::list<SolveAttempt>;
};

struct Sector2 : tt::ConformsTo<imex::protocols::ImplicitSector> {
  using tensors = tmpl::list<Var2, Var3>;
  using initial_guess = imex::GuessExplicitResult;

  struct InitialAttempt {
    struct source : tt::ConformsTo<imex::protocols::ImplicitSource>,
                    tt::ConformsTo<::protocols::StaticReturnApplyable> {
      using return_tags = tmpl::list<Tags::Source<Var3>, Tags::Source<Var2>>;
      using argument_tags = tmpl::list<Var1>;
      static void apply(const gsl::not_null<Scalar<DataVector>*> source3,
                        const gsl::not_null<Scalar<DataVector>*> source2,
                        const Scalar<DataVector>& var1) {
        get(*source2) = 3.0 * get(var1);
        get(*source3) = 5.0 * get(var1);
      }
    };

    struct jacobian : tt::ConformsTo<imex::protocols::ImplicitSourceJacobian>,
                      tt::ConformsTo<::protocols::StaticReturnApplyable> {
      using return_tags = tmpl::list<>;
      using argument_tags = tmpl::list<>;
      static void apply();
    };

    using tags_from_evolution = tmpl::list<>;
    using simple_tags = tmpl::list<>;
    using compute_tags = tmpl::list<>;
    using source_prep = tmpl::list<>;
    using jacobian_prep = tmpl::list<>;
  };

  struct Fallback {
    struct source : tt::ConformsTo<imex::protocols::ImplicitSource>,
                    tt::ConformsTo<::protocols::StaticReturnApplyable> {
      using return_tags = tmpl::list<Tags::Source<Var3>, Tags::Source<Var2>>;
      using argument_tags = tmpl::list<>;
      static void apply(const gsl::not_null<Scalar<DataVector>*> /*source3*/,
                        const gsl::not_null<Scalar<DataVector>*> /*source2*/) {
        CHECK(false);
      }
    };

    struct jacobian : tt::ConformsTo<imex::protocols::ImplicitSourceJacobian>,
                      tt::ConformsTo<::protocols::StaticReturnApplyable> {
      using return_tags = tmpl::list<>;
      using argument_tags = tmpl::list<>;
      static void apply();
    };

    using tags_from_evolution = tmpl::list<>;
    using simple_tags = tmpl::list<>;
    using compute_tags = tmpl::list<>;
    using source_prep = tmpl::list<>;
    using jacobian_prep = tmpl::list<>;
  };
  using solve_attempts = tmpl::list<InitialAttempt, Fallback>;
};

struct System : tt::ConformsTo<imex::protocols::ImexSystem> {
  using variables_tag = Tags::Variables<tmpl::list<Var1, Var2, Var3>>;
  using implicit_sectors = tmpl::list<Sector1, Sector2>;
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using simple_tags =
      db::AddSimpleTags<Tags::TimeStepId, System::variables_tag, OtherVar,
                        imex::Tags::ImplicitHistory<Sector1>,
                        imex::Tags::ImplicitHistory<Sector2>>;
  using compute_tags = db::AddComputeTags<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, compute_tags>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<imex::Actions::RecordTimeStepperData<System>>>>;
};

struct Metavariables {
  using system = System;
  using component_list = tmpl::list<Component<Metavariables>>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Imex.Actions.RecordTimeStepperData",
                  "[Unit][Evolution][Actions]") {
  using component = Component<Metavariables>;

  const size_t number_of_grid_points = 5;

  const Slab slab(1., 3.);
  const TimeStepId time_step_id(true, 0, slab.start());

  System::variables_tag::type vars(number_of_grid_points);
  get(get<Var1>(vars)) = 2.0;
  get(get<Var2>(vars)) = 3.0;
  get(get<Var3>(vars)) = 4.0;

  const Scalar<DataVector> other_var(DataVector(number_of_grid_points, 5.0));

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};

  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {time_step_id, vars, other_var,
       imex::Tags::ImplicitHistory<Sector1>::type(2),
       imex::Tags::ImplicitHistory<Sector2>::type(2)});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  runner.next_action<component>(0);

  const auto& box = ActionTesting::get_databox<component>(runner, 0);
  const auto& history1 = db::get<imex::Tags::ImplicitHistory<Sector1>>(box);
  const auto& history2 = db::get<imex::Tags::ImplicitHistory<Sector2>>(box);

  CHECK(history1.size() == 1);
  CHECK(history1[0].time_step_id == time_step_id);
  CHECK(not history1[0].value.has_value());
  CHECK(get<Tags::dt<Var1>>(history1[0].derivative) == other_var);

  CHECK(history2.size() == 1);
  CHECK(history2[0].time_step_id == time_step_id);
  CHECK(not history2[0].value.has_value());
  CHECK(get(get<Tags::dt<Var2>>(history2[0].derivative)) ==
        3.0 * get(get<Var1>(vars)));
  CHECK(get(get<Tags::dt<Var3>>(history2[0].derivative)) ==
        5.0 * get(get<Var1>(vars)));
}
