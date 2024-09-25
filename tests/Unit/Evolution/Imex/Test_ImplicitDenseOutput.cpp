// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <random>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Imex/GuessResult.hpp"
#include "Evolution/Imex/ImplicitDenseOutput.hpp"
#include "Evolution/Imex/Protocols/ImexSystem.hpp"
#include "Evolution/Imex/Protocols/ImplicitSector.hpp"
#include "Evolution/Imex/Tags/ImplicitHistory.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/Heun2.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
template <typename Var>
struct Sector : tt::ConformsTo<imex::protocols::ImplicitSector> {
  using tensors = tmpl::list<Var>;

  struct initial_guess {
    using return_tags = tmpl::list<Var>;
    using argument_tags = tmpl::list<>;
    static imex::GuessResult apply(
        const gsl::not_null<Scalar<DataVector>*> var,
        const Variables<tmpl::list<Var>>& inhomogeneous_terms,
        const double implicit_weight) {
      get(*var) = get(get<Var>(inhomogeneous_terms)) / (1.0 + implicit_weight);
      return imex::GuessResult::ExactSolution;
    }
  };

  struct SolveAttempt {
    struct source : tt::ConformsTo<imex::protocols::ImplicitSource>,
                    tt::ConformsTo<::protocols::StaticReturnApplyable> {
      using return_tags = tmpl::list<Tags::Source<Var>>;
      using argument_tags = tmpl::list<Var>;
      static void apply(const gsl::not_null<Scalar<DataVector>*> source,
                        const Scalar<DataVector>& var) {
        get(*source) = -get(var);
      }
    };

    using jacobian = imex::NoJacobianBecauseSolutionIsAnalytic;

    using tags_from_evolution = tmpl::list<>;
    using simple_tags = tmpl::list<>;
    using compute_tags = tmpl::list<>;
    using source_prep = tmpl::list<>;
    using jacobian_prep = tmpl::list<>;
  };
  using solve_attempts = tmpl::list<SolveAttempt>;
};

struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct Var2 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct System : tt::ConformsTo<imex::protocols::ImexSystem> {
  using variables_tag = Tags::Variables<tmpl::list<Var1, Var2>>;
  using implicit_sectors = tmpl::list<Sector<Var1>, Sector<Var2>>;
};

struct Metavariables {
  using component_list = tmpl::list<>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Imex.ImplicitDenseOutput",
                  "[Unit][Evolution][Actions]") {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<double> dist(-5.0, 5.0);

  const Slab slab(3.0, 5.0);
  const double test_time = 3.7;

  const size_t number_of_grid_points = 5;
  const auto initial_vars =
      make_with_random_values<Variables<tmpl::list<Var1, Var2>>>(
          make_not_null(&gen), make_not_null(&dist), number_of_grid_points);
  const auto explicit_vars =
      make_with_random_values<Variables<tmpl::list<Var1, Var2>>>(
          make_not_null(&gen), make_not_null(&dist), number_of_grid_points);

  TimeSteppers::History<Variables<tmpl::list<Var1, Var2>>> full_history(2);
  TimeSteppers::History<Variables<tmpl::list<Var1>>> history1(2);
  TimeSteppers::History<Variables<tmpl::list<Var2>>> history2(2);

  // Values of past sources, arbitrary.
  const auto insert_values = [&](const TimeStepId& time_step_id) {
    const auto source = make_with_random_values<
        Variables<tmpl::list<Tags::dt<Var1>, Tags::dt<Var2>>>>(
        make_not_null(&gen), make_not_null(&dist), number_of_grid_points);
    full_history.insert(time_step_id, initial_vars, -source);
    history1.insert(time_step_id, decltype(history1)::no_value,
                    -get(get<Tags::dt<Var1>>(source)));
    history2.insert(time_step_id, decltype(history2)::no_value,
                    -get(get<Tags::dt<Var2>>(source)));
  };
  insert_values(TimeStepId(true, 0, slab.start()));
  insert_values(TimeStepId(true, 0, slab.start(), 1, slab.duration(),
                           slab.end().value()));
  // All RK steppers are implemented with FSAL bookkeeping, even
  // though Heun2 doesn't use FSAL.
  insert_values(TimeStepId(true, 1, slab.end()));

  auto box = db::create<
      db::AddSimpleTags<System::variables_tag,
                        imex::Tags::ImplicitHistory<Sector<Var1>>,
                        imex::Tags::ImplicitHistory<Sector<Var2>>,
                        Tags::ConcreteTimeStepper<TimeStepper>, Tags::Time>,
      time_stepper_ref_tags<TimeStepper>>(
      explicit_vars, std::move(history1), std::move(history2),
      static_cast<std::unique_ptr<TimeStepper>>(
          std::make_unique<TimeSteppers::Heun2>()),
      test_time);

  tuples::TaggedTuple<> unused_inboxes{};
  Parallel::GlobalCache<Metavariables> unused_cache{};
  const int unused_array_index{};
  const void* const unused_component{};
  CHECK(imex::ImplicitDenseOutput<System>::is_ready(
      make_not_null(&box), make_not_null(&unused_inboxes), unused_cache,
      unused_array_index, unused_component));

  auto expected = explicit_vars;
  const bool output_succeeded =
      db::get<Tags::TimeStepper<TimeStepper>>(box).dense_update_u(
          make_not_null(&expected), full_history, test_time);
  REQUIRE(output_succeeded);

  db::mutate_apply<imex::ImplicitDenseOutput<System>>(make_not_null(&box));

  CHECK_ITERABLE_APPROX(db::get<Var1>(box), get<Var1>(expected));
  CHECK_ITERABLE_APPROX(db::get<Var2>(box), get<Var2>(expected));
}
