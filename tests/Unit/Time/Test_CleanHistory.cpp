// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Time/CleanHistory.hpp"
#include "Time/CleanHistory.tpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Var : db::SimpleTag {
  using type = double;
};

struct AlternativeVar : db::SimpleTag {
  using type = double;
};

struct SingleVariableSystem {
  using variables_tag = Var;
};

struct TwoVariableSystem {
  using variables_tag = tmpl::list<Var, AlternativeVar>;
};

template <bool TwoVars>
void test() {
  using system =
      tmpl::conditional_t<TwoVars, TwoVariableSystem, SingleVariableSystem>;

  const Slab slab(1., 3.);
  TimeSteppers::History<double> history{2};
  history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
  history.insert(TimeStepId(true, 0, slab.end()), 0.0, 0.0);

  auto box = db::create<
      db::AddSimpleTags<Tags::ConcreteTimeStepper<TimeStepper>,
                        Tags::HistoryEvolvedVariables<Var>,
                        Tags::HistoryEvolvedVariables<AlternativeVar>>,
      time_stepper_ref_tags<TimeStepper>>(
      static_cast<std::unique_ptr<TimeStepper>>(
          std::make_unique<TimeSteppers::AdamsBashforth>(2)),
      history, history);

  db::mutate_apply<CleanHistory<system>>(make_not_null(&box));

  CHECK(db::get<Tags::HistoryEvolvedVariables<Var>>(box).size() == 1);
  CHECK(db::get<Tags::HistoryEvolvedVariables<AlternativeVar>>(box).size() ==
        (TwoVars ? 1 : 2));
}

SPECTRE_TEST_CASE("Unit.Time.CleanHistory", "[Unit][Time]") {
  test<false>();
  test<true>();
}
}  // namespace
