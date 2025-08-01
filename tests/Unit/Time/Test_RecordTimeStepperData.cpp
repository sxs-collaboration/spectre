// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Time/RecordTimeStepperData.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/TimeStepId.hpp"
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

template <typename System, bool AlternativeUpdates>
void run_test() {
  using history_tag = Tags::HistoryEvolvedVariables<Var>;
  using alternative_history_tag = Tags::HistoryEvolvedVariables<AlternativeVar>;

  const Slab slab(1., 3.);
  const TimeStepId slab_start_id(true, 0, slab.start());
  const TimeStepId slab_end_id(true, 0, slab.end());

  typename history_tag::type history{};
  history.insert(slab_start_id, -3., 3.);
  typename alternative_history_tag::type alternative_history{};
  alternative_history.insert(slab_start_id, -3., 3.);

  const double initial_value = 4.;
  auto box = db::create<
      db::AddSimpleTags<Tags::TimeStepId, Var, ::Tags::dt<Var>,
                        Tags::HistoryEvolvedVariables<Var>, AlternativeVar,
                        ::Tags::dt<AlternativeVar>,
                        Tags::HistoryEvolvedVariables<AlternativeVar>>>(
      slab_end_id, initial_value, 5., std::move(history), initial_value, 5.,
      std::move(alternative_history));

  db::mutate_apply<RecordTimeStepperData<System>>(make_not_null(&box));

  const auto check_history = [&initial_value, &slab_end_id,
                              &slab_start_id](const auto& updated_history) {
    CHECK(updated_history.size() == 2);
    CHECK(updated_history[0].time_step_id == slab_start_id);
    CHECK(updated_history[0].value == std::optional{-3.});
    CHECK(updated_history[0].derivative == 3.);
    CHECK(updated_history[1].time_step_id == slab_end_id);
    CHECK(updated_history[1].value == std::optional{initial_value});
    CHECK(updated_history[1].derivative == 5.);
  };
  check_history(db::get<history_tag>(box));
  if (AlternativeUpdates) {
    check_history(db::get<alternative_history_tag>(box));
  }
}

SPECTRE_TEST_CASE("Unit.Time.RecordTimeStepperData", "[Unit][Time]") {
  run_test<SingleVariableSystem, false>();
  run_test<TwoVariableSystem, true>();
}
}  // namespace
