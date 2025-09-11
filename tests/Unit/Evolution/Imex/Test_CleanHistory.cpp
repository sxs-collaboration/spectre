// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Imex/CleanHistory.hpp"
#include "Evolution/Imex/Tags/ImplicitHistory.hpp"
#include "Helpers/Evolution/Imex/DoImplicitStepSector.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/Heun2.hpp"
#include "Time/TimeSteppers/ImexTimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename Var>
TimeSteppers::History<Variables<tmpl::list<Var>>> create_history() {
  const Slab slab(1., 3.);
  TimeSteppers::History<Variables<tmpl::list<Var>>> history{2};
  history.insert(TimeStepId(true, 0, slab.start()), {5, 0.0}, {5, 0.0});
  history.insert(
      TimeStepId(true, 0, slab.start(), 1, slab.duration(), slab.end().value()),
      {5, 0.0}, {5, 0.0});
  history.insert(TimeStepId(true, 1, slab.end()), {5, 0.0}, {5, 0.0});
  return history;
}

SPECTRE_TEST_CASE("Unit.Evolution.Imex.CleanHistory", "[Unit][Evolution]") {
  namespace helpers = do_implicit_step_helpers;

  auto box = db::create<
      db::AddSimpleTags<
          Tags::ConcreteTimeStepper<ImexTimeStepper>,
          imex::Tags::ImplicitHistory<helpers::Sector<helpers::Var1>>,
          imex::Tags::ImplicitHistory<helpers::Sector<helpers::Var2>>>,
      time_stepper_ref_tags<ImexTimeStepper>>(
      static_cast<std::unique_ptr<ImexTimeStepper>>(
          std::make_unique<TimeSteppers::Heun2>()),
      create_history<helpers::Var1>(), create_history<helpers::Var2>());

  db::mutate_apply<imex::CleanHistory<helpers::System>>(make_not_null(&box));

  CHECK(
      db::get<imex::Tags::ImplicitHistory<helpers::Sector<helpers::Var1>>>(box)
          .size() == 1);
  CHECK(
      db::get<imex::Tags::ImplicitHistory<helpers::Sector<helpers::Var2>>>(box)
          .size() == 1);
}
}  // namespace
