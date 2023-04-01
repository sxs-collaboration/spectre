// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/InitializeEvolvedVariables.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"

namespace CurvedScalarWave::Worldtube {
namespace {

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.CSW.Worldtube.InitializeEvolvedVariables",
    "[Unit][Evolution]") {
  using variables_tag = ::Tags::Variables<tmpl::list<Tags::Psi0, Tags::dtPsi0>>;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;

  auto box = db::create<
      db::AddSimpleTags<variables_tag, dt_variables_tag,
                        ::Tags::HistoryEvolvedVariables<variables_tag>,
                        ::Tags::TimeStepper<TimeSteppers::AdamsBashforth>>>(
      variables_tag::type{}, dt_variables_tag::type{},
      TimeSteppers::History<variables_tag::type>{},
      std::make_unique<TimeSteppers::AdamsBashforth>(4));

  db::mutate_apply<Initialization::InitializeEvolvedVariables>(
      make_not_null(&box));
  CHECK(db::get<variables_tag>(box) == variables_tag::type(size_t(1), 0.));
  CHECK(db::get<dt_variables_tag>(box) ==
        dt_variables_tag::type(size_t(1), 0.));
  CHECK(db::get<::Tags::HistoryEvolvedVariables<variables_tag>>(box) ==
        TimeSteppers::History<variables_tag::type>(1));
}
}  // namespace
}  // namespace CurvedScalarWave::Worldtube
