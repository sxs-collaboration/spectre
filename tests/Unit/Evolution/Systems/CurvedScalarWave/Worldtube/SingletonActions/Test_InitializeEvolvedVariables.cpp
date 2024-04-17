// Distributed under the MIT License.
// See LICENSE.txt for details.
#include <array>
#include <memory>
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/InitializeEvolvedVariables.hpp"
#include "Framework/TestingFramework.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
namespace CurvedScalarWave::Worldtube {
namespace {
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.CSW.Worldtube.InitializeEvolvedVariables",
    "[Unit][Evolution]") {
  using variables_tag = ::Tags::Variables<
      tmpl::list<Tags::EvolvedPosition<3>, Tags::EvolvedVelocity<3>>>;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
  const tnsr::I<double, 3> initial_pos{{1., 2., 3.}};
  const tnsr::I<double, 3> initial_vel{{4., 5., 6.}};
  const size_t current_iteration = 77;
  auto box =
      db::create<db::AddSimpleTags<
                     variables_tag, dt_variables_tag,
                     ::Tags::HistoryEvolvedVariables<variables_tag>,
                     ::Tags::ConcreteTimeStepper<TimeStepper>,
                     Tags::InitialPositionAndVelocity, Tags::CurrentIteration>,
                 time_stepper_ref_tags<TimeStepper>>(
          variables_tag::type{}, dt_variables_tag::type{},
          TimeSteppers::History<variables_tag::type>{},
          static_cast<std::unique_ptr<TimeStepper>>(
              std::make_unique<TimeSteppers::AdamsBashforth>(4)),
          std::array<tnsr::I<double, 3>, 2>{{initial_pos, initial_vel}},
          current_iteration);

  db::mutate_apply<Initialization::InitializeEvolvedVariables>(
      make_not_null(&box));
  const auto vars = db::get<variables_tag>(box);
  for (size_t i = 0; i < 3; ++i) {
    CHECK(get<Tags::EvolvedPosition<3>>(vars).get(i)[0] == initial_pos.get(i));
    CHECK(get<Tags::EvolvedVelocity<3>>(vars).get(i)[0] == initial_vel.get(i));
  }
  CHECK(db::get<dt_variables_tag>(box) ==
        dt_variables_tag::type(size_t(1), 0.));
  CHECK(db::get<::Tags::HistoryEvolvedVariables<variables_tag>>(box) ==
        TimeSteppers::History<variables_tag::type>(1));
  CHECK(get<Tags::CurrentIteration>(box) == 0);
}
}  // namespace
}  // namespace CurvedScalarWave::Worldtube
