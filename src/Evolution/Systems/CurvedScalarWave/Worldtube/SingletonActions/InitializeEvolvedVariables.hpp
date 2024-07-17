// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once
#include <cstddef>
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
/// \cond
namespace Tags {
template <typename StepperInterface>
struct TimeStepper;
}  // namespace Tags
/// \endcond
namespace CurvedScalarWave::Worldtube::Initialization {
/*!
 * \brief Initializes the time stepper and evolved variables used by the
 * worldtube system. Also sets `Tags::CurrentIteration` to 0.
 *
 * \details Sets the initial position and velocity of the particle to the values
 * specified in the input file. The time stepper history is set analogous to the
 * elements which use the same time stepper.
 */
struct InitializeEvolvedVariables {
  static constexpr size_t Dim = 3;
  using variables_tag = ::Tags::Variables<
      tmpl::list<Tags::EvolvedPosition<Dim>, Tags::EvolvedVelocity<Dim>>>;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;

  using simple_tags =
      tmpl::list<variables_tag, dt_variables_tag, Tags::CurrentIteration,
                 Tags::ExpirationTime, Tags::WorldtubeRadius,
                 ::Tags::HistoryEvolvedVariables<variables_tag>>;
  using return_tags = simple_tags;

  using compute_tags = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  using mutable_global_cache_tags = tmpl::list<>;
  using simple_tags_from_options = tmpl::list<Tags::InitialPositionAndVelocity>;
  using argument_tags = tmpl::list<::Tags::TimeStepper<TimeStepper>,
                                   Tags::InitialPositionAndVelocity,
                                   ::Tags::Time, Tags::ExcisionSphere<Dim>>;
  static void apply(
      const gsl::not_null<Variables<
          tmpl::list<Tags::EvolvedPosition<Dim>, Tags::EvolvedVelocity<Dim>>>*>
          evolved_vars,
      const gsl::not_null<
          Variables<tmpl::list<::Tags::dt<Tags::EvolvedPosition<Dim>>,
                               ::Tags::dt<Tags::EvolvedVelocity<Dim>>>>*>
          dt_evolved_vars,
      const gsl::not_null<size_t*> current_iteration,
      const gsl::not_null<double*> expiration_time,
      const gsl::not_null<double*> worldtube_radius,
      const gsl::not_null<::Tags::HistoryEvolvedVariables<variables_tag>::type*>
          time_stepper_history,
      const TimeStepper& time_stepper,
      const std::array<tnsr::I<double, Dim>, 2>& initial_pos_and_vel,
      const double initial_time, const ExcisionSphere<Dim>& excision_sphere) {
    *current_iteration = 0;

    // the functions of time should be ready during the first step but not the
    // second. We choose the arbitrary value of 1e-10 here to ensure this.
    *expiration_time = initial_time + 1e-10;
    *worldtube_radius = excision_sphere.radius();

    const size_t starting_order =
        visit(
            []<typename Tag>(
                const std::pair<tmpl::type_<Tag>, typename Tag::type&&> order) {
              if constexpr (std::is_same_v<Tag,
                                           TimeSteppers::Tags::FixedOrder>) {
                return order.second;
              } else {
                return order.second.minimum;
              }
            },
            time_stepper.order()) -
        time_stepper.number_of_past_steps();
    *time_stepper_history =
        typename ::Tags::HistoryEvolvedVariables<variables_tag>::type{
            starting_order};
    evolved_vars->initialize(size_t(1), 0.);
    dt_evolved_vars->initialize(size_t(1), 0.);
    for (size_t i = 0; i < Dim; ++i) {
      get<Tags::EvolvedPosition<Dim>>(*evolved_vars).get(i)[0] =
          initial_pos_and_vel.at(0).get(i);
      get<Tags::EvolvedVelocity<Dim>>(*evolved_vars).get(i)[0] =
          initial_pos_and_vel.at(1).get(i);
    }
  }
};
}  // namespace CurvedScalarWave::Worldtube::Initialization
