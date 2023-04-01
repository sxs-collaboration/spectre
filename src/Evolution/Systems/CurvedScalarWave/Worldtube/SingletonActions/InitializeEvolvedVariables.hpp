// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube::Initialization {
/*!
 * \brief Initializes the time stepper and evolved variables used by the
 * worldtube system.
 *
 * \details Sets `Tags::Psi0` and `::Tags::dt<Tags::Psi0>` to size 1 with
 * initial value 0. The time stepper history is set analogous to the elements
 * which use the same time stepper.
 */
struct InitializeEvolvedVariables {
  using variables_tag = ::Tags::Variables<tmpl::list<Tags::Psi0, Tags::dtPsi0>>;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;

  using simple_tags =
      tmpl::list<variables_tag, dt_variables_tag,
                 ::Tags::HistoryEvolvedVariables<variables_tag>>;
  using return_tags = simple_tags;

  using compute_tags = tmpl::list<>;
  using simple_tags_from_options = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  using mutable_global_cache_tags = tmpl::list<>;
  using argument_tags = tmpl::list<::Tags::TimeStepper<>>;
  static void apply(
      const gsl::not_null<Variables<tmpl::list<Tags::Psi0, Tags::dtPsi0>>*>
          evolved_vars,
      const gsl::not_null<Variables<
          tmpl::list<::Tags::dt<Tags::Psi0>, ::Tags::dt<Tags::dtPsi0>>>*>
          dt_evolved_vars,
      const gsl::not_null<::Tags::HistoryEvolvedVariables<variables_tag>::type*>
          time_stepper_history,
      const TimeStepper& time_stepper) {
    const size_t starting_order =
        time_stepper.number_of_past_steps() == 0 ? time_stepper.order() : 1;
    *time_stepper_history =
        typename ::Tags::HistoryEvolvedVariables<variables_tag>::type{
            starting_order};
    evolved_vars->initialize(size_t(1), 0.);
    dt_evolved_vars->initialize(size_t(1), 0.);
  }
};
}  // namespace CurvedScalarWave::Worldtube::Initialization
