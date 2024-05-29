// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>
#include <type_traits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Imex/Mode.hpp"
#include "Evolution/Imex/Protocols/ImplicitSector.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
class ImexTimeStepper;
class TimeDelta;
template <typename TagsList>
class Variables;
namespace Tags {
struct TimeStep;
template <typename StepperInterface>
struct TimeStepper;
}  // namespace Tags
namespace TimeSteppers {
template <typename Vars>
class History;
}  // namespace TimeSteppers
namespace imex::Tags {
template <typename ImplicitSector>
struct ImplicitHistory;
struct Mode;
template <typename Sector>
struct SolveFailures;
struct SolveTolerance;
}  // namespace imex::Tags
/// \endcond

namespace imex {
namespace solve_implicit_sector_detail {
template <typename Tags>
using ForwardTuple = tmpl::wrap<
    tmpl::transform<Tags, std::add_lvalue_reference<std::add_const<
                              tmpl::bind<tmpl::type_from, tmpl::_1>>>>,
    std::tuple>;
}  // namespace solve_implicit_sector_detail

/// Perform the implicit solve for one implicit sector.
///
/// This will update the tensors in the implicit sector and clean up
/// the corresponding time stepper history.  A new history entry is
/// not added, because that should be done with the same values of the
/// variables used for the explicit portion of the time derivative,
/// which may still undergo variable-fixing-like corrections.
///
/// \warning
/// This will use the value of `::Tags::Time` from the DataBox.  Most
/// of the time, the value appropriate for evaluating the explicit RHS
/// is stored there, so it will likely need to be set to the
/// appropriate value for the implicit RHS for the duration of this
/// mutation.
template <typename SystemVariablesTag, typename ImplicitSector>
struct SolveImplicitSector {
  static_assert(
      tt::assert_conforms_to_v<ImplicitSector, protocols::ImplicitSector>);

 public:
  using SystemVariables = typename SystemVariablesTag::type;
  using SectorVariables = Variables<typename ImplicitSector::tensors>;

 private:
  template <typename Attempt>
  struct get_tags_from_evolution {
    using type = typename Attempt::tags_from_evolution;
  };

  using tags_for_each_attempt =
      tmpl::transform<typename ImplicitSector::solve_attempts,
                      get_tags_from_evolution<tmpl::_1>>;
  // List of tags used for the initial guess followed by lists of tags
  // used for each solve attempt.
  using evolution_data_tags =
      tmpl::push_front<tags_for_each_attempt,
                       typename ImplicitSector::initial_guess::argument_tags>;

  using EvolutionDataTuple = solve_implicit_sector_detail::ForwardTuple<
      tmpl::join<evolution_data_tags>>;

  static void apply_impl(
      gsl::not_null<SystemVariables*> system_variables,
      gsl::not_null<Scalar<DataVector>*> solve_failures,
      const ImexTimeStepper& time_stepper, const TimeDelta& time_step,
      const TimeSteppers::History<SectorVariables>& implicit_history,
      Mode implicit_solve_mode, double implicit_solve_tolerance,
      const EvolutionDataTuple& joined_evolution_data);

 public:
  using return_tags =
      tmpl::list<SystemVariablesTag, Tags::SolveFailures<ImplicitSector>>;
  using argument_tags = tmpl::append<
      tmpl::list<::Tags::TimeStepper<ImexTimeStepper>, ::Tags::TimeStep,
                 imex::Tags::ImplicitHistory<ImplicitSector>, Tags::Mode,
                 Tags::SolveTolerance>,
      tmpl::join<evolution_data_tags>>;

  template <typename... ForwardArgs>
  static void apply(
      const gsl::not_null<SystemVariables*> system_variables,
      const gsl::not_null<Scalar<DataVector>*> solve_failures,
      const ImexTimeStepper& time_stepper, const TimeDelta& time_step,
      const TimeSteppers::History<SectorVariables>& implicit_history,
      const Mode implicit_solve_mode, const double implicit_solve_tolerance,
      const ForwardArgs&... forward_args) {
    apply_impl(system_variables, solve_failures, time_stepper, time_step,
               implicit_history, implicit_solve_mode, implicit_solve_tolerance,
               std::forward_as_tuple(forward_args...));
  }
};
}  // namespace imex
