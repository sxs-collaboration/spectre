// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <pup.h>
#include <type_traits>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/MathWrapper.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

/// \cond
class TimeDelta;
class TimeStepId;
/// \endcond

/// \cond
#define LTS_TIME_STEPPER_WRAPPED_TYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define LTS_TIME_STEPPER_DERIVED_CLASS(data) BOOST_PP_TUPLE_ELEM(1, data)
/// \endcond

/// \ingroup TimeSteppersGroup
///
/// Base class for TimeSteppers with local time-stepping support,
/// derived from TimeStepper.
///
/// Several of the member functions of this class are templated and
/// perform type erasure before forwarding their arguments to the
/// derived classes.  This is implemented using the macros \ref
/// LTS_TIME_STEPPER_DECLARE_OVERLOADS, which must be placed in a
/// private section of the class body, and
/// LTS_TIME_STEPPER_DEFINE_OVERLOADS(derived_class), which must be
/// placed in the cpp file.
class LtsTimeStepper : public TimeStepper {
 public:
  WRAPPED_PUPable_abstract(LtsTimeStepper);  // NOLINT

  // These two are defined as separate type aliases to keep the
  // doxygen page width somewhat under control.
  template <typename LocalVars, typename RemoteVars, typename Coupling>
  using BoundaryHistoryType = TimeSteppers::BoundaryHistory<
      LocalVars, RemoteVars,
      std::result_of_t<const Coupling&(LocalVars, RemoteVars)>>;

  /// Return type of boundary-related functions.  The coupling returns
  /// the derivative of the variables, but this is multiplied by the
  /// time step so the return type should not have `dt` prefixes.
  template <typename LocalVars, typename RemoteVars, typename Coupling>
  using BoundaryReturn = db::unprefix_variables<
      std::result_of_t<const Coupling&(LocalVars, RemoteVars)>>;

/// \cond
#define LTS_TIME_STEPPER_DECLARE_VIRTUALS_IMPL(_, data)           \
  virtual void add_boundary_delta_forward(                        \
      gsl::not_null<LTS_TIME_STEPPER_WRAPPED_TYPE(data)*> result, \
      const TimeSteppers::BoundaryHistoryEvaluator<               \
          LTS_TIME_STEPPER_WRAPPED_TYPE(data)>& coupling,         \
      const TimeSteppers::BoundaryHistoryCleaner& cleaner,        \
      const TimeDelta& time_step) const = 0;                      \
  virtual void boundary_dense_output_forward(                     \
      gsl::not_null<LTS_TIME_STEPPER_WRAPPED_TYPE(data)*> result, \
      const TimeSteppers::BoundaryHistoryEvaluator<               \
          LTS_TIME_STEPPER_WRAPPED_TYPE(data)>& coupling,         \
      const double time) const = 0;

  GENERATE_INSTANTIATIONS(LTS_TIME_STEPPER_DECLARE_VIRTUALS_IMPL,
                          (MATH_WRAPPER_TYPES))
#undef LTS_TIME_STEPPER_DECLARE_VIRTUALS_IMPL
  /// \endcond

  /// \brief Compute the change in a boundary quantity due to the
  /// coupling on the interface.
  ///
  /// The coupling function `coupling` should take the local and
  /// remote flux data and compute the derivative of the boundary
  /// quantity.  These values may be used to form a linear combination
  /// internally, so the result should have appropriate mathematical
  /// operators defined to allow that.
  ///
  /// Derived classes must implement this as a function with signature
  ///
  /// ```
  /// template <typename T>
  /// void add_boundary_delta_impl(
  ///     gsl::not_null<T*> result,
  ///     const TimeSteppers::BoundaryHistoryEvaluator<T>& coupling,
  ///     const TimeSteppers::BoundaryHistoryCleaner& cleaner,
  ///     const TimeDelta& time_step) const;
  /// ```
  ///
  /// \note
  /// Unlike the `update_u` methods, which overwrite the `result`
  /// argument, this function adds the result to the existing value.
  template <typename LocalVars, typename RemoteVars, typename Coupling>
  void add_boundary_delta(
      const gsl::not_null<BoundaryReturn<LocalVars, RemoteVars, Coupling>*>
          result,
      const gsl::not_null<BoundaryHistoryType<LocalVars, RemoteVars, Coupling>*>
          history,
      const TimeDelta& time_step, const Coupling& coupling) const {
    return add_boundary_delta_forward(&*make_math_wrapper(result),
                                      history->evaluator(coupling),
                                      history->cleaner(), time_step);
  }

  /// Derived classes must implement this as a function with signature
  ///
  /// ```
  /// template <typename T>
  /// void boundary_dense_output_impl(
  ///     gsl::not_null<T*> result,
  ///     const TimeSteppers::BoundaryHistoryEvaluator<T>& coupling,
  ///     const double time) const;
  /// ```
  template <typename LocalVars, typename RemoteVars, typename Coupling>
  void boundary_dense_output(
      const gsl::not_null<BoundaryReturn<LocalVars, RemoteVars, Coupling>*>
          result,
      const BoundaryHistoryType<LocalVars, RemoteVars, Coupling>& history,
      const double time, const Coupling& coupling) const {
    return boundary_dense_output_forward(&*make_math_wrapper(result),
                                         history.evaluator(coupling), time);
  }

  /// Substep LTS integrators are not supported, so this is always 1.
  uint64_t number_of_substeps() const final { return 1; }

  /// Substep LTS integrators are not supported, so this is always 1.
  uint64_t number_of_substeps_for_error() const final { return 1; }

  TimeStepId next_time_id_for_error(const TimeStepId& current_id,
                                    const TimeDelta& time_step) const final {
    return next_time_id(current_id, time_step);
  }
};

/// \cond
#define LTS_TIME_STEPPER_DECLARE_OVERLOADS_IMPL(_, data)          \
  void add_boundary_delta_forward(                                \
      gsl::not_null<LTS_TIME_STEPPER_WRAPPED_TYPE(data)*> result, \
      const TimeSteppers::BoundaryHistoryEvaluator<               \
          LTS_TIME_STEPPER_WRAPPED_TYPE(data)>& coupling,         \
      const TimeSteppers::BoundaryHistoryCleaner& cleaner,        \
      const TimeDelta& time_step) const override;                 \
  void boundary_dense_output_forward(                             \
      gsl::not_null<LTS_TIME_STEPPER_WRAPPED_TYPE(data)*> result, \
      const TimeSteppers::BoundaryHistoryEvaluator<               \
          LTS_TIME_STEPPER_WRAPPED_TYPE(data)>& coupling,         \
      const double time) const override;

#define LTS_TIME_STEPPER_DEFINE_OVERLOADS_IMPL(_, data)                     \
  void LTS_TIME_STEPPER_DERIVED_CLASS(data)::add_boundary_delta_forward(    \
      const gsl::not_null<LTS_TIME_STEPPER_WRAPPED_TYPE(data)*> result,     \
      const TimeSteppers::BoundaryHistoryEvaluator<                         \
          LTS_TIME_STEPPER_WRAPPED_TYPE(data)>& coupling,                   \
      const TimeSteppers::BoundaryHistoryCleaner& cleaner,                  \
      const TimeDelta& time_step) const {                                   \
    return add_boundary_delta_impl(result, coupling, cleaner, time_step);   \
  }                                                                         \
  void LTS_TIME_STEPPER_DERIVED_CLASS(data)::boundary_dense_output_forward( \
      const gsl::not_null<LTS_TIME_STEPPER_WRAPPED_TYPE(data)*> result,     \
      const TimeSteppers::BoundaryHistoryEvaluator<                         \
          LTS_TIME_STEPPER_WRAPPED_TYPE(data)>& coupling,                   \
      const double time) const {                                            \
    return boundary_dense_output_impl(result, coupling, time);              \
  }
/// \endcond

/// \ingroup TimeSteppersGroup
/// Macro declaring overloaded detail methods in classes derived from
/// TimeStepper.  Must be placed in a private section of the class
/// body.
#define LTS_TIME_STEPPER_DECLARE_OVERLOADS                         \
  GENERATE_INSTANTIATIONS(LTS_TIME_STEPPER_DECLARE_OVERLOADS_IMPL, \
                          (MATH_WRAPPER_TYPES))

/// \ingroup TimeSteppersGroup
/// Macro defining overloaded detail methods in classes derived from
/// TimeStepper.  Must be placed in the cpp file for the derived
/// class.
#define LTS_TIME_STEPPER_DEFINE_OVERLOADS(derived_class)          \
  GENERATE_INSTANTIATIONS(LTS_TIME_STEPPER_DEFINE_OVERLOADS_IMPL, \
                          (MATH_WRAPPER_TYPES), (derived_class))
