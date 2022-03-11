// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <pup.h>
#include <type_traits>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/MathWrapper.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/History.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class TimeDelta;
/// \endcond

/// \ingroup TimeSteppersGroup
///
/// Holds classes that take time steps.
namespace TimeSteppers {}

/// \cond
#define TIME_STEPPER_WRAPPED_TYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TIME_STEPPER_DERIVED_CLASS(data) BOOST_PP_TUPLE_ELEM(1, data)
/// \endcond

/// \ingroup TimeSteppersGroup
///
/// Abstract base class for TimeSteppers.
///
/// Several of the member functions of this class are templated and
/// perform type erasure before forwarding their arguments to the
/// derived classes.  This is implemented using the macros \ref
/// TIME_STEPPER_DECLARE_OVERLOADS, which must be placed in a private
/// section of the class body, and
/// TIME_STEPPER_DEFINE_OVERLOADS(derived_class), which must be placed
/// in the cpp file.
class TimeStepper : public PUP::able {
 public:
  WRAPPED_PUPable_abstract(TimeStepper);  // NOLINT

  /// \cond
#define TIME_STEPPER_DECLARE_VIRTUALS_IMPL(_, data)                        \
  virtual void update_u_forward(                                           \
      const gsl::not_null<TIME_STEPPER_WRAPPED_TYPE(data)*> u,             \
      const gsl::not_null<                                                 \
          TimeSteppers::UntypedHistory<TIME_STEPPER_WRAPPED_TYPE(data)>*>  \
          history,                                                         \
      const TimeDelta& time_step) const = 0;                               \
  virtual bool update_u_forward(                                           \
      const gsl::not_null<TIME_STEPPER_WRAPPED_TYPE(data)*> u,             \
      const gsl::not_null<TIME_STEPPER_WRAPPED_TYPE(data)*> u_error,       \
      const gsl::not_null<                                                 \
          TimeSteppers::UntypedHistory<TIME_STEPPER_WRAPPED_TYPE(data)>*>  \
          history,                                                         \
      const TimeDelta& time_step) const = 0;                               \
  virtual bool dense_update_u_forward(                                     \
      const gsl::not_null<TIME_STEPPER_WRAPPED_TYPE(data)*> u,             \
      const TimeSteppers::UntypedHistory<TIME_STEPPER_WRAPPED_TYPE(data)>& \
          history,                                                         \
      const double time) const = 0;                                        \
  virtual bool can_change_step_size_forward(                               \
      const TimeStepId& time_id,                                           \
      const TimeSteppers::UntypedHistory<TIME_STEPPER_WRAPPED_TYPE(data)>& \
          history) const = 0;

  GENERATE_INSTANTIATIONS(TIME_STEPPER_DECLARE_VIRTUALS_IMPL,
                          (MATH_WRAPPER_TYPES))
#undef TIME_STEPPER_DECLARE_VIRTUALS_IMPL
  /// \endcond

  /// Add the change for the current substep to u.
  ///
  /// Derived classes must implement this as a function with signature
  ///
  /// ```
  /// template <typename T>
  /// void update_u_impl(gsl::not_null<T*> u,
  ///                    gsl::not_null<UntypedHistory<T>*> history,
  ///                    const TimeDelta& time_step) const;
  /// ```
  template <typename Vars>
  void update_u(const gsl::not_null<Vars*> u,
                const gsl::not_null<TimeSteppers::History<Vars>*> history,
                const TimeDelta& time_step) const {
    return update_u_forward(&*make_math_wrapper(u), &*history, time_step);
  }

  /// Add the change for the current substep to u; report the error measure when
  /// available.
  ///
  /// For a substep method, the error measure will only be available on full
  /// steps. For a multistep method, the error measure will only be available
  /// when a sufficient number of steps are available in the `history` to
  /// compare two orders of step. Whenever the error measure is unavailable,
  /// `u_error` is unchanged and the function return is `false`.
  ///
  /// Derived classes must implement this as a function with signature
  ///
  /// ```
  /// template <typename T>
  /// bool update_u_impl(gsl::not_null<T*> u, gsl::not_null<T*> u_error,
  ///                    gsl::not_null<UntypedHistory<T>*> history,
  ///                    const TimeDelta& time_step) const;
  /// ```
  template <typename Vars, typename ErrVars>
  bool update_u(const gsl::not_null<Vars*> u,
                const gsl::not_null<ErrVars*> u_error,
                const gsl::not_null<TimeSteppers::History<Vars>*> history,
                const TimeDelta& time_step) const {
    static_assert(
        std::is_same_v<math_wrapper_type<Vars>, math_wrapper_type<ErrVars>>);
    return update_u_forward(&*make_math_wrapper(u),
                            &*make_math_wrapper(u_error), &*history, time_step);
  }

  /// Compute the solution value at a time between steps.  To evaluate
  /// at a time within a given step, call this method at the start of
  /// the step containing the time.  The function returns true on
  /// success, otherwise the call should be retried after the next
  /// substep.
  ///
  /// Derived classes must implement this as a function with signature
  ///
  /// ```
  /// template <typename T>
  /// bool dense_update_u_impl(
  ///     gsl::not_null<T*> u,
  ///     const UntypedHistory<T>& history, double time) const;
  /// ```
  template <typename Vars>
  bool dense_update_u(const gsl::not_null<Vars*> u,
                      const TimeSteppers::History<Vars>& history,
                      const double time) const {
    return dense_update_u_forward(&*make_math_wrapper(u), history, time);
  }

  /// The convergence order of the stepper
  virtual size_t order() const = 0;

  /// The convergence order of the stepper error measure
  virtual size_t error_estimate_order() const = 0;

  /// Number of substeps in this TimeStepper
  virtual uint64_t number_of_substeps() const = 0;

  /// Number of substeps in this TimeStepper when providing an error measure for
  /// adaptive time-stepping
  ///
  /// \details Certain substep methods (e.g. embedded RK4(3)) require additional
  /// steps when providing an error measure of the integration.
  virtual uint64_t number_of_substeps_for_error() const = 0;

  /// Number of past time steps needed for multi-step method
  virtual size_t number_of_past_steps() const = 0;

  /// Rough estimate of the maximum step size this method can take
  /// stably as a multiple of the step for Euler's method.
  virtual double stable_step() const = 0;

  /// The TimeStepId after the current substep
  virtual TimeStepId next_time_id(const TimeStepId& current_id,
                                  const TimeDelta& time_step) const = 0;

  /// The TimeStepId after the current substep when providing an error measure
  /// for adaptive time-stepping.
  ///
  /// Certain substep methods (e.g. embedded RK4(3)) require additional
  /// steps when providing an error measure of the integration.
  virtual TimeStepId next_time_id_for_error(
      const TimeStepId& current_id, const TimeDelta& time_step) const = 0;

  /// Whether a change in the step size is allowed before taking
  /// a step.
  ///
  /// Derived classes must implement this as a function with signature
  ///
  /// ```
  /// template <typename T>
  /// bool can_change_step_size_impl(const TimeStepId& time_id,
  ///                                const UntypedHistory<T>& history) const;
  /// ```
  template <typename Vars>
  bool can_change_step_size(const TimeStepId& time_id,
                            const TimeSteppers::History<Vars>& history) const {
    return can_change_step_size_forward(time_id, history);
  }
};

/// \cond
#define TIME_STEPPER_DECLARE_OVERLOADS_IMPL(_, data)                       \
  void update_u_forward(                                                   \
      gsl::not_null<TIME_STEPPER_WRAPPED_TYPE(data)*> u,                   \
      gsl::not_null<                                                       \
          TimeSteppers::UntypedHistory<TIME_STEPPER_WRAPPED_TYPE(data)>*>  \
          history,                                                         \
      const TimeDelta& time_step) const override;                          \
  bool update_u_forward(                                                   \
      gsl::not_null<TIME_STEPPER_WRAPPED_TYPE(data)*> u,                   \
      gsl::not_null<TIME_STEPPER_WRAPPED_TYPE(data)*> u_error,             \
      gsl::not_null<                                                       \
          TimeSteppers::UntypedHistory<TIME_STEPPER_WRAPPED_TYPE(data)>*>  \
          history,                                                         \
      const TimeDelta& time_step) const override;                          \
  bool dense_update_u_forward(                                             \
      gsl::not_null<TIME_STEPPER_WRAPPED_TYPE(data)*> u,                   \
      const TimeSteppers::UntypedHistory<TIME_STEPPER_WRAPPED_TYPE(data)>& \
          history,                                                         \
      double time) const override;                                         \
  bool can_change_step_size_forward(                                       \
      const TimeStepId& time_id,                                           \
      const TimeSteppers::UntypedHistory<TIME_STEPPER_WRAPPED_TYPE(data)>& \
          history) const override;

#define TIME_STEPPER_DEFINE_OVERLOADS_IMPL(_, data)                        \
  void TIME_STEPPER_DERIVED_CLASS(data)::update_u_forward(                 \
      const gsl::not_null<TIME_STEPPER_WRAPPED_TYPE(data)*> u,             \
      const gsl::not_null<                                                 \
          TimeSteppers::UntypedHistory<TIME_STEPPER_WRAPPED_TYPE(data)>*>  \
          history,                                                         \
      const TimeDelta& time_step) const {                                  \
    return update_u_impl(u, history, time_step);                           \
  }                                                                        \
  bool TIME_STEPPER_DERIVED_CLASS(data)::update_u_forward(                 \
      const gsl::not_null<TIME_STEPPER_WRAPPED_TYPE(data)*> u,             \
      const gsl::not_null<TIME_STEPPER_WRAPPED_TYPE(data)*> u_error,       \
      const gsl::not_null<                                                 \
          TimeSteppers::UntypedHistory<TIME_STEPPER_WRAPPED_TYPE(data)>*>  \
          history,                                                         \
      const TimeDelta& time_step) const {                                  \
    return update_u_impl(u, u_error, history, time_step);                  \
  }                                                                        \
  bool TIME_STEPPER_DERIVED_CLASS(data)::dense_update_u_forward(           \
      const gsl::not_null<TIME_STEPPER_WRAPPED_TYPE(data)*> u,             \
      const TimeSteppers::UntypedHistory<TIME_STEPPER_WRAPPED_TYPE(data)>& \
          history,                                                         \
      const double time) const {                                           \
    return dense_update_u_impl(u, history, time);                          \
  }                                                                        \
  bool TIME_STEPPER_DERIVED_CLASS(data)::can_change_step_size_forward(     \
      const TimeStepId& time_id,                                           \
      const TimeSteppers::UntypedHistory<TIME_STEPPER_WRAPPED_TYPE(data)>& \
          history) const {                                                 \
    return can_change_step_size_impl(time_id, history);                    \
  }
/// \endcond

/// \ingroup TimeSteppersGroup
/// Macro declaring overloaded detail methods in classes derived from
/// TimeStepper.  Must be placed in a private section of the class
/// body.
#define TIME_STEPPER_DECLARE_OVERLOADS                         \
  GENERATE_INSTANTIATIONS(TIME_STEPPER_DECLARE_OVERLOADS_IMPL, \
                          (MATH_WRAPPER_TYPES))

/// \ingroup TimeSteppersGroup
/// Macro defining overloaded detail methods in classes derived from
/// TimeStepper.  Must be placed in the cpp file for the derived
/// class.
#define TIME_STEPPER_DEFINE_OVERLOADS(derived_class)          \
  GENERATE_INSTANTIATIONS(TIME_STEPPER_DEFINE_OVERLOADS_IMPL, \
                          (MATH_WRAPPER_TYPES), (derived_class))

// LtsTimeStepper cannot be split out into its own file because the
// LtsTimeStepper -> TimeStepper -> AdamsBashforthN -> LtsTimeStepper
// include loop causes AdamsBashforthN to be defined before its base
// class if LtsTimeStepper is included first.

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
#define LTS_TIME_STEPPER_DECLARE_VIRTUALS_IMPL(_, data)                       \
  virtual void add_boundary_delta_forward(                                    \
      gsl::not_null<TIME_STEPPER_WRAPPED_TYPE(data)*> result,                 \
      const TimeSteppers::BoundaryHistoryEvaluator<TIME_STEPPER_WRAPPED_TYPE( \
          data)>& coupling,                                                   \
      const TimeSteppers::BoundaryHistoryCleaner& cleaner,                    \
      const TimeDelta& time_step) const = 0;                                  \
  virtual void boundary_dense_output_forward(                                 \
      gsl::not_null<TIME_STEPPER_WRAPPED_TYPE(data)*> result,                 \
      const TimeSteppers::BoundaryHistoryEvaluator<TIME_STEPPER_WRAPPED_TYPE( \
          data)>& coupling,                                                   \
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
#define LTS_TIME_STEPPER_DECLARE_OVERLOADS_IMPL(_, data)                      \
  void add_boundary_delta_forward(                                            \
      gsl::not_null<TIME_STEPPER_WRAPPED_TYPE(data)*> result,                 \
      const TimeSteppers::BoundaryHistoryEvaluator<TIME_STEPPER_WRAPPED_TYPE( \
          data)>& coupling,                                                   \
      const TimeSteppers::BoundaryHistoryCleaner& cleaner,                    \
      const TimeDelta& time_step) const override;                             \
  void boundary_dense_output_forward(                                         \
      gsl::not_null<TIME_STEPPER_WRAPPED_TYPE(data)*> result,                 \
      const TimeSteppers::BoundaryHistoryEvaluator<TIME_STEPPER_WRAPPED_TYPE( \
          data)>& coupling,                                                   \
      const double time) const override;

#define LTS_TIME_STEPPER_DEFINE_OVERLOADS_IMPL(_, data)                       \
  void TIME_STEPPER_DERIVED_CLASS(data)::add_boundary_delta_forward(          \
      const gsl::not_null<TIME_STEPPER_WRAPPED_TYPE(data)*> result,           \
      const TimeSteppers::BoundaryHistoryEvaluator<TIME_STEPPER_WRAPPED_TYPE( \
          data)>& coupling,                                                   \
      const TimeSteppers::BoundaryHistoryCleaner& cleaner,                    \
      const TimeDelta& time_step) const {                                     \
    return add_boundary_delta_impl(result, coupling, cleaner, time_step);     \
  }                                                                           \
  void TIME_STEPPER_DERIVED_CLASS(data)::boundary_dense_output_forward(       \
      const gsl::not_null<TIME_STEPPER_WRAPPED_TYPE(data)*> result,           \
      const TimeSteppers::BoundaryHistoryEvaluator<TIME_STEPPER_WRAPPED_TYPE( \
          data)>& coupling,                                                   \
      const double time) const {                                              \
    return boundary_dense_output_impl(result, coupling, time);                \
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
