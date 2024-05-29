// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/MathWrapper.hpp"
#include "Time/History.hpp"
#include "Time/Time.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

/// \cond
#define IMEX_TIME_STEPPER_WRAPPED_TYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define IMEX_TIME_STEPPER_DERIVED_CLASS(data) BOOST_PP_TUPLE_ELEM(1, data)
/// \endcond

/*!
 * \ingroup TimeSteppersGroup
 *
 * Base class for TimeSteppers with IMEX support, derived from
 * TimeStepper.
 *
 * All supported time-stepping algorithms (both implicit and
 * explicit) consist of substeps that add linear combinations of
 * derivative values.  For implicit substeps, it is convenient to
 * split out the contribution of the final implicit term from
 * contributions from the history.  We therefore write an implicit
 * substep as
 *
 * \f{equation}{
 *   Y_{n+1} =
 *   Y_{n,\text{explicit}} + Y_{n,\text{inhomogeneous}} + w_n S(Y_{n+1})
 * \f}
 *
 * Here \f$Y_{n,\text{explicit}}\f$ is the value of the evolved
 * variables before the implicit substep is applied,
 * \f$Y_{n,\text{inhomogeneous}}\f$ is the contribution of the past
 * values from the implicit derivative history, and \f$S(\cdot)\f$ is
 * the implicit portion of the right-hand-side (generally source
 * terms for PDEs).  We call \f$w_n\f$ the *implicit weight*.
 * This split form is convenient for most consumers of this class, so
 * we present methods for calculating
 * \f$Y_{n,\text{inhomogeneous}}\f$ and \f$w_n\f$ individually
 * instead of a method to perform a full step update.
 *
 * History cleanup is the same for the explicit and implicit parts,
 * and should be done together.
 *
 * Dense output formulae are the same for the explicit and implicit
 * parts of any conservative IMEX stepper.  To evaluate dense output,
 * call `dense_update_u` with the implicit history after a successful
 * call to `dense_update_u` with the explicit history.
 *
 * Several of the member functions of this class are templated and
 * perform type erasure before forwarding their arguments to the
 * derived classes.  This is implemented using the macros \ref
 * IMEX_TIME_STEPPER_DECLARE_OVERLOADS, which must be placed in a
 * private section of the class body, and
 * IMEX_TIME_STEPPER_DEFINE_OVERLOADS(derived_class), which must be
 * placed in the cpp file.
 */
class ImexTimeStepper : public virtual TimeStepper {
 public:
  static constexpr bool imex = true;
  using provided_time_stepper_interfaces =
      tmpl::list<ImexTimeStepper, TimeStepper>;

  WRAPPED_PUPable_abstract(ImexTimeStepper);  // NOLINT

/// \cond
#define IMEX_TIME_STEPPER_DECLARE_VIRTUALS_IMPL(_, data)                      \
  virtual void add_inhomogeneous_implicit_terms_forward(                      \
      gsl::not_null<IMEX_TIME_STEPPER_WRAPPED_TYPE(data)*> u,                 \
      const TimeSteppers::ConstUntypedHistory<IMEX_TIME_STEPPER_WRAPPED_TYPE( \
          data)>& implicit_history,                                           \
      const TimeDelta& time_step) const = 0;                                  \
  virtual double implicit_weight_forward(                                     \
      const TimeSteppers::ConstUntypedHistory<IMEX_TIME_STEPPER_WRAPPED_TYPE( \
          data)>& implicit_history,                                           \
      const TimeDelta& time_step) const = 0;

  GENERATE_INSTANTIATIONS(IMEX_TIME_STEPPER_DECLARE_VIRTUALS_IMPL,
                          (MATH_WRAPPER_TYPES))
#undef IMEX_TIME_STEPPER_DECLARE_VIRTUALS_IMPL
  /// \endcond

  /// Convergence order of the integrator when used in IMEX mode.
  virtual size_t imex_order() const = 0;

  /// Add the change for the current implicit substep,
  /// \f$Y_{n,\text{inhomogeneous}}\f$, to u, given a past history of
  /// the implicit derivatives.
  ///
  /// Derived classes must implement this as a function with signature
  ///
  /// ```
  /// template <typename T>
  /// void add_inhomogeneous_implicit_terms_impl(
  ///     gsl::not_null<T*> u,
  ///     const ConstUntypedHistory<T>& implicit_history,
  ///     const TimeDelta& time_step) const;
  /// ```
  ///
  /// \note
  /// Unlike the `update_u` methods, which overwrite their result
  /// arguments, this function adds the result to the existing value.
  template <typename Vars>
  void add_inhomogeneous_implicit_terms(
      const gsl::not_null<Vars*> u,
      const TimeSteppers::History<Vars>& implicit_history,
      const TimeDelta& time_step) const {
    return add_inhomogeneous_implicit_terms_forward(
        &*make_math_wrapper(u), implicit_history.untyped(), time_step);
  }

  /// The coefficient \f$w_n\f$ of the implicit derivative for the
  /// current substep.  For a Runge-Kutta method, this is the
  /// coefficient on the diagonal of the Butcher tableau.
  ///
  /// Derived classes must implement this as a function with signature
  ///
  /// ```
  /// template <typename T>
  /// double implicit_weight_impl(
  ///     const ConstUntypedHistory<T>& implicit_history,
  ///     const TimeDelta& time_step) const;
  /// ```
  template <typename Vars>
  double implicit_weight(const TimeSteppers::History<Vars>& implicit_history,
                         const TimeDelta& time_step) const {
    return implicit_weight_forward(implicit_history.untyped(), time_step);
  }
};

/// \cond
#define IMEX_TIME_STEPPER_DECLARE_OVERLOADS_IMPL(_, data)                     \
  void add_inhomogeneous_implicit_terms_forward(                              \
      gsl::not_null<IMEX_TIME_STEPPER_WRAPPED_TYPE(data)*> u,                 \
      const TimeSteppers::ConstUntypedHistory<IMEX_TIME_STEPPER_WRAPPED_TYPE( \
          data)>& implicit_history,                                           \
      const TimeDelta& time_step) const override;                             \
  double implicit_weight_forward(                                             \
      const TimeSteppers::ConstUntypedHistory<IMEX_TIME_STEPPER_WRAPPED_TYPE( \
          data)>& implicit_history,                                           \
      const TimeDelta& time_step) const override;

#define IMEX_TIME_STEPPER_DEFINE_OVERLOADS_IMPL(_, data)                      \
  void IMEX_TIME_STEPPER_DERIVED_CLASS(data)::                                \
      add_inhomogeneous_implicit_terms_forward(                               \
          const gsl::not_null<IMEX_TIME_STEPPER_WRAPPED_TYPE(data)*> u,       \
          const TimeSteppers::ConstUntypedHistory<                            \
              IMEX_TIME_STEPPER_WRAPPED_TYPE(data)>& implicit_history,        \
          const TimeDelta& time_step) const {                                 \
    return add_inhomogeneous_implicit_terms_impl(u, implicit_history,         \
                                                 time_step);                  \
  }                                                                           \
  double IMEX_TIME_STEPPER_DERIVED_CLASS(data)::implicit_weight_forward(      \
      const TimeSteppers::ConstUntypedHistory<IMEX_TIME_STEPPER_WRAPPED_TYPE( \
          data)>& implicit_history,                                           \
      const TimeDelta& time_step) const {                                     \
    return implicit_weight_impl(implicit_history, time_step);                 \
  }
/// \endcond

/// \ingroup TimeSteppersGroup
/// Macro declaring overloaded detail methods in classes derived from
/// ImexTimeStepper.  Must be placed in a private section of the class
/// body.
#define IMEX_TIME_STEPPER_DECLARE_OVERLOADS                         \
  GENERATE_INSTANTIATIONS(IMEX_TIME_STEPPER_DECLARE_OVERLOADS_IMPL, \
                          (MATH_WRAPPER_TYPES))

/// \ingroup TimeSteppersGroup
/// Macro defining overloaded detail methods in classes derived from
/// ImexTimeStepper.  Must be placed in the cpp file for the derived
/// class.
#define IMEX_TIME_STEPPER_DEFINE_OVERLOADS(derived_class)          \
  GENERATE_INSTANTIATIONS(IMEX_TIME_STEPPER_DEFINE_OVERLOADS_IMPL, \
                          (MATH_WRAPPER_TYPES), (derived_class))
