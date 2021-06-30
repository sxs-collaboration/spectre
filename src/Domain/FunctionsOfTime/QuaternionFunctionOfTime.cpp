// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"

// BoostMultiArray is used internally in odeint, so BoostMultiArray MUST be
// included before odeint
#include "DataStructures/BoostMultiArray.hpp"

#include <boost/numeric/odeint.hpp>
#include <pup_stl.h>

#include "Domain/FunctionsOfTime/QuaternionHelpers.hpp"
#include "NumericalAlgorithms/OdeIntegration/OdeIntegration.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdHelpers.hpp"

namespace domain::FunctionsOfTime {
template <size_t MaxDerivReturned>
QuaternionFunctionOfTime<MaxDerivReturned>::QuaternionFunctionOfTime(
    double t, std::array<DataVector, 1> initial_func,
    gsl::not_null<
        domain::FunctionsOfTime::PiecewisePolynomial<MaxDerivReturned>*>
        omega_f_of_t_ptr,
    double expiration_time) noexcept
    : stored_quaternions_and_times_{{t, std::move(initial_func)}},
      omega_f_of_t_ptr_(omega_f_of_t_ptr),
      expiration_time_(expiration_time) {}

template <size_t MaxDerivReturned>
std::unique_ptr<FunctionOfTime>
QuaternionFunctionOfTime<MaxDerivReturned>::get_clone() const noexcept {
  return std::make_unique<QuaternionFunctionOfTime>(*this);
}

template <size_t MaxDerivReturned>
void QuaternionFunctionOfTime<MaxDerivReturned>::pup(PUP::er& p) {
  FunctionOfTime::pup(p);
  p | stored_quaternions_and_times_;
  p | expiration_time_;
  if (p.isUnpacking()) {
    omega_f_of_t_ptr_ =
        new domain::FunctionsOfTime::PiecewisePolynomial<MaxDerivReturned>;
  }
  p | omega_f_of_t_ptr_;
}

template <size_t MaxDerivReturned>
void QuaternionFunctionOfTime<MaxDerivReturned>::update_stored_info()
    const noexcept {
  const auto& omega_deriv_info = omega_f_of_t_ptr_->get_deriv_info();

  ASSERT(omega_deriv_info.size() >= stored_quaternions_and_times_.size(),
         "There are more stored quaternions than there are stored omegas. "
         "Currently there are "
             << omega_deriv_info.size() << " stored omegas and "
             << stored_quaternions_and_times_.size()
             << " stored quaternions. There must be at least the same number "
                "of stored quaternions as omegas otherwise we cannot solve the "
                "ode and populate the missing quaternions.");

  // If they are already the same size, we don't have to update anything so exit
  // now
  if (omega_deriv_info.size() == stored_quaternions_and_times_.size()) {
    return;
  }

  // Will store the solution of the ode
  std::array<DataVector, 1> quaternion_to_integrate;
  for (size_t i = stored_quaternions_and_times_.size();
       i < omega_deriv_info.size(); i++) {
    // Final time, initial time, and quaternion
    const double t = omega_deriv_info[i].time;
    const double t0 = omega_deriv_info[i - 1].time;
    quaternion_to_integrate =
        stored_quaternions_and_times_[i - 1].stored_quantities;

    solve_quaternion_ode(&quaternion_to_integrate, t, t0);

    // normalize quaternion
    boost::math::quaternion<double> quat =
        datavector_to_quaternion(quaternion_to_integrate[0]);
    normalize_quaternion(make_not_null(&quat));

    stored_quaternions_and_times_.emplace_back(
      //  FunctionOfTimeHelpers::StoredInfo<1, false>{
            t, std::array<DataVector, 1>{quaternion_to_datavector(quat)});
  }

  ASSERT(omega_deriv_info.size() == stored_quaternions_and_times_.size(),
         "The number of stored omegas must be the same as the number of stored "
         "quaternions after updating the missing quaternions. Now there are "
             << omega_deriv_info.size() << " stored omegas and "
             << stored_quaternions_and_times_.size() << " stored quaternions.");
}

template <size_t MaxDerivReturned>
void QuaternionFunctionOfTime<MaxDerivReturned>::solve_quaternion_ode(
    const gsl::not_null<std::array<DataVector, 1>*> quaternion_to_integrate,
    const double t, const double t0) const noexcept {
  // lambda that stores the internals of the ode
  const auto quaternion_ode_system =
      [this](const std::array<DataVector, 1>& state,
             std::array<DataVector, 1>& dt_state, const double time) noexcept {
        const std::array<DataVector, 1> omega = omega_f_of_t_ptr_->func(time);
        dt_state[0] = DataVector{
            {-0.5 * (state[0][1] * omega[0][0] + state[0][2] * omega[0][1] +
                     state[0][3] * omega[0][2]),
             0.5 * (state[0][0] * omega[0][0] + state[0][2] * omega[0][2] -
                    state[0][3] * omega[0][1]),
             0.5 * (state[0][0] * omega[0][1] + state[0][3] * omega[0][0] -
                    state[0][1] * omega[0][2]),
             0.5 * (state[0][0] * omega[0][2] + state[0][1] * omega[0][1] -
                    state[0][2] * omega[0][0])}};
      };

  // Dense stepper
  boost::numeric::odeint::dense_output_runge_kutta<
      boost::numeric::odeint::controlled_runge_kutta<
          boost::numeric::odeint::runge_kutta_dopri5<
              std::array<DataVector, 1>>>>
      dense_stepper = boost::numeric::odeint::make_dense_output(
          1.0e-15, 1.0e-15,
          boost::numeric::odeint::runge_kutta_dopri5<
              std::array<DataVector, 1>>{});

  // Initialize the stepper with initial quaternion (which is the quaternion
  // passed in), initial time, and initial time step = 1e-3
  dense_stepper.initialize(*quaternion_to_integrate, t0, 1e-3);

  // Do an initial step
  std::pair<double, double> step_range =
      dense_stepper.do_step(quaternion_ode_system);
  while (step_range.first <= t) {
    step_range = dense_stepper.do_step(quaternion_ode_system);
  }

  // Store result in quaternion_to_integrate
  // Have to calculate state at t, because stepper has now advanced past t so
  // the current state isn't correct
  dense_stepper.calc_state(t, *quaternion_to_integrate);
}

template <size_t MaxDerivReturned>
boost::math::quaternion<double>
QuaternionFunctionOfTime<MaxDerivReturned>::setup_func(
    double t) const noexcept {
  // Populate missing quaternions
  update_stored_info();

  // Get quaternion and time at closest time before t
  const auto& stored_info_at_t0 =
      stored_info_from_upper_bound(t, stored_quaternions_and_times_);
  std::array<DataVector, 1> quat_to_integrate =
      stored_info_at_t0.stored_quantities;

  // Solve the ode and store the result in quat_to_integrate
  solve_quaternion_ode(&quat_to_integrate, t, stored_info_at_t0.time);

  // Transform DataVector quaternion to boost quaternion for easy manipulation
  boost::math::quaternion<double> quat =
      datavector_to_quaternion(quat_to_integrate[0]);
  normalize_quaternion(make_not_null(&quat));

  return quat;
}

template <size_t MaxDerivReturned>
std::array<DataVector, 1> QuaternionFunctionOfTime<MaxDerivReturned>::func(
    double t) const noexcept {
  return std::array<DataVector, 1>{quaternion_to_datavector(setup_func(t))};
}

template <size_t MaxDerivReturned>
std::array<DataVector, 2>
QuaternionFunctionOfTime<MaxDerivReturned>::func_and_deriv(
    double t) const noexcept {
  ASSERT(MaxDerivReturned >= 1,
         "Asking for too many derivatives than templated.");
  boost::math::quaternion<double> quat = setup_func(t);

  // Get omega and however many derivatives we need
  std::array<DataVector, 1> omega_func = omega_f_of_t_ptr_->func(t);

  boost::math::quaternion<double> omega =
      datavector_to_quaternion(omega_func[0]);

  boost::math::quaternion<double> dtquat = 0.5 * quat * omega;

  return std::array<DataVector, 2>{quaternion_to_datavector(quat),
                                   quaternion_to_datavector(dtquat)};
}

template <size_t MaxDerivReturned>
std::array<DataVector, 3>
QuaternionFunctionOfTime<MaxDerivReturned>::func_and_2_derivs(
    double t) const noexcept {
  ASSERT(MaxDerivReturned >= 2,
         "Asking for too many derivatives than templated.");
  boost::math::quaternion<double> quat = setup_func(t);

  // Get omega and however many derivatives we need
  std::array<DataVector, 2> omega_func_and_deriv =
      omega_f_of_t_ptr_->func_and_deriv(t);

  boost::math::quaternion<double> omega =
      datavector_to_quaternion(omega_func_and_deriv[0]);
  boost::math::quaternion<double> dtomega =
      datavector_to_quaternion(omega_func_and_deriv[1]);

  boost::math::quaternion<double> dtquat = 0.5 * quat * omega;
  boost::math::quaternion<double> dt2quat =
      0.5 * (dtquat * omega + quat * dtomega);

  return std::array<DataVector, 3>{quaternion_to_datavector(quat),
                                   quaternion_to_datavector(dtquat),
                                   quaternion_to_datavector(dt2quat)};
}

template <size_t MaxDerivReturned>
std::array<DataVector, 4>
QuaternionFunctionOfTime<MaxDerivReturned>::func_and_3_derivs(
    double t) const noexcept {
  ASSERT(MaxDerivReturned >= 3,
         "Asking for too many derivatives than templated.");
  boost::math::quaternion<double> quat = setup_func(t);

  // Get omega and however many derivatives we need
  std::array<DataVector, 3> omega_func_and_2_derivs =
      omega_f_of_t_ptr_->func_and_2_derivs(t);

  boost::math::quaternion<double> omega =
      datavector_to_quaternion(omega_func_and_2_derivs[0]);
  boost::math::quaternion<double> dtomega =
      datavector_to_quaternion(omega_func_and_2_derivs[1]);
  boost::math::quaternion<double> dt2omega =
      datavector_to_quaternion(omega_func_and_2_derivs[2]);

  boost::math::quaternion<double> dtquat = 0.5 * quat * omega;
  boost::math::quaternion<double> dt2quat =
      0.5 * (dtquat * omega + quat * dtomega);
  boost::math::quaternion<double> dt3quat =
      0.5 * (dt2quat * omega + 2.0 * dtquat * dtomega + quat * dt2omega);

  return std::array<DataVector, 4>{
      quaternion_to_datavector(quat), quaternion_to_datavector(dtquat),
      quaternion_to_datavector(dt2quat), quaternion_to_datavector(dt3quat)};
}

template <size_t MaxDerivReturned>
void QuaternionFunctionOfTime<MaxDerivReturned>::reset_expiration_time(
    double next_expiration_time) noexcept {
  FunctionOfTimeHelpers::reset_expiration_time(make_not_null(&expiration_time_),
                                               next_expiration_time);
}

// do explicit instantiation of MaxDerivReturned = {0,1,2,3}
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) template class QuaternionFunctionOfTime<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (0, 1, 2, 3))

#undef DIM
#undef INSTANTIATE
}  // namespace domain::FunctionsOfTime
