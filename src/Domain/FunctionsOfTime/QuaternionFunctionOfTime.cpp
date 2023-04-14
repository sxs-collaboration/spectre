// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"

// BoostMultiArray is used internally in odeint, so BoostMultiArray MUST be
// included before odeint
#include "DataStructures/BoostMultiArray.hpp"

#include <boost/numeric/odeint.hpp>
#include <ostream>
#include <pup_stl.h>

#include "Domain/FunctionsOfTime/QuaternionHelpers.hpp"
#include "NumericalAlgorithms/OdeIntegration/OdeIntegration.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdHelpers.hpp"

namespace domain::FunctionsOfTime {
template <size_t MaxDeriv>
QuaternionFunctionOfTime<MaxDeriv>::QuaternionFunctionOfTime(
    const double t, std::array<DataVector, 1> initial_quat_func,
    std::array<DataVector, MaxDeriv + 1> initial_angle_func,
    const double expiration_time)
    : stored_quaternions_and_times_{{t, std::move(initial_quat_func)}},
      angle_f_of_t_(t, std::move(initial_angle_func), expiration_time) {}

template <size_t MaxDeriv>
std::unique_ptr<FunctionOfTime> QuaternionFunctionOfTime<MaxDeriv>::get_clone()
    const {
  return std::make_unique<QuaternionFunctionOfTime>(*this);
}

template <size_t MaxDeriv>
void QuaternionFunctionOfTime<MaxDeriv>::pup(PUP::er& p) {
  FunctionOfTime::pup(p);
  size_t version = 0;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 0) {
    p | stored_quaternions_and_times_;
    p | angle_f_of_t_;
  }
}

template <size_t MaxDeriv>
void QuaternionFunctionOfTime<MaxDeriv>::update(
    const double time_of_update, DataVector updated_max_deriv,
    const double next_expiration_time) {
  angle_f_of_t_.update(time_of_update, std::move(updated_max_deriv),
                       next_expiration_time);
  update_stored_info();
}

template <size_t MaxDeriv>
void QuaternionFunctionOfTime<MaxDeriv>::update_stored_info() {
  const auto& angle_deriv_info = angle_f_of_t_.get_deriv_info();

  ASSERT(
      angle_deriv_info.size() == stored_quaternions_and_times_.size() + 1,
      "The number of stored quaternions is not one less than the number of "
      "stored angles (which it should be after an update). Currently there are "
          << angle_deriv_info.size() << " stored angles and "
          << stored_quaternions_and_times_.size() << " stored quaternions.");

  const size_t last_index = stored_quaternions_and_times_.size();
  // Final time, initial time, and quaternion
  const double t = angle_deriv_info[last_index].time;
  const double t0 = angle_deriv_info[last_index - 1].time;
  boost::math::quaternion<double> quaternion_to_integrate =
      datavector_to_quaternion(
          stored_quaternions_and_times_[last_index - 1].stored_quantities[0]);

  solve_quaternion_ode(make_not_null(&quaternion_to_integrate), t0, t);

  // normalize quaternion
  normalize_quaternion(make_not_null(&quaternion_to_integrate));

  stored_quaternions_and_times_.emplace_back(
      t, std::array<DataVector, 1>{
             quaternion_to_datavector(quaternion_to_integrate)});

  ASSERT(angle_deriv_info.size() == stored_quaternions_and_times_.size(),
         "The number of stored angles must be the same as the number of stored "
         "quaternions after updating the missing quaternion. Now there are "
             << angle_deriv_info.size() << " stored angles and "
             << stored_quaternions_and_times_.size() << " stored quaternions.");
}

template <size_t MaxDeriv>
void QuaternionFunctionOfTime<MaxDeriv>::solve_quaternion_ode(
    const gsl::not_null<boost::math::quaternion<double>*>
        quaternion_to_integrate,
    const double t0, const double t) const {
  // Boost is stupid and assumes the times it's integrating between are order
  // unity. So if t or t0 is > 10, then you can't take t-t0 to machine precision
  // which causes roundoff error within boost. It was found experimentally that
  // an infinite loop *within boost* is possible if such a condition is met.
  // Thus, we rescale the times and the RHS by this factor to ensure the times
  // are of order unity to avoid this infinite loop. The resulting quaternion is
  // left unchanged.
  const double factor = std::max(1.0, std::max(t0, t));

  // lambda that stores the internals of the ode
  const auto quaternion_ode_system =
      [this, &factor](const boost::math::quaternion<double>& state,
                      boost::math::quaternion<double>& dt_state,
                      const double time) {
        // multiply time and rhs by factor for reasons explained above
        const boost::math::quaternion<double> omega = datavector_to_quaternion(
            angle_f_of_t_.func_and_deriv(time * factor)[1]);
        dt_state = factor * 0.5 * state * omega;
      };

  // Dense stepper
  auto dense_stepper = boost::numeric::odeint::make_dense_output(
      1.0e-12, 1.0e-14,
      boost::numeric::odeint::runge_kutta_dopri5<
          boost::math::quaternion<double>, double,
          boost::math::quaternion<double>, double,
          boost::numeric::odeint::vector_space_algebra>{});

  // Integrate from t0 / factor to t / factor (see explanation above), storing
  // result in quaternion_to_integrate
  boost::numeric::odeint::integrate_adaptive(
      dense_stepper, quaternion_ode_system, *quaternion_to_integrate,
      t0 / factor, t / factor, 1e-4);
}

template <size_t MaxDeriv>
boost::math::quaternion<double> QuaternionFunctionOfTime<MaxDeriv>::setup_func(
    const double t) const {
  // Get quaternion and time at closest time before t
  const auto& stored_info_at_t0 =
      stored_info_from_upper_bound(t, stored_quaternions_and_times_);
  boost::math::quaternion<double> quat_to_integrate =
      datavector_to_quaternion(stored_info_at_t0.stored_quantities[0]);

  // Solve the ode and store the result in quat_to_integrate
  solve_quaternion_ode(make_not_null(&quat_to_integrate),
                       stored_info_at_t0.time, t);

  // Make unit quaternion
  normalize_quaternion(make_not_null(&quat_to_integrate));

  return quat_to_integrate;
}

template <size_t MaxDeriv>
std::array<DataVector, 1> QuaternionFunctionOfTime<MaxDeriv>::quat_func(
    const double t) const {
  return std::array<DataVector, 1>{quaternion_to_datavector(setup_func(t))};
}

template <size_t MaxDeriv>
std::array<DataVector, 2>
QuaternionFunctionOfTime<MaxDeriv>::quat_func_and_deriv(const double t) const {
  boost::math::quaternion<double> quat = setup_func(t);

  // Get angle and however many derivatives we need
  std::array<DataVector, 2> angle_and_deriv = angle_f_of_t_.func_and_deriv(t);

  boost::math::quaternion<double> omega =
      datavector_to_quaternion(angle_and_deriv[1]);

  boost::math::quaternion<double> dtquat = 0.5 * quat * omega;

  return std::array<DataVector, 2>{quaternion_to_datavector(quat),
                                   quaternion_to_datavector(dtquat)};
}

template <size_t MaxDeriv>
std::array<DataVector, 3>
QuaternionFunctionOfTime<MaxDeriv>::quat_func_and_2_derivs(
    const double t) const {
  boost::math::quaternion<double> quat = setup_func(t);

  // Get angle and however many derivatives we need
  std::array<DataVector, 3> angle_and_2_derivs =
      angle_f_of_t_.func_and_2_derivs(t);

  boost::math::quaternion<double> omega =
      datavector_to_quaternion(angle_and_2_derivs[1]);
  boost::math::quaternion<double> dtomega =
      datavector_to_quaternion(angle_and_2_derivs[2]);

  boost::math::quaternion<double> dtquat = 0.5 * quat * omega;
  boost::math::quaternion<double> dt2quat =
      0.5 * (dtquat * omega + quat * dtomega);

  return std::array<DataVector, 3>{quaternion_to_datavector(quat),
                                   quaternion_to_datavector(dtquat),
                                   quaternion_to_datavector(dt2quat)};
}

template <size_t MaxDeriv>
bool operator==(const QuaternionFunctionOfTime<MaxDeriv>& lhs,
                const QuaternionFunctionOfTime<MaxDeriv>& rhs) {
  return lhs.stored_quaternions_and_times_ ==
             rhs.stored_quaternions_and_times_ and
         lhs.angle_f_of_t_ == rhs.angle_f_of_t_;
}

template <size_t MaxDeriv>
bool operator!=(const QuaternionFunctionOfTime<MaxDeriv>& lhs,
                const QuaternionFunctionOfTime<MaxDeriv>& rhs) {
  return not(lhs == rhs);
}

template <size_t MaxDeriv>
std::ostream& operator<<(
    std::ostream& os,
    const QuaternionFunctionOfTime<MaxDeriv>& quaternion_f_of_t) {
  const size_t size = quaternion_f_of_t.stored_quaternions_and_times_.size();

  os << "Quaternion:\n";
  for (size_t i = 0; i < size; ++i) {
    os << quaternion_f_of_t.stored_quaternions_and_times_[i];
    os << "\n";
  }
  os << "Angle:\n";
  os << quaternion_f_of_t.angle_f_of_t_;
  return os;
}

// do explicit instantiation of MaxDeriv = {2,3,4}
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                   \
  template class QuaternionFunctionOfTime<DIM(data)>;          \
  template bool operator==                                     \
      <DIM(data)>(const QuaternionFunctionOfTime<DIM(data)>&,  \
                  const QuaternionFunctionOfTime<DIM(data)>&); \
  template bool operator!=                                     \
      <DIM(data)>(const QuaternionFunctionOfTime<DIM(data)>&,  \
                  const QuaternionFunctionOfTime<DIM(data)>&); \
  template std::ostream& operator<<(                           \
      std::ostream& os, const QuaternionFunctionOfTime<DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3, 4))

#undef DIM
#undef INSTANTIATE
}  // namespace domain::FunctionsOfTime
