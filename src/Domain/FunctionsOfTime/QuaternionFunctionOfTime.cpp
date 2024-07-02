// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"

// BoostMultiArray is used internally in odeint, so BoostMultiArray MUST be
// included before odeint
#include "DataStructures/BoostMultiArray.hpp"

#include <algorithm>
#include <array>
#include <boost/math/quaternion.hpp>
#include <boost/numeric/odeint.hpp>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/QuaternionHelpers.hpp"
#include "Domain/FunctionsOfTime/ThreadsafeList.tpp"
#include "NumericalAlgorithms/OdeIntegration/OdeIntegration.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupBoost.hpp"
#include "Utilities/StdHelpers.hpp"

namespace domain::FunctionsOfTime {
template <size_t MaxDeriv>
QuaternionFunctionOfTime<MaxDeriv>::QuaternionFunctionOfTime() = default;

template <size_t MaxDeriv>
QuaternionFunctionOfTime<MaxDeriv>::QuaternionFunctionOfTime(
    QuaternionFunctionOfTime&&) = default;

template <size_t MaxDeriv>
QuaternionFunctionOfTime<MaxDeriv>::QuaternionFunctionOfTime(
    const QuaternionFunctionOfTime&) = default;

template <size_t MaxDeriv>
auto QuaternionFunctionOfTime<MaxDeriv>::operator=(QuaternionFunctionOfTime&&)
    -> QuaternionFunctionOfTime& = default;

template <size_t MaxDeriv>
auto QuaternionFunctionOfTime<MaxDeriv>::operator=(
    const QuaternionFunctionOfTime&) -> QuaternionFunctionOfTime& = default;

template <size_t MaxDeriv>
QuaternionFunctionOfTime<MaxDeriv>::~QuaternionFunctionOfTime() = default;

template <size_t MaxDeriv>
QuaternionFunctionOfTime<MaxDeriv>::QuaternionFunctionOfTime(
    const double t, const std::array<DataVector, 1>& initial_quat_func,
    std::array<DataVector, MaxDeriv + 1> initial_angle_func,
    const double expiration_time)
    : stored_quaternions_and_times_(t),
      angle_f_of_t_(t, std::move(initial_angle_func), expiration_time) {
  stored_quaternions_and_times_.insert(
      t, datavector_to_quaternion(initial_quat_func[0]), expiration_time);
}

template <size_t MaxDeriv>
QuaternionFunctionOfTime<MaxDeriv>::QuaternionFunctionOfTime(
    CkMigrateMessage* /*unused*/) {}

template <size_t MaxDeriv>
std::unique_ptr<FunctionOfTime> QuaternionFunctionOfTime<MaxDeriv>::get_clone()
    const {
  return std::make_unique<QuaternionFunctionOfTime>(*this);
}

template <size_t MaxDeriv>
std::array<double, 2> QuaternionFunctionOfTime<MaxDeriv>::time_bounds() const {
  return std::array{stored_quaternions_and_times_.initial_time(),
                    stored_quaternions_and_times_.expiration_time()};
}

template <size_t MaxDeriv>
double QuaternionFunctionOfTime<MaxDeriv>::expiration_after(
    const double time) const {
  return stored_quaternions_and_times_.expiration_after(time);
}

template <size_t MaxDeriv>
void QuaternionFunctionOfTime<MaxDeriv>::pup(PUP::er& p) {
  FunctionOfTime::pup(p);
  size_t version = 5;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.

  if (version < 4) {
    unpack_old_version(p, version);
    return;
  }

  p | stored_quaternions_and_times_;
  p | angle_f_of_t_;

  // Just use empty map when unpacking version 4
  if (version >= 5) {
    p | update_backlog_;
  }
}

namespace {
struct LegacyStoredInfo {
  double time{std::numeric_limits<double>::signaling_NaN()};
  std::array<DataVector, 1> stored_quantities;

  void pup(PUP::er& p) {
    p | time;
    p | stored_quantities;
  }
};
}  // namespace

template <size_t MaxDeriv>
void QuaternionFunctionOfTime<MaxDeriv>::unpack_old_version(
    PUP::er& p, const size_t version) {
  ASSERT(p.isUnpacking(), "Can't serialize old version");

  std::vector<LegacyStoredInfo> quaternions{};
  double expiration_time{};

  // For versions 0 and 1, we stored the data first, then angle fot, then
  // possibly the size
  if (version <= 1) {
    if (version == 0) {
      // Version 0 had a std::vector
      p | quaternions;
    } else {
      // Version 1 had a std::deque
      std::deque<LegacyStoredInfo> pupped_quaternions{};
      p | pupped_quaternions;
      quaternions.assign(std::move_iterator(pupped_quaternions.begin()),
                         std::move_iterator(pupped_quaternions.end()));
    }

    // Same for v0 and v1
    p | angle_f_of_t_;
    expiration_time = angle_f_of_t_.time_bounds()[1];

    if (version == 1) {
      uint64_t stored_quaternion_size{};
      p | stored_quaternion_size;
    }
  } else if (version >= 2) {
    // However, for v2+, we store angle fot, expiration time, size, then data
    // for thread-safety reasons
    p | angle_f_of_t_;

    // For v3+ we pup our own expiration time, while for 2 we have to get it
    // from the angle_f_of_t_.
    if (version >= 3) {
      p | expiration_time;
    } else {
      expiration_time = angle_f_of_t_.time_bounds()[1];
    }

    size_t size = 0;
    p | size;
    quaternions.resize(size);
    for (auto& stored_quaternion : quaternions) {
      p | stored_quaternion;
    }
  }

  stored_quaternions_and_times_ =
      decltype(stored_quaternions_and_times_)(quaternions.front().time);
  for (size_t i = 0; i < quaternions.size() - 1; ++i) {
    stored_quaternions_and_times_.insert(
        quaternions[i].time,
        datavector_to_quaternion(quaternions[i].stored_quantities[0]),
        quaternions[i + 1].time);
  }
  stored_quaternions_and_times_.insert(
      quaternions.back().time,
      datavector_to_quaternion(quaternions.back().stored_quantities[0]),
      expiration_time);
}

template <size_t MaxDeriv>
void QuaternionFunctionOfTime<MaxDeriv>::update(
    const double time_of_update, DataVector updated_max_deriv,
    const double next_expiration_time) {
  angle_f_of_t_.update(time_of_update, std::move(updated_max_deriv),
                       next_expiration_time);

  // If the stored PP didn't update, then don't compute the new quaternion, just
  // store the times.
  const double angle_expiration_time = angle_f_of_t_.time_bounds()[1];
  if (angle_expiration_time < next_expiration_time) {
    update_backlog_[time_of_update] = next_expiration_time;
    return;
  }

  // We add this time to the backlog even though it doesn't need to be added
  // just to make the loop easier. It'll be removed
  update_backlog_[time_of_update] = next_expiration_time;

  while (not update_backlog_.empty() and
         update_backlog_.begin()->second <= angle_expiration_time) {
    auto entry = update_backlog_.begin();
    const double stored_time_of_update = entry->first;
    const double stored_expiration_time = entry->second;

    const auto old_interval =
        stored_quaternions_and_times_(stored_time_of_update);
    boost::math::quaternion<double> quaternion_to_integrate = old_interval.data;

    solve_quaternion_ode(make_not_null(&quaternion_to_integrate),
                         old_interval.update, old_interval.expiration);

    normalize_quaternion(make_not_null(&quaternion_to_integrate));

    stored_quaternions_and_times_.insert(
        stored_time_of_update, quaternion_to_integrate, stored_expiration_time);

    update_backlog_.erase(entry);
  }
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
  const auto stored_info_at_t0 = stored_quaternions_and_times_(t);
  boost::math::quaternion<double> quat_to_integrate = stored_info_at_t0.data;

  // Solve the ode and store the result in quat_to_integrate
  solve_quaternion_ode(make_not_null(&quat_to_integrate),
                       stored_info_at_t0.update, t);

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
std::array<DataVector, 4>
QuaternionFunctionOfTime<MaxDeriv>::quat_func_and_3_derivs(
    const double t) const {
  boost::math::quaternion<double> quat = setup_func(t);

  // Get angle and however many derivatives we need
  std::vector<DataVector> angle_and_all_derivs =
      angle_f_of_t_.func_and_all_derivs(t);

  if (angle_and_all_derivs.size() < 4) {
    ERROR(
        "Need more angle derivs to compute the third derivative of the "
        "quaternion. Currently only have "
        << MaxDeriv);
  }

  boost::math::quaternion<double> omega =
      datavector_to_quaternion(angle_and_all_derivs[1]);
  boost::math::quaternion<double> dtomega =
      datavector_to_quaternion(angle_and_all_derivs[2]);

  boost::math::quaternion<double> dt2omega =
      datavector_to_quaternion(angle_and_all_derivs[3]);

  boost::math::quaternion<double> dtquat = 0.5 * quat * omega;
  boost::math::quaternion<double> dt2quat =
      0.5 * (dtquat * omega + quat * dtomega);
  boost::math::quaternion<double> dt3quat =
      0.5 * (dt2quat * omega + 2.0 * dtquat * dtomega + quat * dt2omega);

  return std::array<DataVector, 4>{
      quaternion_to_datavector(quat), quaternion_to_datavector(dtquat),
      quaternion_to_datavector(dt2quat), quaternion_to_datavector(dt3quat)};
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
  const auto quaternions_begin =
      quaternion_f_of_t.stored_quaternions_and_times_.begin();
  const auto quaternions_end =
      quaternion_f_of_t.stored_quaternions_and_times_.end();
  // We want to write the entries in order, but the iterator goes the
  // other way.
  std::vector<std::decay_t<decltype(quaternions_begin)>> iters{};
  if (quaternions_begin != quaternions_end) {
    iters.push_back(quaternions_begin);
    for (;;) {
      auto next = std::next(iters.back());
      if (next == quaternions_end) {
        break;
      }
      iters.push_back(std::move(next));
    }
  }
  std::reverse(iters.begin(), iters.end());

  os << "Quaternion:\n";
  for (const auto& entry : iters) {
    os << "t=" << entry->update << ": " << entry->data << "\n";
  }
  using ::operator<<;
  os << "backlog=" << quaternion_f_of_t.update_backlog_ << "\n";
  os << "Angle:\n";
  os << quaternion_f_of_t.angle_f_of_t_;
  return os;
}

// do explicit instantiation of MaxDeriv = {2,3}
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

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3))

#undef DIM
#undef INSTANTIATE
}  // namespace domain::FunctionsOfTime
