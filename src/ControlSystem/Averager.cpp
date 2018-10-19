// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/Averager.hpp"

#include <boost/none.hpp>
#include <ostream>

#include "ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

template <size_t DerivOrder>
Averager<DerivOrder>::Averager(const double avg_timescale_frac,
                               const bool average_0th_deriv_of_q) noexcept
    : avg_tscale_frac_(avg_timescale_frac),
      average_0th_deriv_of_q_(average_0th_deriv_of_q) {
  if (avg_tscale_frac_ <= 0.0) {
    ERROR(
        "The specified avg_timescale_frac, the fraction of the damping "
        "timescale over which averaging is performed, ("
        << avg_tscale_frac_ << ") must be positive");
  }
}

template <size_t DerivOrder>
Averager<DerivOrder>::Averager(Averager&& rhs) noexcept
    : avg_tscale_frac_(std::move(rhs.avg_tscale_frac_)),
      average_0th_deriv_of_q_(std::move(rhs.average_0th_deriv_of_q_)),
      averaged_values_(std::move(rhs.averaged_values_)),
      boost_none_(std::move(rhs.boost_none_)),
      times_(std::move(rhs.times_)),
      raw_qs_(std::move(rhs.raw_qs_)),
      weight_k_(std::move(rhs.weight_k_)),
      tau_k_(std::move(rhs.tau_k_)) {}

template <size_t DerivOrder>
Averager<DerivOrder>& Averager<DerivOrder>::operator=(Averager&& rhs) noexcept {
  if (this != &rhs) {
    avg_tscale_frac_ = std::move(rhs.avg_tscale_frac_);
    average_0th_deriv_of_q_ = std::move(rhs.average_0th_deriv_of_q_);
    averaged_values_ = std::move(rhs.averaged_values_);
    boost_none_ = std::move(rhs.boost_none_);
    times_ = std::move(rhs.times_);
    raw_qs_ = std::move(rhs.raw_qs_);
    weight_k_ = std::move(rhs.weight_k_);
    tau_k_ = std::move(rhs.tau_k_);
  }
  return *this;
}

template <size_t DerivOrder>
const boost::optional<std::array<DataVector, DerivOrder + 1>>&
Averager<DerivOrder>::operator()(const double time) const noexcept {
  if (times_.size() > DerivOrder and time == times_[0]) {
    return averaged_values_;
  }
  return boost_none_;
}

template <size_t DerivOrder>
void Averager<DerivOrder>::clear() noexcept {
  averaged_values_ = boost::none;
  times_.clear();
  raw_qs_.clear();
  weight_k_ = 0.0;
  tau_k_ = 0.0;
}

template <size_t DerivOrder>
void Averager<DerivOrder>::update(const double time, const DataVector& raw_q,
                                  const DataVector& timescales) noexcept {
  if (not raw_qs_.empty()) {
    if (UNLIKELY(raw_q.size() != raw_qs_[0].size())) {
      ERROR("The number of components in the raw_q provided ("
            << raw_q.size()
            << ") does not match the size of previously supplied raw_q ("
            << raw_qs_[0].size() << ").");
    }
  } else {
    // This is the first call to update: initialize averaged values, weights and
    // effective time (with proper number of components)
    averaged_values_ =
        make_array<DerivOrder + 1>(DataVector(raw_q.size(), 0.0));
    weight_k_ = 0.0;
    tau_k_ = 0.0;
  }

  // Ensure that the number of timescales matches the number of components
  if (UNLIKELY(timescales.size() != raw_q.size())) {
    ERROR("The number of supplied timescales ("
          << timescales.size() << ") does not match the number of components ("
          << raw_q.size() << ").");
  }
  // Get the minimum damping time from all component timescales. This will be
  // used to determine the averaging timescale for ALL components.
  const double min_timescale = min(timescales);

  // Do not allow updates at or before last update time
  if (UNLIKELY(not times_.empty() and time <= last_time_updated())) {
    ERROR("The specified time t=" << time << " is at or before the last time "
                                             "updated, t_update="
                                  << last_time_updated() << ".");
  }

  // update deques
  times_.emplace_front(time);
  raw_qs_.emplace_front(raw_q);
  if (times_.size() > DerivOrder + 1) {
    // get rid of old data once we have a sufficient number of points
    times_.pop_back();
    raw_qs_.pop_back();
  }

  if (times_.size() > DerivOrder) {
    // we now have enough data points to begin averaging
    const double tau_avg = min_timescale * avg_tscale_frac_;
    const double tau_m = times_[0] - times_[1];

    // update the weights and effective time
    const double old_weight = weight_k_;
    weight_k_ = (tau_m + old_weight) * tau_avg / (tau_m + tau_avg);
    tau_k_ = (time * tau_m + tau_k_ * old_weight) * tau_avg /
             (weight_k_ * (tau_m + tau_avg));

    std::array<DataVector, DerivOrder + 1> raw_derivs = get_derivs();
    // use raw value if not using `average_0th_deriv_of_q`
    size_t start_ind = 0;
    if (not average_0th_deriv_of_q_) {
      (*averaged_values_)[0] = raw_qs_[0];
      start_ind = 1;
    }

    for (size_t i = start_ind; i <= DerivOrder; i++) {
      gsl::at(*averaged_values_, i) =
          (gsl::at(raw_derivs, i) * tau_m +
           gsl::at(*averaged_values_, i) * old_weight) *
          tau_avg / (weight_k_ * (tau_m + tau_avg));
    }
  }
}

template <size_t DerivOrder>
double Averager<DerivOrder>::last_time_updated() const noexcept {
  if (UNLIKELY(times_.empty())) {
    ERROR("The time history has not been updated yet.");
  }
  return times_[0];
}

template <size_t DerivOrder>
double Averager<DerivOrder>::average_time(const double time) const noexcept {
  if (LIKELY(times_.size() > DerivOrder and time == times_[0])) {
    return tau_k_;
  }
  ERROR(
      "Cannot return averaged values because the averager does not have "
      "up-to-date information.");
}

template <size_t DerivOrder>
std::array<DataVector, DerivOrder + 1> Averager<DerivOrder>::get_derivs() const
    noexcept {
  static_assert(DerivOrder == 2,
                "Finite differencing currently only implemented for DerivOrder "
                "= 2. If a different DerivOrder is desired, please add stencil "
                "coefficients for that order.");

  // initialize the finite difference coefs
  std::array<std::array<double, 3>, 3> coefs{};

  const double delta_t1 = times_[0] - times_[1];
  const double delta_t2 = times_[1] - times_[2];
  const double delta_t = delta_t1 + delta_t2;
  const double denom = 1.0 / delta_t;
  const double dts = delta_t1 * delta_t2;

  // set coefs for function value
  coefs[0] = {{1.0, 0.0, 0.0}};
  // first deriv coefs
  coefs[1] = {{(2.0 + delta_t2 / delta_t1) * denom, -delta_t / dts,
               delta_t1 / delta_t2 * denom}};
  // second deriv coefs
  coefs[2] = {{2.0 / delta_t1 * denom, -2.0 / dts, 2.0 / delta_t2 * denom}};

  std::array<DataVector, 3> result =
      make_array<3>(DataVector(raw_qs_[0].size(), 0.0));
  // compute derivatives
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      gsl::at(result, i) += raw_qs_[j] * gsl::at(gsl::at(coefs, i), j);
    }
  }
  return result;
}

// explicit instantiations
/// \cond
template class Averager<2>;
/// \endcond
