// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <vector>

#include "ErrorHandling/Error.hpp"

/// A function that has a piecewise-constant `MaxDeriv`th
/// derivative.
template <size_t MaxDeriv>
class FunctionVsTimeNthDeriv {
 public:
  constexpr static size_t max_deriv() noexcept { return MaxDeriv; }

  // this has outer length = number of components.
  // each component has its own set of derivs.
  using value_type = std::vector<std::array<double, MaxDeriv + 1>>;

  FunctionVsTimeNthDeriv(double time,
                         const value_type& initial_func_and_derivs);

  FunctionVsTimeNthDeriv(const FunctionVsTimeNthDeriv&) = delete;
  FunctionVsTimeNthDeriv& operator=(const FunctionVsTimeNthDeriv&) = delete;

  /// Returns the function and its `MaxDeriv` derivatives at time t.
  /// The function has multiple components.
  value_type operator()(double t) const noexcept;
  /// Updates the `MaxDeriv`th derivative of the function at the given time.
  // updated_max_deriv is a vector of the `MaxDeriv`th to update
  // for each component
  void update(double time_of_update,
              const std::vector<double>& updated_max_deriv) noexcept;

 private:
  // Return k such that mT[k] < t < mT[k+1].
  // For t < mT[0], return 0.
  // For t > mT[mT.size()-1] : return mT.size()-1
  size_t ClosestTimeIndex(const double t) const;

  // history of the function times and derivs
  std::vector<double> times_;
  std::vector<value_type> derivs_;
};

/// Function implementations
// constructor, initialize times and derivs
template <size_t MaxDeriv>
FunctionVsTimeNthDeriv<MaxDeriv>::FunctionVsTimeNthDeriv(
    double time, const value_type& initial_func_and_derivs)
    : times_{time}, derivs_{initial_func_and_derivs} {}

// operator, returns function and derivs at time t
template <size_t MaxDeriv>
typename FunctionVsTimeNthDeriv<MaxDeriv>::value_type
FunctionVsTimeNthDeriv<MaxDeriv>::operator()(double t) const noexcept {

  const size_t k = ClosestTimeIndex(t);
  const double dt = t - times_[k];

  // initialize the return variable
  // (with derivs_, since gets overwritten anyways)
  value_type result = derivs_[k];

  // loop over components
  // (the number of components is same size as what is at the end of derivs)
  for (size_t i = 0; i < derivs_[k].size(); i++) {
    // loop over derivs
    for (size_t d = 0; d < MaxDeriv; ++d) {
      result[i][d] = derivs_[k][i][MaxDeriv];
      for (size_t dd = MaxDeriv; dd --> d ;) {
        result[i][d] *= dt / (dd - d + 1);
        result[i][d] += derivs_[k][i][dd];
      }
    }
    // the MaxDeriv is the updatable piece, no computation needed
    result[i][MaxDeriv] = derivs_[k][i][MaxDeriv];
  }

  return result;
}

// update function
template <size_t MaxDeriv>
void FunctionVsTimeNthDeriv<MaxDeriv>::update(
    double time_of_update,
    const std::vector<double>& updated_max_deriv) noexcept {

  // check that we are updating
  if (time_of_update <= times_.back()) {
    ERROR("ResetNthDeriv: t must be increasing from call to call.");
  }

  // get the current values, before updating the `MaxDeriv'th deriv
  value_type func =
      FunctionVsTimeNthDeriv<MaxDeriv>::operator()(time_of_update);
  // This push_back must occur AFTER the Func() call.
  times_.push_back(time_of_update);

  // loop over compnents and update the MaxDerivs
  for (size_t i = 0; i < updated_max_deriv.size(); i++) {
    // update max deriv
    func[i][MaxDeriv] = updated_max_deriv[i];
  }
  // add to time series of derivs
  derivs_.push_back(func);
}

// closest time index function, returns index near time t
template <size_t MaxDeriv>
size_t FunctionVsTimeNthDeriv<MaxDeriv>::ClosestTimeIndex(
    const double t) const {
  // Try to find k such that times_[k] < t < times_[k+1].
  // Special cases:
  //   t < times_[0]:  If just roundoff, ok to use k=0, else trigger an error.
  //   t > times_[times_.size()-1] : Extrapolate, unless it's too far.

  size_t k;
  std::vector<double>::const_iterator it;
  it = std::lower_bound(times_.begin(), times_.end(), t);

  // MG: might just want to check that times_ is monotonically increasing
  //     then can simplify logic below.

  if (it == times_.end()) {  // t is beyond last point
    k = times_.size() - 1;   // treat later
    if (k >= times_.size()) {
      ERROR("internal consistency error");
    }
    if (times_[k] > t) {
      ERROR("internal consistency error");
    }
  } else {
    if (*it > t) {
      if (it - times_.begin() == 0) {  // before first point
        if (*it - t < t * 1.e-15) {    // is it just roundoff?
          k = it - times_.begin();     // Ok.
        } else {
          ERROR("Error T[0]");
        }
        if (k >= times_.size()) {
          ERROR("internal consistency error");
        }
      } else {                        // Ok, it's not before the first point.
        k = it - times_.begin() - 1;  // one earlier
        if (k >= times_.size()) {
          ERROR("internal consistency error");
        }
        if (times_[k] > t) {
          ERROR("internal consistency error");
        }
      }
    } else {
      k = it - times_.begin();
      if (k >= times_.size()) {
        ERROR("internal consistency error");
      }
      if (times_[k] > t) {
        ERROR("internal consistency error");
      }
    }
  }
  if (k + 1 < times_.size()) {
    if (times_[k + 1] <= t) {
      ERROR("internal consistency error");
    }
  }

  return k;
}
