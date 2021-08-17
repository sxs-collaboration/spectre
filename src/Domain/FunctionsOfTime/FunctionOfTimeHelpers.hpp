// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// \brief Helper struct and functions for FunctionsOfTime

#pragma once

#include <array>
#include <limits>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "Utilities/Gsl.hpp"

namespace domain::FunctionsOfTime::FunctionOfTimeHelpers {
/// \brief Stores info at a particlar time about a function and its derivatives
///
/// \details `MaxDerivPlusOne` template parameter is intended to be related to
/// the maximum derivative that a FunctionOfTime is supposed to store. For a
/// generic PiecewisePolynomial this will be the MaxDeriv + 1 (because we store
/// the function as well as its derivatives).
/// `StoreCoefs` template parameter indicates that the coefficients of the
/// polynomial used for evaluation are stored instead of the polynomial itself.
/// The coefficient of \f$ x^N \f$ is the Nth deriv rescaled by 1/factorial(N)
template <size_t MaxDerivPlusOne, bool StoreCoefs = true>
struct StoredInfo {
  double time{std::numeric_limits<double>::signaling_NaN()};
  std::array<DataVector, MaxDerivPlusOne> stored_quantities;

  StoredInfo() = default;

  StoredInfo(double t,
             std::array<DataVector, MaxDerivPlusOne> init_quantities) noexcept;
  ~StoredInfo() = default;
  StoredInfo(StoredInfo&&) noexcept = default;
  StoredInfo& operator=(StoredInfo&&) noexcept = default;
  StoredInfo(const StoredInfo&) = default;
  StoredInfo& operator=(const StoredInfo&) = default;

  void pup(PUP::er& p) noexcept;
};

template <size_t MaxDerivPlusOne>
struct StoredInfo<MaxDerivPlusOne, false> {
  double time{std::numeric_limits<double>::signaling_NaN()};
  std::array<DataVector, MaxDerivPlusOne> stored_quantities;

  StoredInfo() = default;

  StoredInfo(double t,
             std::array<DataVector, MaxDerivPlusOne> init_quantities) noexcept;

  void pup(PUP::er& p) noexcept;
};

/// Resets the previous expiration time if the next expiration time is after,
/// otherwise throws an error
void reset_expiration_time(const gsl::not_null<double*> prev_expiration_time,
                           const double next_expiration_time) noexcept;

/// Returns a StoredInfo corresponding to the closest element in the range of
/// `StoredInfo.time`s that is less than `t`. The function throws an
/// error if `t` is less than all `StoredInfo.time`s (unless `t` is just less
/// than the earliest `StoredInfo.time` by roundoff, in which case it returns
/// the earliest StoredInfo.)
template <size_t MaxDerivPlusOne, bool StoreCoefs>
const StoredInfo<MaxDerivPlusOne, StoreCoefs>& stored_info_from_upper_bound(
    const double t, const std::vector<StoredInfo<MaxDerivPlusOne, StoreCoefs>>&
                        all_stored_infos) noexcept;

template <size_t MaxDerivPlusOne, bool StoreCoefs>
bool operator==(
    const domain::FunctionsOfTime::FunctionOfTimeHelpers::StoredInfo<
        MaxDerivPlusOne, StoreCoefs>& lhs,
    const domain::FunctionsOfTime::FunctionOfTimeHelpers::StoredInfo<
        MaxDerivPlusOne, StoreCoefs>& rhs) noexcept;

template <size_t MaxDerivPlusOne, bool StoreCoefs>
bool operator!=(
    const domain::FunctionsOfTime::FunctionOfTimeHelpers::StoredInfo<
        MaxDerivPlusOne, StoreCoefs>& lhs,
    const domain::FunctionsOfTime::FunctionOfTimeHelpers::StoredInfo<
        MaxDerivPlusOne, StoreCoefs>& rhs) noexcept;
}  // namespace domain::FunctionsOfTime::FunctionOfTimeHelpers
