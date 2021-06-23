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

/// \brief Stores info at a particlar time about a function and its derivatives
///
/// \details `MaxDerivPlusOne` template parameter is intended to be related to
/// the maximum derivative that a FunctionOfTime is supposed to store. For a
/// generic PiecewisePolynomial this will be the MaxDeriv + 1 (because we store
/// the function as well as its derivatives)
template <size_t MaxDerivPlusOne>
struct StoredInfo {
  double time{std::numeric_limits<double>::signaling_NaN()};
  std::array<DataVector, MaxDerivPlusOne> stored_quantities;

  StoredInfo() = default;

  StoredInfo(double t, std::array<DataVector, MaxDerivPlusOne> init_quantities,
             bool StoringCoefs = true) noexcept;

  void pup(PUP::er& p) noexcept;

  bool operator==(const StoredInfo<MaxDerivPlusOne>& rhs) const noexcept;
};

/// Resets the previous expiration time if the next expiration time is after,
/// otherwise thows and error
void reset_fot_expiration_time(gsl::not_null<double*> prev_expiration_time,
                               const double next_expiration_time) noexcept;

/// Returns a StoredInfo corresponding to the closest element in the range of
/// DerivInfos with an update time that is less than or equal to `t`.
/// The function throws an error if `t` is less than all DerivInfo update
/// times. (unless `t` is just less than the earliest update time by roundoff,
/// in which case it returns the DerivInfo at the earliest update time.)
template <size_t MaxDerivPlusOne>
StoredInfo<MaxDerivPlusOne>& stored_info_from_upper_bound(
    const double t,
    std::vector<StoredInfo<MaxDerivPlusOne>>& all_stored_infos) noexcept;
