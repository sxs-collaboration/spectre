// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/FunctionsOfTime/FunctionOfTimeHelpers.hpp"

#include <pup.h>
#include <pup_stl.h>

#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace domain::FunctionsOfTime::FunctionOfTimeHelpers {
template <size_t MaxDerivPlusOne, bool StoreCoefs>
StoredInfo<MaxDerivPlusOne, StoreCoefs>::StoredInfo(
    const double t,
    std::array<DataVector, MaxDerivPlusOne> init_quantities) noexcept
    : time(t), stored_quantities{std::move(init_quantities)} {
  // Convert derivs to coefficients for polynomial evaluation
  // The coefficient of x^N is the Nth deriv rescaled by 1/factorial(N)
  double fact = 1.0;
  if (MaxDerivPlusOne > 2) {
    for (size_t j = 2; j < MaxDerivPlusOne; j++) {
      fact *= j;
      gsl::at(stored_quantities, j) /= fact;
    }
  }
}

template <size_t MaxDerivPlusOne>
StoredInfo<MaxDerivPlusOne, false>::StoredInfo(
    const double t,
    std::array<DataVector, MaxDerivPlusOne> init_quantities) noexcept
    : time(t), stored_quantities{std::move(init_quantities)} {}

template <size_t MaxDerivPlusOne, bool StoreCoefs>
void StoredInfo<MaxDerivPlusOne, StoreCoefs>::pup(PUP::er& p) noexcept {
  p | time;
  p | stored_quantities;
}

template <size_t MaxDerivPlusOne>
void StoredInfo<MaxDerivPlusOne, false>::pup(PUP::er& p) noexcept {
  p | time;
  p | stored_quantities;
}

void reset_expiration_time(const gsl::not_null<double*> prev_expiration_time,
                           const double next_expiration_time) noexcept {
  if (next_expiration_time < *prev_expiration_time) {
    ERROR("Attempted to change expiration time to "
          << next_expiration_time
          << ", which precedes the previous expiration time of "
          << *prev_expiration_time << ".");
  }
  *prev_expiration_time = next_expiration_time;
}

template <size_t MaxDerivPlusOne, bool StoreCoefs>
const StoredInfo<MaxDerivPlusOne, StoreCoefs>& stored_info_from_upper_bound(
    const double t, const std::vector<StoredInfo<MaxDerivPlusOne, StoreCoefs>>&
                        all_stored_infos) noexcept {
  // this function assumes that the times in stored_info_at_update_times is
  // sorted, which is enforced by the update function of a piecewise polynomial.

  ASSERT(not all_stored_infos.empty(),
         "Vector of StoredInfos you are trying to access is empty. Was it "
         "constructed properly?");

  const auto upper_bound_stored_info = std::lower_bound(
      all_stored_infos.begin(), all_stored_infos.end(), t,
      [](const StoredInfo<MaxDerivPlusOne, StoreCoefs>& d, double t0) {
        return d.time < t0;
      });

  if (upper_bound_stored_info == all_stored_infos.begin()) {
    // all elements of times are greater than t
    // check if t is just less than the min element by roundoff
    if (not equal_within_roundoff(upper_bound_stored_info->time, t)) {
      ERROR("requested time " << t << " precedes earliest time "
                              << all_stored_infos.begin()->time
                              << " of times.");
    }
    return *upper_bound_stored_info;
  }
  // t is either greater than all elements of times
  // or t is within the range of times.
  // In both cases, 'upper_bound_deriv_info' currently points to one index past
  // the desired index.
  return *std::prev(upper_bound_stored_info, 1);
}

template <size_t MaxDerivPlusOne, bool StoreCoefs>
bool operator==(
    const domain::FunctionsOfTime::FunctionOfTimeHelpers::StoredInfo<
        MaxDerivPlusOne, StoreCoefs>& lhs,
    const domain::FunctionsOfTime::FunctionOfTimeHelpers::StoredInfo<
        MaxDerivPlusOne, StoreCoefs>& rhs) noexcept {
  return lhs.time == rhs.time and
         lhs.stored_quantities == rhs.stored_quantities;
}

template <size_t MaxDerivPlusOne, bool StoreCoefs>
bool operator!=(
    const domain::FunctionsOfTime::FunctionOfTimeHelpers::StoredInfo<
        MaxDerivPlusOne, StoreCoefs>& lhs,
    const domain::FunctionsOfTime::FunctionOfTimeHelpers::StoredInfo<
        MaxDerivPlusOne, StoreCoefs>& rhs) noexcept {
  return not(lhs == rhs);
}

// explicit instantiation of StoredInfo class and stored_info_from_upper_bound
// function for MaxDerivPlusOne = {1,2,3,4,5}
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define STORECOEF(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                   \
  template class StoredInfo<DIM(data), STORECOEF(data)>;       \
  template bool operator==<DIM(data), STORECOEF(data)>(        \
      const StoredInfo<DIM(data), STORECOEF(data)>&,           \
      const StoredInfo<DIM(data), STORECOEF(data)>&) noexcept; \
  template bool operator!=<DIM(data), STORECOEF(data)>(        \
      const StoredInfo<DIM(data), STORECOEF(data)>&,           \
      const StoredInfo<DIM(data), STORECOEF(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3, 4, 5), (true, false))

#undef INSTANTIATE

#define INSTANTIATE(_, data)                             \
  template const StoredInfo<DIM(data), STORECOEF(data)>& \
  stored_info_from_upper_bound(                          \
      const double,                                      \
      const std::vector<StoredInfo<DIM(data), STORECOEF(data)>>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3, 4, 5), (true, false))

#undef INSTANTIATE
#undef STORECOEF
#undef DIM
}  // namespace domain::FunctionsOfTime::FunctionOfTimeHelpers
