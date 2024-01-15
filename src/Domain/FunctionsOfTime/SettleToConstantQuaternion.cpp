// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/FunctionsOfTime/SettleToConstantQuaternion.hpp"

#include <cmath>
#include <limits>
#include <memory>

#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace {
template <size_t MaxDerivReturned>
std::array<double, MaxDerivReturned + 1> quat_norm_and_derivs(
    const std::array<DataVector, MaxDerivReturned + 1>&
        unnormalized_func_and_derivs) {
  std::array<double, MaxDerivReturned + 1> result =
      make_array<MaxDerivReturned + 1>(0.0);
  gsl::at(result, 0) = norm(gsl::at(unnormalized_func_and_derivs, 0));
  if constexpr (MaxDerivReturned > 0) {
    gsl::at(result, 1) = sum(gsl::at(unnormalized_func_and_derivs, 0) *
                             gsl::at(unnormalized_func_and_derivs, 1)) /
                         gsl::at(result, 0);
    if constexpr (MaxDerivReturned > 1) {
      gsl::at(result, 2) =
          sum(square(gsl::at(unnormalized_func_and_derivs, 1)) +
              gsl::at(unnormalized_func_and_derivs, 0) *
                  gsl::at(unnormalized_func_and_derivs, 2)) /
          gsl::at(result, 0);
    }
  }
  return result;
}
}  // namespace

namespace domain::FunctionsOfTime {
SettleToConstantQuaternion::SettleToConstantQuaternion(
    const std::array<DataVector, 3>& initial_func_and_derivs,
    const double match_time, const double decay_time)
    : unnormalized_function_of_time_(initial_func_and_derivs, match_time,
                                     decay_time),
      match_time_(match_time),
      decay_time_(decay_time) {
  if (initial_func_and_derivs[0].size() != 4) {
    ERROR(
        "SettleToConstantQuaternion requires the initial function and its time "
        "derivatives to be quaternions stored as DataVectors of size 4, not "
        "size "
        << initial_func_and_derivs[0].size());
  }
  if (not equal_within_roundoff(
          gsl::at(quat_norm_and_derivs<2>(initial_func_and_derivs), 0), 1.0)) {
    ERROR(
        "SettleToConstantQuaternion requires that the initial quaternion "
        "should be a quaternion with a norm of 1.0, not "
        << gsl::at(quat_norm_and_derivs<2>(initial_func_and_derivs), 0));
  }
}

std::unique_ptr<FunctionOfTime> SettleToConstantQuaternion::get_clone() const {
  return std::make_unique<SettleToConstantQuaternion>(*this);
}

template <size_t MaxDerivReturned>
std::array<DataVector, MaxDerivReturned + 1>
SettleToConstantQuaternion::func_and_derivs(const double t) const {
  static_assert(MaxDerivReturned < 3, "The maximum available derivative is 2.");

  std::array<DataVector, MaxDerivReturned + 1> result{};
  if constexpr (MaxDerivReturned == 0) {
    result = unnormalized_function_of_time_.func(t);
  } else if constexpr (MaxDerivReturned == 1) {
    result = unnormalized_function_of_time_.func_and_deriv(t);
  } else if constexpr (MaxDerivReturned == 2) {
    result = unnormalized_function_of_time_.func_and_2_derivs(t);
  }
  std::array<double, MaxDerivReturned + 1> norm_and_derivs =
      quat_norm_and_derivs<MaxDerivReturned>(result);
  result /= gsl::at(norm_and_derivs, 0);
  if constexpr (MaxDerivReturned > 0) {
    gsl::at(result, 1) -= gsl::at(result, 0) * gsl::at(norm_and_derivs, 1) /
                          gsl::at(norm_and_derivs, 0);
    if constexpr (MaxDerivReturned > 1) {
      gsl::at(result, 2) -= gsl::at(result, 1) * 2.0 *
                            gsl::at(norm_and_derivs, 1) /
                            gsl::at(norm_and_derivs, 0);
      gsl::at(result, 2) -= gsl::at(result, 0) * gsl::at(norm_and_derivs, 2) /
                            gsl::at(norm_and_derivs, 0);
    }
  }

  return result;
}

void SettleToConstantQuaternion::pup(PUP::er& p) {
  FunctionOfTime::pup(p);
  size_t version = 0;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 0) {
    p | unnormalized_function_of_time_;
    p | match_time_;
    p | decay_time_;
  }
}

bool operator==(const SettleToConstantQuaternion& lhs,
                const SettleToConstantQuaternion& rhs) {
  return lhs.unnormalized_function_of_time_ ==
             rhs.unnormalized_function_of_time_ and
         lhs.match_time_ == rhs.match_time_ and
         lhs.decay_time_ == rhs.decay_time_;
}

bool operator!=(const SettleToConstantQuaternion& lhs,
                const SettleToConstantQuaternion& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID SettleToConstantQuaternion::my_PUP_ID = 0;  // NOLINT

#define DERIV(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                             \
  template std::array<DataVector, DERIV(data) + 1>                       \
  SettleToConstantQuaternion::func_and_derivs<DERIV(data)>(const double) \
      const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (0, 1, 2))

#undef DERIV
#undef INSTANTIATE
}  // namespace domain::FunctionsOfTime
