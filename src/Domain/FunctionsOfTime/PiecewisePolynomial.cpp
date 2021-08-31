// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"

#include <algorithm>
#include <iterator>
#include <memory>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>
#include <utility>  // IWYU pragma: keep

#include "DataStructures/DataVector.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"

namespace domain::FunctionsOfTime {
template <size_t MaxDeriv>
PiecewisePolynomial<MaxDeriv>::PiecewisePolynomial(
    const double t, value_type initial_func_and_derivs,
    const double expiration_time) noexcept
    : deriv_info_at_update_times_{{t, std::move(initial_func_and_derivs)}},
      expiration_time_(expiration_time) {}

template <size_t MaxDeriv>
std::unique_ptr<FunctionOfTime> PiecewisePolynomial<MaxDeriv>::get_clone()
    const noexcept {
  return std::make_unique<PiecewisePolynomial>(*this);
}

template <size_t MaxDeriv>
template <size_t MaxDerivReturned>
std::array<DataVector, MaxDerivReturned + 1>
PiecewisePolynomial<MaxDeriv>::func_and_derivs(const double t) const noexcept {
  if (t > expiration_time_) {
    ERROR("Attempt to evaluate PiecewisePolynomial at a time "
          << t << " that is after the expiration time " << expiration_time_
          << ". The difference between times is " << t - expiration_time_
          << ".");
  }
  const auto& deriv_info_at_t =
      stored_info_from_upper_bound(t, deriv_info_at_update_times_);
  const double dt = t - deriv_info_at_t.time;
  const value_type& coefs = deriv_info_at_t.stored_quantities;

  // initialize result for the number of derivs requested
  std::array<DataVector, MaxDerivReturned + 1> result =
      make_array<MaxDerivReturned + 1>(DataVector(coefs.back().size(), 0.0));

  // evaluate the polynomial using ddpoly (Numerical Recipes sec 5.1)
  result[0] = coefs[MaxDeriv];
  for (size_t j = MaxDeriv; j-- > 0;) {
    const size_t min_deriv = std::min(MaxDerivReturned, MaxDeriv - j);
    for (size_t k = min_deriv; k > 0; k--) {
      gsl::at(result, k) = gsl::at(result, k) * dt + gsl::at(result, k - 1);
    }
    result[0] = result[0] * dt + gsl::at(coefs, j);
  }
  // after the first derivative, factorial constants come in
  double fact = 1.0;
  for (size_t j = 2; j < MaxDerivReturned + 1; j++) {
    fact *= j;
    gsl::at(result, j) *= fact;
  }

  return result;
}

template <size_t MaxDeriv>
void PiecewisePolynomial<MaxDeriv>::update(
    // Clang-tidy says to use 'const DataVector& updated_max_deriv'.
    // However, updated_max_deriv is std::moved out of inside this function.
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    const double time_of_update, DataVector updated_max_deriv,
    const double next_expiration_time) noexcept {
  if (time_of_update <= deriv_info_at_update_times_.back().time) {
    ERROR("t must be increasing from call to call. "
          << "Attempted to update at time " << time_of_update
          << ", which precedes the previous update time of "
          << deriv_info_at_update_times_.back().time << ".");
  }
  if (next_expiration_time < expiration_time_) {
    ERROR("expiration_time must be nondecreasing from call to call. "
          << "Attempted to change expiration time to " << next_expiration_time
          << ", which precedes the previous expiration time of "
          << expiration_time_ << ".");
  }
  if (time_of_update < expiration_time_) {
    ERROR("Attempt to update PiecewisePolynomial at a time "
          << time_of_update
          << " that is earlier than the previous expiration time of "
          << expiration_time_
          << ". This is bad because some asynchronous process may have already "
             "used PiecewisePolynomial at a time later than the current time "
          << time_of_update << ".");
  }
  if (time_of_update > next_expiration_time) {
    ERROR(
        "Attempt to set the expiration time of PiecewisePolynomial "
        "to a value "
        << next_expiration_time << " that is earlier than the current time "
        << time_of_update << ".");
  }

  // Normally, func_and_derivs(t) throws an error if t>expiration_time_.
  // But here, we want to allow time_of_update to
  // be greater than the *previous* expiration time, so we need to
  // reset expiration_time_ before the call to func_and_derivs.
  expiration_time_ = next_expiration_time;

  // get the current values, before updating the `MaxDeriv'th deriv
  value_type func = func_and_derivs(time_of_update);

  if (updated_max_deriv.size() != func.back().size()) {
    ERROR("the number of components trying to be updated ("
          << updated_max_deriv.size()
          << ") does "
             "not match the number of components ("
          << func.back().size() << ") in the PiecewisePolynomial.");
  }

  func[MaxDeriv] = std::move(updated_max_deriv);
  deriv_info_at_update_times_.emplace_back(time_of_update, std::move(func));
}

template <size_t MaxDeriv>
void PiecewisePolynomial<MaxDeriv>::reset_expiration_time(
    const double next_expiration_time) noexcept {
  FunctionOfTimeHelpers::reset_expiration_time(make_not_null(&expiration_time_),
                                               next_expiration_time);
}

template <size_t MaxDeriv>
void PiecewisePolynomial<MaxDeriv>::pup(PUP::er& p) {
  FunctionOfTime::pup(p);
  p | deriv_info_at_update_times_;
  p | expiration_time_;
}

template <size_t MaxDeriv>
bool operator==(const PiecewisePolynomial<MaxDeriv>& lhs,
                const PiecewisePolynomial<MaxDeriv>& rhs) noexcept {
  return lhs.deriv_info_at_update_times_ == rhs.deriv_info_at_update_times_ and
         lhs.expiration_time_ == rhs.expiration_time_;
}

template <size_t MaxDeriv>
bool operator!=(const PiecewisePolynomial<MaxDeriv>& lhs,
                const PiecewisePolynomial<MaxDeriv>& rhs) noexcept {
  return not(lhs == rhs);
}

template <size_t MaxDeriv>
std::ostream& operator<<(
    std::ostream& os,
    const PiecewisePolynomial<MaxDeriv>& piecewise_polynomial) noexcept {
  const auto size = piecewise_polynomial.deriv_info_at_update_times_.size();

  for (size_t i = 0; i < size - 1; ++i) {
    os << piecewise_polynomial.deriv_info_at_update_times_[i];
    os << "\n";
  }
  os << piecewise_polynomial.deriv_info_at_update_times_[size - 1];
  return os;
}

// do explicit instantiation of MaxDeriv = {2,3,4}
// along with all combinations of MaxDerivReturned = {0,...,MaxDeriv}
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIMRETURNED(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                       \
  template class PiecewisePolynomial<DIM(data)>;                   \
  template bool operator==                                         \
      <DIM(data)>(const PiecewisePolynomial<DIM(data)>&,           \
                  const PiecewisePolynomial<DIM(data)>&) noexcept; \
  template bool operator!=                                         \
      <DIM(data)>(const PiecewisePolynomial<DIM(data)>&,           \
                  const PiecewisePolynomial<DIM(data)>&) noexcept; \
  template std::ostream& operator<<(                               \
      std::ostream& os,                                            \
      const PiecewisePolynomial<DIM(data)>& piecewise_polynomial) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (0, 1, 2, 3, 4))

#undef INSTANTIATE

#define INSTANTIATE(_, data)                                          \
  template std::array<DataVector, DIMRETURNED(data) + 1>              \
  PiecewisePolynomial<DIM(data)>::func_and_derivs<DIMRETURNED(data)>( \
      const double) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (0), (0))
GENERATE_INSTANTIATIONS(INSTANTIATE, (1), (0, 1))
GENERATE_INSTANTIATIONS(INSTANTIATE, (2), (0, 1, 2))
GENERATE_INSTANTIATIONS(INSTANTIATE, (3), (0, 1, 2, 3))
GENERATE_INSTANTIATIONS(INSTANTIATE, (4), (0, 1, 2, 3, 4))

#undef DIM
#undef DIMRETURNED
#undef INSTANTIATE
}  // namespace domain::FunctionsOfTime
