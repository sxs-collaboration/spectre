// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/PiecewisePolynomial.hpp"

#include <algorithm>
#include <iterator>
#include <ostream>
#include <utility>  // IWYU pragma: keep

#include "DataStructures/DataVector.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"

template <size_t MaxDeriv>
FunctionsOfTime::PiecewisePolynomial<MaxDeriv>::PiecewisePolynomial(
    double t, value_type initial_func_and_derivs) noexcept
    : deriv_info_at_update_times_{{t, std::move(initial_func_and_derivs)}} {}

template <size_t MaxDeriv>
template <size_t MaxDerivReturned>
std::array<DataVector, MaxDerivReturned + 1>
FunctionsOfTime::PiecewisePolynomial<MaxDeriv>::func_and_derivs(
    const double t) const noexcept {
  const auto& deriv_info_at_t = deriv_info_from_upper_bound(t);
  const double dt = t - deriv_info_at_t.time;
  const value_type& coefs = deriv_info_at_t.derivs_coefs;

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
void FunctionsOfTime::PiecewisePolynomial<MaxDeriv>::update(
    const double time_of_update, DataVector updated_max_deriv) noexcept {
  if (time_of_update <= deriv_info_at_update_times_.back().time) {
    ERROR("t must be increasing from call to call. "
          << "Attempted to update at time " << time_of_update
          << ", which precedes the previous update time of "
          << deriv_info_at_update_times_.back().time << ".");
  }

  // get the current values, before updating the `MaxDeriv'th deriv
  value_type func = func_and_derivs(time_of_update);

  if (updated_max_deriv.size() != func.back().size()) {
    ERROR("the number of components trying to be updated ("
          << updated_max_deriv.size() << ") does "
                                         "not match the number of components ("
          << func.back().size() << ") in the PiecewisePolynomial.");
  }

  func[MaxDeriv] = std::move(updated_max_deriv);
  deriv_info_at_update_times_.emplace_back(time_of_update, std::move(func));
}

template <size_t MaxDeriv>
FunctionsOfTime::PiecewisePolynomial<MaxDeriv>::DerivInfo::DerivInfo(
    double t, value_type deriv) noexcept : time(t),
                                           derivs_coefs(std::move(deriv)) {
  // convert derivs to coefficients for polynomial evaluation.
  // the coefficient of x^N is the Nth deriv rescaled by 1/factorial(N)
  double fact = 1.0;
  for (size_t j = 2; j < MaxDeriv + 1; j++) {
    fact *= j;
    gsl::at(derivs_coefs, j) /= fact;
  }
}

template <size_t MaxDeriv>
const typename FunctionsOfTime::PiecewisePolynomial<MaxDeriv>::DerivInfo&
FunctionsOfTime::PiecewisePolynomial<MaxDeriv>::deriv_info_from_upper_bound(
    const double t) const noexcept {
  // this function assumes that the times in deriv_info_at_update_times is
  // sorted, which is enforced by the update function.

  const auto upper_bound_deriv_info = std::upper_bound(
      deriv_info_at_update_times_.begin(), deriv_info_at_update_times_.end(), t,
      [](double t0, const DerivInfo& d) { return d.time > t0; });

  if (upper_bound_deriv_info == deriv_info_at_update_times_.begin()) {
    // all elements of times are greater than t
    // check if t is just less than the min element by roundoff
    if (not equal_within_roundoff(upper_bound_deriv_info->time, t)) {
      ERROR("requested time " << t << " precedes earliest time "
                              << deriv_info_at_update_times_.begin()->time
                              << " of times.");
    }
    return *upper_bound_deriv_info;
  }

  // t is either greater than all elements of times
  // or t is within the range of times.
  // In both cases, 'upper_bound_deriv_info' currently points to one index past
  // the desired index.
  return *std::prev(upper_bound_deriv_info, 1);
}

// do explicit instantiation of MaxDeriv = {2,3,4}
// along with all combinations of MaxDerivReturned = {0,...,MaxDeriv}
/// \cond
template class FunctionsOfTime::PiecewisePolynomial<2_st>;
template class FunctionsOfTime::PiecewisePolynomial<3_st>;
template class FunctionsOfTime::PiecewisePolynomial<4_st>;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIMRETURNED(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                             \
  template std::array<DataVector, DIMRETURNED(data) + 1> \
  FunctionsOfTime::PiecewisePolynomial<DIM(              \
      data)>::func_and_derivs<DIMRETURNED(data)>(const double) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (2), (0, 1, 2))
GENERATE_INSTANTIATIONS(INSTANTIATE, (3), (0, 1, 2, 3))
GENERATE_INSTANTIATIONS(INSTANTIATE, (4), (0, 1, 2, 3, 4))

#undef DIM
#undef DIMRETURNED
#undef INSTANTIATE
/// \endcond
