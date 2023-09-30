// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <limits>
#include <memory>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace domain::FunctionsOfTime {
template <size_t MaxDeriv>
PiecewisePolynomial<MaxDeriv>::PiecewisePolynomial(
    const double t,
    std::array<DataVector, MaxDeriv + 1> initial_func_and_derivs,
    const double expiration_time)
    : deriv_info_at_update_times_(t) {
  store_entry(t, std::move(initial_func_and_derivs), expiration_time);
}

template <size_t MaxDeriv>
std::unique_ptr<FunctionOfTime> PiecewisePolynomial<MaxDeriv>::get_clone()
    const {
  return std::make_unique<PiecewisePolynomial>(*this);
}

template <size_t MaxDeriv>
template <size_t MaxDerivReturned>
std::array<DataVector, MaxDerivReturned + 1>
PiecewisePolynomial<MaxDeriv>::func_and_derivs(const double t) const {
  const auto deriv_info_at_t = deriv_info_at_update_times_(t);
  const double dt = t - deriv_info_at_t.update;
  const auto& coefs = deriv_info_at_t.data;

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
void PiecewisePolynomial<MaxDeriv>::store_entry(
    const double time_of_update,
    std::array<DataVector, MaxDeriv + 1> func_and_derivs,
    const double next_expiration_time) {
  // Convert derivs to coefficients for polynomial evaluation
  // The coefficient of x^N is the Nth deriv rescaled by 1/factorial(N)
  double fact = 1.0;
  for (size_t j = 2; j <= MaxDeriv; j++) {
    fact *= static_cast<double>(j);
    gsl::at(func_and_derivs, j) /= fact;
  }
  deriv_info_at_update_times_.insert(time_of_update, std::move(func_and_derivs),
                                     next_expiration_time);
}

template <size_t MaxDeriv>
void PiecewisePolynomial<MaxDeriv>::update(const double time_of_update,
                                           DataVector updated_max_deriv,
                                           const double next_expiration_time) {
  // This check is just to give a better error message.  The same
  // condition will be checked internally by some of the operations
  // below.  If this method is improperly called in parallel, the
  // value of deriv_info_at_update_times_.expiration_time() can change
  // during this function, but it is obtained atomically so at worst
  // you get a nonsense error message.
  if (time_of_update != deriv_info_at_update_times_.expiration_time()) {
    ERROR("Attempted to update at time "
          << time_of_update << " instead of the expiration time "
          << deriv_info_at_update_times_.expiration_time());
  }

  auto func = func_and_derivs(time_of_update);

  if (updated_max_deriv.size() != func.back().size()) {
    ERROR("the number of components trying to be updated ("
          << updated_max_deriv.size()
          << ") does not match the number of components (" << func.back().size()
          << ") in the PiecewisePolynomial.");
  }

  func[MaxDeriv] = std::move(updated_max_deriv);
  store_entry(time_of_update, std::move(func), next_expiration_time);
}

template <size_t MaxDeriv>
void PiecewisePolynomial<MaxDeriv>::pup(PUP::er& p) {
  FunctionOfTime::pup(p);
  size_t version = 3;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.

  if (version < 3) {
    unpack_old_version(p, version);
    return;
  }

  p | deriv_info_at_update_times_;
}

namespace {
template <size_t MaxDerivPlusOne>
struct LegacyStoredInfo {
  double time{std::numeric_limits<double>::signaling_NaN()};
  std::array<DataVector, MaxDerivPlusOne> stored_quantities;

  void pup(PUP::er& p) {
    p | time;
    p | stored_quantities;
  }
};
}  // namespace

template <size_t MaxDeriv>
void PiecewisePolynomial<MaxDeriv>::unpack_old_version(PUP::er& p,
                                                       const size_t version) {
  ASSERT(p.isUnpacking(), "Can't serialize old version");

  std::vector<LegacyStoredInfo<MaxDeriv + 1>> info_vector{};
  double expiration_time{};

  // For versions 0 and 1, we stored the data first, then expiration time,
  // then possibly the size
  if (version <= 1) {
    if (version == 0) {
      // Version 0 had a std::vector
      p | info_vector;
    } else {
      // Version 1 had a std::deque
      std::deque<LegacyStoredInfo<MaxDeriv + 1>> pupped_info{};
      p | pupped_info;
      info_vector.assign(std::move_iterator(pupped_info.begin()),
                         std::move_iterator(pupped_info.end()));
    }

    // Same for v0 and v1
    p | expiration_time;

    if (version == 1) {
      uint64_t deriv_info_size{};
      p | deriv_info_size;
    }
  } else if (version >= 2) {
    // However, for v2+, we store expiration time, size, then data for
    // thread-safety reasons
    p | expiration_time;
    size_t size = 0;
    p | size;
    info_vector.resize(size);
    for (auto& deriv_info : info_vector) {
      p | deriv_info;
    }
  }

  deriv_info_at_update_times_ =
      decltype(deriv_info_at_update_times_)(info_vector.front().time);
  for (size_t i = 0; i < info_vector.size() - 1; ++i) {
    deriv_info_at_update_times_.insert(
        info_vector[i].time, std::move(info_vector[i].stored_quantities),
        info_vector[i + 1].time);
  }
  deriv_info_at_update_times_.insert(
      info_vector.back().time, std::move(info_vector.back().stored_quantities),
      expiration_time);
}

template <size_t MaxDeriv>
bool operator==(const PiecewisePolynomial<MaxDeriv>& lhs,
                const PiecewisePolynomial<MaxDeriv>& rhs) {
  return lhs.deriv_info_at_update_times_ == rhs.deriv_info_at_update_times_;
}

template <size_t MaxDeriv>
bool operator!=(const PiecewisePolynomial<MaxDeriv>& lhs,
                const PiecewisePolynomial<MaxDeriv>& rhs) {
  return not(lhs == rhs);
}

template <size_t MaxDeriv>
std::ostream& operator<<(
    std::ostream& os,
    const PiecewisePolynomial<MaxDeriv>& piecewise_polynomial) {
  const auto updates_begin =
      piecewise_polynomial.deriv_info_at_update_times_.begin();
  const auto updates_end =
      piecewise_polynomial.deriv_info_at_update_times_.end();
  if (updates_begin == updates_end) {
    return os;
  }
  // We want to write the entries in order, but the iterator goes the
  // other way.
  std::vector iters{updates_begin};
  {
    for (;;) {
      auto next = std::next(iters.back());
      if (next == updates_end) {
        break;
      }
      iters.push_back(std::move(next));
    }
  }
  std::reverse(iters.begin(), iters.end());

  for (size_t entry = 0; entry < iters.size(); ++entry) {
    os << "t=" << iters[entry]->update << ": ";
    for (size_t i = 0; i < MaxDeriv; ++i) {
      os << gsl::at(iters[entry]->data, i) << " ";
    }
    os << iters[entry]->data[MaxDeriv];
    if (entry != iters.size() - 1) {
      os << "\n";
    }
  }
  return os;
}

// do explicit instantiation of MaxDeriv = {0,1,2,3,4}
// along with all combinations of MaxDerivReturned = {0,...,MaxDeriv}
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIMRETURNED(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                  \
  template class PiecewisePolynomial<DIM(data)>;                              \
  template bool operator==<DIM(data)>(const PiecewisePolynomial<DIM(data)>&,  \
                                      const PiecewisePolynomial<DIM(data)>&); \
  template bool operator!=<DIM(data)>(const PiecewisePolynomial<DIM(data)>&,  \
                                      const PiecewisePolynomial<DIM(data)>&); \
  template std::ostream& operator<<(                                          \
      std::ostream& os,                                                       \
      const PiecewisePolynomial<DIM(data)>& piecewise_polynomial);

GENERATE_INSTANTIATIONS(INSTANTIATE, (0, 1, 2, 3, 4))

#undef INSTANTIATE

#define INSTANTIATE(_, data)                                          \
  template std::array<DataVector, DIMRETURNED(data) + 1>              \
  PiecewisePolynomial<DIM(data)>::func_and_derivs<DIMRETURNED(data)>( \
      const double) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (0, 1, 2, 3, 4), (0, 1, 2))

#undef DIM
#undef DIMRETURNED
#undef INSTANTIATE
}  // namespace domain::FunctionsOfTime
