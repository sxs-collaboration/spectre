// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/AdamsLts.hpp"

#include <cstddef>
#include <utility>

#include "DataStructures/MathWrapper.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/TimeSteppers/AdamsCoefficients.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace TimeSteppers::adams_lts {
namespace {
template <typename T>
using OrderVector = adams_coefficients::OrderVector<T>;

template <typename Op>
LtsCoefficients& add_assign_impl(LtsCoefficients& a, const LtsCoefficients& b,
                                 const Op& op) {
  const auto key_equal = [](const LtsCoefficients::value_type& l,
                            const LtsCoefficients::value_type& r) {
    return get<0>(l) == get<0>(r) and get<1>(l) == get<1>(r);
  };
  const auto key_less = [](const LtsCoefficients::value_type& l,
                           const LtsCoefficients::value_type& r) {
    return get<0>(l) < get<0>(r) or
           (get<0>(l) == get<0>(r) and get<1>(l) < get<1>(r));
  };

  // Two passes: first the common entries, and then the new entries.
  size_t common_entries = 0;
  {
    auto a_it = a.begin();
    auto b_it = b.begin();
    while (a_it != a.end() and b_it != b.end()) {
      if (key_less(*a_it, *b_it)) {
        ++a_it;
      } else if (key_less(*b_it, *a_it)) {
        ++b_it;
      } else {
        ++common_entries;
        get<2>(*a_it) += op(get<2>(*b_it));
        ++a_it;
        ++b_it;
      }
    }
  }
  a.resize(a.size() + b.size() - common_entries);
  {
    auto write = a.rbegin();
    auto a_it = write + static_cast<LtsCoefficients::difference_type>(
                            b.size() - common_entries);
    auto b_it = b.rbegin();
    while (b_it != b.rend()) {
      if (a_it == a.rend() or key_less(*a_it, *b_it)) {
        *write = *b_it;
        get<2>(*write) = op(get<2>(*write));
        ++b_it;
      } else {
        if (key_equal(*a_it, *b_it)) {
          ++b_it;
        }
        *write = *a_it;
        ++a_it;
      }
      ++write;
    }
  }
  return a;
}
}  // namespace

LtsCoefficients& operator+=(LtsCoefficients& a, const LtsCoefficients& b) {
  return add_assign_impl(a, b, [](const double x) { return x; });
}
LtsCoefficients& operator-=(LtsCoefficients& a, const LtsCoefficients& b) {
  return add_assign_impl(a, b, [](const double x) { return -x; });
}

LtsCoefficients operator+(LtsCoefficients&& a, LtsCoefficients&& b) {
  return std::move(a += b);
}
LtsCoefficients operator+(LtsCoefficients&& a, const LtsCoefficients& b) {
  return std::move(a += b);
}
LtsCoefficients operator+(const LtsCoefficients& a, LtsCoefficients&& b) {
  return std::move(b += a);
}
LtsCoefficients operator+(const LtsCoefficients& a, const LtsCoefficients& b) {
  auto a2 = a;
  return std::move(a2 += b);
}

LtsCoefficients operator-(LtsCoefficients&& a, LtsCoefficients&& b) {
  return std::move(a -= b);
}
LtsCoefficients operator-(LtsCoefficients&& a, const LtsCoefficients& b) {
  return std::move(a -= b);
}
LtsCoefficients operator-(const LtsCoefficients& a, const LtsCoefficients& b) {
  auto a2 = a;
  return std::move(a2 -= b);
}
LtsCoefficients operator-(const LtsCoefficients& a, LtsCoefficients&& b) {
  alg::for_each(b, [](auto& coef) { get<2>(coef) *= -1.0; });
  return std::move(b += a);
}

template <typename T>
void apply_coefficients(const gsl::not_null<T*> result,
                        const LtsCoefficients& coefficients,
                        const BoundaryHistoryEvaluator<T>& coupling) {
  for (const auto& term : coefficients) {
    *result += get<2>(term) * *coupling(get<0>(term), get<1>(term));
  }
}

#define MATH_WRAPPER_TYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                          \
  template void apply_coefficients(                   \
      gsl::not_null<MATH_WRAPPER_TYPE(data)*> result, \
      const LtsCoefficients& coefficients,            \
      const BoundaryHistoryEvaluator<MATH_WRAPPER_TYPE(data)>& coupling);

GENERATE_INSTANTIATIONS(INSTANTIATE, (MATH_WRAPPER_TYPES))
#undef INSTANTIATE
#undef MATH_WRAPPER_TYPE
}  // namespace TimeSteppers::adams_lts
