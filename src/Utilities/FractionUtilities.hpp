// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include "ErrorHandling/Assert.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TypeTraits.hpp"

/// Type trait to check if a type looks like a fraction (specifically,
/// if it has numerator and denominator methods)
//@{
template <typename T, typename = cpp17::void_t<>>
struct is_fraction : std::false_type {};
template <typename T>
struct is_fraction<T, cpp17::void_t<decltype(std::declval<T>().numerator()),
                                    decltype(std::declval<T>().denominator())>>
    : std::true_type {};

template <typename T>
constexpr bool is_fraction_v = is_fraction<T>::value;
//@}

/// \ingroup UtilitiesGroup
/// \brief Compute the continued fraction representation of a number
///
/// The template argument may be a fraction type or a floating point
/// type.  If the expansion is computed exactly then it will be the
/// shorter of the expansions for the given value.
template <class T>
class ContinuedFraction {
  template <typename U, typename = typename is_fraction<U>::type>
  struct value_type_helper;

  template <typename U>
  struct value_type_helper<U, std::false_type> {
    using type = int64_t;
  };

  template <typename U>
  struct value_type_helper<U, std::true_type> {
    using type = std::decay_t<decltype(std::declval<U>().numerator())>;
  };

 public:
  /// The decayed return type of T::numerator() for a fraction,
  /// int64_t for other types.
  using value_type = typename value_type_helper<T>::type;

  explicit ContinuedFraction(const T& value) noexcept
      : term_(ifloor(value)), remainder_(value - term_) {}

  /// Obtain the current element in the expansion
  value_type operator*() const noexcept { return term_; }

  /// Check if the expansion is incomplete.  If T is a fraction type
  /// this will be true until the full exact representation of the
  /// fraction is produced.  If T is a floating point type this will
  /// be true until the estimated numerical error is larger than the
  /// remaining error in the representation.
  explicit operator bool() const noexcept { return not done_; }

  /// Advance to the next element in the expansion
  ContinuedFraction& operator++() noexcept {
    if (remainder_ == 0 or (error_ /= square(remainder_)) > 1) {
      done_ = true;
      return *this;
    }
    remainder_ = 1 / remainder_;
    term_ = ifloor(remainder_);
    remainder_ -= term_;
    return *this;
  }

 private:
  template <typename U, Requires<is_fraction_v<U>> = nullptr>
  static value_type ifloor(const U& x) noexcept {
    return static_cast<value_type>(x.numerator() / x.denominator());
  }

  template <typename U, Requires<not is_fraction_v<U>> = nullptr>
  static value_type ifloor(const U& x) noexcept {
    return static_cast<value_type>(std::floor(x));
  }

  value_type term_;
  T remainder_;
  // Estimate of error in the term.  For any non-fundamental type
  // epsilon() returns a default-constructed value, which should be
  // zero for fractions.
  T error_{std::numeric_limits<T>::epsilon()};
  bool done_{false};
};

/// \ingroup UtilitiesGroup
/// \brief Sum a continued fraction
///
/// \tparam Fraction the result type, which must be a fraction type
template <class Fraction>
class ContinuedFractionSummer {
 public:
  using Term_t = std::decay_t<decltype(std::declval<Fraction>().numerator())>;

  /// Sum of the supplied continued fraction terms
  Fraction value() const noexcept { return Fraction(numerator_, denominator_); }

  /// Insert a new term.  Terms should be supplied from most to least
  /// significant.
  void insert(Term_t term) noexcept {
    const auto new_numerator = term * numerator_ + prev_numerator_;
    prev_numerator_ = numerator_;
    numerator_ = new_numerator;

    const auto new_denominator = term * denominator_ + prev_denominator_;
    prev_denominator_ = denominator_;
    denominator_ = new_denominator;
  }

 private:
  Term_t numerator_{1};
  Term_t denominator_{0};
  Term_t prev_numerator_{0};
  Term_t prev_denominator_{1};
};

/// \ingroup UtilitiesGroup
/// \brief Find the fraction in the supplied interval with the
/// smallest denominator
///
/// The endpoints are considered to be in the interval.  The order of
/// the arguments is not significant.  The answer is unique as long as
/// the interval has length less than 1; for longer intervals, an
/// integer in the range will be returned.
template <typename Fraction, typename T1, typename T2>
Fraction simplest_fraction_in_interval(const T1& end1,
                                       const T2& end2) noexcept {
  ContinuedFractionSummer<Fraction> result;
  using Term_t = typename decltype(result)::Term_t;
  ContinuedFraction<T1> cf1(end1);
  ContinuedFraction<T2> cf2(end2);
  using InputTerm_t = std::common_type_t<typename decltype(cf1)::value_type,
                                         typename decltype(cf2)::value_type>;
  InputTerm_t term1 = *cf1;
  InputTerm_t term2 = *cf2;
  for (; cf1 and cf2; term1 = *++cf1, term2 = *++cf2) {
    if (term1 != term2) {
      if (++cf1) {
        ++term1;
      }
      if (++cf2) {
        ++term2;
      }
      result.insert(gsl::narrow<Term_t>(std::min(term1, term2)));
      return result.value();
    }
    result.insert(gsl::narrow<Term_t>(term1));
  }
  return result.value();
}
