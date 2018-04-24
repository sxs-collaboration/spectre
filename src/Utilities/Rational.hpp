// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// \ingroup UtilitiesGroup
/// A rational number
///
/// This serves as a faster replacement for
/// `boost::rational<std::int32_t>`.  As of Boost 1.65.0, arithmetic
/// operators average about twice as fast, and ordering operators are
/// about eight times as fast.
class Rational {
 public:
  // clang-tidy: google-explicit-constructor
  // Treating integers as rationals is desired.
  Rational(std::int32_t numerator = 0,  // NOLINT
           std::int32_t denominator = 1) noexcept;

  std::int32_t numerator() const noexcept { return numerator_; }
  std::int32_t denominator() const noexcept { return denominator_; }

  double value() const noexcept;

  Rational inverse() const noexcept;

  Rational& operator+=(const Rational& other) noexcept;
  Rational& operator-=(const Rational& other) noexcept;
  Rational& operator*=(const Rational& other) noexcept;
  Rational& operator/=(const Rational& other) noexcept;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

  friend Rational operator-(Rational r) noexcept;

 private:
  std::int32_t numerator_{0};
  std::int32_t denominator_{1};
};

Rational operator+(const Rational& a, const Rational& b) noexcept;
Rational operator-(const Rational& a, const Rational& b) noexcept;
Rational operator*(const Rational& a, const Rational& b) noexcept;
Rational operator/(const Rational& a, const Rational& b) noexcept;

bool operator==(const Rational& a, const Rational& b) noexcept;
bool operator!=(const Rational& a, const Rational& b) noexcept;
bool operator<(const Rational& a, const Rational& b) noexcept;
bool operator>(const Rational& a, const Rational& b) noexcept;
bool operator<=(const Rational& a, const Rational& b) noexcept;
bool operator>=(const Rational& a, const Rational& b) noexcept;

std::ostream& operator<<(std::ostream& os, const Rational& r) noexcept;

size_t hash_value(const Rational& r) noexcept;

namespace std {
template <>
struct hash<Rational> {
  size_t operator()(const Rational& r) const noexcept;
};
}  // namespace std
