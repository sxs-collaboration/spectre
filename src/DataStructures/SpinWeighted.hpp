// Distributed under the MIT License.
// See LICENSE.txt for details

#pragma once

#include "Utilities/ForceInline.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"

/*!
 * \ingroup DataStructuresGroup
 * \brief Make a spin-weighted type `T` with spin-weight `Spin`. Mathematical
 * operators are restricted to addition, subtraction, multiplication and
 * division, with spin-weights checked for validity.
 *
 * \details For a spin-weighted object, we limit operations to those valid for a
 * pair of spin-weighted quantities - i.e. addition only makes sense when the
 * two summands possess the same spin weight, and multiplication (or division)
 * result in a summed (or subtracted) spin weight.
 */
template <typename T, int Spin>
struct SpinWeighted {
  T data;
  using value_type = T;
  constexpr static int spin = Spin;
};

// {@
/// \brief Add two spin-weighted quantities if the types are compatible and
/// spins are the same. Un-weighted quantities are assumed to be spin 0.
template <typename T1, typename T2, int Spin>
SPECTRE_ALWAYS_INLINE
    SpinWeighted<decltype(std::declval<T1>() + std::declval<T2>()), Spin>
    operator+(const SpinWeighted<T1, Spin>& lhs,
              const SpinWeighted<T2, Spin>& rhs) noexcept {
  return {lhs.data + rhs.data};
}
template <typename T>
SPECTRE_ALWAYS_INLINE SpinWeighted<T, 0> operator+(
    const SpinWeighted<T, 0>& lhs, const T& rhs) noexcept {
  return {lhs.data + rhs};
}
template <typename T>
SPECTRE_ALWAYS_INLINE SpinWeighted<T, 0> operator+(
    const T& lhs, const SpinWeighted<T, 0>& rhs) noexcept {
  return {lhs + rhs.data};
}
// @}

// @{
/// \brief Subtract two spin-weighted quantities if the types are compatible and
/// spins are the same. Un-weighted quantities are assumed to be spin 0.
template <typename T1, typename T2, int Spin>
SPECTRE_ALWAYS_INLINE
    SpinWeighted<decltype(std::declval<T1>() - std::declval<T2>()), Spin>
    operator-(const SpinWeighted<T1, Spin>& lhs,
              const SpinWeighted<T2, Spin>& rhs) noexcept {
  return {lhs.data - rhs.data};
}
template <typename T>
SPECTRE_ALWAYS_INLINE SpinWeighted<T, 0> operator-(
    const SpinWeighted<T, 0>& lhs, const T& rhs) noexcept {
  return {lhs.data - rhs};
}
template <typename T>
SPECTRE_ALWAYS_INLINE SpinWeighted<T, 0> operator-(
    const T& lhs, const SpinWeighted<T, 0>& rhs) noexcept {
  return {lhs - rhs.data};
}
// @}

// @{
/// \brief Multiply two spin-weighted quantities if the types are compatible and
/// add the spins. Un-weighted quantities are assumed to be spin 0.
template <typename T1, typename T2, int Spin1, int Spin2>
SPECTRE_ALWAYS_INLINE SpinWeighted<
    decltype(std::declval<T1>() * std::declval<T2>()), Spin1 + Spin2>
operator*(const SpinWeighted<T1, Spin1>& lhs,
          const SpinWeighted<T2, Spin2>& rhs) noexcept {
  return {lhs.data * rhs.data};
}
template <typename T, int Spin>
SPECTRE_ALWAYS_INLINE SpinWeighted<T, Spin> operator*(
    const SpinWeighted<T, Spin>& lhs, const T& rhs) noexcept {
  return {lhs.data * rhs};
}
template <typename T, int Spin>
SPECTRE_ALWAYS_INLINE SpinWeighted<T, Spin> operator*(
    const T& lhs, const SpinWeighted<T, Spin>& rhs) noexcept {
  return {lhs * rhs.data};
}
// @}

// @{
/// \brief Divide two spin-weighted quantities if the types are compatible and
/// subtract the spins. Un-weighted quantities are assumed to be spin 0.
template <typename T1, typename T2, int Spin1, int Spin2>
SPECTRE_ALWAYS_INLINE SpinWeighted<
    decltype(std::declval<T1>() / std::declval<T2>()), Spin1 - Spin2>
operator/(const SpinWeighted<T1, Spin1>& lhs,
          const SpinWeighted<T2, Spin2>& rhs) noexcept {
  return {lhs.data / rhs.data};
}
template <typename T, int Spin>
SPECTRE_ALWAYS_INLINE SpinWeighted<T, Spin> operator/(
    const SpinWeighted<T, Spin>& lhs, const T& rhs) noexcept {
  return {lhs.data / rhs};
}
template <typename T, int Spin>
SPECTRE_ALWAYS_INLINE SpinWeighted<T, -Spin> operator/(
    const T& lhs, const SpinWeighted<T, Spin>& rhs) noexcept {
  return {lhs / rhs.data};
}
// @}

// @{
/// \brief Test equivalence of spin-weighted quantities if the types are
/// compatible and spins are the same. Un-weighted quantities are assumed to be
/// spin 0.
template <typename T1, typename T2, int Spin>
SPECTRE_ALWAYS_INLINE bool operator==(
    const SpinWeighted<T1, Spin>& lhs,
    const SpinWeighted<T2, Spin>& rhs) noexcept {
  return lhs.data == rhs.data;
}
template <typename T>
SPECTRE_ALWAYS_INLINE bool operator==(const SpinWeighted<T, 0>& lhs,
                                      const T& rhs) noexcept {
  return lhs.data == rhs;
}
template <typename T>
SPECTRE_ALWAYS_INLINE bool operator==(const T& lhs,
                                      const SpinWeighted<T, 0>& rhs) noexcept {
  return lhs == rhs.data;
}
// @}

// @{
/// \brief Test inequivalence of spin-weighted quantities if the types are
/// compatible and spins are the same. Un-weighted quantities are assumed to be
/// spin 0.
template <typename T1, typename T2, int Spin>
SPECTRE_ALWAYS_INLINE bool operator!=(
    const SpinWeighted<T1, Spin>& lhs,
    const SpinWeighted<T2, Spin>& rhs) noexcept {
  return not(lhs == rhs);
}
template <typename T>
SPECTRE_ALWAYS_INLINE bool operator!=(const SpinWeighted<T, 0>& lhs,
                                      const T& rhs) noexcept {
  return not(lhs == rhs);
}
template <typename T>
SPECTRE_ALWAYS_INLINE bool operator!=(const T& lhs,
                                      const SpinWeighted<T, 0>& rhs) noexcept {
  return not(lhs == rhs);
}
// @}
