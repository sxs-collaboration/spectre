// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines arithmetic operators for std::array and other helpful functions.

#pragma once

#include <array>
#include <cmath>
#include <utility>

#include "Utilities/Gsl.hpp"

// Arithmetic operators for std::array<T, Dim>

template <size_t Dim, typename T, typename U>
inline std::array<T, Dim>& operator+=(std::array<T, Dim>& lhs,
                                      const std::array<U, Dim>& rhs) noexcept {
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(lhs, i) += gsl::at(rhs, i);
  }
  return lhs;
}

template <size_t Dim, typename T, typename U>
inline auto operator+(const std::array<T, Dim>& lhs,
                      const std::array<U, Dim>& rhs) noexcept
    -> std::array<decltype(lhs[0] + rhs[0]), Dim> {
  std::array<decltype(lhs[0] + rhs[0]), Dim> result{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(result, i) = gsl::at(lhs, i) + gsl::at(rhs, i);
  }
  return result;
}

template <size_t Dim, typename T, typename U>
inline std::array<T, Dim>& operator-=(std::array<T, Dim>& lhs,
                                      const std::array<U, Dim>& rhs) noexcept {
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(lhs, i) -= gsl::at(rhs, i);
  }
  return lhs;
}

template <size_t Dim, typename T, typename U>
inline auto operator-(const std::array<T, Dim>& lhs,
                      const std::array<U, Dim>& rhs) noexcept
    -> std::array<decltype(lhs[0] - rhs[0]), Dim> {
  std::array<decltype(lhs[0] - rhs[0]), Dim> result{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(result, i) = gsl::at(lhs, i) - gsl::at(rhs, i);
  }
  return result;
}

template <size_t Dim, typename T, typename U>
inline std::array<T, Dim> operator*(const std::array<T, Dim>& lhs,
                                    const U& scale) noexcept {
  std::array<T, Dim> result{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(result, i) = gsl::at(lhs, i) * scale;
  }
  return result;
}

template <size_t Dim, typename T, typename U>
inline std::array<T, Dim> operator*(const U& scale,
                                    const std::array<T, Dim>& rhs) noexcept {
  return rhs * scale;
}

template <size_t Dim, typename T, typename U>
inline std::array<T, Dim> operator/(const std::array<T, Dim>& lhs,
                                    const U& scale) noexcept {
  std::array<T, Dim> result{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(result, i) = gsl::at(lhs, i) / scale;
  }
  return result;
}

template <size_t Dim, typename T>
inline std::array<T, Dim> operator-(const std::array<T, Dim>& rhs) noexcept {
  std::array<T, Dim> result{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(result, i) = -gsl::at(rhs, i);
  }
  return result;
}

/// \ingroup UtilitiesGroup
/// \brief Construct an array from an existing array omitting one element
template <size_t ElementToRemove, typename T, size_t Dim>
inline std::array<T, Dim - 1> all_but_specified_element_of(
    const std::array<T, Dim>& a) noexcept {
  static_assert(ElementToRemove < Dim, "Specified element does not exist");
  std::array<T, Dim - 1> result{};
  for (size_t i = 0; i < ElementToRemove; ++i) {
    gsl::at(result, i) = gsl::at(a, i);
  }
  for (size_t i = ElementToRemove + 1; i < Dim; ++i) {
    gsl::at(result, i - 1) = gsl::at(a, i);
  }
  return result;
}

/// \ingroup UtilitiesGroup
/// \brief Construct an array from an existing array prepending a value
template <typename T, size_t Dim>
inline constexpr std::array<T, Dim + 1> prepend(const std::array<T, Dim>& a,
                                                T value) noexcept {
  std::array<T, Dim + 1> result{};
  gsl::at(result, 0) = std::move(value);
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(result, i + 1) = gsl::at(a, i);
  }
  return result;
}

//@{
/// \ingroup UtilitiesGroup
/// \brief Euclidean magnitude of the elements of the array.
///
/// \details If T is a container the magnitude is computed separately for each
/// element of the container.
///
/// \requires If T is a container, T must have following mathematical operators:
/// abs(), sqrt(), and element-wise addition and multiplication.  In addition,
/// each T in the array must have the same size.
template <typename T>
inline T magnitude(const std::array<T, 1>& a) noexcept {
  return abs(a[0]);
}

template <>
inline double magnitude(const std::array<double, 1>& a) noexcept {
  return std::abs(a[0]);
}

template <typename T>
inline T magnitude(const std::array<T, 2>& a) noexcept {
  return sqrt(a[0] * a[0] + a[1] * a[1]);
}

template <typename T>
inline T magnitude(const std::array<T, 3>& a) noexcept {
  return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}
//@}
