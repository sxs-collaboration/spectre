// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <ostream>
#include <utility>

#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"

/*!
 * \ingroup UtilitiesGroup
 * \brief Wraps an error to construct an `Expected` that holds the error.
 */
template <typename E>
struct Unexpected {
  E error;
};

template <typename E>
Unexpected(E) -> Unexpected<E>;

namespace Expected_detail {
template <typename E>
concept streamable =
    requires(std::ostream& os, const E& error) { os << error; };
}  // namespace Expected_detail

/*!
 * \ingroup UtilitiesGroup
 * \brief Holds either a value of type `T` or an error of type `E`, like a
 * minimal version of C++23's `std::expected`.
 *
 * Construct an `Expected` from a `T` to hold a value, or from an
 * `Unexpected<E>` to hold an error. Return an `Expected` from functions that
 * can fail to let the caller decide how to handle the failure, e.g. in device
 * code where exceptions are not available.
 *
 * \snippet Test_Expected.cpp expected_example
 *
 * Calling `value()` when there is an error, or `error()` when there is a
 * value, is an error (also in release builds). On the host, the message
 * includes the held error if `E` can be streamed.
 *
 * Unlike `std::expected`, this class stores both a `T` and an `E`, so both must
 * be default-constructible and the one that isn't held is default-constructed.
 * This keeps the class simple. The member functions are `constexpr` instead of
 * `KOKKOS_FUNCTION`s, so they can be called in device code (with nvcc's relaxed
 * constexpr) when `T` and `E` can be used on the device, without making nvcc
 * treat host-only types such as `DataVector` as device-callable.
 */
template <typename T, typename E>
class Expected {
 public:
  // NOLINTNEXTLINE(google-explicit-constructor): converts like std::expected
  constexpr Expected(T value) : value_(std::move(value)), has_value_(true) {}
  // NOLINTNEXTLINE(google-explicit-constructor): converts like std::expected
  constexpr Expected(Unexpected<E> unexpected)
      : error_(std::move(unexpected.error)), has_value_(false) {}

  constexpr bool has_value() const { return has_value_; }
  constexpr explicit operator bool() const { return has_value_; }

  /// @{
  /// The value. It is an error to call this if `has_value()` is false.
  constexpr const T& value() const& {
    check_has_value();
    return value_;
  }
  constexpr T& value() & {
    check_has_value();
    return value_;
  }
  constexpr T value() && {
    check_has_value();
    return std::move(value_);
  }
  /// @}

  /// The error. It is an error to call this if `has_value()` is true.
  constexpr const E& error() const {
    if (has_value_) {
      KOKKOS_IF_ON_HOST((ERROR("Expected holds a value, not an error.");))
      KOKKOS_IF_ON_DEVICE(
          (Kokkos::abort("Expected holds a value, not an error.");))
    }
    return error_;
  }

 private:
  constexpr void check_has_value() const {
    if (not has_value_) {
      KOKKOS_IF_ON_HOST((
          if constexpr (Expected_detail::streamable<E>) {
            ERROR("Expected holds an error, not a value. The error is:\n"
                  << error_);
          } else { ERROR("Expected holds an error, not a value."); }))
      KOKKOS_IF_ON_DEVICE(
          (Kokkos::abort("Expected holds an error, not a value.");))
    }
  }

  T value_{};
  E error_{};
  bool has_value_;
};
