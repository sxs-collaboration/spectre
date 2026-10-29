// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <ostream>
#include <type_traits>

#include "Utilities/Kokkos/KokkosMacros.hpp"
#include "Utilities/TypeTraits/IsStdArray.hpp"

#ifdef SPECTRE_KOKKOS
#include <Kokkos_Printf.hpp>
#endif  // SPECTRE_KOKKOS

/*!
 * \ingroup ErrorHandlingGroup
 * \brief A minimal output stream that can be used in device code.
 *
 * Because the standard library streams can't be used on GPUs, `ASSERT` and
 * `ERROR` stream their message into a `DeviceMessage` when they fail in
 * device code, i.e. in a `KOKKOS_FUNCTION` or Kokkos kernel running on
 * a GPU. The message is collected in a fixed-size buffer and printed with a
 * single `Kokkos::printf`, so messages from different GPU threads don't
 * interleave.
 *
 * Common types are supported:
 * - string literals and `const char*`
 * - `char` (also `signed char` and `unsigned char`) as a character
 * - `bool` as `1` or `0`, like `std::ostream` without `std::boolalpha`
 * - integers
 * - floating-point numbers in scientific notation with 17 significant digits.
 *   The last digit can differ from the exact decimal representation.
 * - enums as their underlying integer
 * - `std::array`s of supported types, as `(a,b,c)` like on the host
 * `std::endl` as a newline. Other manipulators such as `std::scientific` and
 * `std::setprecision` are ignored.
 *
 * Values of other types are printed as `<?>`. Add an overload
 * `KOKKOS_FUNCTION DeviceMessage& operator<<(DeviceMessage&, const T&)` to
 * print them on the device. Messages longer than `capacity` characters are
 * truncated, which is marked by `...` at the end.
 */
class DeviceMessage {
 public:
  /// The maximum number of characters in the message
  static constexpr size_t capacity = 512;

  KOKKOS_FUNCTION DeviceMessage() = default;

  /// The message as a null-terminated string
  KOKKOS_FUNCTION const char* c_str() const { return buffer_.data(); }

  /// The number of characters in the message
  KOKKOS_FUNCTION size_t size() const { return size_; }

  template <typename T>
  KOKKOS_FUNCTION DeviceMessage& operator<<(const T& value) {
    if constexpr (std::is_same_v<T, bool>) {
      put(value ? '1' : '0');
    } else if constexpr (std::is_same_v<T, char> or
                         std::is_same_v<T, signed char> or
                         std::is_same_v<T, unsigned char>) {
      put(static_cast<char>(value));
    } else if constexpr (std::is_integral_v<T> and std::is_signed_v<T>) {
      append_signed(static_cast<long long>(value));
    } else if constexpr (std::is_integral_v<T>) {
      append_unsigned(static_cast<unsigned long long>(value));
    } else if constexpr (std::is_floating_point_v<T>) {
      append_double(static_cast<double>(value));
    } else if constexpr (std::is_enum_v<T>) {
      // Promote so enums based on `char` types print as numbers
      *this << +static_cast<std::underlying_type_t<T>>(value);
    } else if constexpr (std::is_same_v<std::remove_extent_t<T>, char> or
                         std::is_same_v<T, const char*> or
                         std::is_same_v<T, char*>) {
      append(static_cast<const char*>(value));
    } else if constexpr (tt::is_std_array_v<T>) {
      put('(');
      bool first = true;
      for (const auto& element : value) {
        if (not first) {
          put(',');
        }
        first = false;
        *this << element;
      }
      put(')');
    } else if constexpr (std::is_same_v<T, decltype(std::setprecision(0))> or
                         std::is_same_v<T, decltype(std::setw(0))> or
                         std::is_same_v<T, decltype(std::setfill(' '))>) {
      // Formatting manipulators are ignored
    } else {
      append("<?>");
    }
    return *this;
  }

  /// `std::endl` (and other `std::ostream` manipulators) add a newline
  KOKKOS_FUNCTION DeviceMessage& operator<<(
      std::ostream& (* /*manipulator*/)(std::ostream&)) {
    put('\n');
    return *this;
  }

  /// Formatting manipulators such as `std::scientific` are ignored
  KOKKOS_FUNCTION DeviceMessage& operator<<(
      std::ios_base& (* /*manipulator*/)(std::ios_base&)) {
    return *this;
  }

 private:
  KOKKOS_FUNCTION void put(const char character) {
    if (truncated_) {
      return;
    }
    // Keep the last character for the null terminator
    if (size_ + 1 < capacity) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
      buffer_[size_] = character;
      ++size_;
    } else {
      // Mark the truncation by replacing the last characters with "..."
      for (size_t i = size_ - 3; i < size_; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        buffer_[i] = '.';
      }
      truncated_ = true;
    }
  }

  KOKKOS_FUNCTION void append(const char* str) {
    if (str == nullptr) {
      str = "(null)";
    }
    for (; *str != '\0'; ++str) {  // NOLINT(*-pro-bounds-pointer-arithmetic)
      put(*str);
    }
  }

  KOKKOS_FUNCTION void append_unsigned(unsigned long long value,
                                       const size_t min_digits = 1) {
    std::array<char, 20> digits{};
    size_t number_of_digits = 0;
    do {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
      digits[number_of_digits] = static_cast<char>('0' + value % 10);
      value /= 10;
      ++number_of_digits;
    } while (value != 0 or number_of_digits < min_digits);
    while (number_of_digits > 0) {
      --number_of_digits;
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
      put(digits[number_of_digits]);
    }
  }

  KOKKOS_FUNCTION void append_signed(const long long value) {
    if (value < 0) {
      put('-');
      // Negate as unsigned so the most negative value doesn't overflow
      append_unsigned(0ULL - static_cast<unsigned long long>(value));
    } else {
      append_unsigned(static_cast<unsigned long long>(value));
    }
  }

  // Formats like "%.16e". Device code has no `snprintf`, so the digits are
  // computed directly. Scaling to [1, 10) rounds, so the last digit can differ
  // from the exact decimal representation.
  KOKKOS_FUNCTION void append_double(double value) {
    if (std::isnan(value)) {
      append("nan");
      return;
    }
    if (std::signbit(value)) {
      put('-');
      value = -value;
    }
    if (std::isinf(value)) {
      append("inf");
      return;
    }
    // The 17 significant digits are `digits / 10^16`, '.', `digits % 10^16`
    constexpr unsigned long long ten_to_16 = 10'000'000'000'000'000ULL;
    int exponent = 0;
    unsigned long long digits = 0;
    if (value != 0.0) {
      exponent = static_cast<int>(std::floor(std::log10(value)));
      // Powers of ten up to 1e22 are exact, so for most values this is a
      // single rounding. Scale in two steps for denormals, where 10^-exponent
      // overflows.
      double scaled = value;
      if (exponent >= 0) {
        scaled /= std::pow(10.0, exponent);
      } else if (exponent >= -300) {
        scaled *= std::pow(10.0, -exponent);
      } else {
        scaled = (scaled * 1.0e300) * std::pow(10.0, -(exponent + 300));
      }
      // `log10` can be off by one near powers of ten
      if (scaled >= 10.0) {
        scaled /= 10.0;
        ++exponent;
      } else if (scaled < 1.0) {
        scaled *= 10.0;
        --exponent;
      }
      digits = static_cast<unsigned long long>(
          std::llround(scaled * static_cast<double>(ten_to_16)));
      // Rounding can carry into the next decade, e.g. 9.99...95 -> 10.0
      if (digits >= 10 * ten_to_16) {
        digits /= 10;
        ++exponent;
      }
    }
    put(static_cast<char>('0' + digits / ten_to_16));
    put('.');
    append_unsigned(digits % ten_to_16, 16);
    put('e');
    put(exponent < 0 ? '-' : '+');
    append_unsigned(
        static_cast<unsigned long long>(exponent < 0 ? -exponent : exponent),
        2);
  }

  std::array<char, capacity> buffer_{};
  size_t size_ = 0;
  bool truncated_ = false;
};

#ifdef SPECTRE_KOKKOS
namespace DeviceMessage_detail {
// Not inlined so the message buffer and formatting code don't add to the
// registers and stack of the kernels that call this in their failure branch
template <typename MessageFunction>
KOKKOS_FUNCTION __attribute__((noinline)) void print_failure(
    const char* const heading, const char* const expression,
    const char* const file, const int line, const char* const pretty_function,
    const MessageFunction& message_function) {
  DeviceMessage message{};
  message_function(message);
  // A single `printf` so messages from different GPU threads don't interleave
  if (expression == nullptr) {
    Kokkos::printf(
        "\n############ %s ON THE DEVICE ############\n%s in %s:%d\n\n%s\n"
        "############ %s ON THE DEVICE ############\n\n",
        heading, pretty_function, file, line, message.c_str(), heading);
  } else {
    Kokkos::printf(
        "\n############ %s ON THE DEVICE ############\n%s in %s:%d\n\n"
        "'%s' violated!\n%s\n"
        "############ %s ON THE DEVICE ############\n\n",
        heading, pretty_function, file, line, expression, message.c_str(),
        heading);
  }
}

// `Kokkos::abort` isn't `[[noreturn]]` in some configurations (e.g. CUDA with
// bounds checking), so the compiler would warn about missing return statements
// after `ERROR`. It doesn't return on CUDA and HIP devices. (On SYCL devices it
// doesn't abort yet, which SpECTRE doesn't support.)
[[noreturn]] KOKKOS_INLINE_FUNCTION void abort(const char* const message) {
  ::Kokkos::abort(message);
  __builtin_unreachable();
}
}  // namespace DeviceMessage_detail

/*!
 * \ingroup ErrorHandlingGroup
 * \brief Prints a failure message in device code and aborts the program.
 *
 * Used by `ASSERT` and `ERROR` in device code. `HEADING` must be a string
 * literal, `EXPRESSION` the failed expression as a string or `nullptr`, and
 * `MESSAGE` an expression that can be streamed into a `DeviceMessage`.
 */
#define SPECTRE_ABORT_ON_DEVICE(HEADING, EXPRESSION, MESSAGE)                  \
  do {                                                                         \
    ::DeviceMessage_detail::print_failure(                                     \
        HEADING, EXPRESSION, __FILE__, __LINE__,                               \
        static_cast<const char*>(__PRETTY_FUNCTION__),                         \
        [&](DeviceMessage& device_message_ABORT_ON_DEVICE) {                   \
          device_message_ABORT_ON_DEVICE << MESSAGE; /* NOLINT */              \
        });                                                                    \
    ::DeviceMessage_detail::abort(HEADING                                      \
                                  " on the device, see the message above.\n"); \
  } while (false)
#endif  // SPECTRE_KOKKOS
