// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <ios>
#include <type_traits>

#include "Utilities/Kokkos/KokkosMacros.hpp"
#include "Utilities/TypeTraits/IsStdArray.hpp"

/*!
 * \ingroup ErrorHandlingGroup
 * \brief A minimal output stream that can be used in device code.
 *
 * Because the standard library streams can't be used on GPUs, `ASSERT` and
 * `ERROR` stream their message into a `DeviceMessage` when they fail in
 * device code, i.e. in a `KOKKOS_FUNCTION` or Kokkos kernel running on a GPU.
 * It has no buffer: each value is printed right away with `Kokkos::printf`
 * (`std::printf` in builds without Kokkos). See `SPECTRE_ABORT_ON_DEVICE` for
 * how messages from different GPU threads are kept from interleaving.
 *
 * Common types are supported:
 * - string literals and `const char*`
 * - `char` (also `signed char` and `unsigned char`) as a character
 * - `bool` as `1` or `0`, like `std::ostream` without `std::boolalpha`
 * - integers
 * - floating-point numbers like `std::scientific` with
 *   `std::setprecision(18)`, which `ERROR` uses on the host
 * - enums as their underlying integer
 * - `std::array`s of supported types, as `(a,b,c)` like on the host
 *
 * Manipulators such as `std::scientific` are ignored. Manipulators that take
 * an argument, such as `std::setprecision`, are host functions and can't
 * be used on the device; do not use them with this stream.
 *
 * Values of other types are printed as `<?>`. Add an overload
 * `KOKKOS_FUNCTION DeviceMessage& operator<<(DeviceMessage&, const T&)` to
 * print them on the device.
 */
class DeviceMessage {
 public:
  template <typename T>
  KOKKOS_FUNCTION DeviceMessage& operator<<(const T& value) {
    if constexpr (std::is_same_v<T, bool>) {
      print_character(value ? '1' : '0');
    } else if constexpr (std::is_same_v<T, char> or
                         std::is_same_v<T, signed char> or
                         std::is_same_v<T, unsigned char>) {
      print_character(static_cast<char>(value));
    } else if constexpr (std::is_integral_v<T> and std::is_signed_v<T>) {
      print_signed(static_cast<long long>(value));
    } else if constexpr (std::is_integral_v<T>) {
      print_unsigned(static_cast<unsigned long long>(value));
    } else if constexpr (std::is_floating_point_v<T>) {
      print_floating_point(static_cast<double>(value));
    } else if constexpr (std::is_enum_v<T>) {
      // Promote so enums based on `char` types print as numbers
      *this << +static_cast<std::underlying_type_t<T>>(value);
    } else if constexpr (std::is_same_v<std::remove_extent_t<T>, char> or
                         std::is_same_v<T, const char*> or
                         std::is_same_v<T, char*>) {
      print_string(static_cast<const char*>(value));
    } else if constexpr (tt::is_std_array_v<T>) {
      print_character('(');
      bool first = true;
      for (const auto& element : value) {
        if (not first) {
          print_character(',');
        }
        first = false;
        *this << element;
      }
      print_character(')');
    } else {
      print_string("<?>");
    }
    return *this;
  }

  /// Manipulators such as `std::scientific` are ignored
  KOKKOS_FUNCTION DeviceMessage& operator<<(
      std::ios_base& (* /*manipulator*/)(std::ios_base&)) {
    return *this;
  }

 private:
  KOKKOS_FUNCTION static void print_character(char character);
  KOKKOS_FUNCTION static void print_string(const char* str);
  KOKKOS_FUNCTION static void print_signed(long long value);
  KOKKOS_FUNCTION static void print_unsigned(unsigned long long value);
  KOKKOS_FUNCTION static void print_floating_point(double value);
};

#ifdef SPECTRE_KOKKOS
namespace DeviceMessage_detail {
// Returns true for the first thread that fails, which prints its message
KOKKOS_FUNCTION bool claim_failure_message();

KOKKOS_FUNCTION void print_failure_heading(const char* heading,
                                           const char* description,
                                           const char* file, int line,
                                           const char* pretty_function);

// Prints the end of the message and aborts
[[noreturn]] KOKKOS_FUNCTION void finish_failure_message(
    const char* heading, const char* abort_message);

// Ends a failing thread other than the first without printing anything
[[noreturn]] KOKKOS_FUNCTION void end_other_failing_thread(
    const char* abort_message);
}  // namespace DeviceMessage_detail

#define SPECTRE_DEVICE_MESSAGE_STRINGIFY_IMPL(x) #x
#define SPECTRE_DEVICE_MESSAGE_STRINGIFY(x) \
  SPECTRE_DEVICE_MESSAGE_STRINGIFY_IMPL(x)
// The message passed to `Kokkos::abort` as a string literal, so it is
// printed even if the full message is lost
#define SPECTRE_DEVICE_MESSAGE_ABORT_MESSAGE(HEADING, DESCRIPTION) \
  HEADING " ON THE DEVICE in " __FILE__                            \
          ":" SPECTRE_DEVICE_MESSAGE_STRINGIFY(__LINE__) " " DESCRIPTION

/*!
 * \ingroup ErrorHandlingGroup
 * \brief Prints a failure message in device code and aborts the program.
 *
 * Used by `ASSERT` and `ERROR` in device code. `HEADING` and `DESCRIPTION`
 * must be string literals, where the `DESCRIPTION` (e.g. the failed
 * expression) can be empty. `MESSAGE` is an expression that can be streamed
 * into a `DeviceMessage`.
 *
 * Often many GPU threads fail at the same time. Only the first of them prints
 * the full message and then aborts the program. On CUDA devices the others end
 * right away without printing anything (with the PTX instruction `exit`, which
 * ends only the calling thread), so the kernel keeps running until the first
 * thread has printed its message. On HIP devices a single thread can't end by
 * itself, so the others wait for the first thread to print its message and
 * then stop silently. If it doesn't finish in time, e.g. because the GPU can't
 * run it while the others wait, they abort as well. This keeps the message
 * from interleaving with output of other threads, and from being overwritten
 * in the circular printf buffer of the GPU. The message passed to
 * `Kokkos::abort` contains the heading, file, line and description, so these
 * are printed even if the full message is lost.
 */
#define SPECTRE_ABORT_ON_DEVICE(HEADING, DESCRIPTION, MESSAGE)         \
  do {                                                                 \
    if (::DeviceMessage_detail::claim_failure_message()) {             \
      ::DeviceMessage_detail::print_failure_heading(                   \
          HEADING, DESCRIPTION, __FILE__, __LINE__,                    \
          static_cast<const char*>(__PRETTY_FUNCTION__));              \
      ::DeviceMessage device_message_ABORT_ON_DEVICE{};                \
      device_message_ABORT_ON_DEVICE << MESSAGE; /* NOLINT */          \
      ::DeviceMessage_detail::finish_failure_message(                  \
          HEADING,                                                     \
          SPECTRE_DEVICE_MESSAGE_ABORT_MESSAGE(HEADING, DESCRIPTION)); \
    }                                                                  \
    ::DeviceMessage_detail::end_other_failing_thread(                  \
        SPECTRE_DEVICE_MESSAGE_ABORT_MESSAGE(HEADING, DESCRIPTION));   \
  } while (false)
#endif  // SPECTRE_KOKKOS
