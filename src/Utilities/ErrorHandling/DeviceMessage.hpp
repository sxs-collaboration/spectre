// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
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
 * The message is collected in a fixed-size buffer and printed all at once,
 * because GPUs print the output of separate `printf` calls in no guaranteed
 * order. See `SPECTRE_ABORT_ON_DEVICE`.
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
 *
 * Manipulators such as `std::scientific` are ignored. Manipulators that take
 * an argument, such as `std::setprecision`, are host functions and can't be
 * used on the device; do not use them with this stream.
 *
 * Values of other types are printed as `<?>`. Add an overload
 * `KOKKOS_FUNCTION DeviceMessage& operator<<(DeviceMessage&, const T&)` to
 * print them on the device. Messages longer than `capacity` characters are
 * truncated, which is marked by `...` at the end.
 */
class DeviceMessage {
 public:
  /// \brief The size of the buffer, including the null terminator
  ///
  /// `ASSERT` and `ERROR` use a single static buffer on the device (see
  /// `SPECTRE_ABORT_ON_DEVICE`), so the capacity doesn't cost memory per GPU
  /// thread. Each message is printed with a single `printf`, which must fit in
  /// the device's printf buffer (1 MB by default on CUDA, see
  /// `cudaLimitPrintfFifoSize`).
  static constexpr size_t capacity = 1024;

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
    } else {
      append("<?>");
    }
    return *this;
  }

  /// Manipulators such as `std::scientific` are ignored
  KOKKOS_FUNCTION DeviceMessage& operator<<(
      std::ios_base& (* /*manipulator*/)(std::ios_base&)) {
    return *this;
  }

 private:
  // Defined in the .cpp file, so the formatting code is compiled only once
  KOKKOS_FUNCTION void put(char character);
  KOKKOS_FUNCTION void append(const char* str);
  KOKKOS_FUNCTION void append_unsigned(unsigned long long value,
                                       size_t min_digits = 1);
  KOKKOS_FUNCTION void append_signed(long long value);
  KOKKOS_FUNCTION void append_double(double value);

  std::array<char, capacity> buffer_{};
  size_t size_ = 0;
  bool truncated_ = false;
};

#ifdef SPECTRE_KOKKOS
namespace DeviceMessage_detail {
// Returns true for the first thread that fails, which prints its message
KOKKOS_FUNCTION bool claim_failure_message();

// Starts the message in the static buffer and returns it
KOKKOS_FUNCTION DeviceMessage& start_failure_message(const char* heading,
                                                     const char* description,
                                                     const char* file,
                                                     int line);

// Ends the message with the function name, prints it and aborts
[[noreturn]] KOKKOS_FUNCTION void finish_failure_message(
    DeviceMessage& message, const char* pretty_function);

// Ends a failing thread other than the first without printing anything
[[noreturn]] KOKKOS_FUNCTION void end_other_failing_thread(
    const char* abort_message);
}  // namespace DeviceMessage_detail

#define SPECTRE_DEVICE_MESSAGE_STRINGIFY_IMPL(x) #x
#define SPECTRE_DEVICE_MESSAGE_STRINGIFY(x) \
  SPECTRE_DEVICE_MESSAGE_STRINGIFY_IMPL(x)

/*!
 * \ingroup ErrorHandlingGroup
 * \brief Prints a failure message in device code and aborts the program.
 *
 * Used by `ASSERT` and `ERROR` in device code. `HEADING` and `DESCRIPTION`
 * must be string literals, where the `DESCRIPTION` (e.g. the failed
 * expression) can be empty. `MESSAGE` is an expression that can be streamed
 * into a `DeviceMessage`.
 *
 * Often many GPU threads fail at the same time. Only the first of them formats
 * its message, in a single static buffer, and passes it to `Kokkos::abort`,
 * which prints it with the indices of the GPU block and thread. Printing it at
 * once keeps it from interleaving with other output, because GPUs print the
 * output of separate `printf` calls in no guaranteed order. The message starts
 * with the heading, file, line and description, and ends with the function
 * name, so truncating a long message cuts the function name first.
 *
 * On CUDA devices the other failing threads end right away without printing
 * anything (with the PTX instruction `exit`, which ends only the calling
 * thread), so the kernel keeps running until the first thread has printed its
 * message. On HIP devices a single thread can't end by itself, so the others
 * wait until the abort of the first thread ends the kernel. If that doesn't
 * happen in time, e.g. because the GPU can't run the first thread while the
 * others wait, they abort with a short message that contains the heading,
 * file, line and description.
 *
 * \warning Kokkos warns that the SYCL backend does not terminate the program on
 * an `abort() `call. For us this is undefined behavior and needs to be
 * revisited before properly support SYCL.
 */
#define SPECTRE_ABORT_ON_DEVICE(HEADING, DESCRIPTION, MESSAGE)                \
  do {                                                                        \
    if (::DeviceMessage_detail::claim_failure_message()) {                    \
      ::DeviceMessage& device_message_ABORT_ON_DEVICE =                       \
          ::DeviceMessage_detail::start_failure_message(HEADING, DESCRIPTION, \
                                                        __FILE__, __LINE__);  \
      device_message_ABORT_ON_DEVICE << MESSAGE; /* NOLINT */                 \
      ::DeviceMessage_detail::finish_failure_message(                         \
          device_message_ABORT_ON_DEVICE,                                     \
          static_cast<const char*>(__PRETTY_FUNCTION__));                     \
    }                                                                         \
    ::DeviceMessage_detail::end_other_failing_thread(                         \
        HEADING                                                               \
        " in " __FILE__                                                       \
        ":" SPECTRE_DEVICE_MESSAGE_STRINGIFY(__LINE__) " " DESCRIPTION);      \
  } while (false)
#endif  // SPECTRE_KOKKOS
