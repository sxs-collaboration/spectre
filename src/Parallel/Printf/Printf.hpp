// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines Parallel::printf for writing to stdout

#pragma once

#include <cstddef>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits/IsStreamable.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

#include "Parallel/Printf/Printf.decl.h"

namespace Parallel {
namespace detail {
/*!
 * Fundamentals and pointers are already printable so there is no conversion
 * to a std::string necessary.
 */
template <typename T,
          Requires<std::is_fundamental<std::decay_t<
                       std::remove_pointer_t<std::decay_t<T>>>>::value or
                   std::is_pointer<T>::value or
                   std::is_pointer<std::decay_t<T>>::value> = nullptr>
inline constexpr T stream_object_to_string(T&& t) {
  return t;
}

/*!
 * Stream an object of type `T` into a std::stringstream and return it as a
 * std::string so that it is printable by calling `.c_str()` on it.
 * We need a 2-phase call so that the std::string doesn't go out of scope before
 * the C-style string is passed to printf.
 */
template <typename T,
          Requires<std::is_class<std::decay_t<T>>::value or
                   std::is_enum<std::decay_t<T>>::value> = nullptr>
inline std::string stream_object_to_string(T&& t) {
  using ::operator<<;
  static_assert(tt::is_streamable<std::stringstream, T>::value,
                "Cannot stream type and therefore it cannot be printed. Please "
                "define a stream operator for the type.");
  std::stringstream ss;
  ss << std::setprecision(std::numeric_limits<double>::digits10 + 4)
     << std::scientific << t;
  return ss.str();
}

/*!
 * Fundamentals are already printable, so nothing to do.
 */
template <typename T,
          Requires<std::is_fundamental<std::decay_t<
              std::remove_pointer_t<std::decay_t<T>>>>::value> = nullptr>
inline constexpr T get_printable_type(T&& t) {
  return t;
}

/*!
 * Get the pointer of the std::string so it can be passed to CkPrintf which
 * only works on fundamentals
 */
inline const typename std::string::value_type* get_printable_type(
    const std::string& t) {
  return t.c_str();
}

void send_message(bool error, const std::vector<char>& message);
void send_message_to_file(const std::string& file,
                          const std::vector<char>& message);

template <typename... Ts>
inline std::vector<char> allocate_message(const char* const format, Ts&&... t) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#pragma GCC diagnostic ignored "-Wformat-security"
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay)
  const auto length = static_cast<size_t>(snprintf(nullptr, 0, format, t...));
  std::vector<char> message(length + 1);  // +1 for the null byte
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay)
  snprintf(message.data(), message.size(), format, t...);
#pragma GCC diagnostic pop
  return message;
}

template <typename... Args>
inline std::vector<char> format_message(const std::string& format,
                                        Args&&... args) {
  const ScopedFpeState disable_fpes(false);
  return allocate_message(
      format.c_str(),
      get_printable_type(stream_object_to_string(std::forward<Args>(args)))...);
}
}  // namespace detail

/*!
 * \ingroup ParallelGroup
 * \brief Print an atomic message to stdout with C printf usage.
 *
 * Similar to Python, you can print any object that's streamable by passing it
 * in as an argument and using the formatter "%s". For example,
 * \code
 * std::vector<double> a{0.8, 73, 9.8};
 * Parallel::printf("%s\n", a);
 * \endcode
 */
template <typename... Args>
inline void printf(const std::string& format, Args&&... args) {
  detail::send_message(
      false, detail::format_message(format, std::forward<Args>(args)...));
}

/*!
 * \ingroup ParallelGroup
 * \brief Print an atomic message to stderr with C printf usage.
 *
 * See Parallel::printf for details.
 */
template <typename... Args>
inline void printf_error(const std::string& format, Args&&... args) {
  detail::send_message(
      true, detail::format_message(format, std::forward<Args>(args)...));
}

/*!
 * \ingroup ParallelGroup
 * \brief Print an atomic message to a file with C printf usage.
 *
 * See Parallel::printf for details.
 */
template <typename... Args>
inline void fprintf(const std::string& file, const std::string& format,
                    Args&&... args) {
  detail::send_message_to_file(
      file, detail::format_message(format, std::forward<Args>(args)...));
}

/// Chare outputting all Parallel::printf results.
class PrinterChare : public CBase_PrinterChare {
 public:
  PrinterChare() = default;
  explicit PrinterChare(CkMigrateMessage* /*msg*/) {}

  /// Prints a message to stdout or stderr.
  static void print(bool error, const std::vector<char>& message);

  /// Prints a message to a file.
  static void print_to_file(const std::string& file,
                            const std::vector<char>& message);

  static void register_with_charm();

  void pup(PUP::er& /*p*/) override {}
};

// Charm readonly variables set in Main.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern CProxy_PrinterChare printer_chare;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern bool printer_chare_is_set;
}  // namespace Parallel
