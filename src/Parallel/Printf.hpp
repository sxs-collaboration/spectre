// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines Parallel::printf for writing to stdout

#pragma once

#include <charm++.h>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits/IsStreamable.hpp"

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

template <typename Printer, typename... Ts>
inline void print_helper(Printer&& printer, const std::string& format,
                         Ts&&... t) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay)
  printer(format.c_str(), get_printable_type(std::forward<Ts>(t))...);
}

template <typename Printer, typename... Args>
inline void call_printer(Printer&& printer, const std::string& format,
                         Args&&... args) {
  disable_floating_point_exceptions();
  print_helper(printer, format,
               stream_object_to_string(std::forward<Args>(args))...);
  enable_floating_point_exceptions();
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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#pragma GCC diagnostic ignored "-Wformat-security"
  detail::call_printer(
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
      [](auto... a) { CkPrintf(a...); }, format, std::forward<Args>(args)...);
#pragma GCC diagnostic pop
}

/*!
 * \ingroup ParallelGroup
 * \brief Print an atomic message to stderr with C printf usage.
 *
 * See Parallel::printf for details.
 */
template <typename... Args>
inline void printf_error(const std::string& format, Args&&... args) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#pragma GCC diagnostic ignored "-Wformat-security"
  detail::call_printer(
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
      [](auto... a) { CkError(a...); }, format, std::forward<Args>(args)...);
#pragma GCC diagnostic pop
}
}  // namespace Parallel
