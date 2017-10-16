// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines helper functions for the standard library

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <ctime>
#include <deque>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/StlStreamDeclarations.hpp"
#include "Utilities/TypeTraits.hpp"

namespace StdHelpers_detail {
// applies the function f(out, it) to each item from begin to end, separated
// by commas and surrounded by parens
template <typename ForwardIt, typename Func>
inline void print_helper(std::ostream& out, ForwardIt&& begin, ForwardIt&& end,
                         Func f) {
  out << "(";
  if (begin != end) {
    while (true) {
      f(out, begin++);
      if (begin == end) {
        break;
      }
      out << ",";
    }
  }
  out << ")";
}

// prints all the items as a comma separated list surrounded by parens
template <typename ForwardIt>
inline void print_helper(std::ostream& out, ForwardIt&& begin,
                         ForwardIt&& end) {
  print_helper(out, std::forward<ForwardIt>(begin),
               std::forward<ForwardIt>(end),
               [](std::ostream& os, const ForwardIt& it) { os << *it; });
}

// Like print_helper, but sorts the string representations
template <typename ForwardIt, typename Func>
inline void unordered_print_helper(std::ostream& out, ForwardIt&& begin,
                                   ForwardIt&& end, Func f) {
  std::vector<std::string> entries;
  while (begin != end) {
    std::ostringstream ss;
    f(ss, begin++);
    entries.push_back(ss.str());
  }
  std::sort(entries.begin(), entries.end());
  print_helper(out, entries.begin(), entries.end());
}

template <typename ForwardIt>
inline void unordered_print_helper(std::ostream& out, ForwardIt&& begin,
                                   ForwardIt&& end) {
  unordered_print_helper(
      out, std::forward<ForwardIt>(begin), std::forward<ForwardIt>(end),
      [](std::ostream& os, const ForwardIt& it) { os << *it; });
}

// Helper classes for operator<< for tuples
template <size_t N>
struct TuplePrinter {
  template <typename... Args>
  static std::ostream& print(std::ostream& os, const std::tuple<Args...>& t) {
    TuplePrinter<N - 1>::print(os, t);
    os << ", " << std::get<N - 1>(t);
    return os;
  }
};

template <>
struct TuplePrinter<1> {
  template <typename... Args>
  static std::ostream& print(std::ostream& os, const std::tuple<Args...>& t) {
    os << std::get<0>(t);
    return os;
  }
};

template <>
struct TuplePrinter<0> {
  template <typename... Args>
  static std::ostream& print(std::ostream& os,
                             const std::tuple<Args...>& /*t*/) {
    return os;
  }
};
}  // namespace StdHelpers_detail

/*!
 * \ingroup Utilities
 * \brief Output the items of a std::list
 */
template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::list<T>& v) {
  StdHelpers_detail::print_helper(os, std::begin(v), std::end(v));
  return os;
}

/*!
 * \ingroup Utilities
 * \brief Output the items of a std::vector
 */
template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
  StdHelpers_detail::print_helper(os, std::begin(v), std::end(v));
  return os;
}

/*!
 * \ingroup Utilities
 * \brief Output the items of a std::deque
 */
template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::deque<T>& v) {
  StdHelpers_detail::print_helper(os, std::begin(v), std::end(v));
  return os;
}

/*!
 * \ingroup Utilities
 * \brief Output the items of a std::array
 */
template <typename T, size_t N>
inline std::ostream& operator<<(std::ostream& os, const std::array<T, N>& a) {
  StdHelpers_detail::print_helper(os, begin(a), end(a));
  return os;
}

/*!
 * \ingroup Utilities
 * \brief Stream operator for tuples
 */
template <typename... Args>
inline std::ostream& operator<<(std::ostream& os,
                                const std::tuple<Args...>& t) {
  os << "(";
  StdHelpers_detail::TuplePrinter<sizeof...(Args)>::print(os, t);
  os << ")";
  return os;
}

/*!
 * \ingroup Utilities
 * \brief Output all the key, value pairs of a std::unordered_map
 */
template <typename K, typename V>
inline std::ostream& operator<<(std::ostream& os,
                                const std::unordered_map<K, V>& m) {
  StdHelpers_detail::unordered_print_helper(
      os, begin(m), end(m),
      [](std::ostream& out,
         typename std::unordered_map<K, V>::const_iterator it) {
        out << "[" << it->first << "," << it->second << "]";
      });
  return os;
}

/*!
 * \ingroup Utilities
 * \brief Output all the key, value pairs of a std::map
 */
template <typename K, typename V, typename C>
inline std::ostream& operator<<(std::ostream& os, const std::map<K, V, C>& m) {
  StdHelpers_detail::print_helper(
      os, begin(m), end(m),
      [](std::ostream& out, typename std::map<K, V, C>::const_iterator it) {
        out << "[" << it->first << "," << it->second << "]";
      });
  return os;
}

/*!
 * \ingroup Utilities
 * \brief Output the items of a std::unordered_set
 */
template <typename T>
inline std::ostream& operator<<(std::ostream& os,
                                const std::unordered_set<T>& v) {
  StdHelpers_detail::unordered_print_helper(os, std::begin(v), std::end(v));
  return os;
}

/*!
 * \ingroup Utilities
 * \brief Output the items of a std::set
 */
template <typename T, typename C>
inline std::ostream& operator<<(std::ostream& os, const std::set<T, C>& v) {
  StdHelpers_detail::print_helper(os, std::begin(v), std::end(v));
  return os;
}

/*!
 * \ingroup Utilities
 * \brief Stream operator for std::unique_ptr
 */
template <typename T, Requires<tt::is_streamable<std::ostream, T>::value>>
inline std::ostream& operator<<(std::ostream& os, const std::unique_ptr<T>& t) {
  return os << *t;
}

/*!
 * \ingroup Utilities
 * \brief Stream operator for std::shared_ptr
 */
template <typename T, Requires<tt::is_streamable<std::ostream, T>::value>>
inline std::ostream& operator<<(std::ostream& os, const std::shared_ptr<T>& t) {
  return os << *t;
}

/*!
 * \ingroup Utilities
 * \brief Stream operator for std::pair
 */
template <typename T, typename U>
inline std::ostream& operator<<(std::ostream& os, const std::pair<T, U>& t) {
  return os << "(" << t.first << ", " << t.second << ")";
}

/*!
 * \ingroup Utilities
 * \brief Construct a string containing the keys of a std::unordered_map
 */
template <typename K, typename V>
inline std::string keys_of(const std::unordered_map<K, V>& m) {
  std::ostringstream os;
  StdHelpers_detail::unordered_print_helper(
      os, begin(m), end(m),
      [](std::ostream& out,
         typename std::unordered_map<K, V>::const_iterator it) {
        out << it->first;
      });
  return os.str();
}

/*!
 * \ingroup Utilities
 * \brief Construct a string containing the keys of a std::map
 */
template <typename K, typename V, typename C>
inline std::string keys_of(const std::map<K, V, C>& m) {
  std::ostringstream os;
  StdHelpers_detail::print_helper(
      os, begin(m), end(m),
      [](std::ostream& out, typename std::map<K, V, C>::const_iterator it) {
        out << it->first;
      });
  return os.str();
}

/*!
 * \ingroup Utilities
 * \brief Format a string like printf
 *
 * Given a formatting string and arguments this returns the corresponding
 * string. Similar to printf but using std::strings.
 */
template <typename... Args>
std::string formatted_string(const std::string& fmt, Args... args) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
  // clang-tidy: do not use snprintf
  auto requiredBytes = static_cast<size_t>(std::snprintf(  // NOLINT
                           nullptr, 0, fmt.c_str(), args...)) +
                       1;
  std::string rtn;
  rtn.resize(requiredBytes);
  // clang-tidy: do not use snprintf
  std::snprintf(&rtn[0], requiredBytes, fmt.c_str(), args...);  // NOLINT
#pragma GCC diagnostic pop
  if (rtn[rtn.size() - 1] == '\0') {
    rtn.resize(rtn.size() - 1);
  }
  return rtn;
}

/*!
 * \ingroup Utilities
 * \brief Get the current date and time
 */
inline std::string current_date_and_time() {
  const auto now =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  return std::ctime(&now);
}

// Arithmetic operators for std::array<T, Dim>

template <size_t Dim, typename T>
inline std::array<T, Dim>& operator+=(
    std::array<T, Dim>& lhs,
    const std::array<T, Dim>& rhs) noexcept(noexcept(lhs[0] += rhs[0])) {
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(lhs, i) += gsl::at(rhs, i);
  }
  return lhs;
}

template <size_t Dim, typename T>
inline std::array<T, Dim> operator+(
    const std::array<T, Dim>& lhs,
    const std::array<T, Dim>&
        rhs) noexcept(noexcept(std::declval<std::array<T, Dim>&>() += rhs) and
                      noexcept(std::array<T, Dim>{lhs})) {
  std::array<T, Dim> result = lhs;
  result += rhs;
  return result;
}

template <size_t Dim, typename T>
inline std::array<T, Dim>& operator-=(
    std::array<T, Dim>& lhs,
    const std::array<T, Dim>& rhs) noexcept(noexcept(lhs[0] -= rhs[0])) {
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(lhs, i) -= gsl::at(rhs, i);
  }
  return lhs;
}

template <size_t Dim, typename T>
inline std::array<T, Dim> operator-(
    const std::array<T, Dim>& lhs,
    const std::array<T, Dim>&
        rhs) noexcept(noexcept(std::declval<std::array<T, Dim>&>() -= rhs) and
                      noexcept(std::array<T, Dim>{lhs})) {
  std::array<T, Dim> result = lhs;
  result -= rhs;
  return result;
}

template <size_t Dim, typename T, typename U>
inline std::array<T, Dim> operator*(
    const std::array<T, Dim>& lhs,
    const U& scale) noexcept(noexcept(lhs[0] * scale) and
                             noexcept(std::array<T, Dim>{})) {
  std::array<T, Dim> result{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(result, i) = gsl::at(lhs, i) * scale;
  }
  return result;
}

template <size_t Dim, typename T, typename U>
inline std::array<T, Dim> operator*(
    const U& scale,
    const std::array<T, Dim>& rhs) noexcept(noexcept(rhs* scale)) {
  return rhs * scale;
}

template <size_t Dim, typename T, typename U>
inline std::array<T, Dim> operator/(
    const std::array<T, Dim>& lhs,
    const U& scale) noexcept(noexcept(lhs[0] / scale) and
                             noexcept(std::array<T, Dim>{})) {
  std::array<T, Dim> result{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(result, i) = gsl::at(lhs, i) / scale;
  }
  return result;
}

template <size_t Dim, typename T>
inline std::array<T, Dim> operator-(const std::array<T, Dim>& rhs) noexcept(
    noexcept(-rhs[0]) and noexcept(std::array<T, Dim>{})) {
  std::array<T, Dim> result{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(result, i) = -gsl::at(rhs, i);
  }
  return result;
}

template <typename T, size_t Dim>
inline std::array<T, Dim - 1> all_but_first_element_of(
    const std::array<T, Dim>& a) noexcept {
  static_assert(0 != Dim, "Cannot remove first element of empty array.");
  std::array<T, Dim - 1> result{};
  for (size_t i = 1; i < Dim; ++i) {
    gsl::at(result, i - 1) = gsl::at(a, i);
  }
  return result;
}
