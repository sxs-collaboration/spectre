// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines helper functions for the standard library.

#pragma once

#include <array>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <deque>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Utilities/PrintHelpers.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/StlStreamDeclarations.hpp"
#include "Utilities/TypeTraits.hpp"

namespace StdHelpers_detail {
// Helper classes for operator<< for tuples
template <size_t N>
struct TuplePrinter {
  template <typename... Args>
  static std::ostream& print(std::ostream& os, const std::tuple<Args...>& t) {
    TuplePrinter<N - 1>::print(os, t);
    os << "," << std::get<N - 1>(t);
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
 * \ingroup UtilitiesGroup
 * \brief Output the items of a std::list
 */
template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::list<T>& v) {
  sequence_print_helper(os, std::begin(v), std::end(v));
  return os;
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Output the items of a std::vector
 */
template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
  sequence_print_helper(os, std::begin(v), std::end(v));
  return os;
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Output the items of a std::deque
 */
template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::deque<T>& v) {
  sequence_print_helper(os, std::begin(v), std::end(v));
  return os;
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Output the items of a std::array
 */
template <typename T, size_t N>
inline std::ostream& operator<<(std::ostream& os, const std::array<T, N>& a) {
  sequence_print_helper(os, begin(a), end(a));
  return os;
}

/*!
 * \ingroup UtilitiesGroup
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
 * \ingroup UtilitiesGroup
 * \brief Output all the key, value pairs of a std::unordered_map
 */
template <typename K, typename V, typename H>
inline std::ostream& operator<<(std::ostream& os,
                                const std::unordered_map<K, V, H>& m) {
  unordered_print_helper(
      os, begin(m), end(m),
      [](std::ostream& out,
         typename std::unordered_map<K, V, H>::const_iterator it) {
        out << "[" << it->first << "," << it->second << "]";
      });
  return os;
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Output all the key, value pairs of a std::map
 */
template <typename K, typename V, typename C>
inline std::ostream& operator<<(std::ostream& os, const std::map<K, V, C>& m) {
  sequence_print_helper(
      os, begin(m), end(m),
      [](std::ostream& out, typename std::map<K, V, C>::const_iterator it) {
        out << "[" << it->first << "," << it->second << "]";
      });
  return os;
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Output the items of a std::unordered_set
 */
template <typename T>
inline std::ostream& operator<<(std::ostream& os,
                                const std::unordered_set<T>& v) {
  unordered_print_helper(os, std::begin(v), std::end(v));
  return os;
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Output the items of a std::set
 */
template <typename T, typename C>
inline std::ostream& operator<<(std::ostream& os, const std::set<T, C>& v) {
  sequence_print_helper(os, std::begin(v), std::end(v));
  return os;
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Stream operator for std::unique_ptr
 */
template <typename T, Requires<tt::is_streamable<std::ostream, T>::value>>
inline std::ostream& operator<<(std::ostream& os, const std::unique_ptr<T>& t) {
  return os << *t;
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Stream operator for std::shared_ptr
 */
template <typename T, Requires<tt::is_streamable<std::ostream, T>::value>>
inline std::ostream& operator<<(std::ostream& os, const std::shared_ptr<T>& t) {
  return os << *t;
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Stream operator for std::pair
 */
template <typename T, typename U>
inline std::ostream& operator<<(std::ostream& os, const std::pair<T, U>& t) {
  return os << "(" << t.first << ", " << t.second << ")";
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Construct a string containing the keys of a std::unordered_map
 */
template <typename K, typename V, typename H>
inline std::string keys_of(const std::unordered_map<K, V, H>& m) {
  std::ostringstream os;
  unordered_print_helper(
      os, begin(m), end(m),
      [](std::ostream& out,
         typename std::unordered_map<K, V, H>::const_iterator it) {
        out << it->first;
      });
  return os.str();
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Construct a string containing the keys of a std::map
 */
template <typename K, typename V, typename C>
inline std::string keys_of(const std::map<K, V, C>& m) {
  std::ostringstream os;
  sequence_print_helper(
      os, begin(m), end(m),
      [](std::ostream& out, typename std::map<K, V, C>::const_iterator it) {
        out << it->first;
      });
  return os.str();
}

/*!
 * \ingroup UtilitiesGroup
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
 * \ingroup UtilitiesGroup
 * \brief Get the current date and time
 */
inline std::string current_date_and_time() {
  const auto now =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  return std::ctime(&now);
}
