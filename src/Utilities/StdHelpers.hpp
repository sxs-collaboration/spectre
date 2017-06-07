// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines helper functions for the standard library

#pragma once

#include <array>
#include <chrono>
#include <ctime>
#include <future>
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
 * \brief Output all the key, value pairs of a map
 */
template <typename Map, typename std::enable_if_t<tt::is_maplike<Map>::value>*>
inline std::ostream& operator<<(std::ostream& os, const Map& m) {
  StdHelpers_detail::print_helper(
      os, begin(m), end(m),
      [](std::ostream& out, typename Map::const_iterator it) {
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
  StdHelpers_detail::print_helper(os, std::begin(v), std::end(v));
  return os;
}

/*!
 * \ingroup Utilities
 * \brief Output the items of a std::unordered_set
 */
template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::set<T>& v) {
  StdHelpers_detail::print_helper(os, std::begin(v), std::end(v));
  return os;
}

/*!
 * \ingroup Utilities
 * \brief Stream operator for std::unique_ptr
 */
template <typename T,
          typename std::enable_if_t<tt::is_streamable<std::ostream, T>::value>*>
inline std::ostream& operator<<(std::ostream& os, const std::unique_ptr<T>& t) {
  return os << *t;
}

/*!
 * \ingroup Utilities
 * \brief Stream operator for std::shared_ptr
 */
template <typename T,
          typename std::enable_if_t<tt::is_streamable<std::ostream, T>::value>*>
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
 * \brief Construct a string containing the keys of a map
 */
template <typename Map,
          typename std::enable_if_t<tt::is_maplike<Map>::value>* = nullptr>
inline std::string keys_of(const Map& m) {
  std::ostringstream os;
  StdHelpers_detail::print_helper(
      os, begin(m), end(m),
      [](std::ostream& out, typename Map::const_iterator it) {
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
  auto requiredBytes = static_cast<size_t>(
      std::snprintf(nullptr, 0, fmt.c_str(), args...) + 1);  // NOLINT
  std::string rtn;
  rtn.resize(requiredBytes);
  std::snprintf(&rtn[0], requiredBytes, fmt.c_str(), args...);  // NOLINT
#pragma GCC diagnostic pop
  if (rtn[rtn.size() - 1] == '\0') {
    rtn.resize(rtn.size() - 1);
  }
  return rtn;
}
