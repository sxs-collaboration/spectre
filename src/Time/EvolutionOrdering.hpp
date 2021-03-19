// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <functional>
#include <pup.h>
#include <utility>

/// \ingroup TimeGroup
/// Implementation of \ref evolution_less, \ref evolution_greater,
/// \ref evolution_less_equal, and \ref evolution_greater_equal.
///
/// This should only be used through the named aliases, but provides
/// the documentation of the members.
//@{
template <typename T, template <typename> typename Comparator>
struct evolution_comparator {
  bool time_runs_forward = true;
  constexpr bool operator()(const T& x, const T& y) const noexcept {
    return time_runs_forward ? Comparator<T>{}(x, y) : Comparator<T>{}(y, x);
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept { p | time_runs_forward; }
};

template <template <typename> typename Comparator>
struct evolution_comparator<void, Comparator> {
  bool time_runs_forward = true;

  template <typename T, typename U>
  constexpr auto operator()(T&& t, U&& u) const noexcept {
    return time_runs_forward
               ? Comparator<void>{}(std::forward<T>(t), std::forward<U>(u))
               : Comparator<void>{}(std::forward<U>(u), std::forward<T>(t));
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept { p | time_runs_forward; }
};
//@}

/// \ingroup TimeGroup
/// Ordering functors that reverse their order when time runs
/// backwards.  See evolution_comparator and
/// evolution_comparator<void,Comparator> for the provided interface.
///
/// \see std::less
template <typename T = void>
using evolution_less = evolution_comparator<T, std::less>;

/// \ingroup TimeGroup
/// \copydoc evolution_less
template <typename T = void>
using evolution_greater = evolution_comparator<T, std::greater>;

/// \ingroup TimeGroup
/// \copydoc evolution_less
template <typename T = void>
using evolution_less_equal = evolution_comparator<T, std::less_equal>;

/// \ingroup TimeGroup
/// \copydoc evolution_less
template <typename T = void>
using evolution_greater_equal = evolution_comparator<T, std::greater_equal>;
