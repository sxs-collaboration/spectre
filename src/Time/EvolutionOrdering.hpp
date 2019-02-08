// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>
#include <utility>

/// \ingroup TimeGroup
/// Ordering functors that reverse their order when time runs
/// backwards.
///
/// \see std::less
//@{
template <typename T = void>
struct evolution_less {
  bool time_runs_forward = true;
  constexpr bool operator()(const T& x, const T& y) const noexcept {
    return time_runs_forward ? x < y : y < x;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept { p | time_runs_forward; }
};

template <typename T = void>
struct evolution_greater {
  bool time_runs_forward = true;
  constexpr bool operator()(const T& x, const T& y) const noexcept {
    return time_runs_forward ? x > y : y > x;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept { p | time_runs_forward; }
};

template <typename T = void>
struct evolution_less_equal {
  bool time_runs_forward = true;
  constexpr bool operator()(const T& x, const T& y) const noexcept {
    return time_runs_forward ? x <= y : y <= x;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept { p | time_runs_forward; }
};

template <typename T = void>
struct evolution_greater_equal {
  bool time_runs_forward = true;
  constexpr bool operator()(const T& x, const T& y) const noexcept {
    return time_runs_forward ? x >= y : y >= x;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept { p | time_runs_forward; }
};

template <>
struct evolution_less<void> {
  bool time_runs_forward = true;
  template <typename T, typename U>
  constexpr auto operator()(T&& t, U&& u) const noexcept {
    return time_runs_forward ? std::forward<T>(t) < std::forward<U>(u)
                             : std::forward<U>(u) < std::forward<T>(t);
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept { p | time_runs_forward; }
};

template <>
struct evolution_greater<void> {
  bool time_runs_forward = true;
  template <typename T, typename U>
  constexpr auto operator()(T&& t, U&& u) const noexcept {
    return time_runs_forward ? std::forward<T>(t) > std::forward<U>(u)
                             : std::forward<U>(u) > std::forward<T>(t);
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept { p | time_runs_forward; }
};

template <>
struct evolution_less_equal<void> {
  bool time_runs_forward = true;
  template <typename T, typename U>
  constexpr auto operator()(T&& t, U&& u) const noexcept {
    return time_runs_forward ? std::forward<T>(t) <= std::forward<U>(u)
                             : std::forward<U>(u) <= std::forward<T>(t);
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept { p | time_runs_forward; }
};

template <>
struct evolution_greater_equal<void> {
  bool time_runs_forward = true;
  template <typename T, typename U>
  constexpr auto operator()(T&& t, U&& u) const noexcept {
    return time_runs_forward ? std::forward<T>(t) >= std::forward<U>(u)
                             : std::forward<U>(u) >= std::forward<T>(t);
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept { p | time_runs_forward; }
};
//@}
