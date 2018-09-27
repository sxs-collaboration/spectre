// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <tuple>

#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/StaticAssert.hpp"

/*!
 * \ingroup UtilitiesGroup
 * \brief Higher order function objects similar to `std::plus`, etc.
 */
namespace funcl {
/// \cond
template <size_t Arity>
struct Functional {
  static constexpr size_t arity = Arity;

 protected:
  template <class C, size_t Offset, class... Ts, size_t... Is>
  static constexpr decltype(auto) helper(
      const std::tuple<Ts...>& t,
      std::index_sequence<Is...> /*meta*/) noexcept {
    return C{}(std::get<Offset + Is>(t)...);
  }
};

struct Identity;
/// \endcond

#define MAKE_BINARY_OPERATOR(NAME, OPERATOR)                            \
  template <class C0 = Identity, class C1 = C0>                         \
  struct NAME : Functional<C0::arity + C1::arity> {                     \
    using base = Functional<C0::arity + C1::arity>;                     \
    template <class... Ts>                                              \
    constexpr auto operator()(const Ts&... ts) noexcept {               \
      return base::template helper<C0, 0>(                              \
          std::tuple<const Ts&...>(ts...),                              \
          std::make_index_sequence<C0::arity>{})                        \
          OPERATOR base::template helper<C1, C0::arity>(                \
              std::tuple<const Ts&...>(ts...),                          \
              std::make_index_sequence<C1::arity>{});                   \
    }                                                                   \
  };                                                                    \
  /** \cond */                                                          \
  template <class C1>                                                   \
  struct NAME<Identity, C1> : Functional<1 + C1::arity> {               \
    template <class T0, class... Ts>                                    \
    constexpr auto operator()(const T0& t0, const Ts&... ts) noexcept { \
      return t0 OPERATOR C1{}(ts...);                                   \
    }                                                                   \
  };                                                                    \
  template <>                                                           \
  struct NAME<Identity, Identity> : Functional<2> {                     \
    template <class T0, class T1>                                       \
    constexpr auto operator()(const T0& t0, const T1& t1) noexcept {    \
      return t0 OPERATOR t1;                                            \
    }                                                                   \
  };                                                                    \
  /** \endcond*/

/// Functional that asserts the first and second arguments are equal and returns
/// the first argument.
template <class C = Identity>
struct AssertEqual : Functional<2> {
  template <class T>
  const T& operator()(const T& t0, const T& t1) noexcept {
    DEBUG_STATIC_ASSERT(
        C::arity == 1,
        "The arity of the functional passed to AssertEqual must be 1");
    ASSERT(t0 == t1, "Values are not equal in funcl::AssertEqual "
                         << t0 << " and " << t1);
    return C{}(t0);
  }
};

/// Functional for dividing two objects
MAKE_BINARY_OPERATOR(Divides, /)

/// Functional to retrieve the `ArgumentIndex`th argument
template <size_t Arity, size_t ArgumentIndex = 0, class C = Identity>
struct GetArgument : Functional<Arity> {
  template <class... Ts>
  constexpr auto operator()(const Ts&... ts) noexcept {
    static_assert(Arity == sizeof...(Ts),
                  "The arity passed to GetArgument must be the same as the "
                  "actually arity of the function.");
    return C{}(std::get<ArgumentIndex>(std::tuple<const Ts&...>(ts...)));
  }
};

/// The identity higher order function object
struct Identity : Functional<1> {
  template <class T>
  constexpr const T& operator()(const T& t) noexcept {
    return t;
  }
};

/// Functional for subtracting two objects
MAKE_BINARY_OPERATOR(Minus, -)

/// Functional for multiplying two objects
MAKE_BINARY_OPERATOR(Multiplies, *)

/// Functional for negating an object
template <class C = Identity>
struct Negate : Functional<C::arity> {
  template <class... Ts>
  constexpr auto operator()(const Ts&... ts) noexcept {
    return -C{}(ts...);
  }
};

/// Functional for adding two objects
MAKE_BINARY_OPERATOR(Plus, +)

/// Functional for taking the square root of a quantity
template <class C = Identity>
struct Sqrt : Functional<C::arity> {
  template <class... Ts>
  constexpr auto operator()(const Ts&... ts) noexcept {
    using std::sqrt;
    return sqrt(C{}(ts...));
  }
};

/// Function for squaring a quantity
template <class C = Identity>
struct Square : Functional<C::arity> {
  template <class... Ts>
  constexpr auto operator()(const Ts&... ts) noexcept {
    decltype(auto) result = C{}(ts...);
    return result * result;
  }
};

#undef MAKE_BINARY_OPERATOR
}  // namespace funcl
