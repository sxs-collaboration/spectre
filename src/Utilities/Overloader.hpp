// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <utility>

#include "Utilities/ForceInline.hpp"

#if defined(__clang__) || __GNUC__ > 5
#define OVERLOADER_CONSTEXPR \
  constexpr
#else
#define OVERLOADER_CONSTEXPR
#endif

namespace overloader_detail {
struct no_such_type;
}  // namespace overloader_detail

/*!
 * \ingroup UtilitiesGroup
 * \brief Used for overloading lambdas, useful for lambda-SFINAE
 *
 * \snippet Utilities/Test_Overloader.cpp overloader_example
 */
template <class... Fs>
class Overloader;

template <class F1, class F2, class F3, class F4, class F5, class F6, class F7,
          class F8, class... Fs>
class Overloader<F1, F2, F3, F4, F5, F6, F7, F8, Fs...>
    : F1, F2, F3, F4, F5, F6, F7, F8, Overloader<Fs...> {
 public:
  OVERLOADER_CONSTEXPR Overloader(F1 f1, F2 f2, F3 f3, F4 f4, F5 f5, F6 f6,
                                  F7 f7, F8 f8, Fs... fs)
      : F1(std::move(f1)),
        F2(std::move(f2)),
        F3(std::move(f3)),
        F4(std::move(f4)),
        F5(std::move(f5)),
        F6(std::move(f6)),
        F7(std::move(f7)),
        F8(std::move(f8)),
        Overloader<Fs...>(std::move(fs)...) {}

  using F1::operator();
  using F2::operator();
  using F3::operator();
  using F4::operator();
  using F5::operator();
  using F6::operator();
  using F7::operator();
  using F8::operator();
  using Overloader<Fs...>::operator();
};

template <class F1, class F2, class F3, class F4, class... Fs>
class Overloader<F1, F2, F3, F4, Fs...> : F1, F2, F3, F4, Overloader<Fs...> {
 public:
  OVERLOADER_CONSTEXPR Overloader(F1 f1, F2 f2, F3 f3, F4 f4, Fs... fs)
      : F1(std::move(f1)),
        F2(std::move(f2)),
        F3(std::move(f3)),
        F4(std::move(f4)),
        Overloader<Fs...>(std::move(fs)...) {}

  using F1::operator();
  using F2::operator();
  using F3::operator();
  using F4::operator();
  using Overloader<Fs...>::operator();
};

template <class F1, class F2, class... Fs>
class Overloader<F1, F2, Fs...> : F1, F2, Overloader<Fs...> {
 public:
  OVERLOADER_CONSTEXPR Overloader(F1 f1, F2 f2, Fs... fs)
      : F1(std::move(f1)),
        F2(std::move(f2)),
        Overloader<Fs...>(std::move(fs)...) {}

  using F1::operator();
  using F2::operator();
  using Overloader<Fs...>::operator();
};

template <class F>
class Overloader<F> : F {
 public:
  explicit OVERLOADER_CONSTEXPR Overloader(F f) : F(std::move(f)) {}

  using F::operator();
};

template <>
class Overloader<> {
 public:
  using type = Overloader;
  SPECTRE_ALWAYS_INLINE void operator()(
      const overloader_detail::no_such_type& /*unused*/) noexcept {}
};

/*!
 * \ingroup UtilitiesGroup
 * \brief Create `Overloader<Fs...>`, see Overloader for details
 */
template <class... Fs>
OVERLOADER_CONSTEXPR Overloader<Fs...> make_overloader(Fs... fs) {
  return Overloader<Fs...>{std::move(fs)...};
}
