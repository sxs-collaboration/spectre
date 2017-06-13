// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Wraps the template metaprogramming library used (brigand)

#pragma once

// Since this header only wraps brigand and several additions to it we mark
// it as a system header file so that clang-tidy ignores it.
#ifdef __GNUC__
#pragma GCC system_header
#endif

/// \cond NEVER
#define BRIGAND_NO_BOOST_SUPPORT
/// \endcond
#include <brigand/brigand.hpp>

#include "Utilities/Digraph.hpp"
#include "Utilities/ForceInline.hpp"

namespace brigand {
namespace detail {
template <bool b, typename O, typename L, std::size_t I, typename R,
          typename U = void>
struct replace_at_impl;

template <template <typename...> class S, typename... Os, typename... Ts,
          typename R>
struct replace_at_impl<false, S<Os...>, S<Ts...>, 0, R> {
  using type = S<Os..., Ts...>;
};

template <template <typename...> class S, typename... Os, typename... Ts,
          typename T, typename R>
struct replace_at_impl<false, S<Os...>, S<T, Ts...>, 1, R>
    : replace_at_impl<false, S<Os..., R>, S<Ts...>, 0, R> {};

template <template <typename...> class S, typename... Os, typename T,
          typename... Ts, std::size_t I, typename R>
struct replace_at_impl<false, S<Os...>, S<T, Ts...>, I, R,
                       typename std::enable_if<(I > 1)>::type>
    : replace_at_impl<false, S<Os..., T>, S<Ts...>, (I - 1), R> {};

template <template <typename...> class S, typename... Os, typename T1,
          typename T2, typename T3, typename T4, typename T5, typename T6,
          typename T7, typename T8, typename T9, typename T10, typename T11,
          typename T12, typename T13, typename T14, typename T15, typename T16,
          typename... Ts, std::size_t I, typename R>
struct replace_at_impl<true, S<Os...>,
                       S<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
                         T14, T15, T16, Ts...>,
                       I, R>
    : replace_at_impl<((I - 16) > 16), S<Os..., T1, T2, T3, T4, T5, T6, T7, T8,
                                         T9, T10, T11, T12, T13, T14, T15, T16>,
                      S<Ts...>, (I - 16), R> {};

template <typename L, typename I, typename R>
struct call_replace_at_impl
    : replace_at_impl<(I::value > 15), brigand::clear<L>, L, I::value + 1, R> {
};
}

namespace lazy {
template <typename L, typename I, typename R>
using replace_at = ::brigand::detail::call_replace_at_impl<L, I, R>;
}
template <typename L, typename I, typename R>
using replace_at = typename ::brigand::lazy::replace_at<L, I, R>::type;
}  // namespace brigand

namespace brigand {
namespace detail {
template <typename Ls, typename Ind1, typename Ind2>
struct swap_at_impl {
  using type = ::brigand::replace_at<
      ::brigand::replace_at<Ls, Ind1, ::brigand::at<Ls, Ind2>>, Ind2,
      ::brigand::at<Ls, Ind1>>;
};
}

template <typename Ls, typename Ind1, typename Ind2>
using swap_at = typename ::brigand::detail::swap_at_impl<Ls, Ind1, Ind2>::type;
}

namespace brigand {
template <typename V>
using abs = std::integral_constant<typename V::value_type,
                                   (V::value < 0 ? -V::value : V::value)>;

template <typename V>
using sign =
    std::integral_constant<typename V::value_type, (V::value < 0 ? -1 : 1)>;

template <int T>
using int_ = std::integral_constant<int, T>;

template <typename V, typename N>
struct power
    : std::integral_constant<
          typename V::value_type,
          V::value * power<V, std::integral_constant<typename N::value_type,
                                                     N::value - 1>>::value> {};

template <typename V, typename T>
struct power<V, std::integral_constant<T, 0>>
    : std::integral_constant<typename V::value_type, 1> {};

template <typename T>
struct factorial : times<T, factorial<uint64_t<T::value - 1>>> {};
template <>
struct factorial<uint64_t<1>> : uint64_t<1> {};
}

namespace brigand {
namespace detail {
template <typename Ls, std::size_t Size = size<Ls>::value>
struct permutations_impl {
  template <typename T, typename Ls1>
  struct helper {
    using type = ::brigand::transform<
        typename permutations_impl<::brigand::remove<Ls1, T>>::type,
        ::brigand::lazy::push_front<_state, T>>;
  };

  using type = ::brigand::fold<
      Ls, list<>,
      ::brigand::lazy::append<::brigand::_state,
                              helper<::brigand::_element, ::brigand::pin<Ls>>>>;
};

template <typename Ls>
struct permutations_impl<Ls, 1> {
  using type = list<Ls>;
};
}

namespace lazy {
template <typename Ls>
using permutations = detail::permutations_impl<Ls>;
}

template <typename Ls>
using permutations = typename lazy::permutations<Ls>::type;
}

namespace brigand {
namespace detail {
template <typename Ls, std::size_t Size = ::brigand::size<Ls>::value>
struct generic_permutations_impl {
  template <typename Lc, typename Ls1>
  struct helper {
    using type = ::brigand::transform<
        typename generic_permutations_impl<::brigand::erase<Ls1, Lc>>::type,
        ::brigand::lazy::push_front<::brigand::_state, ::brigand::at<Ls1, Lc>>>;
  };
  using type = ::brigand::fold<
      ::brigand::make_sequence<brigand::uint32_t<0>, Size>, ::brigand::list<>,
      ::brigand::lazy::append<::brigand::_state,
                              helper<::brigand::_element, ::brigand::pin<Ls>>>>;
};

template <typename Ls>
struct generic_permutations_impl<Ls, 1> {
  using type = ::brigand::list<Ls>;
};
}

namespace lazy {
template <typename Ls>
using generic_permutations = detail::generic_permutations_impl<Ls>;
}

template <typename Ls>
using generic_permutations = typename lazy::generic_permutations<Ls>::type;
}

namespace brigand {
namespace detail {
template <typename Ls, typename Number = uint32_t<1>>
struct combinations_impl_helper {
  using split_list = split_at<Ls, Number>;
  using type =
      fold<back<split_list>, list<>,
           lazy::append<
               _state,
               bind<list, bind<push_back, pin<front<split_list>>, _element>>>>;
};

template <typename Ls, typename OutSize, typename = void>
struct combinations_impl {
  using type =
      append<list<>, typename combinations_impl_helper<Ls, prev<OutSize>>::type,
             typename combinations_impl<pop_front<Ls>, OutSize>::type>;
};
template <typename Ls, typename OutSize>
struct combinations_impl<
    Ls, OutSize,
    typename std::enable_if<OutSize::value == size<Ls>::value>::type> {
  using type = typename combinations_impl_helper<Ls, prev<OutSize>>::type;
};
}

namespace lazy {
template <typename Ls, typename OutSize = uint32_t<2>>
using combinations = detail::combinations_impl<Ls, OutSize>;
}

template <typename Ls, typename OutSize = uint32_t<2>>
using combinations = typename lazy::combinations<Ls, OutSize>::type;
}

namespace brigand {
namespace detail {
template <typename Seq, typename T>
struct equal_members_helper
    : std::is_same<count_if<Seq, std::is_same<T, _1>>, size_t<1>> {};
}

template <typename Ls1, typename Ls2>
using equal_members =
    and_<fold<Ls1, bool_<true>,
              and_<_state, detail::equal_members_helper<pin<Ls2>, _element>>>,
         fold<Ls2, bool_<true>,
              and_<_state, detail::equal_members_helper<pin<Ls1>, _element>>>>;
}

namespace brigand {
namespace detail {
template <typename Functor, typename State, typename I, typename Sequence>
struct enumerated_fold_impl {
  using type = State;
};

template <typename Functor, typename State, typename I,
          template <typename...> class Sequence, typename T0>
struct enumerated_fold_impl<Functor, State, I, Sequence<T0>> {
  using type = brigand::apply<Functor, State, T0, I>;
};

template <typename Functor, typename State, typename I,
          template <typename...> class Sequence, typename T0, typename T1>
struct enumerated_fold_impl<Functor, State, I, Sequence<T0, T1>> {
  using type = brigand::apply<Functor, brigand::apply<Functor, State, T0, I>,
                              T1, brigand::next<I>>;
};

template <typename Functor, typename State, typename I,
          template <typename...> class Sequence, typename T0, typename T1,
          typename T2>
struct enumerated_fold_impl<Functor, State, I, Sequence<T0, T1, T2>> {
  using type = brigand::apply<
      Functor, brigand::apply<Functor, brigand::apply<Functor, State, T0, I>,
                              T1, brigand::next<I>>,
      T2, brigand::plus<I, brigand::int32_t<2>>>;
};

template <typename Functor, typename State, typename I,
          template <typename...> class Sequence, typename T0, typename T1,
          typename T2, typename T3>
struct enumerated_fold_impl<Functor, State, I, Sequence<T0, T1, T2, T3>> {
  using type = brigand::apply<
      Functor,
      brigand::apply<
          Functor,
          brigand::apply<Functor, brigand::apply<Functor, State, T0, I>, T1,
                         brigand::next<I>>,
          T2, brigand::plus<I, brigand::int32_t<2>>>,
      T3, brigand::plus<I, brigand::int32_t<3>>>;
};

template <typename Functor, typename State, typename I,
          template <typename...> class Sequence, typename T0, typename T1,
          typename T2, typename T3, typename T4>
struct enumerated_fold_impl<Functor, State, I, Sequence<T0, T1, T2, T3, T4>> {
  using type = brigand::apply<
      Functor,
      brigand::apply<
          Functor,
          brigand::apply<
              Functor,
              brigand::apply<Functor, brigand::apply<Functor, State, T0, I>, T1,
                             brigand::next<I>>,
              T2, brigand::plus<I, brigand::int32_t<2>>>,
          T3, brigand::plus<I, brigand::int32_t<3>>>,
      T4, brigand::plus<I, brigand::int32_t<4>>>;
};

template <typename Functor, typename State, typename I,
          template <typename...> class Sequence, typename T0, typename T1,
          typename T2, typename T3, typename T4, typename T5>
struct enumerated_fold_impl<Functor, State, I,
                            Sequence<T0, T1, T2, T3, T4, T5>> {
  using type = brigand::apply<
      Functor,
      brigand::apply<
          Functor,
          brigand::apply<
              Functor,
              brigand::apply<
                  Functor,
                  brigand::apply<Functor, brigand::apply<Functor, State, T0, I>,
                                 T1, brigand::next<I>>,
                  T2, brigand::plus<I, brigand::int32_t<2>>>,
              T3, brigand::plus<I, brigand::int32_t<3>>>,
          T4, brigand::plus<I, brigand::int32_t<4>>>,
      T5, brigand::plus<I, brigand::int32_t<5>>>;
};

template <typename Functor, typename State, typename I,
          template <typename...> class Sequence, typename T0, typename T1,
          typename T2, typename T3, typename T4, typename T5, typename T6>
struct enumerated_fold_impl<Functor, State, I,
                            Sequence<T0, T1, T2, T3, T4, T5, T6>> {
  using type = brigand::apply<
      Functor,
      brigand::apply<
          Functor,
          brigand::apply<
              Functor,
              brigand::apply<
                  Functor, brigand::apply<
                               Functor, brigand::apply<
                                            Functor, brigand::apply<
                                                         Functor, State, T0, I>,
                                            T1, brigand::next<I>>,
                               T2, brigand::plus<I, brigand::int32_t<2>>>,
                  T3, brigand::plus<I, brigand::int32_t<3>>>,
              T4, brigand::plus<I, brigand::int32_t<4>>>,
          T5, brigand::plus<I, brigand::int32_t<5>>>,
      T6, brigand::plus<I, brigand::int32_t<6>>>;
};

template <typename Functor, typename State, typename I,
          template <typename...> class Sequence, typename T0, typename T1,
          typename T2, typename T3, typename T4, typename T5, typename T6,
          typename T7>
struct enumerated_fold_impl<Functor, State, I,
                            Sequence<T0, T1, T2, T3, T4, T5, T6, T7>> {
  using type = brigand::apply<
      Functor,
      brigand::apply<
          Functor,
          brigand::apply<
              Functor,
              brigand::apply<
                  Functor,
                  brigand::apply<
                      Functor,
                      brigand::apply<
                          Functor,
                          brigand::apply<Functor,
                                         brigand::apply<Functor, State, T0, I>,
                                         T1, brigand::next<I>>,
                          T2, brigand::plus<I, brigand::int32_t<2>>>,
                      T3, brigand::plus<I, brigand::int32_t<3>>>,
                  T4, brigand::plus<I, brigand::int32_t<4>>>,
              T5, brigand::plus<I, brigand::int32_t<5>>>,
          T6, brigand::plus<I, brigand::int32_t<6>>>,
      T7, brigand::plus<I, brigand::int32_t<7>>>;
};

template <typename Functor, typename State, typename I,
          template <typename...> class Sequence, typename T0, typename T1,
          typename T2, typename T3, typename T4, typename T5, typename T6,
          typename T7, typename... T>
struct enumerated_fold_impl<Functor, State, I,
                            Sequence<T0, T1, T2, T3, T4, T5, T6, T7, T...>>
    : enumerated_fold_impl<
          Functor,
          brigand::apply<
              Functor,
              brigand::apply<
                  Functor,
                  brigand::apply<
                      Functor,
                      brigand::apply<
                          Functor,
                          brigand::apply<
                              Functor,
                              brigand::apply<
                                  Functor,
                                  brigand::apply<
                                      Functor,
                                      brigand::apply<Functor, State, T0, I>, T1,
                                      brigand::next<I>>,
                                  T2, brigand::plus<I, brigand::int32_t<2>>>,
                              T3, brigand::plus<I, brigand::int32_t<3>>>,
                          T4, brigand::plus<I, brigand::int32_t<4>>>,
                      T5, brigand::plus<I, brigand::int32_t<5>>>,
                  T6, brigand::plus<I, brigand::int32_t<6>>>,
              T7, brigand::plus<I, brigand::int32_t<7>>>,
          brigand::plus<I, brigand::int32_t<8>>, Sequence<T...>> {};
}

namespace lazy {
template <typename Sequence, typename State, typename Functor,
          typename I = brigand::int32_t<0>>
using enumerated_fold =
    typename detail::enumerated_fold_impl<Functor, State, I, Sequence>;
}

template <typename Sequence, typename State, typename Functor,
          typename I = brigand::int32_t<0>>
using enumerated_fold =
    typename lazy::enumerated_fold<Sequence, State, Functor, I>::type;
}

namespace brigand {
namespace detail {
template <typename S, typename E>
struct remove_duplicates_helper {
  using type = typename std::conditional<
      std::is_same<index_of<S, E>, no_such_type_>::value, push_back<S, E>,
      S>::type;
};
}

template <typename Ls>
using remove_duplicates =
    fold<Ls, list<>, detail::remove_duplicates_helper<_state, _element>>;
}

namespace tmpl = brigand;

/// \ingroup Utilities
/// \brief Construct a typelist of types `Ts`.
template <typename... Ts>
using typelist = tmpl::list<Ts...>;
