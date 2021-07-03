// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>
#include <tuple>

#include "Parallel/CharmRegistration.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Section.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

namespace Parallel {
/// \cond
template <class... Ts>
struct ReductionData;
/// \endcond

namespace detail {
/*!
 * \ingroup ParallelGroup
 * \brief Convert a `ReductionData` to a `CkReductionMsg`. Used in custom
 * reducers.
 */
template <class... Ts>
CkReductionMsg* new_reduction_msg(
    ReductionData<Ts...>& reduction_data) noexcept {
  return CkReductionMsg::buildNew(static_cast<int>(reduction_data.size()),
                                  reduction_data.packed().get());
}
}  // namespace detail

/*!
 * \ingroup ParallelGroup
 * \brief The data to be reduced, and invokables to be called whenever two
 * reduction messages are combined and after the reduction has been completed.
 *
 * `InvokeCombine` is a binary invokable that maps `(T current_state, T element)
 * -> T`, where the `current_state` is the result of reductions so far. The
 * `InvokeFinal` is an n-ary that takes as its first argument a `T
 * result_of_reduction` and is invoked once after the reduction is completed.
 * The additional arguments correspond to the resultant data of earlier
 * `ReductionDatum` template parameters in the `ReductionData`, and are
 * identified via the `InvokeFinalExtraArgsIndices`, which must be a
 * `std::index_sequence`. Specifically, say you want the third
 * `ReductionDatum`'s `InvokeFinal` to be passed the first `ReductionDatum` then
 * `std::index_sequence<0>` would be passed for `InvokeFinalExtraArgsIndices`.
 * Here is an example of computing the RMS error of the evolved variables `u`
 * and `v`:
 *
 * \snippet Test_AlgorithmReduction.cpp contribute_to_rms_reduction
 *
 * with the receiving action:
 *
 * \snippet Test_AlgorithmReduction.cpp reduce_rms_action
 */
template <class T, class InvokeCombine, class InvokeFinal = funcl::Identity,
          class InvokeFinalExtraArgsIndices = std::index_sequence<>>
struct ReductionDatum {
  using value_type = T;
  using invoke_combine = InvokeCombine;
  using invoke_final = InvokeFinal;
  using invoke_final_extra_args_indices = InvokeFinalExtraArgsIndices;
  T value;
};

/*!
 * \ingroup ParallelGroup
 * \brief Used for reducing a possibly heterogeneous collection of types in a
 * single reduction call
 */
template <class... Ts, class... InvokeCombines, class... InvokeFinals,
          class... InvokeFinalExtraArgsIndices>
struct ReductionData<ReductionDatum<Ts, InvokeCombines, InvokeFinals,
                                    InvokeFinalExtraArgsIndices>...> {
  static_assert(sizeof...(Ts) > 0,
                "Must be reducing at least one piece of data.");
  static constexpr size_t pack_size() noexcept { return sizeof...(Ts); }
  using datum_list = tmpl::list<ReductionDatum<Ts, InvokeCombines, InvokeFinals,
                                               InvokeFinalExtraArgsIndices>...>;

  explicit ReductionData(
      ReductionDatum<Ts, InvokeCombines, InvokeFinals,
                     InvokeFinalExtraArgsIndices>... args) noexcept;

  explicit ReductionData(Ts... args) noexcept;

  ReductionData() = default;
  ReductionData(const ReductionData& /*rhs*/) = default;
  ReductionData& operator=(const ReductionData& /*rhs*/) = default;
  ReductionData(ReductionData&& /*rhs*/) = default;
  ReductionData& operator=(ReductionData&& /*rhs*/) = default;
  ~ReductionData() = default;

  explicit ReductionData(CkReductionMsg* const message) noexcept {
    PUP::fromMem creator(message->getData());
    creator | *this;
  }

  static CkReductionMsg* combine(int number_of_messages,
                                 CkReductionMsg** msgs) noexcept;

  ReductionData& combine(ReductionData&& t) noexcept {
    ReductionData::combine_helper(this, std::move(t),
                                  std::make_index_sequence<sizeof...(Ts)>{});
    return *this;
  }

  ReductionData& finalize() noexcept {
    invoke_final_loop_over_tuple(std::make_index_sequence<sizeof...(Ts)>{});
    return *this;
  }

  /// \cond
  // clang-tidy: non-const reference
  void pup(PUP::er& p) noexcept { p | data_; }  // NOLINT

  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  std::unique_ptr<char[]> packed() noexcept;

  size_t size() noexcept;

  const std::tuple<Ts...>& data() const noexcept { return data_; }

  std::tuple<Ts...>& data() noexcept { return data_; }

 private:
  template <size_t... Is>
  static void combine_helper(gsl::not_null<ReductionData*> reduced,
                             ReductionData&& current,
                             std::index_sequence<Is...> /*meta*/) noexcept;

  template <size_t I, class InvokeFinal, size_t... Js>
  void invoke_final_helper(std::index_sequence<Js...> /*meta*/) noexcept;

  template <size_t... Is>
  void invoke_final_loop_over_tuple(
      std::index_sequence<Is...> /*meta*/) noexcept;

  std::tuple<Ts...> data_;
  /// \endcond
};

/// \cond
template <class... Ts, class... InvokeCombines, class... InvokeFinals,
          class... InvokeFinalExtraArgsIndices>
ReductionData<ReductionDatum<Ts, InvokeCombines, InvokeFinals,
                             InvokeFinalExtraArgsIndices>...>::
    ReductionData(ReductionDatum<Ts, InvokeCombines, InvokeFinals,
                                 InvokeFinalExtraArgsIndices>... args) noexcept
    : data_(std::move(args.value)...) {}

template <class... Ts, class... InvokeCombines, class... InvokeFinals,
          class... InvokeFinalExtraArgsIndices>
ReductionData<ReductionDatum<Ts, InvokeCombines, InvokeFinals,
                             InvokeFinalExtraArgsIndices>...>::
    ReductionData(Ts... args) noexcept
    : data_(std::move(args)...) {}

template <class... Ts, class... InvokeCombines, class... InvokeFinals,
          class... InvokeFinalExtraArgsIndices>
CkReductionMsg* ReductionData<ReductionDatum<Ts, InvokeCombines, InvokeFinals,
                                             InvokeFinalExtraArgsIndices>...>::
    combine(const int number_of_messages,
            CkReductionMsg** const msgs) noexcept {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  ReductionData reduced(msgs[0]);
  for (int msg_id = 1; msg_id < number_of_messages; ++msg_id) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    ReductionData current(msgs[msg_id]);
    ReductionData::combine_helper(&reduced, std::move(current),
                                  std::make_index_sequence<sizeof...(Ts)>{});
  }
  return detail::new_reduction_msg(reduced);
}

template <class... Ts, class... InvokeCombines, class... InvokeFinals,
          class... InvokeFinalExtraArgsIndices>
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
std::unique_ptr<char[]> ReductionData<
    ReductionDatum<Ts, InvokeCombines, InvokeFinals,
                   InvokeFinalExtraArgsIndices>...>::packed() noexcept {
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  auto result = std::make_unique<char[]>(size());
  PUP::toMem packer(result.get());
  packer | *this;
  return result;
}

template <class... Ts, class... InvokeCombines, class... InvokeFinals,
          class... InvokeFinalExtraArgsIndices>
size_t
ReductionData<ReductionDatum<Ts, InvokeCombines, InvokeFinals,
                             InvokeFinalExtraArgsIndices>...>::size() noexcept {
  PUP::sizer size_pup;
  size_pup | *this;
  return size_pup.size();
}

template <class... Ts, class... InvokeCombines, class... InvokeFinals,
          class... InvokeFinalExtraArgsIndices>
template <size_t... Is>
void ReductionData<ReductionDatum<Ts, InvokeCombines, InvokeFinals,
                                  InvokeFinalExtraArgsIndices>...>::
    combine_helper(const gsl::not_null<ReductionData*> reduced,
                   ReductionData&& current,
                   std::index_sequence<Is...> /*meta*/) noexcept {
  EXPAND_PACK_LEFT_TO_RIGHT((std::get<Is>(reduced->data_) = InvokeCombines{}(
                                 std::move(std::get<Is>(reduced->data_)),
                                 std::move(std::get<Is>(current.data_)))));
}

template <class... Ts, class... InvokeCombines, class... InvokeFinals,
          class... InvokeFinalExtraArgsIndices>
template <size_t I, class InvokeFinal, size_t... Js>
void ReductionData<ReductionDatum<Ts, InvokeCombines, InvokeFinals,
                                  InvokeFinalExtraArgsIndices>...>::
    invoke_final_helper(std::index_sequence<Js...> /*meta*/) noexcept {
  std::get<I>(data_) = InvokeFinal{}(std::move(std::get<I>(data_)),
                                     std::as_const(std::get<Js>(data_))...);
}

template <class... Ts, class... InvokeCombines, class... InvokeFinals,
          class... InvokeFinalExtraArgsIndices>
template <size_t... Is>
void ReductionData<ReductionDatum<Ts, InvokeCombines, InvokeFinals,
                                  InvokeFinalExtraArgsIndices>...>::
    invoke_final_loop_over_tuple(std::index_sequence<Is...> /*meta*/) noexcept {
  EXPAND_PACK_LEFT_TO_RIGHT(
      invoke_final_helper<Is, InvokeFinals>(InvokeFinalExtraArgsIndices{}));
}
/// \endcond

/// Can be used instead of a `Parallel::Section` when no section is desired.
///
/// \see Parallel::contribute_to_reduction
/// @{
struct NoSection {};
NoSection& no_section() noexcept;
/// @}

/*!
 * \ingroup ParallelGroup
 * \brief Perform a reduction from the `sender_component` (typically your own
 * parallel component) to the `target_component`, performing the `Action` upon
 * receiving the reduction.
 *
 * \par Section reductions
 * This function supports section reductions (see `Parallel::Section`). Pass
 * the `Parallel::Section` as the \p section argument, or pass
 * `Parallel::no_section()` to perform a reduction over the entire parallel
 * component (default). Here's an example of a section reduction:
 *
 * \snippet Test_SectionReductions.cpp section_reduction
 *
 * \warning Section reductions currently don't support migrating elements, i.e.
 * either load-balancing or restoring a checkpoint to a different number of PEs.
 * Support for migrating elements may require [updating the "section
 * cookie"](https://charm.readthedocs.io/en/latest/charm++/manual.html#section-operations-with-migrating-elements).
 * One possibility to update the section cookie is to broadcast a CkMulticast
 * message to the section elements and invoke `CkGetSectionInfo` within the
 * message.
 */
template <class Action, class SenderProxy, class TargetProxy, class... Ts,
          class SectionType = NoSection>
void contribute_to_reduction(ReductionData<Ts...> reduction_data,
                             const SenderProxy& sender_component,
                             const TargetProxy& target_component,
                             [[maybe_unused]] const gsl::not_null<SectionType*>
                                 section = &no_section()) noexcept {
  (void)Parallel::charmxx::RegisterReducerFunction<
      &ReductionData<Ts...>::combine>::registrar;
  CkCallback callback(
      TargetProxy::index_t::template redn_wrapper_reduction_action<
          Action, std::decay_t<ReductionData<Ts...>>>(nullptr),
      target_component);
  const auto& charm_reducer_function =
      Parallel::charmxx::charm_reducer_functions.at(
          std::hash<Parallel::charmxx::ReducerFunctions>{}(
              &ReductionData<Ts...>::combine));
  if constexpr (std::is_same_v<SectionType, NoSection>) {
    Parallel::local(sender_component)
        ->contribute(static_cast<int>(reduction_data.size()),
                     reduction_data.packed().get(), charm_reducer_function,
                     callback);
  } else {
    static_assert(
        tt::is_a_v<Section, SectionType>,
        "Either pass a 'Parallel::Section' for the 'section' argument or "
        "'Parallel::NoSection{}'. For the latter you can just omit the "
        "argument.");
    using SectionProxy = typename SectionType::cproxy_section;
    // Retrieve the section cookie that keeps track of the reduction
    auto& section_cookie = section->cookie();
    // Ideally we would update the section cookie here using
    // `CkGetSectionInfo()`. However, that only works with CkMulticast messages
    // (see
    // https://github.com/UIUC-PPL/charm/blob/99cda7a11108f503b89dc847b58e62bc74267440/src/ck-core/ckmulticast.C#L1180).
    // Dispatching a message to the `sender_component` doesn't help because
    // sending a message to a single element doesn't go through CkMulticast. Not
    // updating the section cookie seems to work, but might break when elements
    // migrate (see
    // https://charm.readthedocs.io/en/latest/charm++/manual.html#section-operations-with-migrating-elements).
    // In that case we can possibly broadcast a CkMulticast message to all
    // elements to update their section cookies.
    SectionProxy::contribute(static_cast<int>(reduction_data.size()),
                             reduction_data.packed().get(),
                             charm_reducer_function, section_cookie, callback);
  }
}
}  // namespace Parallel
