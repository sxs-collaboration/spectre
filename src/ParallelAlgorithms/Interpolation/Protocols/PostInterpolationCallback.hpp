// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
/// \endcond

namespace intrp::protocols {
namespace detail {
struct DummyMetavariables;
struct DummyTag : db::SimpleTag {
  using type = int;
};

template <typename T, typename = std::void_t<>>
struct has_signature_1 : std::false_type {};

template <typename T>
struct has_signature_1<
    T, std::void_t<decltype(T::apply(
           std::declval<const db::DataBox<DummyTag>&>(),
           std::declval<const Parallel::GlobalCache<DummyMetavariables>&>(),
           std::declval<const double&>()))>> : std::true_type {
  static_assert(
      std::is_same_v<
          void,
          decltype(T::apply(
              std::declval<const db::DataBox<DummyTag>&>(),
              std::declval<const Parallel::GlobalCache<DummyMetavariables>&>(),
              std::declval<const double&>()))>);
};

template <typename T, typename = std::void_t<>>
struct has_signature_2 : std::false_type {};

template <typename T>
struct has_signature_2<
    T, std::void_t<decltype(
           T::apply(std::declval<const db::DataBox<DummyTag>&>(),
                    std::declval<Parallel::GlobalCache<DummyMetavariables>&>(),
                    std::declval<const double&>()))>> : std::true_type {
  static_assert(
      std::is_same_v<
          void, decltype(T::apply(
                    std::declval<const db::DataBox<DummyTag>&>(),
                    std::declval<Parallel::GlobalCache<DummyMetavariables>&>(),
                    std::declval<const double&>()))>);
};

template <typename T, typename = std::void_t<>>
struct has_signature_3 : std::false_type {};

template <typename T>
struct has_signature_3<
    T, std::void_t<decltype(
           T::apply(std::declval<const gsl::not_null<db::DataBox<DummyTag>*>>(),
                    std::declval<const gsl::not_null<
                        Parallel::GlobalCache<DummyMetavariables>*>>(),
                    std::declval<const double&>()))>> : std::true_type {
  static_assert(
      std::is_same_v<
          bool, decltype(T::apply(
                    std::declval<const gsl::not_null<db::DataBox<DummyTag>*>>(),
                    std::declval<const gsl::not_null<
                        Parallel::GlobalCache<DummyMetavariables>*>>(),
                    std::declval<const double&>()))>);
};

}  // namespace detail

/*!
 * \brief A protocol for the type alias `post_interpolation_callbacks` found in
 * an InterpolationTargetTag.
 *
 * \details A struct conforming to the `PostInterpolationCallback` protocol must
 * have
 *
 * - a function `apply` with one of the 3 signatures in the example. This apply
 *   function will be called once the interpolation is complete. `DbTags`
 *   includes everything in the `vars_to_interpolate_to_target` alias and the
 *   `compute_items_on_target` alias of the InterpolationTargetTag. The `apply`
 *   that returns a bool should return false only if it calls another
 *   `intrp::Action` that still needs the volume data at this temporal_id (such
 *   as another iteration of the horizon finder). These functions must be able
 *   to take any type for the `TemporalId`. If a specific temporal ID type is
 *   required, it should be `static_assert`ed in the function itself.
 *
 * A struct conforming to this protocol can also have an optional `static
 * constexpr double fill_invalid_points_with`. Any points outside the Domain
 * will be filled with this value. If this variable is not defined, then the
 * `apply` function must check for invalid points, and should typically exit
 * with an error message if it finds any.
 *
 * Here is an example of a class that conforms to this protocols:
 *
 * \snippet Helpers/ParallelAlgorithms/Interpolation/Examples.hpp PostInterpolationCallback
 */
struct PostInterpolationCallback {
  template <typename ConformingType>
  struct test {
    static_assert(detail::has_signature_1<ConformingType>::value or
                  detail::has_signature_2<ConformingType>::value or
                  detail::has_signature_3<ConformingType>::value);
  };
};
}  // namespace intrp::protocols
