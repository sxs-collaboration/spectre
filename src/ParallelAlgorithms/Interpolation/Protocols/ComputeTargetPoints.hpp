// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
struct DataVector;
/// \endcond

namespace intrp::protocols {
/*!
 * \brief A protocol for the type alias `compute_target_points` found in an
 * InterpolationTargetTag.
 *
 * \details A struct conforming to the `ComputeTargetPoints` protocol must have
 *
 * - a type alias `is_sequential` that is either `std::true_type` or
 *   `std::false_type` which indicates if interpolations depend on previous
 *   interpolations' results
 *
 * - a type alias `frame` that denotes the frame the target points are computed
 *   in
 *
 * - a function `points` with signature matching the one in the example that
 *   will compute the points in the given `frame`.
 *
 * A struct that conforms to this protocol can optionally have any of these
 * members as well:
 *
 * - a type alias `simple_tags` that list simple tags to be added to the DataBox
 *   of the InterpolationTarget
 *
 * - a type alias `compute_tags` that list compute tags to be added to the
 *   DataBox of the InterpolationTarget
 *
 * - a type alias `const_global_cache_tags` with tags to be put in the
 *   GlobalCache
 *
 * - a function `initialize` with signature matching the one in the example that
 *   is run during the Initialization phase of the InterpolationTarget and can
 *   initialize any of the `simple_tags` added.
 *
 * Here is an example of a class that conforms to this protocols:
 *
 * \snippet Helpers/ParallelAlgorithms/Interpolation/Examples.hpp ComputeTargetPoints
 */
struct ComputeTargetPoints {
  template <typename ConformingType>
  struct test {
    struct DummyMetavariables;
    struct DummyTag : db::SimpleTag {
      using type = int;
    };

    using is_sequential = typename ConformingType::is_sequential;
    static_assert(std::is_same_v<is_sequential, std::true_type> or
                  std::is_same_v<is_sequential, std::false_type>);

    using frame = typename ConformingType::frame;

    template <size_t Dim>
    static constexpr bool conforms = std::is_same_v<
        tnsr::I<DataVector, Dim, frame>,
        decltype(ConformingType::points(
            std::declval<const db::DataBox<DummyTag>&>(),
            std::declval<const tmpl::type_<DummyMetavariables>&>(),
            std::declval<const double&>()))>;

    static_assert(conforms<1> or conforms<2> or conforms<3>);

    // We don't check the initialize() function because it is optional
  };
};
}  // namespace intrp::protocols
