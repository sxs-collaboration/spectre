// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "ParallelAlgorithms/ApparentHorizonFinder/Destination.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/Callback.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace ah::protocols {
/*!
 * \brief A protocol for `HorizonMetavars`s that are used in the
 * `ah::Component` parallel component.
 *
 * \details A struct conforming to the `HorizonMetavars` protocol must
 * have
 *
 * - a type alias `time_tag` to a tag that tells the horizon finder
 *   which time tag to use (for example, `::Tags::TimeAndPrevious`).
 *
 * - a type alias `frame` to the frame that the horizon find happens in (e.g.
 *   `::Frame::Distorted`).
 *
 * - a type alias `compute_tags_on_element` which is a `tmpl::list` of compute
 *   tags used in the `ObservationBox` on the elements.
 *
 * - a type alias `horizon_find_callbacks` which is a `tmpl::list` of callbacks
 *   that conform to `ah::protocols::Callback`.
 *
 * - a type alias `horizon_find_failure_callbacks` which is a `tmpl::list` of
 *   callbacks that conform to `ah::protocols::Callback`.
 *
 * - a static function `name()` that returns a `std::string`.
 *
 * - a static constexpr `ah::Destination` named `destination` which tells what
 *   this horizon find is for.
 *
 * \snippet Helpers/ParallelAlgorithms/ApparentHorizonFinder/TestHelpers.hpp HorizonMetavars
 */
struct HorizonMetavars {
  template <typename ConformingType>
  struct test {
    using time_tag = typename ConformingType::time_tag;

    using frame = typename ConformingType::frame;

    using horizon_find_callbacks =
        typename ConformingType::horizon_find_callbacks;
    static_assert(tmpl::all<horizon_find_callbacks,
                            tt::assert_conforms_to<tmpl::_1, Callback>>::value);
    using horizon_find_failure_callbacks =
        typename ConformingType::horizon_find_failure_callbacks;
    static_assert(tmpl::all<horizon_find_failure_callbacks,
                            tt::assert_conforms_to<tmpl::_1, Callback>>::value);

    using compute_tags_on_element =
        typename ConformingType::compute_tags_on_element;

    static constexpr Destination destination = ConformingType::destination;

    static std::string name() { return ConformingType::name(); }
  };
};
}  // namespace ah::protocols
