// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Protocols/Measurement.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace control_system::TestHelpers {
namespace TestStructs_detail {
struct LabelA;
}  // namespace TestStructs_detail

template <typename Label>
struct Measurement
    : tt::ConformsTo<control_system::protocols::Measurement> {
  using submeasurements = tmpl::list<>;
};

static_assert(
    tt::assert_conforms_to<Measurement<TestStructs_detail::LabelA>,
                           control_system::protocols::Measurement>);

template <typename Label, typename Measurement>
struct System : tt::ConformsTo<control_system::protocols::ControlSystem> {
  static std::string name() { return pretty_type::short_name<Label>(); }
  using measurement = Measurement;
};

static_assert(
    tt::assert_conforms_to<System<TestStructs_detail::LabelA,
                                  Measurement<TestStructs_detail::LabelA>>,
                           control_system::protocols::ControlSystem>);
}  // namespace control_system::TestHelpers
