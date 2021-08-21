// Distributed under the MIT License.
// See LICENSE.txt for details.

// NOTE: The event itself is tested in Test_Measurement.cpp .  This
// file only tests the metafunction.

#include "ControlSystem/Event.hpp"
#include "Helpers/ControlSystem/TestStructs.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct LabelA;
struct LabelB;
struct LabelC;

using SystemA0 = control_system::TestHelpers::System<
    LabelA, control_system::TestHelpers::Measurement<LabelA>>;
using SystemA1 = control_system::TestHelpers::System<
    LabelB, control_system::TestHelpers::Measurement<LabelA>>;
using SystemB0 = control_system::TestHelpers::System<
    LabelC, control_system::TestHelpers::Measurement<LabelB>>;

using result = control_system::control_system_events<
    tmpl::list<SystemA0, SystemB0, SystemA1>>;

using expectedA_option0 = control_system::Event<tmpl::list<SystemA0, SystemA1>>;
using expectedA_option1 = control_system::Event<tmpl::list<SystemA1, SystemA0>>;
using expectedB = control_system::Event<tmpl::list<SystemB0>>;

// The orders of the lists is unspecified.
static_assert(tmpl::size<result>::value == 2);
static_assert(tmpl::list_contains_v<result, expectedB>);
static_assert(tmpl::list_contains_v<result, expectedA_option0> or
              tmpl::list_contains_v<result, expectedA_option1>);
}  // namespace
