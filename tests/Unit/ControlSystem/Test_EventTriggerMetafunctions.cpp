// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/Event.hpp"
#include "ControlSystem/Trigger.hpp"
#include "Helpers/ControlSystem/TestStructs.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct LabelA;
struct LabelB;
struct LabelC;

using SystemA0 = control_system::TestHelpers::System<
    2, LabelA, control_system::TestHelpers::Measurement<LabelA>>;
using SystemA1 = control_system::TestHelpers::System<
    2, LabelB, control_system::TestHelpers::Measurement<LabelA>>;
using SystemB0 = control_system::TestHelpers::System<
    2, LabelC, control_system::TestHelpers::Measurement<LabelB>>;

template <template <typename> typename Metafunction,
          template <typename> typename ResultObject>
constexpr bool check() {
  using result = Metafunction<tmpl::list<SystemA0, SystemB0, SystemA1>>;

  using expectedA_option0 = ResultObject<tmpl::list<SystemA0, SystemA1>>;
  using expectedA_option1 = ResultObject<tmpl::list<SystemA1, SystemA0>>;
  using expectedB = ResultObject<tmpl::list<SystemB0>>;

  // The orders of the lists is unspecified.
  static_assert(tmpl::size<result>::value == 2);
  static_assert(tmpl::list_contains_v<result, expectedB>);
  static_assert(tmpl::list_contains_v<result, expectedA_option0> or
                tmpl::list_contains_v<result, expectedA_option1>);
  return true;
}

static_assert(
    check<control_system::control_system_events, control_system::Event>());
static_assert(
    check<control_system::control_system_triggers, control_system::Trigger>());
}  // namespace
