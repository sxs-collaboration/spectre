// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Elliptic/Actions/InitializeTemporalId.hpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace {

struct TemporalId : db::SimpleTag {
  static std::string name() noexcept { return "TemporalId"; };
  using type = int;
};

}  // namespace

namespace Tags {
template <>
struct NextCompute<TemporalId> : Next<TemporalId>, db::ComputeTag {
  using argument_tags = tmpl::list<TemporalId>;
  static int function(const int& temporal_id) noexcept {
    return temporal_id + 3;
  }
};
}  // namespace Tags

namespace {

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list = tmpl::list<>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Testing,
      tmpl::list<elliptic::Actions::InitializeTemporalId>>>;
};

struct Metavariables {
  using temporal_id = TemporalId;
  using component_list = tmpl::list<ElementArray<Metavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialization, Testing, Exit };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Actions.InitializeTemporalId",
                  "[Unit][Elliptic][Actions]") {
  using element_array = ElementArray<Metavariables>;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_component<element_array>(make_not_null(&runner), 0);
  runner.set_phase(Metavariables::Phase::Testing);
  ActionTesting::next_action<element_array>(make_not_null(&runner), 0);
  const auto get_tag = [&runner](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner, 0);
  };

  CHECK(get_tag(TemporalId{}) == 0);
  CHECK(get_tag(Tags::Next<TemporalId>{}) == 3);
}
