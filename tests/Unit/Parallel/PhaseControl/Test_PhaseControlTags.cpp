// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/Tag.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Options/Options.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/PhaseControl/PhaseChange.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/Triggers/Slabs.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Val, typename PhaseChangeRegistrars>
struct TestCreatable;

namespace Registrars {
template <size_t Val>
struct TestCreatable {
  template <typename PhaseChangeRegistrars>
  using f = ::TestCreatable<Val, PhaseChangeRegistrars>;
};
}  // namespace Registrars

namespace Tags {
struct DummyDecisionTag1 {
  using type = int;
  using combine_method = funcl::Plus<>;
  using main_combine_method = combine_method;
};

struct DummyDecisionTag2 {
  using type = size_t;
  using combine_method = funcl::Minus<>;
  using main_combine_method = combine_method;
};
}  // namespace Tags

template <size_t Val, typename PhaseChangeRegistrars>
struct TestCreatable : public PhaseChange<PhaseChangeRegistrars> {

  TestCreatable() = default;
  explicit TestCreatable(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(TestCreatable);  // NOLINT

  static std::string name() {
    if constexpr(Val == 1_st) {
      return "TestCreatable(1)";
    } else if constexpr(Val == 2_st) {
      return "TestCreatable(2)";
    } else {
      return "TestCreatable(Unknown)";
    }
  }

  template <typename Metavariables>
  using participating_components = tmpl::list<>;

  static constexpr Options::String help{"Creatable for test"};

  struct IntOption {
    using type = int;
    static constexpr Options::String help{"Option parameter"};
  };

  using options = tmpl::list<IntOption>;
  using phase_change_tags_and_combines =
      tmpl::conditional_t<(1_st < Val), tmpl::list<Tags::DummyDecisionTag1>,
                          tmpl::list<Tags::DummyDecisionTag2>>;

  explicit TestCreatable(int option_value) : option_value_{option_value} {}

  void pup(PUP::er& p) override { p | option_value_; }  // NOLINT

  int option_value_ = 0;
};

template <size_t Val, typename PhaseChangeRegistrars>
PUP::able::PUP_ID TestCreatable<Val, PhaseChangeRegistrars>::my_PUP_ID = 0;

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<TimeSequence<std::uint64_t>,
                             TimeSequences::all_time_sequences<std::uint64_t>>,
                  tmpl::pair<Trigger, tmpl::list<Triggers::Slabs>>>;
  };
};

SPECTRE_TEST_CASE("Unit.Parallel.PhaseControl.PhaseControlTags",
                  "[Unit][Parallel]") {
  using phase_changes = tmpl::list<Registrars::TestCreatable<1_st>,
                                   Registrars::TestCreatable<2_st>>;
  Parallel::register_derived_classes_with_charm<PhaseChange<phase_changes>>();
  Parallel::register_factory_classes_with_charm<Metavariables>();

  TestHelpers::db::test_simple_tag<
      PhaseControl::Tags::PhaseChangeAndTriggers<phase_changes>>(
      "PhaseChangeAndTriggers");

  const auto created_phase_changes = TestHelpers::test_option_tag<
      PhaseControl::OptionTags::PhaseChangeAndTriggers<phase_changes>,
      Metavariables>(
      " - - Slabs:\n"
      "       EvenlySpaced:\n"
      "         Interval: 2\n"
      "         Offset: 0\n"
      "   - - TestCreatable(1):\n"
      "         IntOption: 4\n"
      "     - TestCreatable(2):\n"
      "         IntOption: 2");
  CHECK(created_phase_changes.size() == 1_st);
  const auto& first_creatable = created_phase_changes[0].second[0];
  const auto& second_creatable = created_phase_changes[0].second[1];
  REQUIRE(dynamic_cast<TestCreatable<1_st, phase_changes>*>(
              first_creatable.get()) != nullptr);
  CHECK(dynamic_cast<TestCreatable<1_st, phase_changes>*>(first_creatable.get())
            ->option_value_ == 4);

  REQUIRE(dynamic_cast<TestCreatable<2_st, phase_changes>*>(
              second_creatable.get()) != nullptr);
  CHECK(
      dynamic_cast<TestCreatable<2_st, phase_changes>*>(second_creatable.get())
          ->option_value_ == 2);

  static_assert(std::is_same_v<
                PhaseControl::get_phase_change_tags<phase_changes>,
                tmpl::list<
                    Tags::DummyDecisionTag2, Tags::DummyDecisionTag1,
                    PhaseControl::TagsAndCombines::UsePhaseChangeArbitration>>);
}
}  // namespace
