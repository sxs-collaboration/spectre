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
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/ExitCode.hpp"
#include "Parallel/PhaseControl/PhaseChange.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/Triggers/Slabs.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
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

template <size_t Val>
struct TestCreatable : public PhaseChange {

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

template <size_t Val>
PUP::able::PUP_ID TestCreatable<Val>::my_PUP_ID = 0;

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<PhaseChange, tmpl::list<TestCreatable<1_st>,
                                                     TestCreatable<2_st>>>,
                  tmpl::pair<TimeSequence<std::uint64_t>,
                             TimeSequences::all_time_sequences<std::uint64_t>>,
                  tmpl::pair<Trigger, tmpl::list<Triggers::Slabs>>>;
  };
};

SPECTRE_TEST_CASE("Unit.Parallel.PhaseControl.PhaseControlTags",
                  "[Unit][Parallel]") {
  register_factory_classes_with_charm<Metavariables>();

  TestHelpers::db::test_simple_tag<PhaseControl::Tags::PhaseChangeAndTriggers>(
      "PhaseChangeAndTriggers");

  const auto created_phase_changes = TestHelpers::test_option_tag<
      PhaseControl::OptionTags::PhaseChangeAndTriggers, Metavariables>(
      " - Trigger:\n"
      "     Slabs:\n"
      "       EvenlySpaced:\n"
      "         Interval: 2\n"
      "         Offset: 0\n"
      "   PhaseChanges:\n"
      "     - TestCreatable(1):\n"
      "         IntOption: 4\n"
      "     - TestCreatable(2):\n"
      "         IntOption: 2");
  CHECK(created_phase_changes.size() == 1_st);
  const auto& first_creatable = created_phase_changes[0].phase_changes[0];
  const auto& second_creatable = created_phase_changes[0].phase_changes[1];
  REQUIRE(dynamic_cast<TestCreatable<1_st>*>(first_creatable.get()) != nullptr);
  CHECK(dynamic_cast<TestCreatable<1_st>*>(first_creatable.get())
            ->option_value_ == 4);

  REQUIRE(dynamic_cast<TestCreatable<2_st>*>(second_creatable.get()) !=
          nullptr);
  CHECK(dynamic_cast<TestCreatable<2_st>*>(second_creatable.get())
            ->option_value_ == 2);

  static_assert(
      std::is_same_v<
          PhaseControl::get_phase_change_tags<Metavariables>,
          tmpl::list<Tags::DummyDecisionTag2, Tags::DummyDecisionTag1,
                     PhaseControl::TagsAndCombines::UsePhaseChangeArbitration,
                     Parallel::Tags::ExitCode>>);
}
}  // namespace
