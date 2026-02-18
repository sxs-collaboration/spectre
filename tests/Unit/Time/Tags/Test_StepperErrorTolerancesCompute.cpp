// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <optional>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Time/ChangeSlabSize/Event.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/ErrorControl.hpp"
#include "Time/StepChoosers/FixedLtsRatio.hpp"
#include "Time/StepChoosers/LimitIncrease.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/Tags/StepChoosers.hpp"
#include "Time/Tags/StepperErrorEstimatesEnabled.hpp"
#include "Time/Tags/StepperErrorTolerances.hpp"
#include "Time/Tags/StepperErrorTolerancesCompute.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Tags/VariableOrderAlgorithm.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/VariableOrderAlgorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

class DataVector;
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel

namespace {
class OtherEvent : public Event {
 public:
  explicit OtherEvent(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(OtherEvent);  // NOLINT
#pragma GCC diagnostic pop

  using compute_tags_for_observation_box = tmpl::list<>;
  using options = tmpl::list<>;
  static constexpr Options::String help = {""};

  OtherEvent() = default;

  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  void operator()(Parallel::GlobalCache<Metavariables>& /*cache*/,
                  const ArrayIndex& /*array_index*/,
                  const Component* const /*meta*/,
                  const ObservationValue& /*observation_value*/) const {}

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return false; }
};

PUP::able::PUP_ID OtherEvent::my_PUP_ID = 0;  // NOLINT

struct EvolvedVar1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct EvolvedVar2 : db::SimpleTag {
  using type = tnsr::i<DataVector, 2>;
};

struct EvolvedVar3 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

using EvolvedVariablesTag =
    Tags::Variables<tmpl::list<EvolvedVar1, EvolvedVar2>>;

using AltEvolvedVariablesTag = Tags::Variables<tmpl::list<EvolvedVar3>>;

struct ErrorControlSelecter {
  static std::string name() { return "SelectorLabel"; }
};

struct Metavariables {
  template <typename Use>
  using step_choosers =
      tmpl::list<StepChoosers::LimitIncrease, StepChoosers::Constant,
                 StepChoosers::ErrorControl<Use, EvolvedVariablesTag>,
                 StepChoosers::ErrorControl<Use, AltEvolvedVariablesTag,
                                            ErrorControlSelecter>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<Event, tmpl::list<Events::ChangeSlabSize, OtherEvent>>,
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                   step_choosers<StepChooserUse::LtsStep>>,
        tmpl::pair<StepChooser<StepChooserUse::Slab>,
                   tmpl::push_back<step_choosers<StepChooserUse::Slab>,
                                   StepChoosers::FixedLtsRatio>>,
        tmpl::pair<Trigger, tmpl::list<Triggers::Always>>>;
  };
};

SPECTRE_TEST_CASE("Unit.Time.Tags.StepperErrorTolerancesCompute",
                  "[Unit][Time]") {
  TestHelpers::db::test_compute_tag<
      Tags::StepperErrorEstimatesEnabledCompute<true>>(
      "StepperErrorEstimatesEnabled");
  TestHelpers::db::test_compute_tag<
      Tags::StepperErrorTolerancesCompute<EvolvedVariablesTag, true>>(
      "StepperErrorTolerances(Variables(EvolvedVar1,EvolvedVar2))");

  {
    INFO("Compute tag LTS test");
    const auto test_lts_tags = [](const auto box, const bool all_orders) {
      const auto expected_estimates =
          all_orders ? StepperErrorTolerances::Estimates::AllOrders
                     : StepperErrorTolerances::Estimates::StepperOrder;

      db::mutate<Tags::StepChoosers,
                 Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>(
          [](const gsl::not_null<Tags::StepChoosers::type*> choosers,
             const gsl::not_null<EventsAndTriggers*> events) {
            *choosers = TestHelpers::test_creation<Tags::StepChoosers::type,
                                                   Metavariables>(
                "- ErrorControl:\n"
                "    SafetyFactor: 0.95\n"
                "    AbsoluteTolerance: 1.0e-5\n"
                "    RelativeTolerance: 1.0e-4\n"
                "    MaxFactor: 2.1\n"
                "    MinFactor: 0.5\n"
                "- LimitIncrease:\n"
                "    Factor: 2\n"
                "- Constant: 0.5");
            *events = EventsAndTriggers{};
          },
          box);
      CHECK(db::get<Tags::StepperErrorEstimatesEnabled>(*box));
      CHECK(db::get<Tags::StepperErrorTolerances<EvolvedVariablesTag>>(*box) ==
            StepperErrorTolerances{.estimates = expected_estimates,
                                   .absolute = 1.0e-5,
                                   .relative = 1.0e-4});
      CHECK(db::get<Tags::StepperErrorTolerances<AltEvolvedVariablesTag>>(*box)
                .estimates == StepperErrorTolerances::Estimates::None);

      db::mutate<Tags::StepChoosers>(
          [](const gsl::not_null<Tags::StepChoosers::type*> choosers) {
            *choosers = TestHelpers::test_creation<Tags::StepChoosers::type,
                                                   Metavariables>(
                "- LimitIncrease:\n"
                "    Factor: 2\n"
                "- Constant: 0.5");
          },
          box);
      CHECK_FALSE(db::get<Tags::StepperErrorEstimatesEnabled>(*box));
      CHECK(db::get<Tags::StepperErrorTolerances<EvolvedVariablesTag>>(*box)
                .estimates == StepperErrorTolerances::Estimates::None);
      CHECK(db::get<Tags::StepperErrorTolerances<AltEvolvedVariablesTag>>(*box)
                .estimates == StepperErrorTolerances::Estimates::None);

      db::mutate<Tags::StepChoosers>(
          [](const gsl::not_null<Tags::StepChoosers::type*> choosers) {
            *choosers = TestHelpers::test_creation<Tags::StepChoosers::type,
                                                   Metavariables>("");
          },
          box);
      CHECK_FALSE(db::get<Tags::StepperErrorEstimatesEnabled>(*box));
      CHECK(db::get<Tags::StepperErrorTolerances<EvolvedVariablesTag>>(*box)
                .estimates == StepperErrorTolerances::Estimates::None);
      CHECK(db::get<Tags::StepperErrorTolerances<AltEvolvedVariablesTag>>(*box)
                .estimates == StepperErrorTolerances::Estimates::None);

      db::mutate<Tags::StepChoosers>(
          [](const gsl::not_null<Tags::StepChoosers::type*> choosers) {
            *choosers = TestHelpers::test_creation<Tags::StepChoosers::type,
                                                   Metavariables>(
                "- ErrorControl:\n"
                "    SafetyFactor: 0.95\n"
                "    AbsoluteTolerance: 1.0e-5\n"
                "    RelativeTolerance: 1.0e-4\n"
                "    MaxFactor: 2.1\n"
                "    MinFactor: 0.5\n"
                "- ErrorControl:\n"
                "    SafetyFactor: 0.8\n"
                "    AbsoluteTolerance: 1.0e-5\n"
                "    RelativeTolerance: 1.0e-4\n"
                "    MaxFactor: 1.1\n"
                "    MinFactor: 0.1\n"
                "- LimitIncrease:\n"
                "    Factor: 2\n"
                "- Constant: 0.5");
          },
          box);
      CHECK(db::get<Tags::StepperErrorEstimatesEnabled>(*box));
      CHECK(db::get<Tags::StepperErrorTolerances<EvolvedVariablesTag>>(*box) ==
            StepperErrorTolerances{.estimates = expected_estimates,
                                   .absolute = 1.0e-5,
                                   .relative = 1.0e-4});
      CHECK(db::get<Tags::StepperErrorTolerances<AltEvolvedVariablesTag>>(*box)
                .estimates == StepperErrorTolerances::Estimates::None);

      db::mutate<Tags::StepChoosers>(
          [](const gsl::not_null<Tags::StepChoosers::type*> choosers) {
            *choosers = TestHelpers::test_creation<Tags::StepChoosers::type,
                                                   Metavariables>(
                "- ErrorControl:\n"
                "    SafetyFactor: 0.95\n"
                "    AbsoluteTolerance: 1.0e-5\n"
                "    RelativeTolerance: 1.0e-4\n"
                "    MaxFactor: 2.1\n"
                "    MinFactor: 0.5\n"
                "- ErrorControl(SelectorLabel):\n"
                "    SafetyFactor: 0.8\n"
                "    AbsoluteTolerance: 1.0e-5\n"
                "    RelativeTolerance: 1.0e-8\n"
                "    MaxFactor: 1.1\n"
                "    MinFactor: 0.1\n"
                "- LimitIncrease:\n"
                "    Factor: 2\n"
                "- Constant: 0.5");
          },
          box);
      CHECK(db::get<Tags::StepperErrorEstimatesEnabled>(*box));
      CHECK(db::get<Tags::StepperErrorTolerances<EvolvedVariablesTag>>(*box) ==
            StepperErrorTolerances{.estimates = expected_estimates,
                                   .absolute = 1.0e-5,
                                   .relative = 1.0e-4});
      CHECK(db::get<Tags::StepperErrorTolerances<AltEvolvedVariablesTag>>(
                *box) == StepperErrorTolerances{.estimates = expected_estimates,
                                                .absolute = 1.0e-5,
                                                .relative = 1.0e-8});

      db::mutate<Tags::StepChoosers>(
          [](const gsl::not_null<Tags::StepChoosers::type*> choosers) {
            *choosers = TestHelpers::test_creation<Tags::StepChoosers::type,
                                                   Metavariables>(
                "- ErrorControl:\n"
                "    SafetyFactor: 0.95\n"
                "    AbsoluteTolerance: 1.0e-5\n"
                "    RelativeTolerance: 1.0e-4\n"
                "    MaxFactor: 2.1\n"
                "    MinFactor: 0.5\n"
                "- ErrorControl:\n"
                "    SafetyFactor: 0.8\n"
                "    AbsoluteTolerance: 1.0e-5\n"
                "    RelativeTolerance: 1.0e-8\n"
                "    MaxFactor: 1.1\n"
                "    MinFactor: 0.1\n"
                "- LimitIncrease:\n"
                "    Factor: 2\n"
                "- Constant: 0.5");
          },
          box);
      CHECK_THROWS_WITH(
          db::get<Tags::StepperErrorTolerances<EvolvedVariablesTag>>(*box),
          Catch::Matchers::ContainsSubstring("All ErrorControl events for ") and
              Catch::Matchers::ContainsSubstring("EvolvedVar1") and
              Catch::Matchers::ContainsSubstring(
                  " must use the same tolerances."));

      db::mutate<Tags::StepChoosers,
                 Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>(
          [](const gsl::not_null<Tags::StepChoosers::type*> choosers,
             const gsl::not_null<EventsAndTriggers*> events) {
            *choosers = TestHelpers::test_creation<Tags::StepChoosers::type,
                                                   Metavariables>("");
            *events =
                TestHelpers::test_creation<EventsAndTriggers, Metavariables>(
                    "- Trigger: Always\n"
                    "  Events:\n"
                    "    - OtherEvent\n"
                    "- Trigger: Always\n"
                    "  Events:\n"
                    "    - OtherEvent\n"
                    "    - ChangeSlabSize:\n"
                    "        DelayChange: 0\n"
                    "        StepChoosers:\n"
                    "          - LimitIncrease:\n"
                    "              Factor: 2\n"
                    "          - FixedLtsRatio:\n"
                    "              StepChoosers:\n"
                    "                - Constant: 0.5");
          },
          box);
      CHECK_FALSE(db::get<Tags::StepperErrorEstimatesEnabled>(*box));
      CHECK(db::get<Tags::StepperErrorTolerances<EvolvedVariablesTag>>(*box)
                .estimates == StepperErrorTolerances::Estimates::None);
      CHECK(db::get<Tags::StepperErrorTolerances<AltEvolvedVariablesTag>>(*box)
                .estimates == StepperErrorTolerances::Estimates::None);

      db::mutate<Tags::StepChoosers,
                 Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>(
          [](const gsl::not_null<Tags::StepChoosers::type*> choosers,
             const gsl::not_null<EventsAndTriggers*> events) {
            *choosers = TestHelpers::test_creation<Tags::StepChoosers::type,
                                                   Metavariables>("");
            *events =
                TestHelpers::test_creation<EventsAndTriggers, Metavariables>(
                    "- Trigger: Always\n"
                    "  Events:\n"
                    "    - OtherEvent\n"
                    "- Trigger: Always\n"
                    "  Events:\n"
                    "    - OtherEvent\n"
                    "    - ChangeSlabSize:\n"
                    "        DelayChange: 0\n"
                    "        StepChoosers:\n"
                    "          - LimitIncrease:\n"
                    "              Factor: 2\n"
                    "          - FixedLtsRatio:\n"
                    "              StepChoosers:\n"
                    "                - ErrorControl:\n"
                    "                    SafetyFactor: 0.95\n"
                    "                    AbsoluteTolerance: 1.0e-5\n"
                    "                    RelativeTolerance: 1.0e-4\n"
                    "                    MaxFactor: 2.1\n"
                    "                    MinFactor: 0.5\n"
                    "                - Constant: 0.5");
          },
          box);
      CHECK(db::get<Tags::StepperErrorEstimatesEnabled>(*box));
      CHECK(db::get<Tags::StepperErrorTolerances<EvolvedVariablesTag>>(*box) ==
            StepperErrorTolerances{.estimates = expected_estimates,
                                   .absolute = 1.0e-5,
                                   .relative = 1.0e-4});
      CHECK(db::get<Tags::StepperErrorTolerances<AltEvolvedVariablesTag>>(*box)
                .estimates == StepperErrorTolerances::Estimates::None);
    };

    auto box = db::create<
        db::AddSimpleTags<
            Tags::StepChoosers,
            Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>,
            Tags::ConcreteTimeStepper<LtsTimeStepper>,
            Tags::VariableOrderAlgorithm>,
        tmpl::push_back<
            time_stepper_ref_tags<LtsTimeStepper>,
            Tags::StepperErrorEstimatesEnabledCompute<true>,
            Tags::StepperErrorTolerancesCompute<EvolvedVariablesTag, true>,
            Tags::StepperErrorTolerancesCompute<AltEvolvedVariablesTag,
                                                true>>>();

    db::mutate<Tags::ConcreteTimeStepper<LtsTimeStepper>,
               Tags::VariableOrderAlgorithm>(
        [](const gsl::not_null<std::unique_ptr<LtsTimeStepper>*> time_stepper,
           const gsl::not_null<VariableOrderAlgorithm*> vo_algorithm) {
          *time_stepper = std::make_unique<TimeSteppers::AdamsBashforth>(4);
          *vo_algorithm = VariableOrderAlgorithm(0.1);
        },
        make_not_null(&box));
    test_lts_tags(make_not_null(&box), false);

    db::mutate<Tags::VariableOrderAlgorithm>(
        [](const gsl::not_null<VariableOrderAlgorithm*> vo_algorithm) {
          *vo_algorithm = VariableOrderAlgorithm(4_st);
        },
        make_not_null(&box));
    test_lts_tags(make_not_null(&box), false);

    db::mutate<Tags::ConcreteTimeStepper<LtsTimeStepper>>(
        [](const gsl::not_null<std::unique_ptr<LtsTimeStepper>*> time_stepper) {
          *time_stepper =
              std::make_unique<TimeSteppers::AdamsBashforth>(std::nullopt);
        },
        make_not_null(&box));
    test_lts_tags(make_not_null(&box), false);

    db::mutate<Tags::VariableOrderAlgorithm>(
        [](const gsl::not_null<VariableOrderAlgorithm*> vo_algorithm) {
          *vo_algorithm = VariableOrderAlgorithm(0.1);
        },
        make_not_null(&box));
    test_lts_tags(make_not_null(&box), true);
  }

  {
    INFO("Compute tag GTS test");
    auto box = db::create<
        db::AddSimpleTags<
            Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>,
        db::AddComputeTags<
            Tags::StepperErrorEstimatesEnabledCompute<false>,
            Tags::StepperErrorTolerancesCompute<EvolvedVariablesTag, false>,
            Tags::StepperErrorTolerancesCompute<AltEvolvedVariablesTag,
                                                false>>>();
    db::mutate<Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>(
        [](const gsl::not_null<EventsAndTriggers*> events) {
          *events =
              TestHelpers::test_creation<EventsAndTriggers, Metavariables>(
                  "- Trigger: Always\n"
                  "  Events:\n"
                  "    - OtherEvent\n"
                  "- Trigger: Always\n"
                  "  Events:\n"
                  "    - OtherEvent\n"
                  "    - ChangeSlabSize:\n"
                  "        DelayChange: 0\n"
                  "        StepChoosers:\n"
                  "          - LimitIncrease:\n"
                  "              Factor: 2\n"
                  "          - ErrorControl:\n"
                  "              SafetyFactor: 0.95\n"
                  "              AbsoluteTolerance: 1.0e-5\n"
                  "              RelativeTolerance: 1.0e-4\n"
                  "              MaxFactor: 2.1\n"
                  "              MinFactor: 0.5\n"
                  "          - Constant: 0.5");
        },
        make_not_null(&box));
    CHECK(db::get<Tags::StepperErrorEstimatesEnabled>(box));
    CHECK(db::get<Tags::StepperErrorTolerances<EvolvedVariablesTag>>(box) ==
          StepperErrorTolerances{
              .estimates = StepperErrorTolerances::Estimates::StepperOrder,
              .absolute = 1.0e-5,
              .relative = 1.0e-4});
    CHECK(db::get<Tags::StepperErrorTolerances<AltEvolvedVariablesTag>>(box)
              .estimates == StepperErrorTolerances::Estimates::None);

    db::mutate<Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>(
        [](const gsl::not_null<EventsAndTriggers*> events) {
          *events =
              TestHelpers::test_creation<EventsAndTriggers, Metavariables>(
                  "- Trigger: Always\n"
                  "  Events:\n"
                  "    - OtherEvent\n"
                  "- Trigger: Always\n"
                  "  Events:\n"
                  "    - OtherEvent\n"
                  "    - ChangeSlabSize:\n"
                  "        DelayChange: 0\n"
                  "        StepChoosers:\n"
                  "          - LimitIncrease:\n"
                  "              Factor: 2\n"
                  "          - Constant: 0.5");
        },
        make_not_null(&box));
    CHECK_FALSE(db::get<Tags::StepperErrorEstimatesEnabled>(box));
    CHECK(db::get<Tags::StepperErrorTolerances<EvolvedVariablesTag>>(box)
              .estimates == StepperErrorTolerances::Estimates::None);
    CHECK(db::get<Tags::StepperErrorTolerances<AltEvolvedVariablesTag>>(box)
              .estimates == StepperErrorTolerances::Estimates::None);

    db::mutate<Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>(
        [](const gsl::not_null<EventsAndTriggers*> events) {
          *events =
              TestHelpers::test_creation<EventsAndTriggers, Metavariables>(
                  "- Trigger: Always\n"
                  "  Events:\n"
                  "    - OtherEvent\n"
                  "- Trigger: Always\n"
                  "  Events:\n"
                  "    - OtherEvent");
        },
        make_not_null(&box));
    CHECK_FALSE(db::get<Tags::StepperErrorEstimatesEnabled>(box));
    CHECK(db::get<Tags::StepperErrorTolerances<EvolvedVariablesTag>>(box)
              .estimates == StepperErrorTolerances::Estimates::None);
    CHECK(db::get<Tags::StepperErrorTolerances<AltEvolvedVariablesTag>>(box)
              .estimates == StepperErrorTolerances::Estimates::None);

    db::mutate<Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>(
        [](const gsl::not_null<EventsAndTriggers*> events) {
          *events =
              TestHelpers::test_creation<EventsAndTriggers, Metavariables>("");
        },
        make_not_null(&box));
    CHECK_FALSE(db::get<Tags::StepperErrorEstimatesEnabled>(box));
    CHECK(db::get<Tags::StepperErrorTolerances<EvolvedVariablesTag>>(box)
              .estimates == StepperErrorTolerances::Estimates::None);
    CHECK(db::get<Tags::StepperErrorTolerances<AltEvolvedVariablesTag>>(box)
              .estimates == StepperErrorTolerances::Estimates::None);
  }
}
}  // namespace
