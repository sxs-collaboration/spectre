// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iomanip>
#include <memory>
#include <random>
#include <unordered_set>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/Random.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
constexpr size_t volume_dim = 2;

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                             tmpl::list<StepChoosers::Random<
                                 StepChooserUse::LtsStep, volume_dim>>>,
                  tmpl::pair<StepChooser<StepChooserUse::Slab>,
                             tmpl::list<StepChoosers::Random<
                                 StepChooserUse::Slab, volume_dim>>>>;
  };
  using component_list = tmpl::list<>;
};

template <typename Use>
double get_suggestion(const double min, const double max, const size_t seed,
                      const Element<volume_dim>& element,
                      const TimeStepId& time_step_id) {
  using Random = StepChoosers::Random<Use, volume_dim>;

  auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>,
                        Tags::TimeStepId, domain::Tags::Element<volume_dim>>>(
      Metavariables{}, time_step_id, element);

  // Not used.
  const double current_step = 1.23;

  const Random random(min, max, seed);
  const std::unique_ptr<StepChooser<Use>> random_base =
      TestHelpers::test_factory_creation<StepChooser<Use>, Random>(
          MakeString{} << std::setprecision(19) << "Random:\n"
                       << "  Minimum: " << min << "\n"
                       << "  Maximum: " << max << "\n"
                       << "  Seed: " << seed);

  CHECK(random.uses_local_data());

  const auto result = random(element, time_step_id, current_step);
  REQUIRE(result.first.size_goal.has_value());
  CHECK(result.first == TimeStepRequest{.size_goal = result.first.size_goal});
  // Should be deterministic
  CHECK(result == random(element, time_step_id, current_step));

  CHECK(result.second);

  CHECK(*result.first.size_goal >= min);
  CHECK(*result.first.size_goal <= max);

  CHECK(serialize_and_deserialize(random)(element, time_step_id,
                                          current_step) == result);
  CHECK(random_base->desired_step(current_step, box) == result);
  CHECK(
      serialize_and_deserialize(random_base)->desired_step(current_step, box) ==
      result);

  return *result.first.size_goal;
}

template <typename Use>
void test_random() {
  const Element<volume_dim> element1(ElementId<volume_dim>(1), {});
  const Element<volume_dim> element2(ElementId<volume_dim>(2), {});
  const TimeStepId time1(true, 0, Slab(0.0, 1.0).start());
  const TimeStepId time2(true, 0, Slab(0.0, 1.0).end());

  MAKE_GENERATOR(gen);
  const double min = std::uniform_real_distribution(1.0e-6, 1.0)(gen);
  const double max = min * std::uniform_real_distribution(1.1, 10.0)(gen);
  const size_t seed = gen();

  std::unordered_set<double> results{};
  for (const auto& element : {element1, element2}) {
    for (const auto& time : {time1, time2}) {
      CHECK(results.insert(get_suggestion<Use>(min, max, seed, element, time))
                .second);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.Random", "[Unit][Time]") {
  register_factory_classes_with_charm<Metavariables>();

  test_random<StepChooserUse::LtsStep>();
  test_random<StepChooserUse::Slab>();
}
