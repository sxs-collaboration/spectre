// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <typeinfo>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/EqualRateRegions.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/EqualRateRegions.tpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/FixedLtsRatio.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/Tags/EqualRateRegions.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Time/RequestsStepperErrorTolerances.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/LimitIncrease.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/Tags/FixedLtsRatio.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace PUP {
class er;
}  // namespace PUP

namespace {
class ErrorChooser : public StepChooser<StepChooserUse::LtsStep> {
 public:
  ErrorChooser() = default;
  explicit ErrorChooser(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(ErrorChooser);  // NOLINT
#pragma GCC diagnostic pop

  static constexpr Options::String help{""};
  using options = tmpl::list<>;

  using argument_tags = tmpl::list<>;

  TimeStepRequest operator()(double /*last_step*/) const {
    ERROR("StepChooser should not be called in fixed LTS region");
  }

  bool uses_local_data() const override { return false; }
  bool can_be_delayed() const override { return true; }
};

PUP::able::PUP_ID ErrorChooser::my_PUP_ID = 0;  // NOLINT

struct Var1 {};
struct Var2 {};

class ToleranceChooser : public StepChooser<StepChooserUse::LtsStep>,
                         public RequestsStepperErrorTolerances {
 public:
  ToleranceChooser() = default;
  explicit ToleranceChooser(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(ToleranceChooser);  // NOLINT
#pragma GCC diagnostic pop

  static constexpr Options::String help{""};
  using options = tmpl::list<>;

  ToleranceChooser(const double var1_tol, const double var2_tol)
      : var1_tol_(var1_tol), var2_tol_(var2_tol) {}

  using argument_tags = tmpl::list<>;

  TimeStepRequest operator()(double /*last_step*/) const { return {}; }

  bool uses_local_data() const override { return false; }
  bool can_be_delayed() const override { return true; }

  std::unordered_map<std::type_index, StepperErrorTolerances> tolerances()
      const override {
    return {{typeid(Var1),
             {.estimates = StepperErrorTolerances::Estimates::StepperOrder,
              .absolute = var1_tol_,
              .relative = var1_tol_ * 10.0}},
            {typeid(Var2),
             {.estimates = StepperErrorTolerances::Estimates::StepperOrder,
              .absolute = var2_tol_,
              .relative = var2_tol_ * 10.0}}};
  }

 private:
  double var1_tol_{};
  double var2_tol_{};
};

PUP::able::PUP_ID ToleranceChooser::my_PUP_ID = 0;  // NOLINT

template <size_t Dim>
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<
            StepChooser<StepChooserUse::LtsStep>,
            tmpl::list<StepChoosers::Constant, StepChoosers::LimitIncrease,
                       ErrorChooser, ToleranceChooser>>,
        tmpl::pair<
            StepChooser<StepChooserUse::Slab>,
            tmpl::list<evolution::dg::StepChoosers::FixedLtsRatio<Dim>>>>;
  };
};

class TestRegion {
 public:
  using creation_tags = tmpl::list<>;

  static std::unordered_map<std::string, size_t> regions() {
    std::unordered_map<std::string, size_t> result{};
    result.emplace("Region", 0);
    result.emplace("OtherRegion", 1);
    return result;
  }

  template <size_t Dim>
  static bool is_in_region(const size_t region_id,
                           const ElementId<Dim>& element_id) {
    return region_id == 0 and element_id.block_id() == 0;
  }

  void pup(PUP::er& p);  // unused
};

template <size_t Dim>
void test(const std::optional<double>& expected_goal,
          const std::optional<double>& expected_size,
          const std::optional<size_t>& fixed_ratio,
          const std::string& lts_choosers, const std::string& active_regions,
          const bool expect_error) {
  CAPTURE(lts_choosers);
  CAPTURE(active_regions);
  const Slab slab(2.0, 6.0);
  const auto time_step = slab.duration() / 2;

  const Element<Dim> element(ElementId<Dim>(fixed_ratio.has_value() ? 0 : 1),
                             {});

  for (const auto& time_sign : {1, -1}) {
    CAPTURE(time_sign);
    auto box = db::create<
        db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables<Dim>>,
                          evolution::dg::Tags::ConcreteEqualRateRegions<
                              Dim, tmpl::list<TestRegion>>,
                          domain::Tags::Element<Dim>, Tags::TimeStep,
                          Tags::FixedLtsRatio>,
        db::AddComputeTags<evolution::dg::Tags::EqualRateRegionsRef<
            Dim, tmpl::list<TestRegion>>>>(
        Metavariables<Dim>{},
        evolution::dg::EqualRateRegions<Dim, tmpl::list<TestRegion>>{}, element,
        time_sign * time_step, fixed_ratio);

    const auto chooser = TestHelpers::test_creation<
        std::unique_ptr<StepChooser<StepChooserUse::Slab>>, Metavariables<Dim>>(
        "FixedLtsRatio:\n"
        "  Regions: " +
        active_regions +
        "\n"
        "  StepChoosers:\n" +
        lts_choosers);

    const auto set_sign = [&](const std::optional<double>& opt) {
      if (opt.has_value()) {
        return std::optional<double>(time_sign * *opt);
      } else {
        return std::optional<double>{};
      }
    };

    const double current_step = time_sign * slab.duration().value();
    if (expect_error) {
      CHECK_THROWS_WITH(chooser->desired_step(current_step, box),
                        Catch::Matchers::ContainsSubstring("Unknown region"));
    } else {
      const TimeStepRequest expected{.size_goal = set_sign(expected_goal),
                                     .size = set_sign(expected_size)};
      CHECK(chooser->desired_step(current_step, box) == expected);
      CHECK(serialize_and_deserialize(chooser)->desired_step(current_step,
                                                             box) == expected);
    }
  }
}

template <size_t Dim>
void test_regions(const std::optional<double>& expected_goal,
                  const std::optional<double>& expected_size,
                  const std::optional<size_t>& fixed_ratio,
                  const std::string& lts_choosers) {
  test<Dim>(expected_goal, expected_size, fixed_ratio, lts_choosers, "All",
            false);
  test<Dim>(expected_goal, expected_size, fixed_ratio, lts_choosers,
            "[Region, OtherRegion]", false);
  test<Dim>(expected_goal, expected_size, fixed_ratio, lts_choosers,
            "[OtherRegion, Region]", false);
  test<Dim>(expected_goal, expected_size, fixed_ratio, lts_choosers, "[Region]",
            false);
  test<Dim>({}, {}, fixed_ratio, lts_choosers, "[OtherRegion]", false);
  test<Dim>({}, {}, fixed_ratio, lts_choosers, "[]", false);
  if (fixed_ratio.has_value()) {
    // Inputs are only checked when in a region.
    test<Dim>({}, {}, fixed_ratio, lts_choosers, "[BadRegion]", true);
    test<Dim>({}, {}, fixed_ratio, lts_choosers, "[Region, BadRegion]", true);
  }
}

template <size_t Dim>
void test_dim() {
  register_factory_classes_with_charm<Metavariables<Dim>>();

  test_regions<Dim>({}, {}, {}, "    - ErrorChooser");
  test_regions<Dim>({}, {}, {8}, "");
  test_regions<Dim>({40.0}, {}, {8},
                    "    - Constant: 5.0\n"
                    "    - Constant: 7.0");
  // Initial step size used in test is 2.0
  test_regions<Dim>({}, {64.0}, {8},
                    "    - LimitIncrease:\n"
                    "        Factor: 4.0\n"
                    "    - LimitIncrease:\n"
                    "        Factor: 9.0");
  test_regions<Dim>({40.0}, {32.0}, {8},
                    "    - Constant: 5.0\n"
                    "    - LimitIncrease:\n"
                    "        Factor: 2.0");
  // Should never give a limit larger than the goal.
  test_regions<Dim>({40.0}, {}, {8},
                    "    - Constant: 5.0\n"
                    "    - LimitIncrease:\n"
                    "        Factor: 4.0");

  CHECK(evolution::dg::StepChoosers::FixedLtsRatio<Dim>{}.uses_local_data());
  CHECK(evolution::dg::StepChoosers::FixedLtsRatio<Dim>{}.can_be_delayed());

  {
    const evolution::dg::StepChoosers::FixedLtsRatio<Dim> no_tolerances{
        make_vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>(
            std::make_unique<StepChoosers::Constant>(1.0)),
        std::nullopt};
    CHECK(no_tolerances.tolerances().empty());
  }

  {
    const evolution::dg::StepChoosers::FixedLtsRatio<Dim> tolerance_chooser{
        make_vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>(
            std::make_unique<ToleranceChooser>(1.0e-4, 1.0e-10),
            std::make_unique<ToleranceChooser>(1.0e-4, 1.0e-10)),
        std::nullopt};
    const auto tolerances = tolerance_chooser.tolerances();
    CHECK(tolerances.size() == 2);
    CHECK(tolerances.at(typeid(Var1)).absolute == 1.0e-4);
    CHECK(tolerances.at(typeid(Var1)).relative == 1.0e-3);
    CHECK(tolerances.at(typeid(Var2)).absolute == 1.0e-10);
    CHECK(tolerances.at(typeid(Var2)).relative == 1.0e-9);
  }

  {
    const evolution::dg::StepChoosers::FixedLtsRatio<Dim> bad_tolerances{
        make_vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>(
            std::make_unique<ToleranceChooser>(1.0e-4, 1.0e-10),
            std::make_unique<ToleranceChooser>(1.0e-4, 1.0e-8)),
        std::nullopt};
    CHECK_THROWS_WITH(bad_tolerances.tolerances(),
                      Catch::Matchers::ContainsSubstring("must be the same"));
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.DG.EqualRateLts.FixedLtsRatio",
                  "[Unit][Evolution]") {
  test_dim<1>();
  test_dim<2>();
  test_dim<3>();
}
}  // namespace
