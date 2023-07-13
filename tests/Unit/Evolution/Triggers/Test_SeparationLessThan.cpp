// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/Triggers/SeparationLessThan.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Metavariables {
  using component_list = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<Trigger, tmpl::list<Triggers::SeparationLessThan>>>;
  };
};

domain::creators::Brick get_creator(const bool include_distorted_frame) {
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dependence{};
  // We don't actually care about a moving mesh, just that we have a distorted
  // frame so different functions are called inside the trigger.
  if (include_distorted_frame) {
    time_dependence = std::make_unique<
        domain::creators::time_dependence::UniformTranslation<3>>(
        0.0, std::array{0.0, 0.0, 0.0}, std::array{0.0, 0.0, 0.0});
  } else {
    time_dependence = std::make_unique<
        domain::creators::time_dependence::UniformTranslation<3>>(
        0.0, std::array{0.0, 0.0, 0.0});
  }

  return domain::creators::Brick{
      std::array{-10.0, -10.0, -10.0}, std::array{10.0, 10.0, 10.0},
      std::array{0_st, 0_st, 0_st},    std::array{5_st, 5_st, 5_st},
      std::array{false, false, false}, std::move(time_dependence)};
}

SPECTRE_TEST_CASE("Unit.Evolution.Triggers.SeparationLessThan",
                  "[Unit][Time]") {
  register_factory_classes_with_charm<Metavariables>();

  Element<3> element{ElementId<3>{0}, {}};

  const auto check = [&element](const bool include_distorted_frame,
                                const double separation, const double time,
                                const double x_center_a,
                                const double x_center_b,
                                const bool expected_is_triggered) {
    const domain::creators::Brick creator =
        get_creator(include_distorted_frame);
    Domain<3> domain = creator.create_domain();
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        functions_of_time = creator.functions_of_time();
    const tnsr::I<double, 3, Frame::Grid> center_a{
        std::array{x_center_a, 0.0, 0.0}};
    const tnsr::I<double, 3, Frame::Grid> center_b{
        std::array{x_center_b, 0.0, 0.0}};

    const Triggers::SeparationLessThan trigger{separation};

    const bool is_triggered =
        trigger(time, domain, element, functions_of_time, center_a, center_b);

    CHECK(is_triggered == expected_is_triggered);
  };

  for (const auto& [include_distorted_frame, separation, x_center] :
       cartesian_product(make_array(true, false), make_array(10.0, 5.0, 2.0),
                         make_array(6.0, 5.0, 2.0, 0.75))) {
    check(include_distorted_frame, separation, 0.0, x_center, -x_center,
          2.0 * x_center <= separation);
  }

  TestHelpers::test_creation<std::unique_ptr<Trigger>, Metavariables>(
      "SeparationLessThan:\n"
      "  Value: 2.3");
}
}  // namespace
