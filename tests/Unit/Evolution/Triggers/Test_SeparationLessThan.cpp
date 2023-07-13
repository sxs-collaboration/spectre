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
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ExcisionSphere.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/Triggers/SeparationLessThan.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
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

using TranslationMap = domain::CoordinateMaps::TimeDependent::Translation<3>;

void test() {
  const std::string f_of_t_name = "LlamasWithHats";
  std::unique_ptr<domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>
      map = std::make_unique<
          domain::CoordinateMap<Frame::Grid, Frame::Inertial, TranslationMap>>(
          TranslationMap{f_of_t_name});

  const auto check = [&map, &f_of_t_name](
                         const double separation, const double time,
                         const double x_center_a, const double x_center_b,
                         const bool expected_is_triggered) {
    const tnsr::I<double, 3, Frame::Grid> center_a{
        std::array{x_center_a, 0.0, 0.0}};
    const tnsr::I<double, 3, Frame::Grid> center_b{
        std::array{x_center_b, 0.0, 0.0}};

    ExcisionSphere<3> excision_sphere_a{2.0, center_a, {}};
    ExcisionSphere<3> excision_sphere_b{2.0, center_b, {}};

    excision_sphere_a.inject_time_dependent_maps(map->get_clone());
    excision_sphere_b.inject_time_dependent_maps(map->get_clone());

    std::unordered_map<std::string, ExcisionSphere<3>> excision_spheres{};
    excision_spheres["ExcisionSphere" + get_output(domain::ObjectLabel::A)] =
        std::move(excision_sphere_a);
    excision_spheres["ExcisionSphere" + get_output(domain::ObjectLabel::B)] =
        std::move(excision_sphere_b);

    const Domain<3> domain{{}, std::move(excision_spheres)};

    const Triggers::SeparationLessThan trigger{separation};

    // The coefs are zero because we want the inertial point to be the same
    // as the grid point for easy checking
    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        functions_of_time{};
    functions_of_time[f_of_t_name] =
        std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
            time, std::array{DataVector{3, 0.0}}, time + 1.0);

    const bool is_triggered =
        trigger(time, domain, functions_of_time, center_a, center_b);

    CHECK(is_triggered == expected_is_triggered);
  };

  for (const auto& [separation, x_center] : cartesian_product(
           make_array(10.0, 5.0, 2.0), make_array(6.0, 5.0, 2.0, 0.75))) {
    check(separation, 0.0, x_center, -x_center, 2.0 * x_center <= separation);
  }

  TestHelpers::test_creation<std::unique_ptr<Trigger>, Metavariables>(
      "SeparationLessThan:\n"
      "  Value: 2.3");
}

void test_errors() {
  const Triggers::SeparationLessThan trigger{1.0};

  std::unordered_map<std::string, ExcisionSphere<3>> excision_spheres{};

  CHECK_THROWS_WITH(trigger(0.0, Domain<3>{{}, excision_spheres}, {}, {}, {}),
                    Catch::Contains("SeparationLessThan trigger expects an "
                                    "excision sphere named 'ExcisionSphere"));

  ExcisionSphere<3> excision_sphere_a{};
  excision_spheres["ExcisionSphere" + get_output(domain::ObjectLabel::A)] =
      std::move(excision_sphere_a);

  CHECK_THROWS_WITH(trigger(0.0, Domain<3>{{}, excision_spheres}, {}, {}, {}),
                    Catch::Contains("to be time dependent, but it is not."));
}

SPECTRE_TEST_CASE("Unit.Evolution.Triggers.SeparationLessThan",
                  "[Unit][Time]") {
  register_factory_classes_with_charm<Metavariables>();
  domain::FunctionsOfTime::register_derived_with_charm();
  PUPable_reg(SINGLE_ARG(
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, TranslationMap>));
  test();
  test_errors();
}
}  // namespace
