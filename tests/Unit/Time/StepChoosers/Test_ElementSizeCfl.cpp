// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/SizeOfElement.hpp"
#include "Domain/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/StepChoosers/ElementSizeCfl.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct CharacteristicSpeed : db::SimpleTag {
  using type = double;
};

template <size_t Dim>
struct Metavariables {
  using component_list = tmpl::list<>;
  using time_stepper_tag = Tags::TimeStepper<TimeStepper>;
  struct system {
    struct largest_characteristic_speed : db::SimpleTag {
      using type = double;
    };
    struct compute_largest_characteristic_speed : db::ComputeTag,
                                                  largest_characteristic_speed {
      using base = largest_characteristic_speed;
      using argument_tags = tmpl::list<CharacteristicSpeed>;
      using return_type = double;
      static void function(const gsl::not_null<double*> return_speed,
                           const double& speed) {
        *return_speed = speed;
      }
    };
  };
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                             tmpl::list<StepChoosers::ElementSizeCfl<
                                 StepChooserUse::LtsStep, Dim, system>>>,
                  tmpl::pair<StepChooser<StepChooserUse::Slab>,
                             tmpl::list<StepChoosers::ElementSizeCfl<
                                 StepChooserUse::Slab, Dim, system>>>>;
  };
};

template <size_t Dim>
std::pair<double, bool> get_suggestion(
    const double safety_factor, const double characteristic_speed,
    ElementMap<Dim, Frame::Grid>&& element_map) {
  const Parallel::GlobalCache<Metavariables<Dim>> cache{};
  auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables<Dim>>,
                        CharacteristicSpeed, Tags::TimeStepper<TimeStepper>,
                        domain::Tags::ElementMap<Dim, Frame::Grid>,
                        domain::CoordinateMaps::Tags::CoordinateMap<
                            Dim, Frame::Grid, Frame::Inertial>,
                        ::Tags::Time, domain::Tags::FunctionsOfTime>,
      db::AddComputeTags<domain::Tags::SizeOfElementCompute<Dim>,
                         typename Metavariables<Dim>::system::
                             compute_largest_characteristic_speed>>(
      Metavariables<Dim>{}, characteristic_speed,
      std::unique_ptr<TimeStepper>{
          std::make_unique<TimeSteppers::AdamsBashforthN>(2)},
      std::move(element_map),
      ::domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          ::domain::CoordinateMaps::Identity<Dim>{}),
      0.0, typename domain::Tags::FunctionsOfTime::type{});
  const StepChoosers::ElementSizeCfl<StepChooserUse::LtsStep, Dim,
                                     typename Metavariables<Dim>::system>
      element_size_cfl{safety_factor};
  const std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>
      element_size_base = std::make_unique<StepChoosers::ElementSizeCfl<
          StepChooserUse::LtsStep, Dim, typename Metavariables<Dim>::system>>(
          element_size_cfl);

  const double speed = get<typename Metavariables<
      Dim>::system::compute_largest_characteristic_speed>(box);
  const std::array<double, Dim> element_size =
      db::get<domain::Tags::SizeOfElement<Dim>>(box);
  const auto& time_stepper = get<Tags::TimeStepper<>>(box);

  const double current_step = std::numeric_limits<double>::infinity();
  const std::pair<double, bool> result =
      element_size_cfl(time_stepper, element_size, speed, current_step, cache);
  CHECK_FALSE(result.second);
  const auto accepted_step_result = element_size_cfl(
      time_stepper, element_size, speed, result.first * 0.7, cache);
  CHECK(accepted_step_result.second);
  CHECK(element_size_base->desired_step(make_not_null(&box), current_step,
                                        cache) == result);
  CHECK(serialize_and_deserialize(element_size_cfl)(
            time_stepper, element_size, speed, current_step, cache) == result);
  CHECK(serialize_and_deserialize(element_size_base)
            ->desired_step(make_not_null(&box), current_step, cache) == result);
  return result;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.ElementSizeCfl", "[Unit][Time]") {
  Parallel::register_factory_classes_with_charm<Metavariables<1>>();
  Parallel::register_factory_classes_with_charm<Metavariables<2>>();
  Parallel::register_factory_classes_with_charm<Metavariables<3>>();

  {
    INFO("Test 1D element size CFL step chooser");
    auto map =
        domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
            domain::CoordinateMaps::Affine(-1.0, 1.0, 0.3, 1.1));
    const ElementId<1> element_id(0, {{{2, 3}}});
    ElementMap<1, Frame::Grid> logical_to_grid_map(element_id, std::move(map));
    CHECK(approx(
              get_suggestion(0.8, 2.0, std::move(logical_to_grid_map)).first) ==
          0.04);
  }
  {
    INFO("Test 2D element size CFL step chooser");
    using Affine = domain::CoordinateMaps::Affine;
    using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
    auto map =
        domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
            Affine2D{Affine(-1.0, 1.0, 0.3, 0.4),
                     Affine(-1.0, 1.0, -0.5, 1.1)});
    const ElementId<2> element_id(0, {{{1, 0}, {2, 3}}});
    ElementMap<2, Frame::Grid> logical_to_grid_map(element_id, std::move(map));
    CHECK(approx(
              get_suggestion(0.8, 2.0, std::move(logical_to_grid_map)).first) ==
          0.005);
  }
  {
    INFO("Test 3D element size CFL step chooser");
    using Affine = domain::CoordinateMaps::Affine;
    using Affine3D =
        domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
    auto map =
        domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
            Affine3D{Affine(-1.0, 1.0, 0.3, 0.4), Affine(-1.0, 1.0, -0.5, 1.1),
                     Affine(-1.0, 1.0, 12.0, 12.4)});
    const ElementId<3> element_id(0, {{{2, 3}, {1, 0}, {3, 4}}});
    ElementMap<3, Frame::Grid> logical_to_grid_map(element_id, std::move(map));
    CHECK(approx(
              get_suggestion(0.8, 2.0, std::move(logical_to_grid_map)).first) ==
          0.005 / 3.0);
  }

  TestHelpers::test_creation<
      std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>, Metavariables<1>>(
      "ElementSizeCfl:\n"
      "  SafetyFactor: 5.0");
  TestHelpers::test_creation<
      std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>, Metavariables<2>>(
      "ElementSizeCfl:\n"
      "  SafetyFactor: 5.0");
  TestHelpers::test_creation<
      std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>, Metavariables<3>>(
      "ElementSizeCfl:\n"
      "  SafetyFactor: 5.0");
}
