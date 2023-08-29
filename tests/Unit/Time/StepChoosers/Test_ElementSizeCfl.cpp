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
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/SizeOfElement.hpp"
#include "Domain/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Time/StepChoosers/ElementSizeCfl.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct CharacteristicSpeed : db::SimpleTag {
  using type = double;
};

template <size_t Dim>
struct Metavariables {
  using component_list = tmpl::list<>;
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
  auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables<Dim>>,
                        CharacteristicSpeed, Tags::TimeStepper<TimeStepper>,
                        domain::Tags::ElementMap<Dim, Frame::Grid>,
                        domain::CoordinateMaps::Tags::CoordinateMap<
                            Dim, Frame::Grid, Frame::Inertial>,
                        ::Tags::Time, domain::Tags::FunctionsOfTimeInitialize>,
      db::AddComputeTags<domain::Tags::SizeOfElementCompute<Dim>,
                         typename Metavariables<Dim>::system::
                             compute_largest_characteristic_speed>>(
      Metavariables<Dim>{}, characteristic_speed,
      std::unique_ptr<TimeStepper>{
          std::make_unique<TimeSteppers::AdamsBashforth>(2)},
      std::move(element_map),
      ::domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          ::domain::CoordinateMaps::Identity<Dim>{}),
      0.0, typename domain::Tags::FunctionsOfTimeInitialize::type{});
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
      element_size_cfl(time_stepper, element_size, speed, current_step);
  CHECK_FALSE(result.second);
  const auto accepted_step_result =
      element_size_cfl(time_stepper, element_size, speed, result.first * 0.7);
  CHECK(accepted_step_result.second);
  CHECK(element_size_base->desired_step(current_step, box) == result);
  CHECK(serialize_and_deserialize(element_size_cfl)(
            time_stepper, element_size, speed, current_step) == result);
  CHECK(serialize_and_deserialize(element_size_base)
            ->desired_step(current_step, box) == result);
  return result;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.ElementSizeCfl", "[Unit][Time]") {
  register_factory_classes_with_charm<Metavariables<1>>();
  register_factory_classes_with_charm<Metavariables<2>>();
  register_factory_classes_with_charm<Metavariables<3>>();

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

  CHECK(StepChoosers::ElementSizeCfl<StepChooserUse::Slab, 1,
                                     Metavariables<1>::system>{}
            .uses_local_data());
}
