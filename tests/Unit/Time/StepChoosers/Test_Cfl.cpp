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
#include "Domain/MinimumGridSpacing.hpp"  // IWYU pragma: keep
#include "Domain/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/StepChoosers/Cfl.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags.hpp"  // IWYU pragma: keep
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <pup.h>

namespace {
struct CharacteristicSpeed : db::SimpleTag {
  using type = double;
};

struct Metavariables {
  using component_list = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
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
                           const double& speed) noexcept {
        *return_speed = speed;
      }
    };
  };
};


constexpr size_t dim = 1;
using frame = Frame::Grid;
using StepChooserType = StepChooser<tmpl::list<
    StepChoosers::Registrars::Cfl<dim, frame, typename Metavariables::system>>>;
using Cfl = StepChoosers::Cfl<dim, frame, typename Metavariables::system>;

std::pair<double, bool> get_suggestion(const size_t stepper_order,
                                       const double safety_factor,
                                       const double characteristic_speed,
                                       const DataVector& coordinates) noexcept {
  const Parallel::GlobalCache<Metavariables> cache{};
  auto box = db::create<
      db::AddSimpleTags<
          CharacteristicSpeed, domain::Tags::Coordinates<dim, frame>,
          domain::Tags::Mesh<dim>, Tags::TimeStepper<TimeStepper>>,
      db::AddComputeTags<domain::Tags::MinimumGridSpacingCompute<dim, frame>,
                         typename Metavariables::system::
                             compute_largest_characteristic_speed>>(
      characteristic_speed, tnsr::I<DataVector, dim, frame>{{{coordinates}}},
      Mesh<dim>(coordinates.size(), Spectral::Basis::Legendre,
                Spectral::Quadrature::GaussLobatto),
      std::unique_ptr<TimeStepper>{
          std::make_unique<TimeSteppers::AdamsBashforthN>(stepper_order)});

  const double grid_spacing =
      get<domain::Tags::MinimumGridSpacing<dim, frame>>(box);
  const double speed =
      get<typename Metavariables::system::compute_largest_characteristic_speed>(
          box);
  const auto& time_stepper = get<Tags::TimeStepper<TimeStepper>>(box);

  const Cfl cfl{safety_factor};
  const std::unique_ptr<StepChooserType> cfl_base = std::make_unique<Cfl>(cfl);

  const double current_step = std::numeric_limits<double>::infinity();
  const auto result =
      cfl(grid_spacing, time_stepper, speed, current_step, cache);
  CHECK_FALSE(result.second);
  const auto accepted_step_result =
      cfl(grid_spacing, time_stepper, speed, result.first * 0.7, cache);
  CHECK(accepted_step_result.second);
  CHECK(cfl_base->desired_step(make_not_null(&box), current_step, cache) ==
        result);
  CHECK(cfl_base->desired_slab(current_step, box, cache) == result.first);
  CHECK(serialize_and_deserialize(cfl)(grid_spacing, time_stepper, speed,
                                       current_step, cache) == result);
  CHECK(serialize_and_deserialize(cfl_base)->desired_step(
            make_not_null(&box), current_step, cache) == result);
  return result;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.Cfl", "[Unit][Time]") {
  Parallel::register_derived_classes_with_charm<StepChooserType>();

  CHECK(get_suggestion(1, 1., 1., {0., 2., 3., 5.}).first == approx(1.));
  CHECK(get_suggestion(2, 1., 1., {0., 2., 3., 5.}).first < 1.);
  CHECK(get_suggestion(1, 2., 1., {0., 2., 3., 5.}).first == approx(2.));
  CHECK(get_suggestion(1, 1., 2., {0., 2., 3., 5.}).first == approx(0.5));
  CHECK(get_suggestion(1, 1., 1., {0., 2., 2.5, 5.}).first == approx(0.5));

  TestHelpers::test_factory_creation<StepChooserType>(
      "Cfl:\n"
      "  SafetyFactor: 5.0");
}
