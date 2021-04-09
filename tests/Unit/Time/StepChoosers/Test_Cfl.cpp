// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
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
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "Time/StepChoosers/Cfl.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags.hpp"  // IWYU pragma: keep
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <pup.h>

namespace {
constexpr size_t dim = 1;
using frame = Frame::Grid;

struct CharacteristicSpeed : db::SimpleTag {
  using type = double;
};

struct Metavariables {
  using component_list = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  using time_stepper_tag = Tags::TimeStepper<TimeStepper>;
  struct system {
    static constexpr size_t volume_dim = dim;
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

  template <typename Use>
  using Cfl = StepChoosers::Cfl<Use, frame, system>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                             tmpl::list<Cfl<StepChooserUse::LtsStep>>>,
                  tmpl::pair<StepChooser<StepChooserUse::Slab>,
                             tmpl::list<Cfl<StepChooserUse::Slab>>>>;
  };
};

template <typename Use>
std::pair<double, bool> get_suggestion(const size_t stepper_order,
                                       const double safety_factor,
                                       const double characteristic_speed,
                                       const DataVector& coordinates) noexcept {
  using Cfl = Metavariables::Cfl<Use>;

  const Parallel::GlobalCache<Metavariables> cache{};
  auto box = db::create<
      db::AddSimpleTags<
          Parallel::Tags::MetavariablesImpl<Metavariables>, CharacteristicSpeed,
          domain::Tags::Coordinates<dim, frame>, domain::Tags::Mesh<dim>,
          Tags::TimeStepper<TimeStepper>>,
      db::AddComputeTags<domain::Tags::MinimumGridSpacingCompute<dim, frame>,
                         typename Metavariables::system::
                             compute_largest_characteristic_speed>>(
      Metavariables{}, characteristic_speed,
      tnsr::I<DataVector, dim, frame>{{{coordinates}}},
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
  const std::unique_ptr<StepChooser<Use>> cfl_base = std::make_unique<Cfl>(cfl);

  const double current_step = std::numeric_limits<double>::infinity();
  const auto result =
      cfl(grid_spacing, time_stepper, speed, current_step, cache);
  CHECK(serialize_and_deserialize(cfl)(grid_spacing, time_stepper, speed,
                                       current_step, cache) == result);
  CHECK_FALSE(result.second);
  const auto accepted_step_result =
      cfl(grid_spacing, time_stepper, speed, result.first * 0.7, cache);
  CHECK(accepted_step_result.second);
  if constexpr (std::is_same_v<Use, StepChooserUse::LtsStep>) {
    CHECK(cfl_base->desired_step(make_not_null(&box), current_step, cache) ==
          result);
    CHECK(serialize_and_deserialize(cfl_base)->desired_step(
              make_not_null(&box), current_step, cache) == result);
  } else {
    CHECK(cfl_base->desired_slab(current_step, box, cache) == result.first);
    CHECK(serialize_and_deserialize(cfl_base)->desired_slab(
              current_step, box, cache) == result.first);
  }
  return result;
}

template <typename Use>
void test_use() noexcept {
  CHECK(get_suggestion<Use>(1, 1., 1., {0., 2., 3., 5.}).first == approx(1.));
  CHECK(get_suggestion<Use>(2, 1., 1., {0., 2., 3., 5.}).first < 1.);
  CHECK(get_suggestion<Use>(1, 2., 1., {0., 2., 3., 5.}).first == approx(2.));
  CHECK(get_suggestion<Use>(1, 1., 2., {0., 2., 3., 5.}).first == approx(0.5));
  CHECK(get_suggestion<Use>(1, 1., 1., {0., 2., 2.5, 5.}).first == approx(0.5));

  TestHelpers::test_creation<std::unique_ptr<StepChooser<Use>>, Metavariables>(
      "Cfl:\n"
      "  SafetyFactor: 5.0");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.Cfl", "[Unit][Time]") {
  Parallel::register_factory_classes_with_charm<Metavariables>();

  test_use<StepChooserUse::LtsStep>();
  test_use<StepChooserUse::Slab>();
}
