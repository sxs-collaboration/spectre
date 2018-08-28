// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <memory>
// IWYU pragma: no_include <pup.h>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/MinimumGridSpacing.hpp"  // IWYU pragma: keep
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/ConstGlobalCache.hpp"
// IWYU pragma: no_include "Parallel/PupStlCpp11.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/StepChoosers/Cfl.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags.hpp"  // IWYU pragma: keep
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
constexpr size_t dim = 1;
using frame = Frame::Grid;
using registrars = tmpl::list<StepChoosers::Register::Cfl<dim, frame>>;
using Cfl = StepChoosers::Cfl<dim, frame, registrars>;

struct CharacteristicSpeed : db::SimpleTag {
  static std::string name() noexcept { return "CharacteristicSpeed"; }
  using type = double;
};

struct Metavariables {
  using component_list = tmpl::list<>;
  using const_global_cache_tag_list = tmpl::list<CacheTags::TimeStepper>;
  struct system {
    struct compute_largest_characteristic_speed {
      using argument_tags = tmpl::list<CharacteristicSpeed>;
      double operator()(const double speed) const noexcept { return speed; }
    };
  };
};

double get_suggestion(const size_t stepper_order, const double safety_factor,
                      const double characteristic_speed,
                      const DataVector& coordinates) noexcept {
  const Parallel::ConstGlobalCache<Metavariables> cache{
      {std::make_unique<TimeSteppers::AdamsBashforthN>(stepper_order)}};
  const auto box = db::create<
      db::AddSimpleTags<CharacteristicSpeed,
                        domain::Tags::Coordinates<dim, frame>,
                        domain::Tags::Mesh<dim>>,
      db::AddComputeTags<domain::Tags::MinimumGridSpacing<dim, frame>>>(
      characteristic_speed, tnsr::I<DataVector, dim, frame>{{{coordinates}}},
      domain::Mesh<dim>(coordinates.size(), Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto));

  const double grid_spacing =
      get<domain::Tags::MinimumGridSpacing<dim, frame>>(box);

  const Cfl cfl{safety_factor};
  const std::unique_ptr<StepChooser<registrars>> cfl_base =
      std::make_unique<Cfl>(cfl);

  const double result = cfl(grid_spacing, box, cache);
  CHECK(cfl_base->desired_step(box, cache) == result);
  CHECK(serialize_and_deserialize(cfl)(grid_spacing, box, cache) == result);
  CHECK(serialize_and_deserialize(cfl_base)->desired_step(box, cache) ==
        result);
  return result;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.Cfl", "[Unit][Time]") {
  Parallel::register_derived_classes_with_charm<StepChooser<registrars>>();

  CHECK(get_suggestion(1, 1., 1., {0., 2., 3., 5.}) == approx(1.));
  CHECK(get_suggestion(2, 1., 1., {0., 2., 3., 5.}) < 1.);
  CHECK(get_suggestion(1, 2., 1., {0., 2., 3., 5.}) == approx(2.));
  CHECK(get_suggestion(1, 1., 2., {0., 2., 3., 5.}) == approx(0.5));
  CHECK(get_suggestion(1, 1., 1., {0., 2., 2.5, 5.}) == approx(0.5));

  test_factory_creation<StepChooser<registrars>>(
      "  Cfl:\n"
      "    SafetyFactor: 5.0");
}
