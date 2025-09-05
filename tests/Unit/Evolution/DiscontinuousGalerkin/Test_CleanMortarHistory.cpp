// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/DiscontinuousGalerkin/CleanMortarHistory.hpp"
#include "Evolution/DiscontinuousGalerkin/CleanMortarHistory.tpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Utilities/Gsl.hpp"

namespace {
struct Var : db::SimpleTag {
  using type = double;
};

struct System {
  static constexpr size_t volume_dim = 2;
  using variables_tag = Var;
};

SPECTRE_TEST_CASE("Unit.Evolution.DG.CleanMortarHistory", "[Unit][Evolution]") {
  const Slab slab(1., 3.);

  TimeSteppers::BoundaryHistory<evolution::dg::MortarData<2>,
                                evolution::dg::MortarData<2>, double>
      boundary_history{};
  boundary_history.local().insert(TimeStepId(true, 0, slab.start()), 2, {});
  boundary_history.local().insert(TimeStepId(true, 0, slab.end()), 2, {});
  boundary_history.remote().insert(TimeStepId(true, 0, slab.start()), 2, {});
  boundary_history.remote().insert(TimeStepId(true, 0, slab.end()), 2, {});
  evolution::dg::Tags::MortarDataHistory<2, double>::type mortar_histories{};
  const std::array mortars{
      DirectionalId<2>{{Direction<2>::Axis::Xi, Side::Lower}, ElementId<2>{}},
      DirectionalId<2>{{Direction<2>::Axis::Xi, Side::Upper}, ElementId<2>{}}};
  for (const auto& mortar : mortars) {
    mortar_histories.emplace(mortar, boundary_history);
  }

  auto box = db::create<
      db::AddSimpleTags<Tags::ConcreteTimeStepper<LtsTimeStepper>,
                        evolution::dg::Tags::MortarDataHistory<2, double>>,
      time_stepper_ref_tags<LtsTimeStepper>>(
      static_cast<std::unique_ptr<LtsTimeStepper>>(
          std::make_unique<TimeSteppers::AdamsBashforth>(2)),
      std::move(mortar_histories));

  db::mutate_apply<evolution::dg::CleanMortarHistory<System>>(
      make_not_null(&box));

  for (const auto& mortar : mortars) {
    CHECK(db::get<evolution::dg::Tags::MortarDataHistory<2, double>>(box)
              .at(mortar)
              .local()
              .size() == 1);
  }
}
}  // namespace
