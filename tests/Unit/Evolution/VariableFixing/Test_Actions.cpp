// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <unordered_map>
#include <utility>
// IWYU pragma: no_include <exception>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/VariableFixing/Actions.hpp"
#include "Evolution/VariableFixing/RadiallyFallingFloor.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_forward_declare Actions::ApplyVariableFixer

namespace {
using VariableFixer = VariableFixing::RadiallyFallingFloor<
    3, hydro::Tags::RestMassDensity<DataVector>,
    hydro::Tags::Pressure<DataVector>>;

template <typename Metavariables>
struct mock_component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<::Actions::ApplyVariableFixer<VariableFixer>>;
  using initial_databox = db::compute_databox_type<
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 ::Tags::Coordinates<3, Frame::Inertial>>>;
};

struct Metavariables {
  using component_list = tmpl::list<mock_component<Metavariables>>;
  using const_global_cache_tag_list =
      tmpl::list<VariableFixing::OptionTags::MaskRadius>;
  enum class Phase { Initialize, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.VariableFixing.Actions",
                  "[Unit][Evolution][VariableFixing]") {
  using TupleOfMockDistributedObjects =
      typename ActionTesting::MockRuntimeSystem<
          Metavariables>::TupleOfMockDistributedObjects;
  using component = mock_component<Metavariables>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using MockDistributedObjectsTag =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<component>;
  TupleOfMockDistributedObjects dist_objects{};
  const DataVector x{-2.0, -1.0, 0.0, 1.0, 2.0};
  const DataVector y{-2.0, -1.0, 0.0, 1.0, 2.0};
  const DataVector z{-2.0, -1.0, 0.0, 1.0, 2.0};

  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(0,
               ActionTesting::MockDistributedObject<component>{db::create<
                   db::AddSimpleTags<hydro::Tags::RestMassDensity<DataVector>,
                                     hydro::Tags::Pressure<DataVector>,
                                     ::Tags::Coordinates<3, Frame::Inertial>>>(
                   Scalar<DataVector>{DataVector{2.3, -4.2, 1.e-10, 0.0, -0.1}},
                   Scalar<DataVector>{DataVector{0.0, 1.e-8, 2.0, -5.5, 3.2}},
                   tnsr::I<DataVector, 3, Frame::Inertial>{{{x, y, z}}})});
  const double radius_at_which_to_begin_applying_floor = 1.e-4;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {radius_at_which_to_begin_applying_floor}, std::move(dist_objects)};
  auto& box = runner.template algorithms<component>()
                  .at(0)
                  .template get_databox<typename component::initial_databox>();
  runner.next_action<component>(0);
  const double root_three = sqrt(3.0);
  constexpr double one_third = 1.0 / 3.0;
  const DataVector fixed_pressure{
      1.e-7 * pow(2.0 * root_three, -2.5) * one_third, 1.e-8, 2.0,
      1.e-7 * pow(3, -1.25) * one_third, 3.2};
  const DataVector fixed_density{
      2.3, 1.e-5 * pow(3, -0.75),
      1.e-10,  // quantities at a radius below
               // `radius_at_which_to_begin_applying_floor` do not get fixed.
      1.e-5 * pow(3, -0.75), 1.e-5 * pow(2.0 * root_three, -1.5)};

  CHECK_ITERABLE_APPROX(db::get<hydro::Tags::Pressure<DataVector>>(box).get(),
                        fixed_pressure);
  CHECK_ITERABLE_APPROX(
      db::get<hydro::Tags::RestMassDensity<DataVector>>(box).get(),
      fixed_density);
}
