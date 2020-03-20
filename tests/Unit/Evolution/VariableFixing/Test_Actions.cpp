// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"  // IWYU pragma: keep
#include "Evolution/VariableFixing/Actions.hpp"
#include "Evolution/VariableFixing/RadiallyFallingFloor.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"      // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <exception>

// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox
// IWYU pragma: no_forward_declare VariableFixing::Actions::FixVariables

namespace {

template <typename Metavariables>
struct mock_component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using simple_tags = tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                                 hydro::Tags::Pressure<DataVector>,
                                 domain::Tags::Coordinates<3, Frame::Inertial>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing,
                             tmpl::list<VariableFixing::Actions::FixVariables<
                                 VariableFixing::RadiallyFallingFloor<3>>>>>;
};

struct Metavariables {
  using component_list = tmpl::list<mock_component<Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.VariableFixing.Actions",
                  "[Unit][Evolution][VariableFixing]") {
  using component = mock_component<Metavariables>;
  const DataVector x{-2.0, -1.0, 0.0, 1.0, 2.0};
  const DataVector y{-2.0, -1.0, 0.0, 1.0, 2.0};
  const DataVector z{-2.0, -1.0, 0.0, 1.0, 2.0};

  using simple_tags = typename component::simple_tags;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      VariableFixing::RadiallyFallingFloor<3>(1.e-4, 1.e-5, -1.5, 1.e-7 / 3.0,
                                              -2.5)};
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {Scalar<DataVector>{DataVector{2.3, -4.2, 1.e-10, 0.0, -0.1}},
       Scalar<DataVector>{DataVector{0.0, 1.e-8, 2.0, -5.5, 3.2}},
       tnsr::I<DataVector, 3, Frame::Inertial>{{{x, y, z}}}});
  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);

  auto& box = ActionTesting::get_databox<component, simple_tags>(runner, 0);
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
