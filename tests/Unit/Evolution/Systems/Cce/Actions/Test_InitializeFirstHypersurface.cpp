// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/Actions/UpdateGauge.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/NumericalAlgorithms/Spectral/SwshTestHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace Cce {

namespace {

using swsh_boundary_tags_to_generate =
    tmpl::list<Tags::BoundaryValue<Tags::BondiJ>,
               Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>,
               Tags::BoundaryValue<Tags::BondiR>>;

using real_boundary_tags_to_compute =
    tmpl::list<Tags::CauchyCartesianCoords, Tags::CauchyAngularCoords,
               Tags::InertialRetardedTime>;

using swsh_boundary_tags_to_compute =
    tmpl::list<Tags::GaugeC, Tags::GaugeD, Tags::GaugeOmega>;

using swsh_volume_tags_to_compute = tmpl::list<Tags::BondiJ>;

template <typename Metavariables>
struct mock_characteristic_evolution {
  using simple_tags = tmpl::push_back<db::AddSimpleTags<
      ::Tags::Variables<real_boundary_tags_to_compute>,
      ::Tags::Variables<tmpl::append<swsh_boundary_tags_to_generate,
                                     swsh_boundary_tags_to_compute>>,
      ::Tags::Variables<swsh_volume_tags_to_compute>>>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing,
                             tmpl::list<Actions::InitializeFirstHypersurface>>>;
};

struct metavariables {
  using component_list =
      tmpl::list<mock_characteristic_evolution<metavariables>>;

  using cce_hypersurface_initialization = InitializeJ<Tags::BoundaryValue>;
  enum class Phase { Initialization, Testing, Exit };
};
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Cce.Actions.InitializeFirstHypersurface",
    "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  // limited l_max distribution because test depends on an analytic
  // basis function with factorials.
  UniformCustomDistribution<size_t> sdist{7, 10};
  const size_t l_max = sdist(gen);
  const size_t number_of_radial_points = sdist(gen);
  UniformCustomDistribution<double> coefficient_distribution{-2.0, 2.0};
  CAPTURE(l_max);

  using component = mock_characteristic_evolution<metavariables>;
  ActionTesting::MockRuntimeSystem<metavariables> runner{
      {l_max, number_of_radial_points}};

  Variables<real_boundary_tags_to_compute> real_variables{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  Variables<tmpl::append<swsh_boundary_tags_to_generate,
                         swsh_boundary_tags_to_compute>>
      swsh_variables{Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  Variables<swsh_volume_tags_to_compute> swsh_volume_variables{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
      number_of_radial_points};

  tmpl::for_each<swsh_boundary_tags_to_generate>([&swsh_variables, &gen,
                                                  &coefficient_distribution,
                                                  &l_max](auto tag_v) noexcept {
    using tag = typename decltype(tag_v)::type;
    SpinWeighted<ComplexModalVector, tag::type::type::spin> generated_modes{
        Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
    Spectral::Swsh::TestHelpers::generate_swsh_modes<tag::type::type::spin>(
        make_not_null(&generated_modes.data()), make_not_null(&gen),
        make_not_null(&coefficient_distribution), 1, l_max);
    Spectral::Swsh::inverse_swsh_transform(
        l_max, 1, make_not_null(&get(get<tag>(swsh_variables))),
        generated_modes);
    // aggressive filter to make the uniformly generated random modes
    // somewhat reasonable
    Spectral::Swsh::filter_swsh_volume_quantity(
        make_not_null(&get(get<tag>(swsh_variables))), l_max, l_max / 2, 32.0,
        8);
  });

  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0, {real_variables, swsh_variables, swsh_volume_variables});
  auto expected_box = db::create<
      tmpl::append<component::simple_tags,
                   db::AddSimpleTags<Tags::LMax, Tags::NumberOfRadialPoints>>>(
      std::move(real_variables), std::move(swsh_variables),
      std::move(swsh_volume_variables), l_max, number_of_radial_points);

  runner.set_phase(metavariables::Phase::Testing);
  // apply the `InitializeFirstHypersurface` action
  ActionTesting::next_action<component>(make_not_null(&runner), 0);

  // apply the corresponding mutators to the `expected_box`
  db::mutate_apply<InitializeJ<Tags::BoundaryValue>>(
      make_not_null(&expected_box));
  db::mutate_apply<InitializeGauge>(make_not_null(&expected_box));
  db::mutate_apply<InitializeScriPlusValue<Tags::InertialRetardedTime>>(
      make_not_null(&expected_box));

  tmpl::for_each<
      tmpl::append<real_boundary_tags_to_compute, swsh_boundary_tags_to_compute,
                   swsh_volume_tags_to_compute>>(
      [&runner, &expected_box](auto tag_v) noexcept {
        using tag = typename decltype(tag_v)::type;
        const auto& test_lhs =
            ActionTesting::get_databox_tag<component, tag>(runner, 0);
        const auto& test_rhs = db::get<tag>(expected_box);
        CHECK_ITERABLE_APPROX(test_lhs, test_rhs);
      });
}
}  // namespace Cce
