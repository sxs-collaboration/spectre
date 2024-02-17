// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/Cce/Actions/InitializeKleinGordonFirstHypersurface.hpp"
#include "Evolution/Systems/Cce/KleinGordonSource.hpp"
#include "Evolution/Systems/Cce/PrecomputeCceDependencies.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/NumericalAlgorithms/Spectral/SwshTestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce {
namespace {

using real_volume_tags_to_generate = tmpl::list<Tags::OneMinusY>;

using swsh_volume_tags_to_generate =
    tmpl::list<Tags::Exp2Beta, Tags::BondiR, Tags::BondiK, Tags::BondiJ,
               Tags::Dy<Tags::KleinGordonPsi>,
               Tags::KleinGordonPsi,
               Spectral::Swsh::Tags::Derivative<Tags::KleinGordonPsi,
                                                Spectral::Swsh::Tags::Eth>>;

using swsh_boundary_tags_to_generate =
    tmpl::list<Tags::BoundaryValue<Tags::KleinGordonPi>, Tags::BondiUAtScri>;

using swsh_volume_tags_to_compute =
    tmpl::list<Tags::KleinGordonSource<Tags::BondiBeta>,
               Tags::KleinGordonSource<Tags::BondiQ>,
               Tags::KleinGordonSource<Tags::BondiU>,
               Tags::KleinGordonSource<Tags::BondiW>,
               Tags::KleinGordonSource<Tags::BondiH>>;

using swsh_boundary_tags_to_compute =
    tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::KleinGordonPi>>;

template <typename Metavariables>
struct mock_kg_characteristic_evolution {
  using simple_tags = db::AddSimpleTags<
      ::Tags::Variables<tmpl::append<real_volume_tags_to_generate,
                                     swsh_volume_tags_to_generate,
                                     swsh_volume_tags_to_compute>>,
      ::Tags::Variables<tmpl::append<swsh_boundary_tags_to_generate,
                                     swsh_boundary_tags_to_compute>>,
      Spectral::Swsh::Tags::SwshInterpolator<Tags::CauchyAngularCoords>>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Evolve,
          tmpl::list<
              tmpl::transform<
                  bondi_hypersurface_step_tags,
                  tmpl::bind<::Actions::MutateApply,
                             tmpl::bind<ComputeKleinGordonSource, tmpl::_1>>>,
              ::Actions::MutateApply<
                  GaugeAdjustedBoundaryValue<Tags::KleinGordonPi>>>>>;

  using const_global_cache_tags =
      tmpl::list<Tags::LMax, Tags::NumberOfRadialPoints>;
};

struct metavariables {
  using component_list =
      tmpl::list<mock_kg_characteristic_evolution<metavariables>>;
};

// This unit test is to validate the automatic calling chain of Klein-Gordon Cce
// calculations, including the mutators `ComputeKleinGordonSource` and
// `GaugeAdjustedBoundaryValue<Tags::KleinGordonPi>`. The test involves
// (a) Fills a bunch of variables with random numbers (filtered so that there is
// no aliasing in highest modes).
// (b) Puts those variables in two places: the MockRuntimeSystem runner and a
// databox called expected_box.
// (c) Calls next_action a few times (which fills the results in the databox of
// MockRuntimeSystem runner).
// (d) Calls all the individual calculations by hand, filling the results in
// expected_box.
// (e) Compares MockRuntimeSystem's databox vs expected_box.
template <typename Generator>
void test_klein_gordon_cce_source(const gsl::not_null<Generator*> gen) {
  UniformCustomDistribution<size_t> sdist{7, 10};
  const size_t l_max = sdist(*gen);
  const size_t number_of_radial_points = sdist(*gen);
  UniformCustomDistribution<double> coefficient_distribution{-2.0, 2.0};

  Variables<
      tmpl::append<real_volume_tags_to_generate, swsh_volume_tags_to_generate,
                   swsh_volume_tags_to_compute>>
      component_volume_variables{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
          number_of_radial_points};

  Variables<tmpl::append<swsh_boundary_tags_to_generate,
                         swsh_boundary_tags_to_compute>>
      component_boundary_variables{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

  tmpl::for_each<swsh_volume_tags_to_generate>(
      [&component_volume_variables, &gen, &coefficient_distribution,
       &number_of_radial_points, &l_max](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        SpinWeighted<ComplexModalVector, tag::type::type::spin> generated_modes{
            number_of_radial_points *
            Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
        Spectral::Swsh::TestHelpers::generate_swsh_modes<tag::type::type::spin>(
            make_not_null(&generated_modes.data()), gen,
            make_not_null(&coefficient_distribution), number_of_radial_points,
            l_max);
        get(get<tag>(component_volume_variables)) =
            Spectral::Swsh::inverse_swsh_transform(
                l_max, number_of_radial_points, generated_modes);
        // aggressive filter to make the uniformly generated random modes
        // somewhat reasonable
        Spectral::Swsh::filter_swsh_volume_quantity(
            make_not_null(&get(get<tag>(component_volume_variables))), l_max,
            l_max / 2, 32.0, 8);
      });

  tmpl::for_each<swsh_boundary_tags_to_generate>(
      [&component_boundary_variables, &gen, &coefficient_distribution,
       &l_max](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        SpinWeighted<ComplexModalVector, tag::type::type::spin> generated_modes{
            Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
        Spectral::Swsh::TestHelpers::generate_swsh_modes<tag::type::type::spin>(
            make_not_null(&generated_modes.data()), gen,
            make_not_null(&coefficient_distribution), 1, l_max);
        get(get<tag>(component_boundary_variables)) =
            Spectral::Swsh::inverse_swsh_transform(l_max, 1, generated_modes);
        // aggressive filter to make the uniformly generated random modes
        // somewhat reasonable
        Spectral::Swsh::filter_swsh_boundary_quantity(
            make_not_null(&get(get<tag>(component_boundary_variables))), l_max,
            l_max / 2);
      });

  tnsr::i<DataVector, 3> cartesian_cauchy_coordinates;
  tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>
      angular_cauchy_coordinates;

  Spectral::Swsh::create_angular_and_cartesian_coordinates(
      make_not_null(&cartesian_cauchy_coordinates),
      make_not_null(&angular_cauchy_coordinates), l_max);
  Spectral::Swsh::SwshInterpolator interpolator(
      get<0>(angular_cauchy_coordinates), get<1>(angular_cauchy_coordinates),
      l_max);

  PrecomputeCceDependencies<Tags::EvolutionGaugeBoundaryValue,
                            Tags::OneMinusY>::
      apply(make_not_null(&get<Tags::OneMinusY>(component_volume_variables)),
            l_max, number_of_radial_points);

  using component = mock_kg_characteristic_evolution<metavariables>;

  // tests start here
  ActionTesting::MockRuntimeSystem<metavariables> runner{
      tuples::tagged_tuple_from_typelist<
          Parallel::get_const_global_cache_tags<metavariables>>{
          l_max, number_of_radial_points}};
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {component_volume_variables, component_boundary_variables, interpolator});
  auto expected_box = db::create<
      tmpl::append<component::simple_tags,
                   db::AddSimpleTags<Tags::LMax, Tags::NumberOfRadialPoints>>>(
      component_volume_variables, component_boundary_variables, interpolator,
      l_max, number_of_radial_points);

  runner.set_phase(Parallel::Phase::Evolve);

  for (int i = 0; i < 6; i++) {
    ActionTesting::next_action<component>(make_not_null(&runner), 0);
  }

  // tests for `ComputeKleinGordonSource`
  tmpl::for_each<bondi_hypersurface_step_tags>([&expected_box,
                                                &runner](auto tag_v) {
    using tag = typename decltype(tag_v)::type;
    db::mutate_apply<ComputeKleinGordonSource<tag>>(
        make_not_null(&expected_box));

    auto computed_result =
        ActionTesting::get_databox_tag<component, Tags::KleinGordonSource<tag>>(
            runner, 0);

    auto expected_result = db::get<Tags::KleinGordonSource<tag>>(expected_box);
    CHECK(computed_result == expected_result);
  });

  // tests for `GaugeAdjustedBoundaryValue<Tags::KleinGordonPi>`
  {
    db::mutate_apply<GaugeAdjustedBoundaryValue<Tags::KleinGordonPi>>(
        make_not_null(&expected_box));
    auto computed_result = ActionTesting::get_databox_tag<
        component, Tags::EvolutionGaugeBoundaryValue<Tags::KleinGordonPi>>(
        runner, 0);

    auto expected_result =
        db::get<Tags::EvolutionGaugeBoundaryValue<Tags::KleinGordonPi>>(
            expected_box);
    CHECK(computed_result == expected_result);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.Actions.KGCceCalculations",
                  "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  test_klein_gordon_cce_source(make_not_null(&gen));
}
}  // namespace Cce
