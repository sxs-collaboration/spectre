// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/Actions/Psi0Matching.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/NumericalAlgorithms/Spectral/SwshTestHelpers.hpp"
#include "Parallel/Phase.hpp"

namespace Cce {

namespace {

struct UpdateCartesianFromAngular {
  using const_global_cache_tags = tmpl::list<Tags::LMax>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate_apply<GaugeUpdateAngularFromCartesian<
        Tags::PartiallyFlatAngularCoords, Tags::PartiallyFlatCartesianCoords>>(
        make_not_null(&box));
    db::mutate_apply<GaugeUpdateInterpolator<Tags::PartiallyFlatAngularCoords>>(
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

using real_tags_to_generate = tmpl::list<Tags::PartiallyFlatCartesianCoords>;

using real_tags_to_compute = tmpl::list<Tags::PartiallyFlatAngularCoords>;

using swsh_volume_tags_to_generate = tmpl::list<Tags::BondiJ, Tags::OneMinusY>;

using swsh_boundary_tags_to_generate =
    tmpl::list<Tags::CauchyGaugeC, Tags::CauchyGaugeD, Tags::CauchyGaugeOmega,
               Tags::BoundaryValue<Tags::BondiR>,
               Tags::BoundaryValue<Tags::BondiBeta>>;

using swsh_volume_tags_to_compute =
    tmpl::list<Tags::BondiJCauchyView, Tags::Dy<Tags::BondiJCauchyView>,
               Tags::Dy<Tags::Dy<Tags::BondiJCauchyView>>, Tags::Psi0Match,
               Tags::Dy<Tags::Psi0Match>>;

using swsh_boundary_tags_to_compute =
    tmpl::list<Tags::BoundaryValue<Tags::Psi0Match>,
               Tags::BoundaryValue<Tags::Dlambda<Tags::Psi0Match>>>;

template <typename Metavariables>
struct mock_characteristic_evolution {
  using simple_tags = tmpl::push_back<
      db::AddSimpleTags<
          ::Tags::Variables<
              tmpl::append<real_tags_to_generate, real_tags_to_compute>>,
          ::Tags::Variables<tmpl::append<swsh_volume_tags_to_generate,
                                         swsh_volume_tags_to_compute>>,
          ::Tags::Variables<tmpl::append<swsh_boundary_tags_to_generate,
                                         swsh_boundary_tags_to_compute>>>,
      Spectral::Swsh::Tags::SwshInterpolator<Tags::PartiallyFlatAngularCoords>>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Evolve,
          tmpl::list<UpdateCartesianFromAngular,
                     Actions::CalculatePsi0AndDerivAtInnerBoundary>>>;
};

struct metavariables {
  using component_list =
      tmpl::list<mock_characteristic_evolution<metavariables>>;
};
}  // namespace

// This unit test is to validate the automatic calling chain of the action
// CalculatePsi0AndDerivAtInnerBoundary. The test involves
// (a) Fills a bunch of variables with random numbers (filtered so that there is
// no aliasing in highest modes).
// (b) Puts those variables in two places: the
// MockRuntimeSystem runner and a databox called expected_box.
// (c) Calls next_action twice (which fills the results in the databox of
// MockRuntimeSystem runner).
// (d) Calls all the individual mutators by hand, filling the results in
// expected_box.
// (e) Compares MockRuntimeSystem's databox vs expected_box.
// This test cannot catch any error in any of the individual mutators. The
// correctness of the mutators is tested in Test_NewmanPenrose.cpp
SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.Actions.Psi0Matching",
                  "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  // limited l_max distribution because test depends on an analytic
  // basis function with factorials.
  UniformCustomDistribution<size_t> sdist{7, 10};
  const size_t l_max = sdist(gen);
  UniformCustomDistribution<size_t> sdist_rad{2, 3};
  const size_t number_of_radial_points = sdist_rad(gen);
  UniformCustomDistribution<double> coefficient_distribution{-2.0, 2.0};
  CAPTURE(l_max);

  using component = mock_characteristic_evolution<metavariables>;
  ActionTesting::MockRuntimeSystem<metavariables> runner{{l_max}};

  // (a) Fills a bunch of variables with random numbers (filtered so that there
  // is no aliasing in highest modes).
  Variables<tmpl::append<real_tags_to_generate, real_tags_to_compute>>
      real_variables{Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  Variables<
      tmpl::append<swsh_volume_tags_to_generate, swsh_volume_tags_to_compute>>
      swsh_volume_variables{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
          number_of_radial_points};
  Variables<tmpl::append<swsh_boundary_tags_to_generate,
                         swsh_boundary_tags_to_compute>>
      swsh_boundary_variables{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

  tmpl::for_each<real_tags_to_generate>([
    &real_variables, &gen, &coefficient_distribution, &l_max
  ](auto tag_v) {
    using tag = typename decltype(tag_v)::type;
    SpinWeighted<ComplexModalVector, 0> generated_modes{
        Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
    SpinWeighted<ComplexDataVector, 0> generated_data{
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    for (auto& tensor_component : get<tag>(real_variables)) {
      Spectral::Swsh::TestHelpers::generate_swsh_modes<0>(
          make_not_null(&generated_modes.data()), make_not_null(&gen),
          make_not_null(&coefficient_distribution), 1, l_max);
      Spectral::Swsh::inverse_swsh_transform(
          l_max, 1, make_not_null(&generated_data), generated_modes);
      // aggressive filter to make the uniformly generated random modes
      // somewhat reasonable
      Spectral::Swsh::filter_swsh_boundary_quantity(
          make_not_null(&generated_data), l_max, l_max / 2);
      tensor_component = real(generated_data.data());
    }
  });

  tmpl::for_each<swsh_volume_tags_to_generate>([&swsh_volume_variables, &gen,
                                                &coefficient_distribution,
                                                &l_max,
                                                &number_of_radial_points](
                                                   auto tag_v) {
    using tag = typename decltype(tag_v)::type;
    const size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    SpinWeighted<ComplexModalVector, 0> generated_modes{
        Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
    SpinWeighted<ComplexDataVector, 0> generated_data{
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    for (auto& tensor_component : get<tag>(swsh_volume_variables)) {
      SpinWeighted<ComplexDataVector, 2> angular_view_j;
      for (size_t i = 0; i < number_of_radial_points; ++i) {
        Spectral::Swsh::TestHelpers::generate_swsh_modes<0>(
            make_not_null(&generated_modes.data()), make_not_null(&gen),
            make_not_null(&coefficient_distribution), 1, l_max);
        Spectral::Swsh::inverse_swsh_transform(
            l_max, 1, make_not_null(&generated_data), generated_modes);
        // aggressive filter to make the uniformly generated random modes
        // somewhat reasonable
        Spectral::Swsh::filter_swsh_boundary_quantity(
            make_not_null(&generated_data), l_max, l_max / 2);
        angular_view_j.set_data_ref(
            tensor_component.data().data() + i * number_of_angular_points,
            number_of_angular_points);
        angular_view_j.data() = generated_data.data();
      }
    }
  });

  tmpl::for_each<swsh_boundary_tags_to_generate>([
    &swsh_boundary_variables, &gen, &coefficient_distribution, &l_max
  ](auto tag_v) {
    using tag = typename decltype(tag_v)::type;
    SpinWeighted<ComplexModalVector, 0> generated_modes{
        Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
    SpinWeighted<ComplexDataVector, 0> generated_data{
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    for (auto& tensor_component : get<tag>(swsh_boundary_variables)) {
      Spectral::Swsh::TestHelpers::generate_swsh_modes<0>(
          make_not_null(&generated_modes.data()), make_not_null(&gen),
          make_not_null(&coefficient_distribution), 1, l_max);
      Spectral::Swsh::inverse_swsh_transform(
          l_max, 1, make_not_null(&generated_data), generated_modes);
      // aggressive filter to make the uniformly generated random modes
      // somewhat reasonable
      Spectral::Swsh::filter_swsh_boundary_quantity(
          make_not_null(&generated_data), l_max, l_max / 2);
      tensor_component = generated_data.data();
    }
  });

  // (b) Puts those variables in two places: the
  // MockRuntimeSystem runner and a databox called expected_box.
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {real_variables, swsh_volume_variables, swsh_boundary_variables,
       Spectral::Swsh::SwshInterpolator{}});
  auto expected_box = db::create<
      tmpl::append<component::simple_tags, db::AddSimpleTags<Tags::LMax>>>(
      std::move(real_variables), std::move(swsh_volume_variables),
      std::move(swsh_boundary_variables), Spectral::Swsh::SwshInterpolator{},
      l_max);

  // (c) Calls next_action twice (which fills the results in the databox of
  // MockRuntimeSystem runner).
  runner.set_phase(Parallel::Phase::Evolve);
  // Obtain inertial angular coordinates and SwshInterpolator that are needed
  // by `CalculatePsi0AndDerivAtInnerBoundary`. This action is tested in
  // Test_UpdateGauge.cpp
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  // Apply the `CalculatePsi0AndDerivAtInnerBoundary` action
  ActionTesting::next_action<component>(make_not_null(&runner), 0);

  // (d) Calls all the individual mutators by hand, filling the results in
  // expected_box.
  db::mutate_apply<GaugeUpdateAngularFromCartesian<
      Tags::PartiallyFlatAngularCoords, Tags::PartiallyFlatCartesianCoords>>(
      make_not_null(&expected_box));
  db::mutate_apply<GaugeUpdateInterpolator<Tags::PartiallyFlatAngularCoords>>(
      make_not_null(&expected_box));

  db::mutate_apply<TransformBondiJToCauchyCoords>(make_not_null(&expected_box));
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::BondiJCauchyView>>>(
      make_not_null(&expected_box));
  db::mutate_apply<
      PreSwshDerivatives<Tags::Dy<Tags::Dy<Tags::BondiJCauchyView>>>>(
      make_not_null(&expected_box));
  db::mutate_apply<VolumeWeyl<Tags::Psi0Match>>(make_not_null(&expected_box));
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::Psi0Match>>>(
      make_not_null(&expected_box));
  db::mutate_apply<InnerBoundaryWeyl>(make_not_null(&expected_box));

  // (e) Compares MockRuntimeSystem's databox vs expected_box.
  tmpl::for_each<swsh_volume_tags_to_compute>(
      [&runner, &expected_box](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        const auto& action_result =
            ActionTesting::get_databox_tag<component, tag>(runner, 0);
        const auto& expected_result = db::get<tag>(expected_box);
        CHECK_ITERABLE_APPROX(action_result, expected_result);
      });

  tmpl::for_each<swsh_boundary_tags_to_compute>(
      [&runner, &expected_box ](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        const auto& action_result =
            ActionTesting::get_databox_tag<component, tag>(runner, 0);
        const auto& expected_result = db::get<tag>(expected_box);
        CHECK_ITERABLE_APPROX(action_result, expected_result);
      });
}
}  // namespace Cce
