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
#include "Evolution/Systems/Cce/Actions/CalculateScriInputs.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/PreSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/PrecomputeCceDependencies.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/Spectral/SwshTestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/VectorAlgebra.hpp"

namespace Cce {

namespace {
// these will be generated in the test
using all_pre_swsh_derivative_scri_input_tags = tmpl::list<
    Tags::BondiH, Tags::DuRDividedByR, Tags::EthRDividedByR,
    Tags::Dy<Tags::BondiJ>, Tags::Dy<Tags::BondiW>, Tags::Dy<Tags::BondiQ>,
    Tags::Dy<Tags::BondiU>, Tags::Dy<Tags::Dy<Tags::BondiBeta>>,
    Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                     Spectral::Swsh::Tags::EthEthbar>>;

// these need to be computed via `PreSwshDerivatives` and
// `PrecomputeCceDependencies`, but are not calculated in the action being
// tested
using extra_pre_swsh_derivative_scri_tags =
    tmpl::list<Tags::Dy<Tags::Dy<Tags::BondiJ>>>;

using extra_precomputation_scri_tags = tmpl::list<Tags::OneMinusY>;

using all_real_boundary_scri_input_tags =
    tmpl::list<Tags::InertialRetardedTime>;

using transform_buffers_for_scri =
    tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
        all_swsh_derivative_tags_for_scri,
        tmpl::bind<Spectral::Swsh::coefficient_buffer_tags_for_derivative_tag,
                   tmpl::_1>>>>;

template <typename Metavariables>
struct mock_characteristic_evolution {
  using simple_tags = db::AddSimpleTags<
      ::Tags::Variables<tmpl::append<all_swsh_derivative_tags_for_scri,
                                     all_pre_swsh_derivative_tags_for_scri,
                                     all_pre_swsh_derivative_scri_input_tags,
                                     extra_pre_swsh_derivative_scri_tags,
                                     extra_precomputation_scri_tags>>,
      ::Tags::Variables<transform_buffers_for_scri>,
      ::Tags::Variables<all_real_boundary_scri_input_tags>,
      ::Tags::Variables<
          tmpl::append<all_boundary_pre_swsh_derivative_tags_for_scri,
                       all_boundary_swsh_derivative_tags_for_scri>>>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  // so that the precomputation mutator can be used in the tmpl::transform
  template <typename Tag>
  using local_precompute = PrecomputeCceDependencies<Tags::BoundaryValue, Tag>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Evolve,
          tmpl::list<tmpl::transform<
                         extra_pre_swsh_derivative_scri_tags,
                         tmpl::bind<::Actions::MutateApply,
                                    tmpl::bind<PreSwshDerivatives, tmpl::_1>>>,
                     tmpl::transform<
                         extra_precomputation_scri_tags,
                         tmpl::bind<::Actions::MutateApply,
                                    tmpl::bind<local_precompute, tmpl::_1>>>,
                     Actions::CalculateScriInputs>>>;
};

struct metavariables {
  using component_list =
      tmpl::list<mock_characteristic_evolution<metavariables>>;
  enum class Phase { Initialization, Evolve, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.Actions.CalculateScriInputs",
                  "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  // limited l_max distribution because test depends on an analytic
  // basis function with factorials.
  UniformCustomDistribution<size_t> sdist{7, 10};
  const size_t l_max = sdist(gen);
  UniformCustomDistribution<double> coefficient_distribution{-2.0, 2.0};
  const size_t number_of_radial_points = sdist(gen);
  CAPTURE(l_max);
  CAPTURE(number_of_radial_points);
  using component = mock_characteristic_evolution<metavariables>;
  ActionTesting::MockRuntimeSystem<metavariables> runner{
      tuples::tagged_tuple_from_typelist<
          Parallel::get_const_global_cache_tags<metavariables>>{
          l_max, number_of_radial_points}};
  Variables<tmpl::append<
      all_swsh_derivative_tags_for_scri, all_pre_swsh_derivative_tags_for_scri,
      all_pre_swsh_derivative_scri_input_tags,
      extra_pre_swsh_derivative_scri_tags, extra_precomputation_scri_tags>>
      component_volume_variables{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
          number_of_radial_points};
  Variables<all_real_boundary_scri_input_tags> component_real_variables{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  Variables<transform_buffers_for_scri> component_transform_variables{
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max) *
      number_of_radial_points};
  Variables<tmpl::append<all_boundary_pre_swsh_derivative_tags_for_scri,
                         all_boundary_swsh_derivative_tags_for_scri>>
      component_boundary_variables{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

  tmpl::for_each<
      all_pre_swsh_derivative_scri_input_tags>([&component_volume_variables,
                                                &gen, &coefficient_distribution,
                                                &number_of_radial_points,
                                                &l_max](auto tag_v) {
    using tag = typename decltype(tag_v)::type;
    SpinWeighted<ComplexModalVector, tag::type::type::spin> generated_modes{
        number_of_radial_points *
        Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
    Spectral::Swsh::TestHelpers::generate_swsh_modes<tag::type::type::spin>(
        make_not_null(&generated_modes.data()), make_not_null(&gen),
        make_not_null(&coefficient_distribution), number_of_radial_points,
        l_max);
    get(get<tag>(component_volume_variables)) =
        Spectral::Swsh::inverse_swsh_transform(l_max, number_of_radial_points,
                                               generated_modes);
    // aggressive filter to make the uniformly generated random modes
    // somewhat reasonable
    Spectral::Swsh::filter_swsh_volume_quantity(
        make_not_null(&get(get<tag>(component_volume_variables))), l_max,
        l_max / 2, 32.0, 8);
  });

  tmpl::for_each<all_real_boundary_scri_input_tags>([&component_real_variables,
                                                     &gen,
                                                     &coefficient_distribution,
                                                     &l_max](auto tag_v) {
    using tag = typename decltype(tag_v)::type;
    SpinWeighted<ComplexModalVector, 0> generated_modes{
        Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
    Spectral::Swsh::TestHelpers::generate_swsh_modes<0>(
        make_not_null(&generated_modes.data()), make_not_null(&gen),
        make_not_null(&coefficient_distribution), 1, l_max);
    SpinWeighted<ComplexDataVector, 0> spin_weighted_buffer =
        Spectral::Swsh::inverse_swsh_transform(l_max, 1, generated_modes);
    // aggressive filter to make the uniformly generated random modes
    // somewhat reasonable
    Spectral::Swsh::filter_swsh_boundary_quantity(
        make_not_null(&spin_weighted_buffer), l_max, l_max / 2);
    get(get<tag>(component_real_variables)) = real(spin_weighted_buffer.data());
  });

  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {component_volume_variables, component_transform_variables,
       component_real_variables, component_boundary_variables});
  auto expected_box = db::create<
      tmpl::append<component::simple_tags,
                   db::AddSimpleTags<Tags::LMax, Tags::NumberOfRadialPoints>>>(
      component_volume_variables, component_transform_variables,
      component_real_variables, component_boundary_variables, l_max,
      number_of_radial_points);
  runner.set_phase(metavariables::Phase::Initialization);

  // run the initialization to get the values into the databox
  tmpl::for_each<extra_pre_swsh_derivative_scri_tags>(
      [&expected_box](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        db::mutate_apply<PreSwshDerivatives<tag>>(make_not_null(&expected_box));
      });

  tmpl::for_each<extra_precomputation_scri_tags>([&expected_box](auto tag_v) {
    using tag = typename decltype(tag_v)::type;
    db::mutate_apply<PrecomputeCceDependencies<Tags::BoundaryValue, tag>>(
        make_not_null(&expected_box));
  });

  runner.set_phase(metavariables::Phase::Evolve);
  // this will execute all of the `MutateApply` actions for the `extra...`
  // typelists
  for (size_t i = 0;
       i < tmpl::size<extra_pre_swsh_derivative_scri_tags>::value +
               tmpl::size<extra_precomputation_scri_tags>::value + 1;
       ++i) {
    ActionTesting::next_action<component>(make_not_null(&runner), 0);
  }

  // execute the desired box manipulations directly without calling the actions.
  // The mutations themselves are tested in other unit tests.
  tmpl::for_each<tmpl::append<all_pre_swsh_derivative_tags_for_scri,
                              all_boundary_pre_swsh_derivative_tags_for_scri>>(
      [&expected_box](auto pre_swsh_derivative_tag_v) {
        using pre_swsh_derivative_tag =
            typename decltype(pre_swsh_derivative_tag_v)::type;
        db::mutate_apply<PreSwshDerivatives<pre_swsh_derivative_tag>>(
            make_not_null(&expected_box));
      });

  db::mutate_apply<
      Spectral::Swsh::AngularDerivatives<all_swsh_derivative_tags_for_scri>>(
      make_not_null(&expected_box));
  tmpl::for_each<all_boundary_swsh_derivative_tags_for_scri>(
      [&expected_box, &l_max](auto swsh_derivative_tag_v) {
        using swsh_derivative_tag =
            typename decltype(swsh_derivative_tag_v)::type;
        db::mutate<swsh_derivative_tag>(
            make_not_null(&expected_box),
            [&l_max](const gsl::not_null<typename swsh_derivative_tag::type*>
                         derivative,
                     const typename swsh_derivative_tag::derivative_of::type&
                         argument) {
              Spectral::Swsh::angular_derivatives<
                  tmpl::list<typename swsh_derivative_tag::derivative_kind>>(
                  l_max, 1, make_not_null(&get(*derivative)), get(argument));
            },
            db::get<typename swsh_derivative_tag::derivative_of>(expected_box));
      });

  tmpl::for_each<all_swsh_derivative_tags_for_scri>([&expected_box](
                                                        auto derivative_tag_v) {
    using derivative_tag = typename decltype(derivative_tag_v)::type;
    detail::apply_swsh_jacobian_helper<derivative_tag>(
        make_not_null(&expected_box),
        typename ApplySwshJacobianInplace<
            derivative_tag>::on_demand_argument_tags{});
  });

  Approx multiple_derivative_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e4)
          .scale(1.0);

  tmpl::for_each<tmpl::append<all_swsh_derivative_tags_for_scri,
                              all_pre_swsh_derivative_tags_for_scri,
                              all_boundary_pre_swsh_derivative_tags_for_scri,
                              all_boundary_swsh_derivative_tags_for_scri>>(
      [&expected_box, &runner, &multiple_derivative_approx](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        const auto& test_lhs =
            ActionTesting::get_databox_tag<component, tag>(runner, 0);
        const auto& test_rhs = db::get<tag>(expected_box);
        CHECK_ITERABLE_CUSTOM_APPROX(test_lhs, test_rhs,
                                     multiple_derivative_approx);
      });
}
}  // namespace Cce
