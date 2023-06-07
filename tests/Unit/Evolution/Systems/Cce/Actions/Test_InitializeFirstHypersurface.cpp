// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/Actions/UpdateGauge.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "Evolution/Systems/Cce/Initialize/ConformalFactor.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/Initialize/InverseCubic.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/NumericalAlgorithms/Spectral/SwshTestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {

namespace {
using swsh_boundary_tags_to_generate =
    tmpl::list<Tags::BoundaryValue<Tags::BondiJ>,
               Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>,
               Tags::BoundaryValue<Tags::BondiBeta>,
               Tags::BoundaryValue<Tags::BondiR>>;

using real_cauchy_boundary_tags_to_compute =
    tmpl::list<Tags::CauchyCartesianCoords, Tags::CauchyAngularCoords,
               Tags::InertialRetardedTime>;

using swsh_cauchy_boundary_tags_to_compute =
    tmpl::list<Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD>;

using real_inertial_boundary_tags_to_compute =
    tmpl::list<Tags::PartiallyFlatCartesianCoords,
               Tags::PartiallyFlatAngularCoords>;

using swsh_inertial_boundary_tags_to_compute =
    tmpl::list<Tags::CauchyGaugeC, Tags::CauchyGaugeD>;

using swsh_volume_tags_to_compute = tmpl::list<Tags::BondiJ>;

template <typename Metavariables>
struct dummy_boundary {};

template <typename Metavariables>
struct mock_observer_writer {
  using chare_type = ActionTesting::MockNodeGroupChare;
  using component_being_mocked = observers::ObserverWriter<Metavariables>;
  using replace_these_simple_actions = tmpl::list<>;
  using with_these_simple_actions = tmpl::list<>;

  using simple_tags = tmpl::list<observers::Tags::H5FileLock>;

  using const_global_cache_tags = tmpl::list<>;

  using metavariables = Metavariables;
  using array_index = size_t;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
};

template <typename Metavariables>
struct mock_characteristic_evolution {
  using simple_tags = db::AddSimpleTags<
      ::Tags::Variables<tmpl::append<real_cauchy_boundary_tags_to_compute,
                                     real_inertial_boundary_tags_to_compute>>,
      ::Tags::Variables<tmpl::append<swsh_boundary_tags_to_generate,
                                     swsh_cauchy_boundary_tags_to_compute,
                                     swsh_inertial_boundary_tags_to_compute>>,
      ::Tags::Variables<swsh_volume_tags_to_compute>, ::Tags::TimeStepId>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tags =
      tmpl::list<Tags::InitializeJ<metavariables::evolve_ccm>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<
              Actions::InitializeFirstHypersurface<
                  metavariables::evolve_ccm, dummy_boundary<Metavariables>>,
              ::Actions::MutateApply<GaugeUpdateAngularFromCartesian<
                  Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>,
              ::Actions::MutateApply<GaugeUpdateJacobianFromCoordinates<
                  Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
                  Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>,
              std::conditional_t<
                  Metavariables::evolve_ccm,
                  tmpl::list<
                      ::Actions::MutateApply<GaugeUpdateAngularFromCartesian<
                          Tags::PartiallyFlatAngularCoords,
                          Tags::PartiallyFlatCartesianCoords>>,
                      ::Actions::MutateApply<GaugeUpdateJacobianFromCoordinates<
                          Tags::CauchyGaugeC, Tags::CauchyGaugeD,
                          Tags::PartiallyFlatAngularCoords,
                          Tags::PartiallyFlatCartesianCoords>>>,
                  tmpl::list<>>>>>;
};

template <bool EvolveCcm>
struct metavariables {
  using component_list =
      tmpl::list<mock_observer_writer<metavariables>,
                 mock_characteristic_evolution<metavariables>>;

  static constexpr bool evolve_ccm = EvolveCcm;
};

template <bool EvolveCcm>
void test_InitializeFirstHypersurface() {
  register_derived_classes_with_charm<InitializeJ::InitializeJ<EvolveCcm>>();

  MAKE_GENERATOR(gen);
  // limited l_max distribution because test depends on an analytic
  // basis function with factorials.
  UniformCustomDistribution<size_t> sdist{7, 10};
  const size_t l_max = sdist(gen);
  const size_t number_of_radial_points = sdist(gen);
  UniformCustomDistribution<double> coefficient_distribution{-2.0, 2.0};
  CAPTURE(l_max);

  using component = mock_characteristic_evolution<metavariables<EvolveCcm>>;
  ActionTesting::MockRuntimeSystem<metavariables<EvolveCcm>> runner{
      {std::make_unique<InitializeJ::InverseCubic<EvolveCcm>>(), l_max,
       number_of_radial_points}};

  Variables<tmpl::append<real_cauchy_boundary_tags_to_compute,
                         real_inertial_boundary_tags_to_compute>>
      real_variables{Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  Variables<tmpl::append<swsh_boundary_tags_to_generate,
                         swsh_cauchy_boundary_tags_to_compute,
                         swsh_inertial_boundary_tags_to_compute>>
      swsh_variables{Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  Variables<swsh_volume_tags_to_compute> swsh_volume_variables{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
      number_of_radial_points};
  TimeStepId time_step_id{true, 0, Time{Slab{1.0, 2.0}, {0, 1}}};

  tmpl::for_each<swsh_boundary_tags_to_generate>([&swsh_variables, &gen,
                                                  &coefficient_distribution,
                                                  &l_max](auto tag_v) {
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

  ActionTesting::emplace_component_and_initialize<
      mock_observer_writer<metavariables<EvolveCcm>>>(&runner, 0,
                                                      {Parallel::NodeLock{}});
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {real_variables, swsh_variables, swsh_volume_variables, time_step_id});

  auto expected_box =
      db::create<tmpl::push_back<typename component::simple_tags, Tags::LMax,
                                 Tags::NumberOfRadialPoints>>(
          std::move(real_variables), std::move(swsh_variables),
          std::move(swsh_volume_variables), time_step_id, l_max,
          number_of_radial_points);

  runner.set_phase(Parallel::Phase::Testing);
  // apply the `InitializeFirstHypersurface` action
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  if constexpr (EvolveCcm) {
    ActionTesting::next_action<component>(make_not_null(&runner), 0);
    ActionTesting::next_action<component>(make_not_null(&runner), 0);
  }

  // apply the corresponding mutators to the `expected_box`
  auto node_lock = Parallel::NodeLock{};
  db::mutate_apply<
      typename Cce::InitializeJ::InitializeJ<EvolveCcm>::mutate_tags,
      typename Cce::InitializeJ::InitializeJ<EvolveCcm>::argument_tags>(
      Cce::InitializeJ::InverseCubic<EvolveCcm>{}, make_not_null(&expected_box),
      make_not_null(&node_lock));
  db::mutate_apply<GaugeUpdateAngularFromCartesian<
      Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
      make_not_null(&expected_box));
  db::mutate_apply<GaugeUpdateJacobianFromCoordinates<
      Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
      Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
      make_not_null(&expected_box));
  if constexpr (EvolveCcm) {
    db::mutate_apply<GaugeUpdateAngularFromCartesian<
        Tags::PartiallyFlatAngularCoords, Tags::PartiallyFlatCartesianCoords>>(
        make_not_null(&expected_box));
    db::mutate_apply<GaugeUpdateJacobianFromCoordinates<
        Tags::CauchyGaugeC, Tags::CauchyGaugeD,
        Tags::PartiallyFlatAngularCoords, Tags::PartiallyFlatCartesianCoords>>(
        make_not_null(&expected_box));
  }
  db::mutate_apply<InitializeScriPlusValue<Tags::InertialRetardedTime>>(
      make_not_null(&expected_box), 1.0);

  tmpl::for_each<tmpl::append<real_cauchy_boundary_tags_to_compute,
                              swsh_cauchy_boundary_tags_to_compute,
                              swsh_volume_tags_to_compute>>(
      [&runner, &expected_box](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        CAPTURE(db::tag_name<tag>());
        const auto& test_lhs =
            ActionTesting::get_databox_tag<component, tag>(runner, 0);
        const auto& test_rhs = db::get<tag>(expected_box);
        CHECK_ITERABLE_APPROX(test_lhs, test_rhs);
      });

  if constexpr (EvolveCcm) {
    tmpl::for_each<tmpl::append<real_inertial_boundary_tags_to_compute,
                                swsh_inertial_boundary_tags_to_compute>>(
        [&runner, &expected_box](auto tag_v) {
          using tag = typename decltype(tag_v)::type;
          CAPTURE(db::tag_name<tag>());
          const auto& test_lhs =
              ActionTesting::get_databox_tag<component, tag>(runner, 0);
          const auto& test_rhs = db::get<tag>(expected_box);
          CHECK_ITERABLE_APPROX(test_lhs, test_rhs);
        });
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Cce.Actions.InitializeFirstHypersurface",
    "[Unit][Cce]") {
  // Evolve inertial coordinates
  test_InitializeFirstHypersurface<true>();
  // Do not evolve inertial coordinates
  test_InitializeFirstHypersurface<false>();
}
}  // namespace Cce
