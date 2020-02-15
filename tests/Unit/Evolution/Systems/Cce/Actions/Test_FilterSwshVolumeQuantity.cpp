// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/Actions/FilterSwshVolumeQuantity.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/VectorAlgebra.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace Cce {

namespace {
template <typename Metavariables>
struct mock_characteristic_evolution {
  using component_being_mocked = CharacteristicEvolution<Metavariables>;
  using replace_these_simple_actions = tmpl::list<>;
  using with_these_simple_actions = tmpl::list<>;

  using simple_tags =
      db::AddSimpleTags<Spectral::Swsh::Tags::LMax,
                        ::Tags::Variables<tmpl::list<Tags::BondiJ>>>;
  using compute_tags = db::AddComputeTags<>;

  using initialize_action_list =
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags, compute_tags>>;
  using initialization_tags =
      Parallel::get_initialization_tags<initialize_action_list>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             initialize_action_list>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Evolve,
          tmpl::list<Actions::FilterSwshVolumeQuantity<Tags::BondiJ>>>>;
  using const_global_cache_tags =
      Parallel::get_const_global_cache_tags_from_actions<
          phase_dependent_action_list>;
};

struct metavariables {
  using component_list =
      tmpl::list<mock_characteristic_evolution<metavariables>>;
  enum class Phase { Initialization, Evolve, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.Actions.FilterSwshVolumeQuantity",
                  "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  // limited l_max distribution because test depends on an analytic
  // basis function with factorials.
  UniformCustomDistribution<size_t> sdist{7, 10};
  const size_t l_max = sdist(gen);
  UniformCustomDistribution<double> coefficient_distribution{-2.0, 2.0};
  const size_t number_of_radial_points = 2;
  CAPTURE(l_max);
  // Generate data uniform in r with all angular modes
  SpinWeighted<ComplexModalVector, 2> generated_modes;
  generated_modes.data() = make_with_random_values<ComplexModalVector>(
      make_not_null(&gen), make_not_null(&coefficient_distribution),
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max));
  for (const auto& mode : Spectral::Swsh::cached_coefficients_metadata(l_max)) {
    if (mode.l < 2) {
      generated_modes.data()[mode.transform_of_real_part_offset] = 0.0;
      generated_modes.data()[mode.transform_of_imag_part_offset] = 0.0;
    }
    if (mode.m == 0) {
      generated_modes.data()[mode.transform_of_real_part_offset] =
          real(generated_modes.data()[mode.transform_of_real_part_offset]);
      generated_modes.data()[mode.transform_of_imag_part_offset] =
          real(generated_modes.data()[mode.transform_of_imag_part_offset]);
    }
  }
  const auto pre_filter_angular_data =
      Spectral::Swsh::inverse_swsh_transform(l_max, 1, generated_modes);
  Scalar<SpinWeighted<ComplexDataVector, 2>> to_filter;
  get(to_filter) = SpinWeighted<ComplexDataVector, 2>{create_vector_of_n_copies(
      pre_filter_angular_data.data(), number_of_radial_points)};
  Variables<tmpl::list<Tags::BondiJ>> component_variables{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
      number_of_radial_points};
  get<Tags::BondiJ>(component_variables) = to_filter;

  using component = mock_characteristic_evolution<metavariables>;
  ActionTesting::MockRuntimeSystem<metavariables> runner{
      tuples::tagged_tuple_from_typelist<
          Parallel::get_const_global_cache_tags<metavariables>>{l_max - 2, 36.0,
                                                                32_st}};

  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0, {l_max, std::move(component_variables)});
  runner.set_phase(metavariables::Phase::Evolve);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);

  Spectral::Swsh::filter_swsh_volume_quantity(make_not_null(&get(to_filter)),
                                              l_max, l_max - 2, 36.0, 32);

  const auto& filtered_from_component =
      ActionTesting::get_databox_tag<component, Tags::BondiJ>(runner, 0);

  CHECK_ITERABLE_APPROX(get(filtered_from_component), get(to_filter));
}
}  // namespace Cce
