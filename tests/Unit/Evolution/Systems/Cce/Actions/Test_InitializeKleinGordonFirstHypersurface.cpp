// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/Cce/Actions/InitializeKleinGordonFirstHypersurface.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTestHelpers.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshFiltering.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce {
namespace {
template <typename Metavariables>
struct mock_kg_characteristic_evolution {
  using simple_tags = db::AddSimpleTags<
      ::Tags::Variables<tmpl::list<Tags::BoundaryValue<Tags::KleinGordonPsi>>>,
      ::Tags::Variables<tmpl::list<Tags::KleinGordonPsi>>, ::Tags::TimeStepId>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Evolve,
          tmpl::list<Actions::InitializeKleinGordonFirstHypersurface>>>;
};

struct metavariables {
  using component_list =
      tmpl::list<mock_kg_characteristic_evolution<metavariables>>;
};

// This function tests that the action
// `Actions::InitializeKleinGordonFirstHypersurface` can be correctly invoked.
//
// The action is invoked by a mock run time system, then the function checks
// that the constructed hypersurface data (`computed_psi`) agree with what we
// expect.
template <typename Generator>
void test_klein_gordon_first_hypersurface(const gsl::not_null<Generator*> gen) {
  UniformCustomDistribution<size_t> sdist{7, 10};
  const size_t l_max = sdist(*gen);
  const size_t number_of_radial_points = sdist(*gen);
  UniformCustomDistribution<double> coefficient_distribution{-2.0, 2.0};

  using component = mock_kg_characteristic_evolution<metavariables>;

  // generate boundary data for the Klein-Gordon variable
  // (`kg_boundary_variable`)
  SpinWeighted<ComplexModalVector, 0> generated_kg_boundary_modes{
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
  SpinWeighted<ComplexDataVector, 0> generated_kg_boundary_data{
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
  Spectral::Swsh::TestHelpers::generate_swsh_modes<0>(
      make_not_null(&generated_kg_boundary_modes.data()), gen,
      make_not_null(&coefficient_distribution), 1, l_max);
  Spectral::Swsh::inverse_swsh_transform(
      l_max, 1, make_not_null(&generated_kg_boundary_data),
      generated_kg_boundary_modes);
  // aggressive filter to make the uniformly generated random modes
  // somewhat reasonable
  Spectral::Swsh::filter_swsh_boundary_quantity(
      make_not_null(&generated_kg_boundary_data), l_max, l_max / 2);

  Variables<tmpl::list<Tags::BoundaryValue<Tags::KleinGordonPsi>>>
      kg_boundary_variable{real(generated_kg_boundary_data.data())};

  const size_t boundary_size =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  Variables<tmpl::list<Tags::KleinGordonPsi>> kg_variable_to_compute{
      boundary_size * number_of_radial_points};

  // required by the action
  TimeStepId time_step_id{true, 0, Time{Slab{1.0, 2.0}, {0, 1}}};

  // tests start here
  ActionTesting::MockRuntimeSystem<metavariables> runner{
      {l_max, number_of_radial_points}};
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0, {kg_boundary_variable, kg_variable_to_compute, time_step_id});

  runner.set_phase(Parallel::Phase::Evolve);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  auto computed_psi =
      ActionTesting::get_databox_tag<component, Tags::KleinGordonPsi>(runner,
                                                                      0);

  const DataVector one_minus_y_collocation =
      1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>(
                number_of_radial_points);

  // compare with the expected values
  for (size_t i = 0; i < number_of_radial_points; i++) {
    ComplexDataVector angular_view_of_computed_psi{
        get(computed_psi).data().data() + boundary_size * i, boundary_size};

    auto expected_psi = real(generated_kg_boundary_data.data()) *
                        one_minus_y_collocation[i] / 2.;
    CHECK(angular_view_of_computed_psi == expected_psi);
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Cce.Actions.InitializeKleinGordonFirstHypersurface",
    "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  test_klein_gordon_first_hypersurface(make_not_null(&gen));
}
}  // namespace Cce
