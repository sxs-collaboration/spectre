// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <limits>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/NonconservativeSystem.hpp"
#include "Evolution/Systems/CurvedScalarWave/BackgroundSpacetime.hpp"
#include "Evolution/Systems/CurvedScalarWave/CalculateGrVars.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Phase.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <size_t Dim, typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using initial_tags =
      tmpl::list<CurvedScalarWave::Tags::BackgroundSpacetime<
                     typename Metavariables::background_spacetime>,
                 domain::Tags::Coordinates<Dim, Frame::Inertial>, ::Tags::Time>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<initial_tags>,
                 CurvedScalarWave::Actions::CalculateGrVars<
                     typename Metavariables::system>>>>;
};

template <size_t Dim, typename System, typename BackgroundSpacetime>
struct Metavariables {
  using background_spacetime = BackgroundSpacetime;
  using component_list = tmpl::list<component<Dim, Metavariables>>;
  using system = System;
  using const_global_cache_tag_list = tmpl::list<>;
};

template <typename BackgroundSpacetime>
void test(const BackgroundSpacetime& background_spacetime,
          const gsl::not_null<std::mt19937*> generator) {
  static constexpr size_t Dim = BackgroundSpacetime::volume_dim;
  using system = CurvedScalarWave::System<Dim>;
  using metavars = Metavariables<Dim, system, BackgroundSpacetime>;
  using comp = component<Dim, metavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{{}};
  const size_t num_points = 42;
  std::uniform_real_distribution dist{-10., 10.};
  const auto random_coords = make_with_random_values<tnsr::I<DataVector, Dim>>(
      generator, make_not_null(&dist), DataVector{num_points});
  const double time = 0.;
  ActionTesting::emplace_component_and_initialize<comp>(
      &runner, 0, {background_spacetime, random_coords, time});
  // invoke CalculateGrVars
  ActionTesting::next_action<comp>(make_not_null(&runner), 0);
  const auto solution_at_coords = background_spacetime.variables(
      random_coords, time, typename system::spacetime_tag_list{});
  // check that each tag corresponds to analytic solution now
  tmpl::for_each<typename system::spacetime_tag_list>(
      [&runner, &solution_at_coords](auto spacetime_tag_v) {
        using spacetime_tag = tmpl::type_from<decltype(spacetime_tag_v)>;
        CHECK(ActionTesting::get_databox_tag<comp, spacetime_tag>(runner, 0) ==
              get<spacetime_tag>(solution_at_coords));
      });
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.CurvedScalarWave.CalculateGrVars",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);
  test(gr::Solutions::Minkowski<1>(), make_not_null(&generator));
  test(gr::Solutions::Minkowski<2>(), make_not_null(&generator));
  test(gr::Solutions::Minkowski<3>(), make_not_null(&generator));
  test(gr::Solutions::KerrSchild(1., {0.5, 0., 0.1}, {0.2, 0.5, -0.7}),
       make_not_null(&generator));
}
}  // namespace
