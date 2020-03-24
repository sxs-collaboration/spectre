// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/variant/get.hpp> // IWYU pragma: keep
#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/RadiationTransport/M1Grey/M1Closure.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/RadiationTransport/Tags.hpp"  // IWYU pragma: keep
#include "Framework/ActionTesting.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox

namespace {

template <typename Metavariables>
struct mock_component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using Closure = typename RadiationTransport::M1Grey::ComputeM1Closure<
      typename Metavariables::neutrino_species>;
  using simple_tags = db::AddSimpleTags<tmpl::flatten<
      tmpl::list<typename Closure::return_tags, typename Closure::argument_tags,
                 domain::Tags::Coordinates<3, Frame::Inertial>>>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<Actions::MutateApply<
              typename RadiationTransport::M1Grey::ComputeM1Closure<
                  typename metavariables::neutrino_species>>>>>;
};

struct Metavariables {
  using component_list = tmpl::list<mock_component<Metavariables>>;
  using neutrino_species = tmpl::list<neutrinos::ElectronNeutrinos<1>,
                                      neutrinos::HeavyLeptonNeutrinos<0>>;
  enum class Phase { Initialization, Testing, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.RadiationTransport.M1Grey.Actions", "[Unit][M1Grey]") {
  using component = mock_component<Metavariables>;

  const DataVector x{-2.0, -1.0, 0.0, 1.0, 2.0};
  const DataVector y{-2.0, -1.0, 0.0, 1.0, 2.0};
  const DataVector z{-2.0, -1.0, 0.0, 1.0, 2.0};

  // Closure output variables (initialization only,
  // use the same value for all species)
  const DataVector xi{0.5, 0.5, 0.5, 0.5, 0.5};
  const auto tildeP = make_with_value<tnsr::II<DataVector, 3>>(x, 0.5);
  const DataVector J{0.5, 0.5, 0.5, 0.5, 0.5};
  const DataVector Hn{0.5, 0.5, 0.5, 0.5, 0.5};
  const DataVector Hx{0.5, 0.5, 0.5, 0.5, 0.5};
  const DataVector Hy{0.5, 0.5, 0.5, 0.5, 0.5};
  const DataVector Hz{0.5, 0.5, 0.5, 0.5, 0.5};
  // Closure input variables
  // First neutrino species (set to optically thick)
  const DataVector E0{1.0, 1.0, 1.0, 1.0, 1.0};
  const DataVector Sx0{0.0, 0.0, 0.0, 0.0, 0.0};
  const DataVector Sy0{0.0, 0.0, 0.0, 0.0, 0.0};
  const DataVector Sz0{0.0, 0.0, 0.0, 0.0, 0.0};
  // Second neutrino species (set to optically thin)
  const DataVector E1{1.0, 1.0, 1.0, 1.0, 1.0};
  const DataVector Sx1{0.0, 0.0, 0.0, 1.0, 1.0};
  const DataVector Sy1{0.0, 0.0, 1.0, 0.0, 0.0};
  const DataVector Sz1{1.0, 1.0, 0.0, 0.0, 0.0};

  // Fluid and metric variables
  const DataVector vx{0.0, 0.0, 0.0, 0.0, 0.0};
  const DataVector vy{0.0, 0.0, 0.0, 0.0, 0.0};
  const DataVector vz{0.0, 0.0, 0.0, 0.0, 0.0};
  const DataVector W{1.0, 1.0, 1.0, 1.0, 1.0};
  auto metric = make_with_value<tnsr::ii<DataVector, 3>>(x, 0.0);
  auto inverse_metric = make_with_value<tnsr::II<DataVector, 3>>(x, 0.0);
  get<0, 0>(metric) = 1.;
  get<1, 1>(metric) = 1.;
  get<2, 2>(metric) = 1.;
  get<0, 0>(inverse_metric) = 1.;
  get<1, 1>(inverse_metric) = 1.;
  get<2, 2>(inverse_metric) = 1.;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {Scalar<DataVector>{xi}, Scalar<DataVector>{xi}, tildeP, tildeP,
       Scalar<DataVector>{J}, Scalar<DataVector>{J}, Scalar<DataVector>{Hn},
       Scalar<DataVector>{Hn},
       tnsr::i<DataVector, 3, Frame::Inertial>{{{Hx, Hy, Hz}}},
       tnsr::i<DataVector, 3, Frame::Inertial>{{{Hx, Hy, Hz}}},
       Scalar<DataVector>{E0}, Scalar<DataVector>{E1},
       tnsr::i<DataVector, 3, Frame::Inertial>{{{Sx0, Sy0, Sz0}}},
       tnsr::i<DataVector, 3, Frame::Inertial>{{{Sx1, Sy1, Sz1}}},
       tnsr::I<DataVector, 3, Frame::Inertial>{{{vx, vy, vz}}},
       Scalar<DataVector>{W}, metric, inverse_metric,
       tnsr::I<DataVector, 3, Frame::Inertial>{{{x, y, z}}}});
  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);

  runner.next_action<component>(0);

  // Check that first species return xi=0 (optically thick)
  const DataVector expected_xi0{0.0, 0.0, 0.0, 0.0, 0.0};
  CHECK_ITERABLE_APPROX(
      (ActionTesting::get_databox_tag<
           component, RadiationTransport::M1Grey::Tags::ClosureFactor<
                          neutrinos::ElectronNeutrinos<1>>>(runner, 0)
           .get()),
      expected_xi0);
  // Check that second species return xi=1 (optically thin)
  const DataVector expected_xi1{1.0, 1.0, 1.0, 1.0, 1.0};
  CHECK_ITERABLE_APPROX(
      (ActionTesting::get_databox_tag<
           component, RadiationTransport::M1Grey::Tags::ClosureFactor<
                          neutrinos::HeavyLeptonNeutrinos<0>>>(runner, 0)
           .get()),
      expected_xi1);
}
