// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/TimeDerivative.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonChare.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace CurvedScalarWave::Worldtube {
namespace {

template <size_t Dim>
void time_derivative_wrapper(
    const gsl::not_null<Scalar<double>*> dt_psi0,
    const gsl::not_null<Scalar<double>*> dt2_psi0,
    const Scalar<double>& psi_monopole,
    const tnsr::i<double, Dim, Frame::Grid>& psi_dipole,
    const tnsr::ii<double, Dim, Frame::Grid>& psi_quadrupole,
    const tnsr::i<double, Dim, Frame::Grid>& dt_psi_dipole,
    const tnsr::AA<double, Dim, Frame::Grid>& inverse_spacetime_metric,
    const tnsr::A<double, Dim, Frame::Grid>& trace_christoffel,
    const tnsr::i<double, Dim, Frame::Grid>& evolved_vars_tnsr,
    const Scalar<double> wt_radius) {
  // we pack the evolved_vars into a tensor so we can test it on the python
  // side. the zeroth element of the tensor is Psi0 and the first element is
  // dtPsi0.
  Variables<tmpl::list<Tags::Psi0, Tags::dtPsi0>> evolved_vars(1);
  get(get<Tags::Psi0>(evolved_vars))[0] = get<0>(evolved_vars_tnsr);
  get(get<Tags::dtPsi0>(evolved_vars))[0] = get<1>(evolved_vars_tnsr);
  Variables<tmpl::list<::Tags::dt<Tags::Psi0>, ::Tags::dt<Tags::dtPsi0>>>
      dt_evolved_vars(1);
  // only the radius is used
  const ::ExcisionSphere<3> excision_sphere(
      get(wt_radius), tnsr::I<double, Dim, Frame::Grid>{{0., 0., 0.}}, {});
  TimeDerivativeMutator::apply(make_not_null(&dt_evolved_vars), evolved_vars,
                               psi_monopole, psi_dipole, psi_quadrupole,
                               dt_psi_dipole, inverse_spacetime_metric,
                               trace_christoffel, excision_sphere);
  get(*dt_psi0) = get(get<::Tags::dt<Tags::Psi0>>(dt_evolved_vars))[0];
  get(*dt2_psi0) = get(get<::Tags::dt<Tags::dtPsi0>>(dt_evolved_vars))[0];
}

template <size_t Dim>
void test_mutator() {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions"};
  MAKE_GENERATOR(generator);
  pypp::check_with_random_values<1>(&time_derivative_wrapper<Dim>,
                                    "TimeDerivatives", {"dt_psi0", "dt2_psi0"},
                                    {{{0.1, 1.}}}, 1, 1.e-12);
}

template <typename Metavariables>
struct MockWorldtubeSingleton {
  using metavariables = Metavariables;
  static constexpr size_t Dim = metavariables::volume_dim;
  using chare_type = ActionTesting::MockSingletonChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
              db::AddSimpleTags<
                  TimeDerivativeMutator::dt_variables_tag,
                  TimeDerivativeMutator::variables_tag,
                  Stf::Tags::StfTensor<Tags::PsiWorldtube, 0, Dim, Frame::Grid>,
                  Stf::Tags::StfTensor<Tags::PsiWorldtube, 1, Dim, Frame::Grid>,
                  Stf::Tags::StfTensor<Tags::PsiWorldtube, 2, Dim, Frame::Grid>,
                  Stf::Tags::StfTensor<::Tags::dt<Tags::PsiWorldtube>, 1, Dim,
                                       Frame::Grid>,
                  gr::Tags::InverseSpacetimeMetric<double, Dim, Frame::Grid>,
                  gr::Tags::TraceSpacetimeChristoffelSecondKind<double, Dim,
                                                                Frame::Grid>,
                  Tags::ExpansionOrder, Tags::ExcisionSphere<Dim>>,
              db::AddComputeTags<>>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing,
                             tmpl::list<Actions::ComputeTimeDerivative>>>;
  using component_being_mocked = WorldtubeSingleton<Metavariables>;
};

template <size_t Dim>
struct MockMetavariables {
  static constexpr size_t volume_dim = Dim;
  using component_list =
      tmpl::list<MockWorldtubeSingleton<MockMetavariables<Dim>>>;
  using const_global_cache_tags = tmpl::list<>;
};

template <size_t Dim>
void test_action() {
  using dt_variables_tag = TimeDerivativeMutator::dt_variables_tag;
  using metavars = MockMetavariables<Dim>;
  using worldtube_chare = MockWorldtubeSingleton<metavars>;
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> wt_radius_dist(0.1, 50.);
  std::uniform_real_distribution<> tensor_dist(-1., 10.);
  const double wt_radius = wt_radius_dist(generator);

  const ::ExcisionSphere<3> excision_sphere(
      wt_radius, tnsr::I<double, Dim, Frame::Grid>{{0., 0., 0.}}, {});

  const auto psi_monopole = make_with_random_values<Scalar<double>>(
      make_not_null(&generator), make_not_null(&tensor_dist), 1);
  const auto psi_dipole =
      make_with_random_values<tnsr::i<double, Dim, Frame::Grid>>(
          make_not_null(&generator), make_not_null(&tensor_dist), 1);
  const auto psi_quadrupole =
      make_with_random_values<tnsr::ii<double, Dim, Frame::Grid>>(
          make_not_null(&generator), make_not_null(&tensor_dist), 1);
  const auto dt_psi_dipole =
      make_with_random_values<tnsr::i<double, Dim, Frame::Grid>>(
          make_not_null(&generator), make_not_null(&tensor_dist), 1);
  const auto inverse_spacetime_metric =
      make_with_random_values<tnsr::AA<double, Dim, Frame::Grid>>(
          make_not_null(&generator), make_not_null(&tensor_dist), 1);
  const auto trace_christoffel =
      make_with_random_values<tnsr::A<double, Dim, Frame::Grid>>(
          make_not_null(&generator), make_not_null(&tensor_dist), 1);
  const auto evolved_vars =
      make_with_random_values<Variables<tmpl::list<Tags::Psi0, Tags::dtPsi0>>>(
          make_not_null(&generator), make_not_null(&tensor_dist),
          DataVector(1));
  const Variables<tmpl::list<::Tags::dt<Tags::Psi0>, ::Tags::dt<Tags::dtPsi0>>>
      dt_evolved_vars_initial(1, -999.);
  for (size_t expansion_order = 0; expansion_order <= 2; ++expansion_order) {
    CAPTURE(expansion_order);
    ActionTesting::MockRuntimeSystem<metavars> runner{{}};
    ActionTesting::emplace_singleton_component_and_initialize<worldtube_chare>(
        &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0},
        {dt_evolved_vars_initial, evolved_vars, psi_monopole, psi_dipole,
         psi_quadrupole, dt_psi_dipole, inverse_spacetime_metric,
         trace_christoffel, expansion_order, excision_sphere});
    const auto& dt_evolved_vars =
        ActionTesting::get_databox_tag<worldtube_chare, dt_variables_tag>(
            runner, 0);
    ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
    ActionTesting::next_action<worldtube_chare>(make_not_null(&runner), 0);
    if (expansion_order < 2) {
      CHECK(dt_evolved_vars == dt_evolved_vars_initial);
    } else {
      dt_variables_tag::type dt_evolved_vars_expected(
          1, std::numeric_limits<double>::signaling_NaN());
      TimeDerivativeMutator::apply(
          make_not_null(&dt_evolved_vars_expected), evolved_vars, psi_monopole,
          psi_dipole, psi_quadrupole, dt_psi_dipole, inverse_spacetime_metric,
          trace_christoffel, excision_sphere);
      CHECK_VARIABLES_APPROX(dt_evolved_vars, dt_evolved_vars_expected);
    }
  }
}

SPECTRE_TEST_CASE("Unit.CurvedScalarWave.Worldtube.TimeDerivative", "[Unit]") {
  MAKE_GENERATOR(generator);
  static constexpr size_t Dim = 3;
  test_mutator<Dim>();
  test_action<Dim>();
}

}  // namespace
}  // namespace CurvedScalarWave::Worldtube
