// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SelfForce.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/UpdateAcceleration.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "Utilities/TMPL.hpp"

namespace CurvedScalarWave::Worldtube {
namespace {

SPECTRE_TEST_CASE("Unit.Evolution.Systems.CSW.Worldtube.UpdateAcceleration",
                  "[Unit][Evolution]") {
  static constexpr size_t Dim = 3;
  using variables_tag = ::Tags::Variables<
      tmpl::list<Tags::EvolvedPosition<Dim>, Tags::EvolvedVelocity<Dim>>>;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1., 1.);
  const DataVector used_for_size(1);
  auto dt_evolved_vars = make_with_random_values<dt_variables_tag::type>(
      make_not_null(&gen), dist, used_for_size);
  const auto geodesic_acc = make_with_random_values<tnsr::I<double, Dim>>(
      make_not_null(&gen), dist, 1);
  const auto vel = make_with_random_values<tnsr::I<double, Dim>>(
      make_not_null(&gen), dist, 1);
  const auto pos = make_with_random_values<tnsr::I<double, Dim>>(
      make_not_null(&gen), dist, 1);
  const std::array<tnsr::I<double, Dim>, 2> pos_vel{{pos, vel}};
  const auto metric = make_with_random_values<tnsr::aa<double, Dim>>(
      make_not_null(&gen), dist, 1);
  const auto inverse_metric = make_with_random_values<tnsr::AA<double, Dim>>(
      make_not_null(&gen), dist, 1);
  const auto dilation =
      make_with_random_values<Scalar<double>>(make_not_null(&gen), dist, 1);
  const Tags::BackgroundQuantities<Dim>::type background_quantities{
      metric, inverse_metric, dilation};
  const auto dt_psi_monopole =
      make_with_random_values<Scalar<double>>(make_not_null(&gen), dist, 1);
  const auto psi_dipole = make_with_random_values<tnsr::i<double, Dim>>(
      make_not_null(&gen), dist, 1);
  const size_t max_iterations = 0;
  auto box = db::create<db::AddSimpleTags<
      dt_variables_tag, Tags::ParticlePositionVelocity<Dim>,
      Tags::BackgroundQuantities<Dim>, Tags::GeodesicAcceleration<Dim>,
      Stf::Tags::StfTensor<::Tags::dt<Tags::PsiWorldtube>, 0, Dim,
                           Frame::Inertial>,
      Stf::Tags::StfTensor<Tags::PsiWorldtube, 1, Dim, Frame::Inertial>,
      Tags::Charge, Tags::Mass, Tags::MaxIterations>>(
      std::move(dt_evolved_vars), pos_vel, background_quantities, geodesic_acc,
      dt_psi_monopole, psi_dipole, 1., std::make_optional(1.), max_iterations);

  db::mutate_apply<UpdateAcceleration>(make_not_null(&box));
  const auto& dt_vars = db::get<dt_variables_tag>(box);
  for (size_t i = 0; i < Dim; ++i) {
    CHECK(get<::Tags::dt<Tags::EvolvedPosition<Dim>>>(dt_vars).get(i)[0] ==
          vel.get(i));
    CHECK(get<::Tags::dt<Tags::EvolvedVelocity<Dim>>>(dt_vars).get(i)[0] ==
          geodesic_acc.get(i));
  }

  db::mutate<Tags::MaxIterations>(
      [](const gsl::not_null<size_t*> max_iterations_arg) {
        *max_iterations_arg = 1;
      },
      make_not_null(&box));

  db::mutate_apply<UpdateAcceleration>(make_not_null(&box));
  const auto self_force_acc = self_force_acceleration(
      dt_psi_monopole, psi_dipole, vel, 1., 1., inverse_metric, dilation);
  for (size_t i = 0; i < Dim; ++i) {
    CHECK(get<::Tags::dt<Tags::EvolvedPosition<Dim>>>(dt_vars).get(i)[0] ==
          vel.get(i));
    CHECK(get<::Tags::dt<Tags::EvolvedVelocity<Dim>>>(dt_vars).get(i)[0] ==
          geodesic_acc.get(i) + self_force_acc.get(i));
  }
}
}  // namespace
}  // namespace CurvedScalarWave::Worldtube
