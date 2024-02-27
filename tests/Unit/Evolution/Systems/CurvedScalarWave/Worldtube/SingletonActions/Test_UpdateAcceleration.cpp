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
  const auto pos = make_with_random_values<tnsr::I<double, 3>>(
      make_not_null(&gen), dist, used_for_size);
  const auto vel = make_with_random_values<tnsr::I<double, 3>>(
      make_not_null(&gen), dist, used_for_size);
  auto dt_evolved_vars = make_with_random_values<dt_variables_tag::type>(
      make_not_null(&gen), dist, used_for_size);
  const auto acceleration =
      make_with_random_values<tnsr::I<double, 3>>(make_not_null(&gen), dist, 1);
  auto box = db::create<
      db::AddSimpleTags<Tags::ParticlePositionVelocity<Dim>, dt_variables_tag,
                        Tags::GeodesicAcceleration<Dim>>>(
      std::array<tnsr::I<double, 3>, 2>{{pos, vel}}, std::move(dt_evolved_vars),
      acceleration);

  db::mutate_apply<UpdateAcceleration>(make_not_null(&box));

  const auto& dt_vars_after_mutate = db::get<dt_variables_tag>(box);
  for (size_t i = 0; i < Dim; ++i) {
    CHECK(get<::Tags::dt<Tags::EvolvedPosition<Dim>>>(dt_vars_after_mutate)
              .get(i)[0] == vel.get(i));
    CHECK(get<::Tags::dt<Tags::EvolvedVelocity<Dim>>>(dt_vars_after_mutate)
              .get(i)[0] == acceleration.get(i));
  }
}
}  // namespace
}  // namespace CurvedScalarWave::Worldtube
