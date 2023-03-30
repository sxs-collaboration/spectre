// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <random>
#include <utility>

#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/AnalyticSolutions/ForceFree/AlfvenWave.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

using InitialData = evolution::initial_data::InitialData;
using AlfvenWave = ForceFree::Solutions::AlfvenWave;

struct AlfvenWaveProxy : AlfvenWave {
  using AlfvenWave::AlfvenWave;
  using variables_tags =
      tmpl::list<ForceFree::Tags::TildeE, ForceFree::Tags::TildeB,
                 ForceFree::Tags::TildePsi, ForceFree::Tags::TildePhi,
                 ForceFree::Tags::TildeQ>;

  tuples::tagged_tuple_from_typelist<variables_tags> return_variables(
      const tnsr::I<DataVector, 3>& x, double t) const {
    return this->variables(x, t, variables_tags{});
  }
};

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.ForceFree.AlfvenWave",
    "[Unit][PointwiseFunctions]") {
  // test creation
  const auto solution =
      TestHelpers::test_creation<AlfvenWave>("WaveSpeed: 0.5");
  CHECK(solution == AlfvenWave(0.5));
  CHECK(solution != AlfvenWave(-0.5));

  // test serialize
  test_serialization(solution);

  // test move
  test_move_semantics(AlfvenWave{0.5}, AlfvenWave{0.5});

  // test derived
  register_classes_with_charm<AlfvenWave>();
  const std::unique_ptr<InitialData> base_ptr =
      std::make_unique<AlfvenWave>(0.3);
  const std::unique_ptr<InitialData> deserialized_base_ptr =
      serialize_and_deserialize(base_ptr)->get_clone();
  CHECK(dynamic_cast<const AlfvenWave&>(*deserialized_base_ptr.get()) ==
        dynamic_cast<const AlfvenWave&>(*base_ptr.get()));

  // test solution for a random wave speed
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist_mu(-0.99, 0.99);
  const double wave_speed{make_with_random_values<double>(
      make_not_null(&gen), make_not_null(&dist_mu))};

  AlfvenWaveProxy alfven_wave(wave_speed);
  const auto member_variables = std::make_tuple(wave_speed);

  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/ForceFree"};
  const DataVector used_for_size{10};

  pypp::check_with_random_values<1>(
      &AlfvenWaveProxy::return_variables, alfven_wave, "AlfvenWave",
      {"TildeE", "TildeB", "TildePsi", "TildePhi", "TildeQ"}, {{{-0.2, 0.2}}},
      member_variables, used_for_size);
}
}  // namespace
