// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <utility>

#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/ForceFree/FastWave.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

using InitialData = evolution::initial_data::InitialData;
using FastWave = ForceFree::Solutions::FastWave;

struct FastWaveProxy : FastWave {
  using FastWave::FastWave;
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
    "Unit.PointwiseFunctions.AnalyticSolutions.ForceFree.FastWave",
    "[Unit][PointwiseFunctions]") {
  // test creation
  const auto solution = TestHelpers::test_creation<FastWave>("");
  CHECK(solution == FastWave());
  // test serialize
  test_serialization(solution);
  // test move
  test_move_semantics(FastWave{}, FastWave{});
  // test derived
  register_classes_with_charm<FastWave>();
  const std::unique_ptr<InitialData> initial_data_ptr =
      std::make_unique<FastWave>();
  const std::unique_ptr<InitialData> deserialized_initial_data_ptr =
      serialize_and_deserialize(initial_data_ptr)->get_clone();
  CHECK(dynamic_cast<FastWave*>(deserialized_initial_data_ptr.get()) !=
        nullptr);

  // test solution
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/ForceFree"};
  const DataVector used_for_size{10};
  pypp::check_with_random_values<1>(
      &FastWaveProxy::return_variables, FastWaveProxy(), "FastWave",
      {"TildeE", "TildeB", "TildePsi", "TildePhi", "TildeQ"}, {{{-10.0, 10.0}}},
      {}, used_for_size);
}
}  // namespace
