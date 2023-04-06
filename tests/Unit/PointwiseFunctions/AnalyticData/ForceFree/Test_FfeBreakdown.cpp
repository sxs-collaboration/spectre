// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <utility>

#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/ForceFree/FfeBreakdown.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

using InitialData = evolution::initial_data::InitialData;
using FfeBreakdown = ForceFree::AnalyticData::FfeBreakdown;

struct FfeBreakdownProxy : FfeBreakdown {
  using FfeBreakdown::FfeBreakdown;
  using variables_tags =
      tmpl::list<ForceFree::Tags::TildeE, ForceFree::Tags::TildeB,
                 ForceFree::Tags::TildePsi, ForceFree::Tags::TildePhi,
                 ForceFree::Tags::TildeQ>;

  tuples::tagged_tuple_from_typelist<variables_tags> return_variables(
      const tnsr::I<DataVector, 3>& x) const {
    return this->variables(x, variables_tags{});
  }
};

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.ForceFree.FfeBreakdown",
    "[Unit][PointwiseFunctions]") {
  // test creation
  const auto solution = TestHelpers::test_creation<FfeBreakdown>("");
  CHECK(solution == FfeBreakdown());
  // test serialize
  test_serialization(solution);
  // test move
  test_move_semantics(FfeBreakdown{}, FfeBreakdown{});
  // test derived
  register_classes_with_charm<FfeBreakdown>();
  const std::unique_ptr<InitialData> initial_data_ptr =
      std::make_unique<FfeBreakdown>();
  const std::unique_ptr<InitialData> deserialized_initial_data_ptr =
      serialize_and_deserialize(initial_data_ptr)->get_clone();
  CHECK(dynamic_cast<FfeBreakdown*>(deserialized_initial_data_ptr.get()) !=
        nullptr);

  // test solution
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticData/ForceFree"};
  const DataVector used_for_size{10};
  pypp::check_with_random_values<1>(
      &FfeBreakdownProxy::return_variables, FfeBreakdownProxy(), "FfeBreakdown",
      {"TildeE", "TildeB", "TildePsi", "TildePhi", "TildeQ"}, {{{-10.0, 10.0}}},
      {}, used_for_size);
}
}  // namespace
