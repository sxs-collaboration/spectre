// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <random>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/AnalyticData/ForceFree/MagnetosphericWald.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

using InitialData = evolution::initial_data::InitialData;
using MagnetosphericWald = ForceFree::AnalyticData::MagnetosphericWald;

struct MagnetosphericWaldProxy : MagnetosphericWald {
  using MagnetosphericWald::MagnetosphericWald;
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
    "Unit.PointwiseFunctions.AnalyticData.ForceFree.MagnetosphericWald",
    "[Unit][PointwiseFunctions]") {
  // test creation
  const auto solution =
      TestHelpers::test_creation<MagnetosphericWald>("Spin: 0.5");
  CHECK(solution == MagnetosphericWald(0.5));
  CHECK(solution != MagnetosphericWald(0.1));

  CHECK_THROWS_WITH(
      []() { const MagnetosphericWald soln(2.0); }(),
      Catch::Matchers::ContainsSubstring("Spin magnitude must be"));

  // test serialize
  test_serialization(solution);

  // test move
  test_move_semantics(MagnetosphericWald{0.5}, MagnetosphericWald{0.5});

  // test derived
  register_classes_with_charm<MagnetosphericWald>();
  const std::unique_ptr<InitialData> base_ptr =
      std::make_unique<MagnetosphericWald>(0.5);
  const std::unique_ptr<InitialData> deserialized_base_ptr =
      serialize_and_deserialize(base_ptr)->get_clone();
  CHECK(dynamic_cast<const MagnetosphericWald&>(*deserialized_base_ptr.get()) ==
        dynamic_cast<const MagnetosphericWald&>(*base_ptr.get()));

  // test solution
  const DataVector used_for_size{10};

  const double bh_spin = 0.5;
  const auto member_variables = std::make_tuple(bh_spin);

  MagnetosphericWaldProxy magnetospheric_wald(bh_spin);

  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticData/ForceFree"};

  // Check for the EM variables
  pypp::check_with_random_values<1>(
      &MagnetosphericWaldProxy::return_variables, magnetospheric_wald,
      "MagnetosphericWald",
      {"TildeE", "TildeB", "TildePsi", "TildePhi", "TildeQ"}, {{{-10.0, 10.0}}},
      member_variables, used_for_size);
}
}  // namespace
