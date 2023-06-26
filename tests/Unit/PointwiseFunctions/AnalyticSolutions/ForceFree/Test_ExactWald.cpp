// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <utility>

#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/ForceFree/ExactWald.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

using InitialData = evolution::initial_data::InitialData;
using ExactWald = ForceFree::Solutions::ExactWald;

struct ExactWaldProxy : ExactWald {
  using ExactWald::ExactWald;
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
    "Unit.PointwiseFunctions.AnalyticSolutions.ForceFree.ExactWald",
    "[Unit][PointwiseFunctions]") {
  // test creation
  const auto solution =
      TestHelpers::test_creation<ExactWald>("MagneticFieldAmplitude: 1.0");
  CHECK(solution == ExactWald(1.0));
  CHECK(solution != ExactWald(-1.0));

  // test serialize
  test_serialization(solution);

  // test move
  test_move_semantics(ExactWald{1.0}, ExactWald{1.0});

  // test derived
  register_classes_with_charm<ExactWald>();
  const std::unique_ptr<InitialData> base_ptr =
      std::make_unique<ExactWald>(0.5);
  const std::unique_ptr<InitialData> deserialized_base_ptr =
      serialize_and_deserialize(base_ptr)->get_clone();
  CHECK(dynamic_cast<const ExactWald&>(*deserialized_base_ptr) ==
        dynamic_cast<const ExactWald&>(*base_ptr));

  // test solution for a random value of B0
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist_mu(-10.0, 10.0);
  const double magnetic_field_amplitude{make_with_random_values<double>(
      make_not_null(&gen), make_not_null(&dist_mu))};

  ExactWaldProxy exact_wald(magnetic_field_amplitude);
  const auto member_variables = std::make_tuple(magnetic_field_amplitude);

  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/ForceFree"};
  const DataVector used_for_size{10};

  pypp::check_with_random_values<1>(
      &ExactWaldProxy::return_variables, exact_wald, "ExactWald",
      {"TildeE", "TildeB", "TildePsi", "TildePhi", "TildeQ"}, {{{-2.0, 2.0}}},
      member_variables, used_for_size);
}
}  // namespace
