// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/ScalarAdvection/Sinusoid.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct ScalarAdvectionSinusoidProxy : ScalarAdvection::AnalyticData::Sinusoid {
  tuples::TaggedTuple<ScalarAdvection::Tags::U> variables(
      const tnsr::I<DataVector, 1, Frame::Inertial>& x) const noexcept {
    return ScalarAdvection::AnalyticData::Sinusoid::variables(
        x, tmpl::list<ScalarAdvection::Tags::U>{});
  }
};

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.ScalarAdvection.Sinusoid",
    "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};

  test_serialization(ScalarAdvection::AnalyticData::Sinusoid{});
  const auto periodic =
      TestHelpers::test_creation<ScalarAdvection::AnalyticData::Sinusoid>("");
  CHECK(periodic == ScalarAdvection::AnalyticData::Sinusoid{});
  test_move_semantics(ScalarAdvection::AnalyticData::Sinusoid{},
                      ScalarAdvection::AnalyticData::Sinusoid{});

  pypp::check_with_random_values<1>(
      &ScalarAdvectionSinusoidProxy::variables, ScalarAdvectionSinusoidProxy{},
      "PointwiseFunctions.AnalyticData.ScalarAdvection.Sinusoid",
      {"u_variable"}, {{{-1.0, 1.0}}}, {}, DataVector(10));
}
}  // namespace
