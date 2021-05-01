// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/ScalarAdvection/Krivodonova.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct ScalarAdvectionKrivodonovaProxy
    : ScalarAdvection::AnalyticData::Krivodonova {
  tuples::TaggedTuple<ScalarAdvection::Tags::U> variables(
      const tnsr::I<DataVector, 1, Frame::Inertial>& x) const noexcept {
    return ScalarAdvection::AnalyticData::Krivodonova::variables(
        x, tmpl::list<ScalarAdvection::Tags::U>{});
  }
};

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.ScalarAdvection.Krivodonova",
    "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};

  test_serialization(ScalarAdvection::AnalyticData::Krivodonova{});
  const auto periodic =
      TestHelpers::test_creation<ScalarAdvection::AnalyticData::Krivodonova>(
          "");
  CHECK(periodic == ScalarAdvection::AnalyticData::Krivodonova{});
  test_move_semantics(ScalarAdvection::AnalyticData::Krivodonova{},
                      ScalarAdvection::AnalyticData::Krivodonova{});

  pypp::check_with_random_values<1>(
      &ScalarAdvectionKrivodonovaProxy::variables,
      ScalarAdvectionKrivodonovaProxy{},
      "PointwiseFunctions.AnalyticData.ScalarAdvection.Krivodonova",
      {"u_variable"}, {{{-1.0, 1.0}}}, {}, DataVector(50));
}
}  // namespace
