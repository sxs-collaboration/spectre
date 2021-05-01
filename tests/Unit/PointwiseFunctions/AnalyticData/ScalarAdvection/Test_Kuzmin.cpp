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
#include "PointwiseFunctions/AnalyticData/ScalarAdvection/Kuzmin.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct ScalarAdvectionKuzminProxy : ScalarAdvection::AnalyticData::Kuzmin {
  tuples::TaggedTuple<ScalarAdvection::Tags::U> variables(
      const tnsr::I<DataVector, 2, Frame::Inertial>& x) const noexcept {
    return ScalarAdvection::AnalyticData::Kuzmin::variables(
        x, tmpl::list<ScalarAdvection::Tags::U>{});
  }
};

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticData.ScalarAdvection.Kuzmin",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};

  test_serialization(ScalarAdvection::AnalyticData::Kuzmin{});
  const auto periodic =
      TestHelpers::test_creation<ScalarAdvection::AnalyticData::Kuzmin>("");
  CHECK(periodic == ScalarAdvection::AnalyticData::Kuzmin{});
  test_move_semantics(ScalarAdvection::AnalyticData::Kuzmin{},
                      ScalarAdvection::AnalyticData::Kuzmin{});

  pypp::check_with_random_values<1>(
      &ScalarAdvectionKuzminProxy::variables, ScalarAdvectionKuzminProxy{},
      "PointwiseFunctions.AnalyticData.ScalarAdvection.Kuzmin", {"u_variable"},
      {{{-1.0, 1.0}}}, {}, DataVector(100));
}
}  // namespace
