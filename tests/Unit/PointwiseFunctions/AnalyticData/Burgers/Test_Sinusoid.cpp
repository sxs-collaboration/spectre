// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"        // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"  // IWYU pragma: keep
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/Burgers/Sinusoid.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct BurgersSinusoidProxy : Burgers::AnalyticData::Sinusoid {
  tuples::TaggedTuple<Burgers::Tags::U> variables(
      const tnsr::I<DataVector, 1, Frame::Inertial>& x) const noexcept {
    return Burgers::AnalyticData::Sinusoid::variables(
        x, tmpl::list<Burgers::Tags::U>{});
  }
};

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticData.Burgers.Sinusoid",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};

  test_serialization(Burgers::AnalyticData::Sinusoid{});
  const auto periodic =
      TestHelpers::test_creation<Burgers::AnalyticData::Sinusoid>("");
  CHECK(periodic == Burgers::AnalyticData::Sinusoid{});
  test_move_semantics(Burgers::AnalyticData::Sinusoid{},
                      Burgers::AnalyticData::Sinusoid{});

  pypp::check_with_random_values<1, tmpl::list<Burgers::Tags::U>>(
      &BurgersSinusoidProxy::variables, BurgersSinusoidProxy{},
      "PointwiseFunctions.AnalyticData.Burgers.Sinusoid", {"u_variable"},
      {{{0.0, M_PI}}}, {}, DataVector(5));
}
}  // namespace
