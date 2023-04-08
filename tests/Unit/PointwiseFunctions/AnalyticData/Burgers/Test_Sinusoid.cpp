// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <memory>
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
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct BurgersSinusoidProxy : Burgers::AnalyticData::Sinusoid {
  tuples::TaggedTuple<Burgers::Tags::U> variables(
      const tnsr::I<DataVector, 1, Frame::Inertial>& x) const {
    return Burgers::AnalyticData::Sinusoid::variables(
        x, tmpl::list<Burgers::Tags::U>{});
  }
};

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticData.Burgers.Sinusoid",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};

  test_serialization(Burgers::AnalyticData::Sinusoid{});
  register_classes_with_charm<Burgers::AnalyticData::Sinusoid>();
  const std::unique_ptr<evolution::initial_data::InitialData> option_solution =
      TestHelpers::test_option_tag_factory_creation<
          evolution::initial_data::OptionTags::InitialData,
          Burgers::AnalyticData::Sinusoid>("Sinusoid:\n")
          ->get_clone();
  const auto deserialized_option_solution =
      serialize_and_deserialize(option_solution);
  const auto& sinusoid = dynamic_cast<const Burgers::AnalyticData::Sinusoid&>(
      *deserialized_option_solution);
  CHECK(sinusoid == Burgers::AnalyticData::Sinusoid{});
  test_move_semantics(Burgers::AnalyticData::Sinusoid{},
                      Burgers::AnalyticData::Sinusoid{});

  pypp::check_with_random_values<1>(
      &BurgersSinusoidProxy::variables, BurgersSinusoidProxy{},
      "PointwiseFunctions.AnalyticData.Burgers.Sinusoid", {"u_variable"},
      {{{0.0, M_PI}}}, {}, DataVector(5));
}
}  // namespace
