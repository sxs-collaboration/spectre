// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <optional>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Punctures/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/Punctures/MultiplePunctures.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Punctures::AnalyticData {
namespace {

using test_tags =
    tmpl::list<Punctures::Tags::TracelessConformalExtrinsicCurvature,
               Punctures::Tags::Alpha, Punctures::Tags::Beta>;

struct MultiplePuncturesProxy {
  tuples::tagged_tuple_from_typelist<test_tags> test_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const {
    return multiple_punctures.variables(x, test_tags{});
  }

  const MultiplePunctures& multiple_punctures;
};

void test_data(const std::vector<Puncture>& punctures,
               const std::string& options_string) {
  const auto created =
      TestHelpers::test_factory_creation<elliptic::analytic_data::Background,
                                         MultiplePunctures>(options_string);
  REQUIRE(dynamic_cast<const MultiplePunctures*>(created.get()) != nullptr);
  const auto& derived = dynamic_cast<const MultiplePunctures&>(*created);
  const auto multiple_punctures = serialize_and_deserialize(derived);
  {
    INFO("Properties");
    CHECK(multiple_punctures.punctures() == punctures);
  }
  {
    const MultiplePuncturesProxy proxy{multiple_punctures};
    pypp::check_with_random_values<1>(
        &MultiplePuncturesProxy::test_variables, proxy, "MultiplePunctures",
        {"traceless_conformal_extrinsic_curvature", "alpha", "beta"},
        {{{-1., 1.}}}, std::make_tuple(), DataVector(5));
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.AnalyticData.Punctures.MultiplePunctures",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticData/Punctures"};
  test_data({{{1., 2., 3.}, 1., {0., 0., 0.}, {0., 0., 0.}},
             {{-1., 2., 3.}, 0.5, {0.1, -0.2, 0.3}, {0.3, -0.2, 0.1}}},
            "MultiplePunctures:\n"
            "  Punctures:\n"
            "   - Position: [1., 2., 3.]\n"
            "     Mass: 1.\n"
            "     Momentum: [0, 0, 0]\n"
            "     Spin: [0, 0, 0]\n"
            "   - Position: [-1., 2., 3.]\n"
            "     Mass: 0.5\n"
            "     Momentum: [0.1, -0.2, 0.3]\n"
            "     Spin: [0.3, -0.2, 0.1]\n");
}

}  // namespace Punctures::AnalyticData
