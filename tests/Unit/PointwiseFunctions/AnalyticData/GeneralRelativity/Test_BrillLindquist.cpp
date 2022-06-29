// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <limits>
#include <random>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/GeneralRelativity/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/GeneralRelativity/BrillLindquist.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct BrillLindquistProxy : gr::AnalyticData::BrillLindquist {
  using gr::AnalyticData::BrillLindquist::BrillLindquist;

  template <typename DataType>
  using variables_tags =
      typename gr::AnalyticDataBase<3>::template tags<DataType>;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<variables_tags<DataType>> test_variables(
      const tnsr::I<DataType, 3>& x) const {
    return this->variables(x, variables_tags<DataType>{});
  }
};

template <typename DataType>
void test_random(const DataType& used_for_size) {
  MAKE_GENERATOR(generator);
  auto mass_distribution = std::uniform_real_distribution<>(0.5, 1.5);
  auto center_distribution = std::uniform_real_distribution<>(0.5, 5.5);
  const double mass_a = mass_distribution(generator);
  const double mass_b = mass_distribution(generator);
  const std::array<double, 3> center_a{{center_distribution(generator),
                                        center_distribution(generator),
                                        center_distribution(generator)}};
  const std::array<double, 3> center_b{{center_distribution(generator),
                                        center_distribution(generator),
                                        center_distribution(generator)}};
  pypp::SetupLocalPythonEnvironment local_python_env{""};
  pypp::check_with_random_values<1>(
      &BrillLindquistProxy::test_variables<DataType>,
      BrillLindquistProxy(mass_a, mass_b, center_a, center_b),
      "PointwiseFunctions.AnalyticData.GeneralRelativity.BrillLindquist",
      {"lapse", "dt_lapse", "d_lapse", "shift", "dt_shift", "d_shift",
       "spatial_metric", "dt_spatial_metric", "d_spatial_metric",
       "sqrt_det_spatial_metric", "extrinsic_curvature",
       "inverse_spatial_metric"},
      {{{1.0, 2.0}}}, std::make_tuple(mass_a, mass_b, center_a, center_b),
      used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticData.Gr.BrillLindquist",
                  "[PointwiseFunctions][Unit]") {
  test_random(std::numeric_limits<double>::signaling_NaN());
}
