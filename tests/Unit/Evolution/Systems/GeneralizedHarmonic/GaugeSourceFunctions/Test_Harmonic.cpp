// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <memory>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Dispatch.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Gauges.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Harmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/RegisterDerived.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace {
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<gh::gauges::GaugeCondition,
                             tmpl::list<gh::gauges::Harmonic>>>;
  };
};

template <size_t Dim>
void test() {
  const auto gauge_condition = serialize_and_deserialize(
      TestHelpers::test_creation<std::unique_ptr<gh::gauges::GaugeCondition>,
                                 Metavariables>("Harmonic:")
          ->get_clone());

  const size_t num_points = 5;
  tnsr::a<DataVector, Dim, Frame::Inertial> gauge_h(num_points);
  tnsr::ab<DataVector, Dim, Frame::Inertial> d4_gauge_h(num_points);

  const double time = std::numeric_limits<double>::signaling_NaN();
  const tnsr::I<DataVector, Dim, Frame::Inertial> inertial_coords(num_points);

  // Used dispatch with defaulted arguments that we don't need for Harmonic
  // gauge.
  gh::gauges::dispatch(make_not_null(&gauge_h), make_not_null(&d4_gauge_h), {},
                       {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, Mesh<Dim>{},
                       time, inertial_coords, {}, *gauge_condition);
  CHECK(gauge_h == tnsr::a<DataVector, Dim, Frame::Inertial>(num_points, 0.0));
  CHECK(d4_gauge_h ==
        tnsr::ab<DataVector, Dim, Frame::Inertial>(num_points, 0.0));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.Gauge.Harmonic",
                  "[Unit][Evolution]") {
  gh::gauges::register_derived_with_charm();
  test<1>();
  test<2>();
  test<3>();
}
