// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <optional>

#include "ControlSystem/FutureMeasurements.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Framework/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.ControlSystem.FutureMeasurements",
                  "[Unit][ControlSystem]") {
  control_system::FutureMeasurements measurements(5, 1.0);
  REQUIRE(measurements.next_measurement() == std::optional(1.0));
  REQUIRE(not measurements.next_update().has_value());

  domain::FunctionsOfTime::PiecewisePolynomial<0> measurement_timescales(
      1.0, {{{0.5}}}, 1.75);
  measurements.update(measurement_timescales);
  REQUIRE(measurements.next_measurement() == std::optional(1.0));
  REQUIRE(not measurements.next_update().has_value());

  measurements.pop_front();
  REQUIRE(measurements.next_measurement() == std::optional(1.5));
  REQUIRE(not measurements.next_update().has_value());

  measurements.pop_front();
  REQUIRE(measurements.next_measurement() == std::optional(2.0));
  REQUIRE(not measurements.next_update().has_value());

  measurements.pop_front();
  REQUIRE(not measurements.next_measurement().has_value());
  REQUIRE(not measurements.next_update().has_value());

  measurement_timescales.update(1.75, {1.0}, 5.0);
  measurements.update(measurement_timescales);
  REQUIRE(measurements.next_measurement() == std::optional(3.0));
  REQUIRE(measurements.next_update() == std::optional(4.0));

  measurements.pop_front();
  REQUIRE(measurements.next_measurement() == std::optional(4.0));
  REQUIRE(measurements.next_update() == std::optional(4.0));

  measurements.pop_front();
  REQUIRE(measurements.next_measurement() == std::optional(5.0));
  REQUIRE(not measurements.next_update().has_value());

  measurements.pop_front();
  REQUIRE(measurements.next_measurement() == std::optional(6.0));
  REQUIRE(not measurements.next_update().has_value());

  measurement_timescales.update(5.0, {0.5}, 10.0);
  measurements.update(measurement_timescales);
  REQUIRE(measurements.next_measurement() == std::optional(6.0));
  REQUIRE(measurements.next_update() == std::optional(7.5));

  measurements.pop_front();
  REQUIRE(measurements.next_measurement() == std::optional(6.5));
  REQUIRE(measurements.next_update() == std::optional(7.5));

  measurements.pop_front();
  REQUIRE(measurements.next_measurement() == std::optional(7.0));
  REQUIRE(measurements.next_update() == std::optional(7.5));

  measurements.pop_front();
  REQUIRE(measurements.next_measurement() == std::optional(7.5));
  REQUIRE(measurements.next_update() == std::optional(7.5));

  measurements.pop_front();
  REQUIRE(measurements.next_measurement() == std::optional(8.0));
  REQUIRE(measurements.next_update() == std::optional(10.0));

  const auto measurements_copy = serialize_and_deserialize(measurements);
  REQUIRE(measurements_copy.next_measurement() == std::optional(8.0));
  REQUIRE(measurements_copy.next_update() == std::optional(10.0));
}
