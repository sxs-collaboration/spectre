// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>

#include "ControlSystem/ControlErrors/GridCenters.hpp"
#include "ControlSystem/Protocols/ControlError.hpp"
#include "ControlSystem/Tags/QueueTags.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "ControlSystem/UpdateTimescaleTuner.hpp"
#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Creators/Tags/ObjectCenter.hpp"
#include "Domain/ExcisionSphere.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct Metavars {
  using const_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTimeInitialize,
                 domain::Tags::ObjectCenter<domain::ObjectLabel::A>,
                 domain::Tags::ObjectCenter<domain::ObjectLabel::B>>;
  using observed_reduction_data_tags = tmpl::list<>;
  using component_list = tmpl::list<observers::ObserverWriter<Metavars>>;
  void pup(PUP::er& /*p*/) {}
};

constexpr double grid_x_coord = 6.0;

void test_grid_centers(
    const double time, const double measured_grid_x_coord,
    const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>&
        function_of_time,
    const DataVector& expected_error) {
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time["GridCenters"] = function_of_time->get_clone();
  const Parallel::GlobalCache<Metavars> cache{
      {std::move(functions_of_time),
       tnsr::I<double, 3, Frame::Grid>{{grid_x_coord, 0.0, 0.0}},
       tnsr::I<double, 3, Frame::Grid>{{-grid_x_coord, 0.0, 0.0}}, "", "",
       std::vector<std::string>{}}};
  using grid_center_A =
      control_system::QueueTags::Center<::domain::ObjectLabel::A, Frame::Grid>;
  using grid_center_B =
      control_system::QueueTags::Center<::domain::ObjectLabel::B, Frame::Grid>;
  const tuples::TaggedTuple<grid_center_A, grid_center_B> measurements{
      std::array{measured_grid_x_coord, 0.0, 0.0},
      std::array{-measured_grid_x_coord, 0.0, 0.0}};

  using GridCentersError = control_system::ControlErrors::GridCenters;

  const auto error_class_creation =
      TestHelpers::test_creation<GridCentersError, Metavars>("");
  GridCentersError error_class =
      serialize_and_deserialize(error_class_creation);

  CHECK_FALSE(error_class.get_suggested_timescale().has_value());
  error_class.reset();
  CHECK_FALSE(error_class.get_suggested_timescale().has_value());

  TimescaleTuner<true> tuner{std::vector<double>{20.0, 20.0}, 20.0, 0.01,
                             1.0e-4, 1.01};

  CHECK_ITERABLE_APPROX(
      error_class(tuner, cache, time, "GridCenters", measurements),
      expected_error);

  REQUIRE(not error_class.get_suggested_timescale().has_value());

  const DataVector old_timescale = tuner.current_timescale();

  control_system::update_timescale_tuner(
      make_not_null(&tuner), make_not_null(&error_class), ::Verbosity::Silent,
      time, "GridCenters"s);

  CHECK(old_timescale == tuner.current_timescale());
  CHECK_FALSE(error_class.get_suggested_timescale().has_value());
}
}  // namespace

// [[TimeOut, 10]]
SPECTRE_TEST_CASE("Unit.ControlSystem.ControlErrors.GridCenters",
                  "[ControlSystem][Unit]") {
  const double initial_time = 0.0;
  const double expiration_time = 5.0;
  const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>
      function_of_time =
          std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
              initial_time,
              std::array{DataVector{6.0, 0.0, 0.0, -6.0, 0.0, 0.0},
                         DataVector{6, 0.0}, DataVector{6, 0.0}},
              expiration_time);

  test_grid_centers(initial_time, grid_x_coord, function_of_time,
                    DataVector{6, 0.0});
  test_grid_centers(initial_time, 0.5 * grid_x_coord, function_of_time,
                    {-3.0, 0.0, 0.0, 3.0, 0.0, 0.0});
}
