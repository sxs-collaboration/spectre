// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>

#include "ControlSystem/ControlErrors/Skew.hpp"
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
#include "Utilities/Serialization/Serialize.hpp"
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
constexpr size_t l_max = 8;
constexpr double radius = 0.5;

void test_skew(const double time,
               const ylm::Strahlkorper<Frame::Distorted>& horizon_a,
               const ylm::Strahlkorper<Frame::Distorted>& horizon_b,
               const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>&
                   function_of_time,
               const DataVector& expected_error,
               const std::optional<double>& expected_timescale,
               const std::optional<Approx>& custom_approx = std::nullopt) {
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time["Skew"] = function_of_time->get_clone();
  const std::array<double, 3> grid_center_a{{grid_x_coord, 0.0, 0.0}};
  const std::array<double, 3> grid_center_b{{-grid_x_coord, 0.0, 0.0}};
  const Parallel::GlobalCache<Metavars> cache{
      {std::move(functions_of_time),
       tnsr::I<double, 3, Frame::Grid>{grid_center_a},
       tnsr::I<double, 3, Frame::Grid>{grid_center_b}, "", "",
       std::vector<std::string>{}}};
  using HorizonA = control_system::QueueTags::Horizon<Frame::Distorted,
                                                      ::domain::ObjectLabel::A>;
  using HorizonB = control_system::QueueTags::Horizon<Frame::Distorted,
                                                      ::domain::ObjectLabel::B>;
  const tuples::TaggedTuple<HorizonA, HorizonB> measurements{horizon_a,
                                                             horizon_b};

  using SkewError = control_system::ControlErrors::Skew;

  const auto error_class_creation =
      TestHelpers::test_creation<SkewError, Metavars>("");
  SkewError error_class = serialize_and_deserialize(error_class_creation);

  CHECK_FALSE(error_class.get_suggested_timescale().has_value());
  error_class.reset();
  CHECK_FALSE(error_class.get_suggested_timescale().has_value());

  TimescaleTuner<true> tuner{std::vector<double>{20.0, 20.0}, 20.0, 0.01,
                             1.0e-4, 1.01};

  auto approx_to_use = custom_approx.value_or(approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      error_class(tuner, cache, time, "Skew", measurements), expected_error,
      approx_to_use);

  REQUIRE(error_class.get_suggested_timescale().has_value() ==
          expected_timescale.has_value());
  if (expected_timescale.has_value()) {
    CHECK(error_class.get_suggested_timescale().value() ==
          approx(expected_timescale.value()));
  }

  const DataVector old_timescale = tuner.current_timescale();

  control_system::update_timescale_tuner(make_not_null(&tuner),
                                         make_not_null(&error_class),
                                         ::Verbosity::Silent, time, "Skew"s);

  if (expected_timescale.has_value()) {
    CHECK(old_timescale != tuner.current_timescale());
  } else {
    CHECK(old_timescale == tuner.current_timescale());
  }
  error_class.reset();
  CHECK_FALSE(error_class.get_suggested_timescale().has_value());
}
}  // namespace

// [[TimeOut, 10]]
SPECTRE_TEST_CASE("Unit.ControlSystem.ControlErrors.Skew",
                  "[ControlSystem][Unit]") {
  const double initial_time = 0.0;
  const double expiration_time = 5.0;
  const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>
      function_of_time =
          std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
              initial_time, make_array<3>(DataVector{2, 0.0}), expiration_time);

  const std::array<double, 3> horizon_center_a{grid_x_coord, 0.0, 0.0};
  const std::array<double, 3> horizon_center_b{-grid_x_coord, 0.0, 0.0};
  {
    ylm::Strahlkorper<Frame::Distorted> horizon_a(l_max, radius,
                                                  horizon_center_a);
    ylm::Strahlkorper<Frame::Distorted> horizon_b(l_max, radius,
                                                  horizon_center_b);

    // Not activated
    test_skew(initial_time, horizon_a, horizon_b, function_of_time,
              DataVector{2, 0.0}, std::nullopt);

    horizon_a = ylm::Strahlkorper<Frame::Distorted>(l_max, 5.0 * radius,
                                                    horizon_center_a);
    horizon_b = ylm::Strahlkorper<Frame::Distorted>(l_max, 5.0 * radius,
                                                    horizon_center_b);

    // Activated, but error is still zero because the horizon isn't distorted.
    // However, we do have a suggested timescale now
    test_skew(initial_time, horizon_a, horizon_b, function_of_time,
              DataVector{2, 0.0}, {6.0});
  }

  // Create surface that is just a sphere that's shifted by a constant offset,
  // aka \vec{r}' = \vec{r} + \vec{c} where c is the center shift away from the
  // original center
  {
    // Need high resolution because of the wonky shape of the surface we are
    // creating
    const size_t l_max_high_res = 6 * l_max;
    ylm::Strahlkorper<Frame::Distorted> horizon_a{
        l_max_high_res, 5.0 * radius, {0.0, 0.0, 0.0}};
    auto coords = ylm::cartesian_coords(horizon_a);
    const std::array<double, 3> center_offset{0.0, 0.0,
                                              5.0 * radius / sqrt(2.0)};
    // Recenter the coords
    for (size_t i = 0; i < 3; i++) {
      coords.get(i) -= gsl::at(center_offset, i);
    }
    const auto radii = pointwise_l2_norm(coords);

    // Redistribute theta so that theta=pi/2 is along the x-axis again
    const auto& ylm = horizon_a.ylm_spherepack();
    const auto original_theta_phis = ylm.theta_phi_points();
    const std::array<DataVector, 2> new_theta_phis{
        square(original_theta_phis[0]) / M_PI, original_theta_phis[1]};

    // Interpolate shifted coordinates to these new theta/phi points
    const auto interpolation_info =
        ylm.set_up_interpolation_info(new_theta_phis);
    std::array<DataVector, 3> new_coords{get<0>(coords), get<1>(coords),
                                         get<2>(coords)};
    for (size_t i = 0; i < 3; i++) {
      ylm.interpolate(make_not_null(&gsl::at(new_coords, i)),
                      make_not_null(coords.get(i).data()), interpolation_info);
    }
    const auto new_radii = magnitude(new_coords);

    horizon_a = ylm::Strahlkorper<Frame::Distorted>(
        l_max_high_res, l_max_high_res, new_radii, horizon_center_a);

    const ylm::Strahlkorper<Frame::Distorted> horizon_b{
        l_max_high_res, 5.0 * radius, horizon_center_b};

    // These numbers are computed by hand give the center offset above
    const double weight_a = exp(5.0 / (12.0 * sqrt(2.0)) - 1.0);
    const DataVector expected_control_error{
        0.0, -0.5 * (1.0 - tanh((19.0 - 5.0 / sqrt(2.0)) / 2.4 - 5.0)) *
                 weight_a * M_PI_4 / (weight_a + exp(-7.0 / 12.0))};

    // Because of all the interpolation, our result isn't as accurate, even with
    // the high LMax resolution
    const Approx custom_approx = Approx::custom().epsilon(1.0e-6).scale(1.0);
    test_skew(initial_time, horizon_a, horizon_b, function_of_time,
              expected_control_error, {6.0}, {custom_approx});
  }
}
