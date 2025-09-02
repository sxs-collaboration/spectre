// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Shape.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Tags/Criteria.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "PointwiseFunctions/GeneralRelativity/KerrHorizon.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace ah::Criteria {
namespace {
template <typename Frame>
struct Metavariables {
  using component_list = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<ah::Criterion, tmpl::list<Shape>>>;
  };
};

template <typename Fr>
ylm::Strahlkorper<Fr> make_strahlkorper(size_t l_max,
                                        const std::array<double, 3>& spin) {
  const ylm::Strahlkorper<Fr>& sphere =
      ylm::Strahlkorper<Fr>(l_max, 4.5, {{0.4, 0.5, 0.6}});
  const std::array<DataVector, 2> theta_phi =
      sphere.ylm_spherepack().theta_phi_points();
  const auto radius = gr::Solutions::kerr_horizon_radius(theta_phi, 1.0, spin);
  return ylm::Strahlkorper<Fr>{l_max, l_max, get(radius), {{0.4, 0.5, 0.6}}};
}

template <typename Fr>
void test_criterion_evaluation(
    const std::unique_ptr<ah::Criterion>& criterion,
    const ObservationBox<tmpl::list<>, db::DataBox<tmpl::list<>>>& box,
    Parallel::GlobalCache<Metavariables<Frame::Inertial>>& cache, size_t l_max,
    const std::array<double, 3>& spin, size_t expected_new_l_max) {
  const ylm::Strahlkorper<Fr> strahlkorper = make_strahlkorper<Fr>(l_max, spin);
  const size_t new_l_max =
      criterion->evaluate(box, cache, strahlkorper, FastFlow::IterInfo{});
  CHECK(new_l_max == expected_new_l_max);
}

void test_shape() {
  TestHelpers::db::test_simple_tag<ah::Criteria::Tags::Criteria>("Criteria");
  const auto criterion{TestHelpers::test_factory_creation<ah::Criterion, Shape>(
      "Shape:\n"
      "  MinTruncationError: 1.0e-10\n"
      "  MaxTruncationError: 1.0e-8\n"
      "  MaxPileUpModes: 5\n"
      "  MinResolutionL: 4\n"
      "  MaxResolutionL: 12")};
  // Test observation_name
  CHECK(criterion->observation_name() == "Shape");

  Parallel::GlobalCache<Metavariables<Frame::Inertial>> empty_cache{};
  auto databox = db::create<tmpl::list<>>();
  const ObservationBox<tmpl::list<>, db::DataBox<tmpl::list<>>> box{
      make_not_null(&databox)};

  constexpr double min_truncation_error{1.0e-10};
  constexpr double max_truncation_error{1.0e-8};
  constexpr size_t max_pile_up_modes{5};
  constexpr size_t min_resolution_l{4};
  constexpr size_t max_resolution_l{12};
  constexpr size_t mid_l{8};
  const std::array<double, 3> zero_spin{{0.0, 0.0, 0.0}};
  const std::array<double, 3> low_aligned_spin{{0.0, 0.0, 0.1}};
  const std::array<double, 3> low_spin{{0.1, 0.2, 0.3}};
  const std::array<double, 3> high_spin{{0.4, 0.5, 0.6}};
  constexpr size_t min_l_allowed_by_power_monitor{4};
  constexpr double eps{std::numeric_limits<double>::epsilon()};

  // Normal case - decrease resolution when truncation error is low
  // Make truncation error low by passing in a spherical Strahlkorper, i.e.
  // a horizon of a black hole with zero spin
  test_criterion_evaluation<Frame::Inertial>(criterion, box, empty_cache, mid_l,
                                             zero_spin, mid_l - 1);

  // Normal case - increase resolution when truncation error is high
  // Make truncation error high (~1e-7) by passing in a spinning Kerr Schild
  // horizon for the strahlkorper
  test_criterion_evaluation<Frame::Inertial>(criterion, box, empty_cache, mid_l,
                                             high_spin, mid_l + 1);

  // Normal case - keep resolution the same when truncation error is in range
  // Do this by using a Strahlkorper of a Kerr Schild black hole spinning
  // but not quite as fast
  test_criterion_evaluation<Frame::Inertial>(criterion, box, empty_cache, mid_l,
                                             low_spin, mid_l);

  // Normal case - keep resolution the same when you would have raised it
  // except you have too many pile up modes. Here make lots of pile up modes by
  // using high resolution for spin=0.1 Kerr-Schild black hole, and use
  // very small truncation error ranges so that the criterion would otherwise
  // want to increase resolution, if it weren't for the pileup modes
  const std::unique_ptr<ah::Criterion> criterion_pile_up =
      std::make_unique<ah::Criteria::Shape>(min_truncation_error * 1.e-10,
                                            max_truncation_error * 1.e-10, 1,
                                            min_resolution_l, mid_l * 4);
  test_criterion_evaluation<Frame::Inertial>(criterion_pile_up, box,
                                             empty_cache, mid_l * 4,
                                             low_aligned_spin, mid_l * 4);

  // Edge case - at minimum resolution, low truncation error
  test_criterion_evaluation<Frame::Inertial>(criterion, box, empty_cache,
                                             min_resolution_l, zero_spin,
                                             min_resolution_l);

  // Edge case - at max resolution, high truncation error
  test_criterion_evaluation<Frame::Inertial>(criterion, box, empty_cache,
                                             max_resolution_l, high_spin,
                                             max_resolution_l);

  // Edge case - at minimum resolution, high truncation error
  test_criterion_evaluation<Frame::Inertial>(criterion, box, empty_cache,
                                             min_resolution_l, high_spin,
                                             min_resolution_l + 1);

  // Edge case - at max resolution, low truncation error
  test_criterion_evaluation<Frame::Inertial>(criterion, box, empty_cache,
                                             max_resolution_l, zero_spin,
                                             max_resolution_l - 1);

  // Edge case - truncation error exactly at min_truncation_error
  const ylm::Strahlkorper<Frame::Inertial>
      strahlkorper_with_known_truncation_error =
          make_strahlkorper<Frame::Inertial>(mid_l, high_spin);
  const DataVector power_monitor_known_truncation_error =
      ylm::power_monitor(strahlkorper_with_known_truncation_error);
  const double known_truncation_error =
      PowerMonitors::relative_truncation_error(
          power_monitor_known_truncation_error,
          power_monitor_known_truncation_error.size());
  const std::unique_ptr<ah::Criterion> criterion_min_truncation_error =
      std::make_unique<ah::Criteria::Shape>(
          known_truncation_error, known_truncation_error * 100.0,
          max_pile_up_modes, min_resolution_l, max_resolution_l);
  test_criterion_evaluation<Frame::Inertial>(criterion_min_truncation_error,
                                             box, empty_cache, mid_l, high_spin,
                                             mid_l);

  // Edge case - truncation error just above min_truncation_error
  const std::unique_ptr<ah::Criterion>
      criterion_min_truncation_error_just_above =
          std::make_unique<ah::Criteria::Shape>(
              known_truncation_error - eps, known_truncation_error * 100.0,
              max_pile_up_modes, min_resolution_l, max_resolution_l);
  test_criterion_evaluation<Frame::Inertial>(
      criterion_min_truncation_error_just_above, box, empty_cache, mid_l,
      high_spin, mid_l);

  // Edge case - truncation error just below min_truncation_error
  const std::unique_ptr<ah::Criterion>
      criterion_min_truncation_error_just_below =
          std::make_unique<ah::Criteria::Shape>(
              known_truncation_error + eps, known_truncation_error * 100.0,
              max_pile_up_modes, min_resolution_l, max_resolution_l);
  test_criterion_evaluation<Frame::Inertial>(
      criterion_min_truncation_error_just_below, box, empty_cache, mid_l,
      high_spin, mid_l - 1);

  // Edge case - truncation error exactly at max_truncation_error
  const std::unique_ptr<ah::Criterion> criterion_max_truncation_error =
      std::make_unique<ah::Criteria::Shape>(
          known_truncation_error / 100.0, known_truncation_error,
          max_pile_up_modes, min_resolution_l, max_resolution_l);
  test_criterion_evaluation<Frame::Inertial>(criterion_max_truncation_error,
                                             box, empty_cache, mid_l, high_spin,
                                             mid_l);

  // Edge case - truncation error just below max_truncation_error
  const std::unique_ptr<ah::Criterion>
      criterion_max_truncation_error_just_below =
          std::make_unique<ah::Criteria::Shape>(
              known_truncation_error / 100.0, known_truncation_error + eps,
              max_pile_up_modes, min_resolution_l, max_resolution_l);
  test_criterion_evaluation<Frame::Inertial>(
      criterion_max_truncation_error_just_below, box, empty_cache, mid_l,
      high_spin, mid_l);

  // Edge case - truncation error just above max_truncation_error
  const std::unique_ptr<ah::Criterion>
      criterion_max_truncation_error_just_above =
          std::make_unique<ah::Criteria::Shape>(
              known_truncation_error / 100.0, known_truncation_error - eps,
              max_pile_up_modes, min_resolution_l, max_resolution_l);
  test_criterion_evaluation<Frame::Inertial>(
      criterion_max_truncation_error_just_above, box, empty_cache, mid_l,
      high_spin, mid_l + 1);

  // Edge case - l_min = 4 (minimum allowed by power monitor)
  const std::unique_ptr<ah::Criterion> criterion_l_min_4 =
      std::make_unique<ah::Criteria::Shape>(
          min_truncation_error, max_truncation_error, max_pile_up_modes,
          min_l_allowed_by_power_monitor, max_resolution_l);
  test_criterion_evaluation<Frame::Inertial>(
      criterion_l_min_4, box, empty_cache, min_l_allowed_by_power_monitor,
      zero_spin, min_l_allowed_by_power_monitor);

  // Edge case - l_min = 4 (minimum allowed by power monitor), high residual
  test_criterion_evaluation<Frame::Inertial>(
      criterion_l_min_4, box, empty_cache, min_l_allowed_by_power_monitor,
      high_spin, min_l_allowed_by_power_monitor + 1);

  // Edge case - min_resolution_l == max_resolution_l
  const std::unique_ptr<ah::Criterion>
      criterion_min_resolution_l_eq_max_resolution_l =
          std::make_unique<ah::Criteria::Shape>(
              min_truncation_error, max_truncation_error, max_pile_up_modes,
              mid_l, mid_l);
  test_criterion_evaluation<Frame::Inertial>(
      criterion_min_resolution_l_eq_max_resolution_l, box, empty_cache, mid_l,
      high_spin, mid_l);

  // Edge case - min_resolution_l == max_resolution_l but
  // different from strahlkorper
#ifdef SPECTRE_DEBUG
  const ylm::Strahlkorper<Frame::Inertial> debug_strahlkorper{
      make_strahlkorper<Frame::Inertial>(mid_l - 2, zero_spin)};
  CHECK_THROWS_WITH(
      criterion_min_resolution_l_eq_max_resolution_l->evaluate(
          box, empty_cache, debug_strahlkorper, FastFlow::IterInfo{}),
      Catch::Matchers::ContainsSubstring(
          "If MinResolutionL == MaxResolutionL"));
#endif

  // Test equality and serialization
  const Shape criterion_one{1.0e-6, 1.0e-4, 5, 4, 12};
  const Shape criterion_two{1.0e-6, 1.0e-4, 5, 4, 12};
  const Shape criterion_three{1.0e-6, 1.0e-3, 5, 4, 12};
  CHECK(criterion_one.is_equal(criterion_two));
  CHECK(not(criterion_one.is_equal(criterion_three)));
  const auto criterion_one_serialized =
      serialize_and_deserialize(criterion_one);
  CHECK(criterion_one.is_equal(criterion_one_serialized));
}

void test_shape_constructor_validation() {
  // Define test constants
  constexpr double min_truncation_error = 1.0e-6;
  constexpr double max_truncation_error = 1.0e-4;
  const double zero = 0.0;
  constexpr size_t max_pile_up_modes = 5;
  constexpr size_t min_resolution_l = 4;
  constexpr size_t max_resolution_l = 12;
  constexpr size_t min_allowed_resolution = 4;

  // Test valid construction
  CHECK_NOTHROW((Shape{min_truncation_error, max_truncation_error,
                       max_pile_up_modes, min_resolution_l, max_resolution_l}));

  // Test invalid construction - min_truncation_error >= max_truncation_error
  CHECK_THROWS_WITH(
      (Shape{max_truncation_error, min_truncation_error, max_pile_up_modes,
             min_resolution_l, max_resolution_l}),
      Catch::Matchers::ContainsSubstring(
          "MinTruncationError must be less than MaxTruncationError"));

  // Test invalid construction - min_truncation_error == max_truncation_error
  CHECK_THROWS_WITH(
      (Shape{min_truncation_error, min_truncation_error, max_pile_up_modes,
             min_resolution_l, max_resolution_l}),
      Catch::Matchers::ContainsSubstring(
          "MinTruncationError must be less than MaxTruncationError"));

  // Test invalid construction - min_resolution_l > max_resolution_l
  CHECK_THROWS_WITH(
      (Shape{min_truncation_error, max_truncation_error, max_pile_up_modes,
             max_resolution_l, min_resolution_l}),
      Catch::Matchers::ContainsSubstring(
          "MinResolutionL must be less than MaxResolutionL"));

  // Test edge case - zero max_truncation_error (should fail)
  CHECK_THROWS_WITH(
      (Shape{min_truncation_error, zero, max_pile_up_modes, min_resolution_l,
             max_resolution_l}),
      Catch::Matchers::ContainsSubstring(
          "MinTruncationError must be less than MaxTruncationError"));

  // Test edge case - min_resolution_l below minimum allowed value (4)
  CHECK_THROWS_WITH(
      (Shape{min_truncation_error, max_truncation_error, max_pile_up_modes,
             min_allowed_resolution - 1, max_resolution_l}),
      Catch::Matchers::ContainsSubstring("MinResolutionL must be at least 4"));

  // Test edge case - max_resolution_l below minimum allowed value (4)
  CHECK_THROWS_WITH(
      (Shape{min_truncation_error, max_truncation_error, max_pile_up_modes,
             min_allowed_resolution, min_allowed_resolution - 1}),
      Catch::Matchers::ContainsSubstring("MaxResolutionL must be at least 4"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.Criteria.Shape",
                  "[ApparentHorizonFinder][Unit]") {
  test_shape();
  test_shape_constructor_validation();
}
}  // namespace ah::Criteria
