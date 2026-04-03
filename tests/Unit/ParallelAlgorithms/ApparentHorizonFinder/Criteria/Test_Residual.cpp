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
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/IncreaseResolution.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Residual.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Tags/Criteria.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ah::Criteria {
namespace {
template <typename Frame>
struct Metavariables {
  using component_list = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<ah::Tags::LMax>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<ah::Criterion, tmpl::list<Residual>>>;
  };
};

FastFlow::IterInfo make_iter_info(double residual_ylm) {
  FastFlow::IterInfo info{};
  info.residual_ylm = residual_ylm;
  return info;
}

template <typename Fr>
ylm::Strahlkorper<Fr> make_strahlkorper(size_t l_max) {
  return ylm::Strahlkorper<Fr>(l_max, 4.5, {{0.4, 0.5, 0.6}});
}

template <typename Fr>
void test_criterion_evaluation(
    const std::unique_ptr<ah::Criterion>& criterion,
    const ObservationBox<tmpl::list<>, db::DataBox<tmpl::list<>>>& box,
    Parallel::GlobalCache<Metavariables<Frame::Inertial>>& cache, size_t l_max,
    double residual_ylm, size_t expected_new_l_max) {
  const ylm::Strahlkorper<Fr> strahlkorper = make_strahlkorper<Fr>(l_max);
  const FastFlow::IterInfo iter_info = make_iter_info(residual_ylm);
  const size_t new_l_max =
      criterion->evaluate(box, cache, strahlkorper, iter_info);
  CHECK(new_l_max == expected_new_l_max);
}

void test_residual() {
  TestHelpers::db::test_simple_tag<ah::Criteria::Tags::Criteria>("Criteria");
  const auto criterion{
      TestHelpers::test_factory_creation<ah::Criterion, Residual>(
          "Residual:\n"
          "  MinResidual: 1.0e-6\n"
          "  MaxResidual: 1.0e-4\n"
          "  MinResolutionL: 4")};
  // Test observation_name
  CHECK(criterion->observation_name() == "Residual");

  auto databox = db::create<tmpl::list<>>();
  const ObservationBox<tmpl::list<>, db::DataBox<tmpl::list<>>> box{
      make_not_null(&databox)};

  constexpr double min_residual{1.0e-6};
  constexpr double max_residual{1.0e-4};
  constexpr size_t min_resolution_l{4};
  constexpr size_t max_resolution_l{12};
  constexpr size_t mid_l{8};
  constexpr size_t min_l_allowed_by_strahlkorper{2};
  constexpr double eps{std::numeric_limits<double>::epsilon()};

  Parallel::GlobalCache<Metavariables<Frame::Inertial>> cache{
      tuples::TaggedTuple<ah::Tags::LMax>{max_resolution_l}};

  // Normal case - decrease resolution when residual is low
  test_criterion_evaluation<Frame::Inertial>(criterion, box, cache, mid_l,
                                             0.5 * min_residual, mid_l - 1);

  // Normal case - increase resolution when residual is high
  test_criterion_evaluation<Frame::Inertial>(criterion, box, cache, mid_l,
                                             2.0 * max_residual, mid_l + 1);

  // Normal case - keep resolution when residual is in range
  const double in_range_residual{(0.3 * min_residual) + (0.7 * max_residual)};
  test_criterion_evaluation<Frame::Inertial>(criterion, box, cache, mid_l,
                                             in_range_residual, mid_l);

  // Edge case - at minimum resolution, low residual
  test_criterion_evaluation<Frame::Inertial>(
      criterion, box, cache, min_resolution_l, 0.5 * min_residual,
      min_resolution_l);

  // Edge case - at maximum resolution, high residual
  test_criterion_evaluation<Frame::Inertial>(
      criterion, box, cache, max_resolution_l, 2.0 * max_residual,
      max_resolution_l);

  // Edge case - at minimum resolution, high residual
  test_criterion_evaluation<Frame::Inertial>(
      criterion, box, cache, min_resolution_l, 2.0 * max_residual,
      min_resolution_l + 1);

  // Edge case - at maximum resolution, low residual
  test_criterion_evaluation<Frame::Inertial>(
      criterion, box, cache, max_resolution_l, 0.5 * min_residual,
      max_resolution_l - 1);

  // Edge case - residual exactly at min_residual
  test_criterion_evaluation<Frame::Inertial>(criterion, box, cache, mid_l,
                                             min_residual, mid_l);

  // Edge case - residual exactly at max_residual
  test_criterion_evaluation<Frame::Inertial>(criterion, box, cache, mid_l,
                                             max_residual, mid_l);

  // Edge case - residual just below min_residual
  test_criterion_evaluation<Frame::Inertial>(criterion, box, cache, mid_l,
                                             min_residual - eps, mid_l - 1);

  // Edge case - residual just above min_residual
  test_criterion_evaluation<Frame::Inertial>(criterion, box, cache, mid_l,
                                             min_residual + eps, mid_l);

  // Edge case - residual just below max_residual
  test_criterion_evaluation<Frame::Inertial>(criterion, box, cache, mid_l,
                                             max_residual - eps, mid_l);

  // Edge case - residual just above max_residual
  test_criterion_evaluation<Frame::Inertial>(criterion, box, cache, mid_l,
                                             max_residual + eps, mid_l + 1);

  // Edge case - l_max = 2 (minimum allowed by Strahlkorper)
  test_criterion_evaluation<Frame::Inertial>(
      criterion, box, cache, min_l_allowed_by_strahlkorper, 0.5 * min_residual,
      min_l_allowed_by_strahlkorper);

  // Edge case - l_max = 2, high residual
  test_criterion_evaluation<Frame::Inertial>(
      criterion, box, cache, min_l_allowed_by_strahlkorper, 2.0 * max_residual,
      min_l_allowed_by_strahlkorper + 1);

  // Edge case - min_resolution_l == max_resolution_l
  const std::unique_ptr<ah::Criterion> fixed_criterion{
      std::make_unique<Residual>(min_residual, max_residual, mid_l)};
  Parallel::GlobalCache<Metavariables<Frame::Inertial>> mid_cache{
      tuples::TaggedTuple<ah::Tags::LMax>{mid_l}};
  test_criterion_evaluation<Frame::Inertial>(fixed_criterion, box, mid_cache,
                                             mid_l, 0.5 * min_residual, mid_l);

  // Edge case - min_resolution_l == max_resolution_l but
  // different from strahlkorper
#ifdef SPECTRE_DEBUG
  const ylm::Strahlkorper<Frame::Inertial> debug_strahlkorper{
      make_strahlkorper<Frame::Inertial>(mid_l - 2)};
  const FastFlow::IterInfo debug_iter_info{make_iter_info(0.5 * min_residual)};
  CHECK_THROWS_WITH(
      fixed_criterion->evaluate(box, mid_cache, debug_strahlkorper,
                                debug_iter_info),
      Catch::Matchers::ContainsSubstring("If MinResolutionL == LMax"));
#endif

  // Very large l_max
  const size_t very_large_l{100};
  test_criterion_evaluation<Frame::Inertial>(criterion, box, cache,
                                             very_large_l, 0.5 * min_residual,
                                             very_large_l - 1);

  // Very small residual
  const double very_small_residual{1.0e-12};
  test_criterion_evaluation<Frame::Inertial>(criterion, box, cache, mid_l,
                                             very_small_residual, mid_l - 1);

  // Very large residual
  const double very_large_residual{1.0};
  test_criterion_evaluation<Frame::Inertial>(criterion, box, cache, mid_l,
                                             very_large_residual, mid_l + 1);

  // Test with different frame
  test_criterion_evaluation<Frame::Grid>(criterion, box, cache, mid_l,
                                         0.5 * min_residual, mid_l - 1);

  // Test serialization
  {
    const Residual original{min_residual, max_residual, min_resolution_l};
    const auto serialized{serialize_and_deserialize(original)};

    // Test that the serialized object produces the same results
    const ylm::Strahlkorper<Frame::Inertial> strahlkorper{
        make_strahlkorper<Frame::Inertial>(mid_l)};
    const FastFlow::IterInfo iter_info{make_iter_info(in_range_residual)};

    const size_t recommended_l_original{
        original.operator()(cache, strahlkorper, iter_info)};
    const size_t recommended_l_serialized{
        serialized.operator()(cache, strahlkorper, iter_info)};
    CHECK(recommended_l_original == recommended_l_serialized);
  }

  const Residual criterion_one{1.0e-6, 1.0e-4, 4};
  const Residual criterion_two{1.0e-6, 1.0e-4, 6};
  const Residual criterion_one_same{1.0e-6, 1.0e-4, 4};
  CHECK(criterion_one.is_equal(criterion_one_same));
  CHECK(not(criterion_one.is_equal(criterion_two)));
  const auto criterion_one_serialized =
      serialize_and_deserialize(criterion_one);
  CHECK(criterion_one.is_equal(criterion_one_serialized));

  const std::unique_ptr<ah::Criterion> criterion_four =
      std::make_unique<ah::Criteria::Residual>(1.0e-6, 1.0e-4, 4);
  const std::unique_ptr<ah::Criterion> criterion_five =
      std::make_unique<ah::Criteria::Residual>(1.0e-6, 1.0e-4, 4);
  const std::unique_ptr<ah::Criterion> criterion_six =
      std::make_unique<ah::Criteria::IncreaseResolution>();
  CHECK(criterion_four->is_equal(*criterion_five));
  CHECK(not(criterion_four->is_equal(*criterion_six)));
}

void test_residual_constructor_validation() {
  // Define test constants
  constexpr double min_residual = 1.0e-6;
  constexpr double max_residual = 1.0e-4;
  constexpr double zero = 0.0;
  constexpr size_t min_resolution_l = 4;
  constexpr size_t min_allowed_resolution = 2;

  // Test valid construction
  CHECK_NOTHROW((Residual{min_residual, max_residual, min_resolution_l}));

  // Test invalid construction - min_residual >= max_residual
  CHECK_THROWS_WITH((Residual{max_residual, min_residual, min_resolution_l}),
                    Catch::Matchers::ContainsSubstring(
                        "MinResidual must be less than MaxResidual"));

  // Test invalid construction - min_residual == max_residual
  CHECK_THROWS_WITH((Residual{min_residual, min_residual, min_resolution_l}),
                    Catch::Matchers::ContainsSubstring(
                        "MinResidual must be less than MaxResidual"));

  // Test edge case - zero max_residual (should fail)
  CHECK_THROWS_WITH((Residual{min_residual, zero, min_resolution_l}),
                    Catch::Matchers::ContainsSubstring(
                        "MinResidual must be less than MaxResidual"));

  // Test edge case - min_resolution_l below minimum allowed value (2)
  CHECK_THROWS_WITH(
      (Residual{min_residual, max_residual, min_allowed_resolution - 1}),
      Catch::Matchers::ContainsSubstring(
          "MinResolutionL must not be less than 2"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.Criteria.Residual",
                  "[ApparentHorizonFinder][Unit]") {
  test_residual();
  test_residual_constructor_validation();
}
}  // namespace ah::Criteria
