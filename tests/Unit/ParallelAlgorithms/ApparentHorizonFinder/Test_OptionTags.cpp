// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <unordered_set>
#include <vector>

#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Domain.hpp"
#include "Framework/TestCreation.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/IO/InitialShapeFromFile.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/InitialShape.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Options/Auto.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Component.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Factory.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/IncreaseResolution.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Residual.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Destination.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/KerrSchild.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/OptionTags.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/HorizonMetavars.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/KerrHorizon.hpp"
#include "Time/Tags/TimeAndPrevious.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct MockHorizonMetavars : tt::ConformsTo<ah::protocols::HorizonMetavars> {
  using time_tag = ::Tags::TimeAndPrevious<0>;

  using frame = ::Frame::Grid;

  // Don't need callbacks
  using horizon_find_callbacks = tmpl::list<>;
  using horizon_find_failure_callbacks = tmpl::list<>;

  using compute_tags_on_element = tmpl::list<>;

  static constexpr ah::Destination destination = ah::Destination::ControlSystem;

  static std::string name() { return "MockHorizonMetavars"; }
};

struct MockMetavariables {
  static constexpr size_t volume_dim = 3;

  using component_list =
      tmpl::list<ah::Component<MockMetavariables, MockHorizonMetavars>>;
};

struct TestCreationMetavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<ah::Criterion, ah::Criteria::standard_criteria>,
        tmpl::pair<ylm::InitialShape<Frame::Grid>,
                   tmpl::list<ylm::InitialShapes::Sphere<Frame::Grid>,
                              ylm::InitialShapes::FromFile<Frame::Grid>,
                              ah::InitialShapes::KerrSchild<Frame::Grid>>>>;
  };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.OptionTags",
                  "[ApparentHorizonFinder][Unit]") {
  (void)MockHorizonMetavars::destination;
  domain::creators::register_derived_with_charm();

  // Constants used in this test.
  const size_t l_max = 12;
  const double radius = 2.0;
  const std::array<double, 3> center = {{0.05, 0.06, 0.07}};

  // Options for ApparentHorizon
  std::vector<std::unique_ptr<ah::Criterion>> criteria;
  criteria.emplace_back(
      std::make_unique<ah::Criteria::Residual>(1.e-12, 1.e-2, 2));
  ah::HorizonOptions<::Frame::Grid> apparent_horizon_opts(
      std::move(criteria),
      ylm::Strahlkorper<Frame::Grid>{l_max, radius, center}, FastFlow{},
      Verbosity::Verbose, 3_st, std::nullopt);

  // Test creation of options
  const auto created_opts =
      TestHelpers::test_creation<ah::HorizonOptions<Frame::Grid>,
                                 TestCreationMetavariables>(
          "Criteria:\n"
          "  - Residual:\n"
          "      MinResidual: 1.e-12\n"
          "      MaxResidual: 1.e-2\n"
          "      MinResolutionL: 2\n"
          "FastFlow:\n"
          "  Flow: Fast\n"
          "  Alpha: 1.0\n"
          "  Beta: 0.5\n"
          "  AbsTol: 1e-12\n"
          "  TruncationTol: 1e-2\n"
          "  DivergenceTol: 1.2\n"
          "  DivergenceIter: 5\n"
          "  MaxIts: 100\n"
          "Verbosity: Verbose\n"
          "InitialGuess:\n"
          "  InitialL: 12\n"
          "  InitialShape:\n"
          "    Sphere:\n"
          "      Center: [0.05, 0.06, 0.07]\n"
          "      Radius: 2.0\n"
          "MaxComputeCoordsRetries: 3\n"
          "BlocksForHorizonFind: All");
  CHECK(created_opts == apparent_horizon_opts);

  const auto domain_creator = domain::creators::Sphere(
      1.8, 2.2, domain::creators::Sphere::Excision{}, 1_st, 5_st, false);

  {
    const auto blocks_for_horizon_find =
        ah::Tags::BlocksForHorizonFind::create_from_options<MockMetavariables>(
            std::make_unique<domain::creators::Sphere>(
                1.8, 2.2, domain::creators::Sphere::Excision{}, 1_st, 5_st,
                false),
            created_opts);
    REQUIRE(blocks_for_horizon_find.contains("MockHorizonMetavars"));
    const auto block_names = domain_creator.block_names();
    CHECK(blocks_for_horizon_find.at("MockHorizonMetavars") ==
          std::unordered_set<std::string>{block_names.begin(),
                                          block_names.end()});
  }
  {
    const auto kerr_schild_shape = TestHelpers::test_factory_creation<
        ylm::InitialShape<Frame::Grid>,
        ah::InitialShapes::KerrSchild<Frame::Grid>>(
        "KerrSchild:\n"
        "  Center: [0.1, -0.2, 0.3]\n"
        "  Mass: 0.5\n"
        "  Spin: [0.0, 0.0, 0.7]\n");
    const size_t kerr_schild_l_max = 16;
    const std::array<double, 3> kerr_schild_center{{0.1, -0.2, 0.3}};
    const double mass = 0.5;
    const std::array<double, 3> dimensionless_spin{{0.0, 0.0, 0.7}};
    const ylm::Spherepack ylm{kerr_schild_l_max, kerr_schild_l_max};
    const ylm::Strahlkorper<Frame::Grid> expected_kerr_schild_horizon{
        kerr_schild_l_max, kerr_schild_l_max,
        get(gr::Solutions::kerr_horizon_radius(ylm.theta_phi_points(), mass,
                                               dimensionless_spin)),
        kerr_schild_center};
    CHECK(kerr_schild_shape->strahlkorper(kerr_schild_l_max, {}) ==
          expected_kerr_schild_horizon);

    const auto kerr_schild_created_opts =
        TestHelpers::test_creation<ah::HorizonOptions<Frame::Grid>,
                                   TestCreationMetavariables>(
            "Criteria:\n"
            "FastFlow:\n"
            "  Flow: Fast\n"
            "  Alpha: 1.0\n"
            "  Beta: 0.5\n"
            "  AbsTol: 1e-12\n"
            "  TruncationTol: 1e-2\n"
            "  DivergenceTol: 1.2\n"
            "  DivergenceIter: 5\n"
            "  MaxIts: 100\n"
            "Verbosity: Verbose\n"
            "InitialGuess:\n"
            "  InitialL: 16\n"
            "  InitialShape:\n"
            "    KerrSchild:\n"
            "      Center: [0.1, -0.2, 0.3]\n"
            "      Mass: 0.5\n"
            "      Spin: [0.0, 0.0, 0.7]\n"
            "MaxComputeCoordsRetries: 3\n"
            "BlocksForHorizonFind: All");

    CHECK(kerr_schild_created_opts.initial_guess.l_max() == kerr_schild_l_max);
    CHECK(kerr_schild_created_opts.initial_guess.expansion_center() ==
          kerr_schild_center);
    CHECK_ITERABLE_APPROX(kerr_schild_created_opts.initial_guess.coefficients(),
                          expected_kerr_schild_horizon.coefficients());
  }
  {
    const auto new_created_opts =
        TestHelpers::test_creation<ah::HorizonOptions<Frame::Grid>,
                                   TestCreationMetavariables>(
            "Criteria:\n"
            "FastFlow:\n"
            "  Flow: Fast\n"
            "  Alpha: 1.0\n"
            "  Beta: 0.5\n"
            "  AbsTol: 1e-12\n"
            "  TruncationTol: 1e-2\n"
            "  DivergenceTol: 1.2\n"
            "  DivergenceIter: 5\n"
            "  MaxIts: 100\n"
            "Verbosity: Verbose\n"
            "InitialGuess:\n"
            "  InitialL: 12\n"
            "  InitialShape:\n"
            "    Sphere:\n"
            "      Center: [0.05, 0.06, 0.07]\n"
            "      Radius: 2.0\n"
            "MaxComputeCoordsRetries: 3\n"
            "BlocksForHorizonFind: [Shell0]");
    const auto blocks_for_horizon_find =
        ah::Tags::BlocksForHorizonFind::create_from_options<MockMetavariables>(
            std::make_unique<domain::creators::Sphere>(
                1.8, 2.2, domain::creators::Sphere::Excision{}, 1_st, 5_st,
                false, std::nullopt, std::vector{2.0}),
            new_created_opts);
    REQUIRE(blocks_for_horizon_find.contains("MockHorizonMetavars"));
    const auto block_names = domain_creator.block_names();
    CHECK(blocks_for_horizon_find.at("MockHorizonMetavars") ==
          std::unordered_set<std::string>{"Shell0UpperZ", "Shell0LowerZ",
                                          "Shell0UpperY", "Shell0LowerY",
                                          "Shell0UpperX", "Shell0LowerX"});
  }

  // Test LMax option tag
  {
    constexpr size_t max_l_20 = 20_st;
    const auto tag_from_20 = ah::Tags::LMax::create_from_options(max_l_20);
    CHECK(tag_from_20 == max_l_20);
  }
}
