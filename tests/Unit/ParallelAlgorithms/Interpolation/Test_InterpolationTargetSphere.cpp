// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Domain.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/ParallelAlgorithms/Interpolation/InterpolationTargetTestHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/AngularOrdering.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/Sphere.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Spherepack.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct MockMetavariables {
  struct InterpolationTargetA
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::TimeStepId;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
    using compute_items_on_target = tmpl::list<>;
    using compute_target_points =
        ::intrp::TargetPoints::Sphere<InterpolationTargetA, ::Frame::Inertial>;
    using post_interpolation_callback =
        intrp::callbacks::ObserveTimeSeriesOnSurface<tmpl::list<>,
                                                     InterpolationTargetA>;
  };
  static constexpr size_t volume_dim = 3;
  using interpolator_source_vars = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using interpolation_target_tags = tmpl::list<InterpolationTargetA>;

  using component_list =
      tmpl::list<InterpTargetTestHelpers::mock_interpolation_target<
                     MockMetavariables, InterpolationTargetA>,
                 InterpTargetTestHelpers::mock_interpolator<MockMetavariables>>;
};

template <typename Generator>
void test_interpolation_target_sphere(
    const gsl::not_null<Generator*> generator, const size_t number_of_spheres,
    const intrp::AngularOrdering angular_ordering) {
  // Keep bounds a bit inside than inner and outer radius of shell below so the
  // offset-sphere is still within the domain
  std::uniform_real_distribution<double> dist{1.2, 4.5};
  std::vector<double> radii(number_of_spheres);
  for (size_t i = 0; i < number_of_spheres; i++) {
    double radius = dist(*generator);
    while (alg::find(radii, radius) != radii.end()) {
      radius = dist(*generator);
    }
    radii[i] = radius;
  }
  const size_t l_max = 18;
  const std::array<double, 3> center = {{0.05, 0.06, 0.07}};

  CAPTURE(l_max);
  CAPTURE(center);
  CAPTURE(radii);
  CAPTURE(number_of_spheres);
  CAPTURE(angular_ordering);

  // Options for Sphere
  std::string radii_str;
  intrp::OptionHolders::Sphere sphere_opts;
  std::stringstream ss;
  ss << std::setprecision(std::numeric_limits<double>::max_digits10);
  if (number_of_spheres == 1) {
    // Test the double variant
    sphere_opts =
        intrp::OptionHolders::Sphere(l_max, center, radii[0], angular_ordering);
    ss << radii[0];
  } else {
    // Test the vector variant
    sphere_opts =
        intrp::OptionHolders::Sphere(l_max, center, radii, angular_ordering);
    ss << "[" << radii[0];
    for (size_t i = 1; i < number_of_spheres; i++) {
      ss << "," << radii[i];
    }
    ss << "]";
  }
  radii_str = ss.str();

  // Test creation of options
  const auto created_opts =
      TestHelpers::test_creation<intrp::OptionHolders::Sphere>(
          "Center: [0.05, 0.06, 0.07]\n"
          "Radius: " +
          radii_str +
          "\n"
          "LMax: 18\n"
          "AngularOrdering: " +
          std::string(MakeString{} << angular_ordering));
  CHECK(created_opts == sphere_opts);

  const auto domain_creator = domain::creators::Sphere(
      0.9, 4.9, domain::creators::Sphere::Excision{}, 1_st, 5_st, false);

  TestHelpers::db::test_simple_tag<
      intrp::Tags::Sphere<MockMetavariables::InterpolationTargetA>>("Sphere");

  const auto expected_block_coord_holders = [&domain_creator, &radii, &center,
                                             &angular_ordering,
                                             &number_of_spheres]() {
    // How many points are supposed to be in a Strahlkorper,
    // reproduced here by hand for the test.
    const size_t n_theta = l_max + 1;
    const size_t n_phi = 2 * l_max + 1;

    // Have to turn this into a set to guarantee ordering
    const std::set<double> radii_set(radii.begin(), radii.end());

    tnsr::I<DataVector, 3, Frame::Inertial> points(number_of_spheres * n_theta *
                                                   n_phi);

    size_t s = 0;
    for (const double radius : radii_set) {
      // The theta points of a Strahlkorper are Gauss-Legendre points.
      const std::vector<double> theta_points = []() {
        std::vector<double> thetas(n_theta);
        std::vector<double> work(n_theta + 1);
        std::vector<double> unused_weights(n_theta);
        int err = 0;
        gaqd_(static_cast<int>(n_theta), thetas.data(), unused_weights.data(),
              work.data(), static_cast<int>(n_theta + 1), &err);
        return thetas;
      }();

      const double two_pi_over_n_phi = 2.0 * M_PI / n_phi;
      if (angular_ordering == intrp::AngularOrdering::Strahlkorper) {
        for (size_t i_phi = 0; i_phi < n_phi; ++i_phi) {
          const double phi = two_pi_over_n_phi * i_phi;
          for (size_t i_theta = 0; i_theta < n_theta; ++i_theta) {
            const double theta = theta_points[i_theta];
            points.get(0)[s] = radius * sin(theta) * cos(phi) + center[0];
            points.get(1)[s] = radius * sin(theta) * sin(phi) + center[1],
            points.get(2)[s] = radius * cos(theta) + center[2];
            ++s;
          }
        }
      } else {
        for (size_t i_theta = 0; i_theta < n_theta; ++i_theta) {
          for (size_t i_phi = 0; i_phi < n_phi; ++i_phi) {
            const double phi = two_pi_over_n_phi * i_phi;
            const double theta = theta_points[i_theta];
            points.get(0)[s] = radius * sin(theta) * cos(phi) + center[0];
            points.get(1)[s] = radius * sin(theta) * sin(phi) + center[1],
            points.get(2)[s] = radius * cos(theta) + center[2];
            ++s;
          }
        }
      }
    }
    return block_logical_coordinates(domain_creator.create_domain(), points);
  }();
  InterpTargetTestHelpers::test_interpolation_target<
      MockMetavariables,
      intrp::Tags::Sphere<MockMetavariables::InterpolationTargetA>>(
      domain_creator, sphere_opts, expected_block_coord_holders);
}

void test_sphere_errors() {
  CHECK_THROWS_WITH(
      ([]() {
        const auto created_opts =
            TestHelpers::test_creation<intrp::OptionHolders::Sphere>(
                "Center: [0.05, 0.06, 0.07]\n"
                "Radius: [1.0, 1.0]\n"
                "LMax: 18\n"
                "AngularOrdering: Cce");
      })(),
      Catch::Contains("into radii for Sphere interpolation target. It already "
                      "exists. Existing radii are"));
  CHECK_THROWS_WITH(
      ([]() {
        const auto created_opts =
            TestHelpers::test_creation<intrp::OptionHolders::Sphere>(
                "Center: [0.05, 0.06, 0.07]\n"
                "Radius: [-1.0]\n"
                "LMax: 18\n"
                "AngularOrdering: Cce");
      })(),
      Catch::Contains("Radius must be positive"));
  CHECK_THROWS_WITH(
      ([]() {
        const auto created_opts =
            TestHelpers::test_creation<intrp::OptionHolders::Sphere>(
                "Center: [0.05, 0.06, 0.07]\n"
                "Radius: -1.0\n"
                "LMax: 18\n"
                "AngularOrdering: Cce");
      })(),
      Catch::Contains("Radius must be positive"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.InterpolationTarget.Sphere",
                  "[Unit]") {
  domain::creators::register_derived_with_charm();
  test_sphere_errors();
  MAKE_GENERATOR(gen);
  for (size_t num_spheres : {1_st, 2_st, 3_st}) {
    test_interpolation_target_sphere(make_not_null(&gen), num_spheres,
                                     intrp::AngularOrdering::Cce);
    test_interpolation_target_sphere(make_not_null(&gen), num_spheres,
                                     intrp::AngularOrdering::Strahlkorper);
  }
}
