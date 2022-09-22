// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Domain.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/ParallelAlgorithms/Interpolation/InterpolationTargetTestHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/AngularOrdering.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/Sphere.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
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
}  // namespace

void test_interpolation_target_sphere(
    const intrp::AngularOrdering angular_ordering) {
  const size_t l_max = 18;
  const double radius = 3.6;
  const std::array<double, 3> center = {{0.05, 0.06, 0.07}};

  // Options for Sphere
  intrp::OptionHolders::Sphere sphere_opts(l_max, center, radius,
                                           angular_ordering);

  // Test creation of options
  const auto created_opts =
      TestHelpers::test_creation<intrp::OptionHolders::Sphere>(
          "Center: [0.05, 0.06, 0.07]\n"
          "Radius: 3.6\n"
          "Lmax: 18\n"
          "AngularOrdering: " +
          std::string(MakeString{} << angular_ordering));
  CHECK(created_opts == sphere_opts);

  const auto domain_creator =
      domain::creators::Shell(0.9, 4.9, 1, {{5, 5}}, false);

  TestHelpers::db::test_simple_tag<
      intrp::Tags::Sphere<MockMetavariables::InterpolationTargetA>>("Sphere");

  const auto expected_block_coord_holders = [&domain_creator, &radius, &center,
                                             &angular_ordering]() {
    // How many points are supposed to be in a Strahlkorper,
    // reproduced here by hand for the test.
    const size_t n_theta = l_max + 1;
    const size_t n_phi = 2 * l_max + 1;

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
    tnsr::I<DataVector, 3, Frame::Inertial> points(n_theta * n_phi);
    size_t s = 0;
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
    return block_logical_coordinates(domain_creator.create_domain(), points);
  }();
  InterpTargetTestHelpers::test_interpolation_target<
      MockMetavariables,
      intrp::Tags::Sphere<MockMetavariables::InterpolationTargetA>>(
      domain_creator, sphere_opts, expected_block_coord_holders);
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.InterpolationTarget.Sphere",
                  "[Unit]") {
  domain::creators::register_derived_with_charm();
  test_interpolation_target_sphere(intrp::AngularOrdering::Cce);
  test_interpolation_target_sphere(intrp::AngularOrdering::Strahlkorper);
}
