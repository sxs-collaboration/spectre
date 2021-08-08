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
#include "Helpers/NumericalAlgorithms/Interpolation/InterpolationTargetTestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetKerrHorizon.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Spherepack.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct MockMetavariables {
  struct InterpolationTargetA {
    using temporal_id = ::Tags::TimeStepId;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
    using compute_items_on_target = tmpl::list<>;
    using compute_target_points =
        ::intrp::TargetPoints::KerrHorizon<InterpolationTargetA,
                                           ::Frame::Inertial>;
  };
  static constexpr size_t volume_dim = 3;
  using interpolator_source_vars = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using interpolation_target_tags = tmpl::list<InterpolationTargetA>;

  using component_list =
      tmpl::list<InterpTargetTestHelpers::mock_interpolation_target<
                     MockMetavariables, InterpolationTargetA>,
                 InterpTargetTestHelpers::mock_interpolator<MockMetavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};
}  // namespace

void test_interpolation_target_kerr_horizon(
    const bool theta_varies_fastest) noexcept {
  // Constants used in this test.
  // We use l_max=18 to get enough points that the surface is
  // represented to roundoff error; for smaller l_max we would need to
  // modify InterpTargetTestHelpers::test_interpolation_target to
  // handle a custom `approx`.
  const size_t l_max = 18;
  const double mass = 1.8;
  const std::array<double, 3> center = {{0.05, 0.06, 0.07}};
  const std::array<double, 3> dimless_spin = {{0.2, 0.3, 0.4}};

  // Options for KerrHorizon
  intrp::OptionHolders::KerrHorizon kerr_horizon_opts(
      l_max, center, mass, dimless_spin, theta_varies_fastest);

  // Test creation of options
  const auto created_opts =
      TestHelpers::test_creation<intrp::OptionHolders::KerrHorizon>(
          "Center: [0.05, 0.06, 0.07]\n"
          "DimensionlessSpin: [0.2, 0.3, 0.4]\n"
          "Lmax: 18\n"
          "Mass: 1.8\n"
          "ThetaVariesFastest: " +
          std::string(theta_varies_fastest ? "true" : "false"));
  CHECK(created_opts == kerr_horizon_opts);

  const auto domain_creator =
      domain::creators::Shell(0.9, 4.9, 1, {{5, 5}}, false);

  const auto expected_block_coord_holders = [&domain_creator, &mass, &center,
                                             &dimless_spin,
                                             &theta_varies_fastest]() noexcept {
    // How many points are supposed to be in a Strahlkorper,
    // reproduced here by hand for the test.
    const size_t n_theta = l_max + 1;
    const size_t n_phi = 2 * l_max + 1;

    // The theta points of a Strahlkorper are Gauss-Legendre points.
    const std::vector<double> theta_points = []() noexcept {
      std::vector<double> thetas(n_theta);
      std::vector<double> work(n_theta + 1);
      std::vector<double> unused_weights(n_theta);
      int err = 0;
      gaqd_(static_cast<int>(n_theta), thetas.data(), unused_weights.data(),
            work.data(), static_cast<int>(n_theta + 1), &err);
      return thetas;
    }();

    // Radius as function of theta, phi
    const auto radius = [&mass, &dimless_spin](const double theta,
                                               const double phi) noexcept {
      // Recoding kerr_horizon_radius in a different way for the test.
      const std::array<double, 3> spin_a = {{mass * dimless_spin[0],
                                             mass * dimless_spin[1],
                                             mass * dimless_spin[2]}};
      const double spin_a_squared =
          square(spin_a[0]) + square(spin_a[1]) + square(spin_a[2]);
      const double a_dot_xhat_squared =
          square(spin_a[0] * sin(theta) * cos(phi) +
                 spin_a[1] * sin(theta) * sin(phi) + spin_a[2] * cos(theta));
      const double r_boyer_lindquist_squared =
          square(mass + sqrt(square(mass) - spin_a_squared));
      return sqrt((r_boyer_lindquist_squared + spin_a_squared) /
                  (1.0 + a_dot_xhat_squared / r_boyer_lindquist_squared));
    };

    const double two_pi_over_n_phi = 2.0 * M_PI / n_phi;
    tnsr::I<DataVector, 3, Frame::Inertial> points(n_theta * n_phi);
    size_t s = 0;
    if (theta_varies_fastest) {
      for (size_t i_phi = 0; i_phi < n_phi; ++i_phi) {
        const double phi = two_pi_over_n_phi * i_phi;
        for (size_t i_theta = 0; i_theta < n_theta; ++i_theta) {
          const double theta = theta_points[i_theta];
          const double r = radius(theta, phi);
          points.get(0)[s] = r * sin(theta) * cos(phi) + center[0];
          points.get(1)[s] = r * sin(theta) * sin(phi) + center[1],
          points.get(2)[s] = r * cos(theta) + center[2];
          ++s;
        }
      }
    } else {
      for (size_t i_theta = 0; i_theta < n_theta; ++i_theta) {
        for (size_t i_phi = 0; i_phi < n_phi; ++i_phi) {
          const double phi = two_pi_over_n_phi * i_phi;
          const double theta = theta_points[i_theta];
          const double r = radius(theta, phi);
          points.get(0)[s] = r * sin(theta) * cos(phi) + center[0];
          points.get(1)[s] = r * sin(theta) * sin(phi) + center[1],
          points.get(2)[s] = r * cos(theta) + center[2];
          ++s;
        }
      }
    }
    return block_logical_coordinates(domain_creator.create_domain(), points);
  }();

  TestHelpers::db::test_simple_tag<
      intrp::Tags::KerrHorizon<MockMetavariables::InterpolationTargetA>>(
      "KerrHorizon");

  InterpTargetTestHelpers::test_interpolation_target<
      MockMetavariables,
      intrp::Tags::KerrHorizon<MockMetavariables::InterpolationTargetA>>(
      domain_creator, kerr_horizon_opts, expected_block_coord_holders);
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.InterpolationTarget.KerrHorizon",
                  "[Unit]") {
  domain::creators::register_derived_with_charm();
  test_interpolation_target_kerr_horizon(true);
  test_interpolation_target_kerr_horizon(false);
}
