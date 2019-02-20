// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "ApparentHorizons/FastFlow.hpp"
#include "ApparentHorizons/Strahlkorper.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Domain.hpp"
#include "Informer/Tags.hpp" // IWYU pragma: keep
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetApparentHorizon.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Spherepack.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/NumericalAlgorithms/Interpolation/InterpolationTargetTestHelpers.hpp"
#include "tests/Unit/TestCreation.hpp"

namespace {
struct MockMetavariables {
  struct InterpolationTargetA {
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
    using compute_target_points =
        ::intrp::Actions::ApparentHorizon<InterpolationTargetA,
                                          ::Frame::Inertial>;
    using type = compute_target_points::options_type;
  };
  using temporal_id = ::Tags::TimeId;
  using domain_frame = Frame::Inertial;
  static constexpr size_t domain_dim = 3;
  using interpolator_source_vars = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using interpolation_target_tags = tmpl::list<InterpolationTargetA>;

  using component_list =
      tmpl::list<InterpTargetTestHelpers::mock_interpolation_target<
                     MockMetavariables, InterpolationTargetA>,
                 InterpTargetTestHelpers::mock_interpolator<MockMetavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
};
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.InterpolationTarget.ApparentHorizon", "[Unit]") {
  // Constants used in this test.
  // We use l_max=12 to get enough points that the surface is
  // represented to roundoff error; for smaller l_max we would need to
  // modify InterpTargetTestHelpers::test_interpolation_target to
  // handle a custom `approx`.
  const size_t l_max = 12;
  const double radius = 2.0;
  const std::array<double, 3> center = {{0.05, 0.06, 0.07}};

  // Options for ApparentHorizon
  intrp::OptionHolders::ApparentHorizon<Frame::Inertial> apparent_horizon_opts(
      Strahlkorper<Frame::Inertial>{l_max, radius, center}, FastFlow{},
      Verbosity::Verbose);

  // Test creation of options
  const auto created_opts =
      test_creation<intrp::OptionHolders::ApparentHorizon<Frame::Inertial>>(
          "  FastFlow:\n"
          "  Verbosity: Verbose\n"
          "  InitialGuess:\n"
          "    Center: [0.05, 0.06, 0.07]\n"
          "    Radius: 2.0\n"
          "    Lmax: 12");
  CHECK(created_opts == apparent_horizon_opts);

  const auto domain_creator =
      domain::creators::Shell<Frame::Inertial>(1.8, 2.2, 1, {{5, 5}}, false);

  const auto expected_block_coord_holders =
      [&domain_creator, &center, &radius ]() noexcept {
    // How many points are supposed to be in a Strahlkorper,
    // reproduced here by hand for the test.
    const auto l_mesh = static_cast<size_t>(std::floor(1.5 * l_max));
    const size_t n_theta = l_mesh + 1;
    const size_t n_phi = 2 * l_mesh + 1;

    // The theta points of a Strahlkorper are Gauss-Legendre points.
    const std::vector<double> theta_points = [&n_theta]() noexcept {
      std::vector<double> thetas(n_theta);
      std::vector<double> work(n_theta + 1);
      std::vector<double> unused_weights(n_theta);
      int err = 0;
      gaqd_(static_cast<int>(n_theta), thetas.data(), unused_weights.data(),
            work.data(), static_cast<int>(n_theta + 1), &err);
      return thetas;
    }
    ();

    const double two_pi_over_n_phi = 2.0 * M_PI / n_phi;
    tnsr::I<DataVector, 3, Frame::Inertial> points(n_theta * n_phi);
    size_t s = 0;
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
    return block_logical_coordinates(domain_creator.create_domain(), points);
  }
  ();

  InterpTargetTestHelpers::test_interpolation_target<MockMetavariables>(
      domain_creator, std::move(apparent_horizon_opts),
      expected_block_coord_holders);
}
