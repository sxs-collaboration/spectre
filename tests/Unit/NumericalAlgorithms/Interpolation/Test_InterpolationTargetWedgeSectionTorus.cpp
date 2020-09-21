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
#include "Domain/Creators/Shell.hpp"
#include "Domain/Domain.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/Interpolation/InterpolationTargetTestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetWedgeSectionTorus.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct MockMetavariables {
  struct InterpolationTargetA {
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
    using compute_items_on_target = tmpl::list<>;
    using compute_target_points =
        ::intrp::TargetPoints::WedgeSectionTorus<InterpolationTargetA>;
  };
  using temporal_id = ::Tags::TimeStepId;
  static constexpr size_t volume_dim = 3;
  using interpolator_source_vars = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using interpolation_target_tags = tmpl::list<InterpolationTargetA>;

  using component_list =
      tmpl::list<InterpTargetTestHelpers::mock_interpolation_target<
                     MockMetavariables, InterpolationTargetA>,
                 InterpTargetTestHelpers::mock_interpolator<MockMetavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

void test_r_theta_lgl() noexcept {
  const size_t num_radial = 3;
  const size_t num_theta = 4;
  const size_t num_phi = 5;
  // Test with a torus that is not symmetric above/below the equator.
  intrp::OptionHolders::WedgeSectionTorus wedge_section_torus_opts(
      1.2, 4.0, 0.35 * M_PI, 0.55 * M_PI, num_radial, num_theta, num_phi, false,
      false);

  const auto domain_creator =
      domain::creators::Shell(0.9, 4.9, 1, {{5, 5}}, false);

  const size_t num_total = num_radial * num_theta * num_phi;
  const auto expected_block_coord_holders = [&domain_creator,
                                             &num_total]() noexcept {
    tnsr::I<DataVector, 3, Frame::Inertial> points(num_total);
    for (size_t r = 0; r < num_radial; ++r) {
      const double radius =
          2.6 +
          1.4 *
              Spectral::collocation_points<Spectral::Basis::Legendre,
                                           Spectral::Quadrature::GaussLobatto>(
                  num_radial)[r];
      for (size_t t = 0; t < num_theta; ++t) {
        const double theta =
            M_PI * (0.45 + 0.1 * Spectral::collocation_points<
                                     Spectral::Basis::Legendre,
                                     Spectral::Quadrature::GaussLobatto>(
                                     num_theta)[t]);
        for (size_t p = 0; p < num_phi; ++p) {
          const double phi = 2.0 * M_PI * p / num_phi;
          const size_t i = r + t * num_radial + p * num_theta * num_radial;
          get<0>(points)[i] = radius * sin(theta) * cos(phi);
          get<1>(points)[i] = radius * sin(theta) * sin(phi);
          get<2>(points)[i] = radius * cos(theta);
        }
      }
    }
    return block_logical_coordinates(domain_creator.create_domain(), points);
  }();

  InterpTargetTestHelpers::test_interpolation_target<
      MockMetavariables,
      intrp::Tags::WedgeSectionTorus<MockMetavariables::InterpolationTargetA>>(
      domain_creator, wedge_section_torus_opts,
      expected_block_coord_holders);
}

void test_r_theta_uniform() noexcept {
  const size_t num_radial = 4;
  const size_t num_theta = 5;
  const size_t num_phi = 6;
  // Test with a torus that is symmetric above/below the equator.
  intrp::OptionHolders::WedgeSectionTorus wedge_section_torus_opts(
      1.8, 3.6, 0.25 * M_PI, 0.75 * M_PI, num_radial, num_theta, num_phi, true,
      true);

  const auto domain_creator =
      domain::creators::Shell(0.9, 4.9, 1, {{5, 5}}, false);

  const size_t num_total = num_radial * num_theta * num_phi;
  const auto expected_block_coord_holders = [&domain_creator,
                                             &num_total]() noexcept {
    tnsr::I<DataVector, 3, Frame::Inertial> points(num_total);
    for (size_t r = 0; r < num_radial; ++r) {
      const double radius = 1.8 + 1.8 * r / (num_radial - 1.0);
      for (size_t t = 0; t < num_theta; ++t) {
        const double theta = M_PI * (0.25 + 0.5 * t / (num_theta - 1.0));
        for (size_t p = 0; p < num_phi; ++p) {
          const double phi = 2.0 * M_PI * p / num_phi;
          const size_t i = r + t * num_radial + p * num_theta * num_radial;
          get<0>(points)[i] = radius * sin(theta) * cos(phi);
          get<1>(points)[i] = radius * sin(theta) * sin(phi);
          get<2>(points)[i] = radius * cos(theta);
        }
      }
    }
    return block_logical_coordinates(domain_creator.create_domain(), points);
  }();

  TestHelpers::db::test_simple_tag<
      intrp::Tags::WedgeSectionTorus<MockMetavariables::InterpolationTargetA>>(
      "WedgeSectionTorus");

  InterpTargetTestHelpers::test_interpolation_target<
      MockMetavariables,
      intrp::Tags::WedgeSectionTorus<MockMetavariables::InterpolationTargetA>>(
      domain_creator, wedge_section_torus_opts,
      expected_block_coord_holders);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.InterpolationTarget.WedgeSectionTorus",
    "[Unit]") {
  // Check creating the options
  const auto created_torus =
      TestHelpers::test_creation<intrp::OptionHolders::WedgeSectionTorus>(
          "MinRadius: 1.8\n"
          "MaxRadius: 20.\n"
          "MinTheta: 0.785\n"
          "MaxTheta: 2.356\n"
          "NumberRadialPoints: 20\n"
          "NumberThetaPoints: 10\n"
          "NumberPhiPoints: 20\n"
          "UniformThetaGrid: true\n");
  CHECK(created_torus == intrp::OptionHolders::WedgeSectionTorus(
                             1.8, 20., 0.785, 2.356, 20, 10, 20, false, true));

  // Check computing the points
  test_r_theta_lgl();
  test_r_theta_uniform();
}
