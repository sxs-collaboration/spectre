// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Domain.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/Interpolation/InterpolationTargetTestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetSpecifiedPoints.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim>
struct MockMetavariables {
  struct InterpolationTargetA {
    using temporal_id = ::Tags::TimeStepId;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
    using compute_items_on_target = tmpl::list<>;
    using compute_target_points =
        ::intrp::TargetPoints::SpecifiedPoints<InterpolationTargetA, Dim>;
  };
  static constexpr size_t volume_dim = Dim;
  using interpolator_source_vars = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using interpolation_target_tags = tmpl::list<InterpolationTargetA>;

  using component_list = tmpl::list<
      InterpTargetTestHelpers::mock_interpolation_target<MockMetavariables<Dim>,
                                                         InterpolationTargetA>,
      InterpTargetTestHelpers::mock_interpolator<MockMetavariables<Dim>>>;
  enum class Phase { Initialization, Testing, Exit };
};

void test_1d() {
  // Options for SpecifiedPoints
  intrp::OptionHolders::SpecifiedPoints<1> points_opts(
      std::vector<std::array<double, 1>>{
          {std::array<double, 1>{{1.0}}, std::array<double, 1>{{0.3}}}});

  // Test creation of options
  const auto created_opts =
      TestHelpers::test_creation<intrp::OptionHolders::SpecifiedPoints<1>>(
          "Points: [[1.0], [0.3]]");
  CHECK(created_opts == points_opts);

  const auto domain_creator = domain::creators::Interval(
      {{-1.0}}, {{1.0}}, {{1}}, {{3}}, {{false}}, nullptr);

  const auto expected_block_coord_holders = [&domain_creator]() noexcept {
    tnsr::I<DataVector, 1, Frame::Inertial> points;
    get<0>(points) = DataVector({{1.0, 0.3}});
    return block_logical_coordinates(domain_creator.create_domain(), points);
  }();

  TestHelpers::db::test_simple_tag<intrp::Tags::SpecifiedPoints<
      MockMetavariables<1>::InterpolationTargetA, 1>>("SpecifiedPoints");

  InterpTargetTestHelpers::test_interpolation_target<
      MockMetavariables<1>, intrp::Tags::SpecifiedPoints<
                                MockMetavariables<1>::InterpolationTargetA, 1>>(
      domain_creator, std::move(points_opts), expected_block_coord_holders);
}

void test_2d() {
  // Options for SpecifiedPoints
  intrp::OptionHolders::SpecifiedPoints<2> points_opts(
      std::vector<std::array<double, 2>>{{std::array<double, 2>{{0.0, 1.0}},
                                          std::array<double, 2>{{-0.2, 0.1}},
                                          std::array<double, 2>{{0.3, 1.9}}}});

  // Test creation of options
  const auto created_opts =
      TestHelpers::test_creation<intrp::OptionHolders::SpecifiedPoints<2>>(
          "Points: [[0.0, 1.0], [-0.2, 0.1], [0.3, 1.9]]");
  CHECK(created_opts == points_opts);

  const auto domain_creator = domain::creators::Rectangle(
      {{-1.0, -1.0}}, {{1.0, 2.0}}, {{1, 1}}, {{3, 4}}, {{false, false}});

  const auto expected_block_coord_holders = [&domain_creator]() noexcept {
    tnsr::I<DataVector, 2, Frame::Inertial> points;
    get<0>(points) = DataVector({{0.0, -0.2, 0.3}});
    get<1>(points) = DataVector({{1.0, 0.1, 1.9}});
    return block_logical_coordinates(domain_creator.create_domain(), points);
  }();

  TestHelpers::db::test_simple_tag<intrp::Tags::SpecifiedPoints<
      MockMetavariables<2>::InterpolationTargetA, 2>>("SpecifiedPoints");

  InterpTargetTestHelpers::test_interpolation_target<
      MockMetavariables<2>, intrp::Tags::SpecifiedPoints<
                                MockMetavariables<2>::InterpolationTargetA, 2>>(
      domain_creator, std::move(points_opts), expected_block_coord_holders);
}

void test_3d() {
  // Options for SpecifiedPoints
  intrp::OptionHolders::SpecifiedPoints<3> points_opts(
      std::vector<std::array<double, 3>>{
          {std::array<double, 3>{{0.0, 0.0, 0.0}},
           std::array<double, 3>{{1.0, -0.3, 0.2}},
           std::array<double, 3>{{-0.8, 1.6, 2.4}},
           std::array<double, 3>{{0.0, 1.0, 0.0}}}});

  // Test creation of options
  const auto created_opts =
      TestHelpers::test_creation<intrp::OptionHolders::SpecifiedPoints<3>>(
          "Points: [[0.0, 0.0, 0.0], [1.0, -0.3, 0.2], "
          "[-0.8, 1.6, 2.4], [0.0, 1.0, 0.0]]");
  CHECK(created_opts == points_opts);

  const auto domain_creator = domain::creators::Brick(
      {{-1.0, -1.0, -1.0}}, {{1.0, 2.0, 3.0}}, {{1, 1, 1}}, {{3, 4, 5}},
      {{false, false, false}});

  const auto expected_block_coord_holders = [&domain_creator]() noexcept {
    tnsr::I<DataVector, 3, Frame::Inertial> points;
    get<0>(points) = DataVector({{0.0, 1.0, -0.8, 0.0}});
    get<1>(points) = DataVector({{0.0, -0.3, 1.6, 1.0}});
    get<2>(points) = DataVector({{0.0, 0.2, 2.4, 0.0}});
    return block_logical_coordinates(domain_creator.create_domain(), points);
  }();

  TestHelpers::db::test_simple_tag<intrp::Tags::SpecifiedPoints<
      MockMetavariables<3>::InterpolationTargetA, 3>>("SpecifiedPoints");

  InterpTargetTestHelpers::test_interpolation_target<
      MockMetavariables<3>, intrp::Tags::SpecifiedPoints<
                                MockMetavariables<3>::InterpolationTargetA, 3>>(
      domain_creator, std::move(points_opts), expected_block_coord_holders);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.InterpolationTarget.SpecifiedPoints", "[Unit]") {
  domain::creators::register_derived_with_charm();

  test_1d();
  test_2d();
  test_3d();
}
