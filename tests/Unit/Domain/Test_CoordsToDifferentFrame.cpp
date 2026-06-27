// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <memory>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordsToDifferentFrame.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/Shape.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/StrahlkorperTransformations.hpp"
#include "NumericalAlgorithms/Strahlkorper/StrahlkorperFunctions.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <typename SrcFrame, typename DestFrame>
void test_coords_to_different_frame() {
  const tnsr::I<DataVector, 3, SrcFrame> src_center{DataVector{0.03}};

  const std::vector<double> radial_partitioning{};
  const std::vector<domain::CoordinateMaps::Distribution> radial_distribution{
      domain::CoordinateMaps::Distribution::Linear};

  const domain::creators::Sphere domain_creator{
      0.001,
      10.0,
      domain::creators::Sphere::Excision{},
      1_st,
      5_st,
      false,
      std::nullopt,
      radial_partitioning,
      radial_distribution,
      ShellWedges::All,
      std::make_unique<
          domain::creators::time_dependence::UniformTranslation<3>>(
          0.0, std::array<double, 3>({{0.0, 0.0, 0.0}}),
          std::array<double, 3>({{0.01, 0.02, 0.03}}))};

  const Domain<3> domain = domain_creator.create_domain();
  const auto functions_of_time = domain_creator.functions_of_time();

  const double time = 0.5;
  tnsr::I<DataVector, 3, DestFrame> dest_center{DataVector{0.0}};

  coords_to_different_frame(make_not_null(&dest_center), src_center, domain,
                            functions_of_time, time);

  tnsr::I<DataVector, 3, DestFrame> expected_center{DataVector{0.0}};

  if constexpr (std::is_same_v<SrcFrame, ::Frame::Inertial>) {
    expected_center[0] = src_center[0] - 0.005;
    expected_center[1] = src_center[1] - 0.01;
    expected_center[2] = src_center[2] - 0.015;
  } else if (std::is_same_v<DestFrame, ::Frame::Inertial>) {
    expected_center[0] = src_center[0] + 0.005;
    expected_center[1] = src_center[1] + 0.01;
    expected_center[2] = src_center[2] + 0.015;
  } else {
    expected_center[0] = src_center[0];
    expected_center[1] = src_center[1];
    expected_center[2] = src_center[2];
  }
  CHECK(expected_center == dest_center);
}
SPECTRE_TEST_CASE("Unit.Domain.CoordsToDifferentFrame", "[Unit]") {
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();
  test_coords_to_different_frame<Frame::Grid, Frame::Inertial>();
  test_coords_to_different_frame<Frame::Inertial, Frame::Distorted>();
  test_coords_to_different_frame<Frame::Inertial, Frame::Grid>();
  test_coords_to_different_frame<Frame::Grid, Frame::Inertial>();
  test_coords_to_different_frame<Frame::Grid, Frame::Distorted>();
}
}  // namespace
