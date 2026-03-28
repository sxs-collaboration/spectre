// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "IO/Exporter/Exporter.hpp"
#include "IO/Exporter/SpacetimeInterpolator.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace spectre::Exporter {

namespace {

void write_data_to_file(const std::string& h5_file_name) {
  // Create a domain
  const std::array<double, 3> velocity{{1.0, 0.0, 0.0}};
  const domain::creators::time_dependence::UniformTranslation<3, 0>
      time_dependence{0.0, velocity};
  const domain::creators::Sphere domain_creator{
      1.0,
      3.0,
      domain::creators::Sphere::Excision{},
      1_st,
      6_st,
      true,
      {},
      {},
      domain::CoordinateMaps::Distribution::Linear,
      ShellWedges::All,
      time_dependence.get_clone()};
  const auto domain = domain_creator.domain();
  const auto functions_of_time = domain_creator.functions_of_time();

  // Generate some volume data in the grid frame
  const auto element_ids =
      initial_element_ids(domain_creator.initial_refinement_levels());
  const Mesh<3> mesh{6, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const auto xi = logical_coordinates(mesh);
  std::vector<ElementVolumeData> element_volume_data{};
  const auto func = [](const auto& x) { return get(magnitude(x)); };
  for (const auto& element_id : element_ids) {
    const ElementMap<3, Frame::Grid> element_map{
        element_id, domain.blocks()[element_id.block_id()]};
    const auto x = element_map(xi);
    DataVector psi = func(x);
    element_volume_data.push_back(ElementVolumeData{
        element_id, {TensorComponent{"Psi", std::move(psi)}}, mesh});
  }

  // Write grid-frame data to file at multiple times. This means the data is
  // moving with the grid, but since the SpacetimeInterpolator works in the
  // grid frame the time interpolation should be exact.
  const size_t num_times = 10;
  std::vector<double> times(num_times);
  std::iota(times.begin(), times.end(), 0.0);
  h5::H5File<h5::AccessType::ReadWrite> h5_file(h5_file_name);
  auto& volfile = h5_file.insert<h5::VolumeData>("/VolumeData", 0);
  size_t obs_id = 0;
  for (const double time : times) {
    volfile.write_volume_data(obs_id, time, element_volume_data,
                              serialize(domain), serialize(functions_of_time),
                              serialize(functions_of_time));
    ++obs_id;
  }
}

void test_time_interpolation(
    const SpacetimeInterpolator<3, Frame::Inertial>& interpolator) {
  auto [tmin, tmax] = interpolator.time_bounds();
  CHECK(tmin < tmax);
  std::vector<double> interpolated_data{};
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      interpolator.interpolate_to_point(make_not_null(&interpolated_data),
                                        tnsr::I<double, 3>{{0.0, 0.0, 0.0}},
                                        tmin - 0.1),
      Catch::Matchers::ContainsSubstring("outside the time bounds."));
#endif  // SPECTRE_DEBUG
  for (const double time : {tmin, tmin + 0.1, tmin + 1.0,
                            tmin + 0.5 * (tmax - tmin), tmax - 0.1, tmax}) {
    CAPTURE(time);
    interpolator.interpolate_to_point(
        make_not_null(&interpolated_data),
        tnsr::I<double, 3>{{time + 1.0, 0.0, 0.0}}, time);
    CHECK(interpolated_data[0] == approx(1.0));
  }
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      interpolator.interpolate_to_point(make_not_null(&interpolated_data),
                                        tnsr::I<double, 3>{{0.0, 0.0, 0.0}},
                                        tmax + 0.1),
      Catch::Matchers::ContainsSubstring("outside the time bounds."));
#endif  // SPECTRE_DEBUG
}

}  // namespace

SPECTRE_TEST_CASE("Unit.IO.Exporter.SpacetimeInterpolator", "[Unit]") {
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();

  // Write sample data to file
  const std::string h5_file_name{"Unit.IO.Exporter.SpacetimeInterpolator.h5"};
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  write_data_to_file(h5_file_name);

  {
    SpacetimeInterpolator<3, Frame::Inertial> interpolator{
        h5_file_name, "VolumeData", {"Psi"}};
    CHECK(interpolator.max_time_bounds() == std::array<double, 2>{{1.0, 8.0}});
    interpolator.load_time_bounds({{1.0, 4.0}});
    CHECK(interpolator.num_loaded_slices() == 6);
    CHECK(interpolator.time_bounds() == std::array<double, 2>{{1.0, 4.0}});
    // Load some time bounds that are already loaded
    interpolator.load_time_bounds({{1.5, 4.0}});
    CHECK(interpolator.num_loaded_slices() == 6);
    CHECK(interpolator.time_bounds() == std::array<double, 2>{{1.0, 4.0}});
    interpolator.load_time_bounds({{2.5, 4.0}});
    CHECK(interpolator.num_loaded_slices() == 5);
    CHECK(interpolator.time_bounds() == std::array<double, 2>{{2.0, 4.0}});
    // Load some time bounds that don't overlap with the current bounds
    interpolator.load_time_bounds({{7.5, 8.0}});
    CHECK(interpolator.num_loaded_slices() == 4);
    CHECK(interpolator.time_bounds() == std::array<double, 2>{{7.0, 8.0}});
    // Interpolate
    interpolator.load_time_bounds({{1.0, 4.0}});
    CHECK(interpolator.num_loaded_slices() == 6);
    CHECK(interpolator.time_bounds() == std::array<double, 2>{{1.0, 4.0}});
    test_time_interpolation(interpolator);
    interpolator.load_time_bounds({{4.0, 7.0}});
    CHECK(interpolator.time_bounds() == std::array<double, 2>{{4.0, 7.0}});
    test_time_interpolation(interpolator);
    {
      INFO("Test move constructor");
      const SpacetimeInterpolator<3, Frame::Inertial> moved_interpolator{
          std::move(interpolator)};
      CHECK(moved_interpolator.max_time_bounds() ==
            std::array<double, 2>{{1.0, 8.0}});
      CHECK(moved_interpolator.num_loaded_slices() == 6);
      CHECK(moved_interpolator.time_bounds() ==
            std::array<double, 2>{{4.0, 7.0}});
      test_time_interpolation(moved_interpolator);
    }
  }

  // Delete the test file
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }

  {
    INFO("Test with time-independent domain");
    const auto domain = domain::creators::Brick{
        {{0.0, 0.0, 0.0}},
        {{1.0, 1.0, 1.0}},
        {{0, 0, 0}},
        {{4, 4, 4}}}.domain();
    const Mesh<3> mesh{4, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};

    h5::H5File<h5::AccessType::ReadWrite> h5_file{h5_file_name};
    auto& volfile = h5_file.insert<h5::VolumeData>("/VolumeData", 0);
    for (size_t i = 0; i < 5; ++i) {
      const double t = 1.0 * static_cast<double>(i);
      ElementVolumeData element_volume_data{
          ElementId<3>{0}, {TensorComponent{"Psi", DataVector(64, t)}}, mesh};
      volfile.write_volume_data(i, t, {std::move(element_volume_data)},
                                serialize(domain));
    }

    SpacetimeInterpolator<3> interpolator{h5_file_name, "VolumeData", {"Psi"}};
    interpolator.load_time_bounds({{1.5, 3}});
    std::vector<double> result{};
    interpolator.interpolate_to_point(
        make_not_null(&result), tnsr::I<double, 3, Frame::Inertial>{{0, 0, 0}},
        2.5);
    CHECK(result[0] == approx(2.5));

    if (file_system::check_if_file_exists(h5_file_name)) {
      file_system::rm(h5_file_name, true);
    }
  }
}

}  // namespace spectre::Exporter
