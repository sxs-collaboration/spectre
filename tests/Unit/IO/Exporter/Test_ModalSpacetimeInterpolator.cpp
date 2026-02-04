// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "IO/Exporter/ModalSpacetimeInterpolator.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace spectre::Exporter {

namespace {

void write_test_volume_data(const std::string& h5_file_name,
                            const std::string& subfile_name,
                            const std::vector<double>& times,
                            const Domain<3>& domain, const Mesh<3>& mesh,
                            const Mesh<3>& projection_mesh,
                            const std::vector<ElementId<3>>& element_ids) {
  const auto logical_coords = logical_coordinates(mesh);

  h5::H5File<h5::AccessType::ReadWrite> h5_file{h5_file_name, true};
  auto& volume_file = h5_file.insert<h5::VolumeData>(subfile_name, 0);

  size_t obs_id = 0;
  for (const double time : times) {
    std::vector<ElementVolumeData> element_volume_data{};
    element_volume_data.reserve(element_ids.size());

    for (const auto& element_id : element_ids) {
      const ElementMap<3, Frame::Inertial> element_map{
          element_id, domain.blocks()[element_id.block_id()]};
      const auto inertial_coords = element_map(logical_coords);

      // linear field
      DataVector psi = get<0>(inertial_coords);
      psi += 2.0 * get<1>(inertial_coords);
      psi += 3.0 * get<2>(inertial_coords);
      psi += time;

      // nonlinear field
      DataVector phi = exp(square(sin(get<0>(inertial_coords))));
      phi *= cos(get<2>(inertial_coords));
      phi *= sin(get<1>(inertial_coords));
      phi *= 100.0 * sin(time / 10.);

      const auto projection_matrices =
          Spectral::p_projection_matrices(mesh, projection_mesh);
      auto projected_psi =
          apply_matrices(projection_matrices, psi, mesh.extents());
      auto projected_phi =
          apply_matrices(projection_matrices, phi, mesh.extents());

      element_volume_data.push_back(
          ElementVolumeData{element_id,
                            {TensorComponent{"Psi", std::move(projected_psi)},
                             TensorComponent{"Phi", std::move(projected_phi)}},
                            projection_mesh});
    }

    volume_file.write_volume_data(obs_id, time, element_volume_data,
                                  serialize(domain));
    ++obs_id;
  }
}

std::array<double, 2> expected_values(
    const tnsr::I<double, 3, Frame::Inertial>& x, const double time) {
  const double psi = x.get(0) + 2. * x.get(1) + 3. * x.get(2) + time;
  const double phi = 100. * sin(time / 10.) * exp(square(sin(x.get(0)))) *
                     cos(x.get(2)) * sin(x.get(1));
  return {psi, phi};
}

}  // namespace

SPECTRE_TEST_CASE("Unit.IO.Exporter.ModalSpacetimeInterpolator", "[Unit]") {
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();

  const std::string h5_file_name_1{"ModalSpacetimeInterpolator_1.h5"};
  const std::string h5_file_name_2{"ModalSpacetimeInterpolator_2.h5"};
  const std::string serialized_interpolator_file{"serialized_interpolator.h5"};
  if (file_system::check_if_file_exists(h5_file_name_1)) {
    file_system::rm(h5_file_name_1, true);
  }
  if (file_system::check_if_file_exists(h5_file_name_2)) {
    file_system::rm(h5_file_name_2, true);
  }
  if (file_system::check_if_file_exists(serialized_interpolator_file)) {
    file_system::rm(serialized_interpolator_file, true);
  }

  const domain::creators::Brick domain_creator{
      {{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}, {{1, 1, 1}}, {{2, 2, 2}}};
  const auto domain = domain_creator.create_domain();
  const auto all_element_ids =
      initial_element_ids(domain_creator.initial_refinement_levels());

  std::vector<ElementId<3>> element_ids_file_1{};
  std::vector<ElementId<3>> element_ids_file_2{};
  element_ids_file_1.reserve(all_element_ids.size() / 2 + 1);
  element_ids_file_2.reserve(all_element_ids.size() / 2 + 1);
  for (size_t i = 0; i < all_element_ids.size(); ++i) {
    if (i % 2 == 0) {
      element_ids_file_1.push_back(all_element_ids[i]);
    } else {
      element_ids_file_2.push_back(all_element_ids[i]);
    }
  }
  const double final_time = 4.0;
  std::vector<double> dense_times{};
  std::vector<double> coarse_times{};

  // use non-overlapping time steps to ensure this is handled correctly
  const double small_step_size = 0.0498765;
  const double large_step_size = 0.4912345;
  const double initial_time_small_step = 0.0001234;
  const double initial_time_large_step = 0.0009876;
  // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
  for (double time = initial_time_small_step; time <= final_time;
       time += small_step_size) {  // NOLINT(cert-flp30-c)
    dense_times.push_back(time);
  }
  // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
  for (double time = initial_time_large_step; time <= final_time;
       time += large_step_size) {  // NOLINT(cert-flp30-c)
    coarse_times.push_back(time);
  }
  const std::string coarse_subfile_name = "/CoarseGrid";
  const std::string fine_subfile_name = "/FineGrid";

  const Mesh<3> coarse_mesh{4, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto};
  const Mesh<3> fine_mesh{12, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};

  write_test_volume_data(h5_file_name_1, coarse_subfile_name, dense_times,
                         domain, fine_mesh, coarse_mesh, element_ids_file_1);
  write_test_volume_data(h5_file_name_1, fine_subfile_name, coarse_times,
                         domain, fine_mesh, fine_mesh, element_ids_file_1);
  write_test_volume_data(h5_file_name_2, coarse_subfile_name, dense_times,
                         domain, fine_mesh, coarse_mesh, element_ids_file_2);
  write_test_volume_data(h5_file_name_2, fine_subfile_name, coarse_times,
                         domain, fine_mesh, fine_mesh, element_ids_file_2);

  std::uniform_real_distribution<double> coord_dist(0.0, 1.0);
  std::uniform_real_distribution<double> time_dist(0.1, 3.9);
  MAKE_GENERATOR(generator);

  const std::vector<std::string> volume_files{h5_file_name_1, h5_file_name_2};

  const ModalSpacetimeInterpolator<3, Frame::Inertial> constructed_interpolator(
      volume_files,
      std::vector<std::string>{coarse_subfile_name, fine_subfile_name},
      {"Psi", "Phi"});

  const double tolerance = 1e-9;

  auto check_interpolator_values =
      [&coord_dist, &time_dist, &generator, tolerance](
          const ModalSpacetimeInterpolator<3, Frame::Inertial>& interpolator)
      -> void {
    for (size_t i = 0; i < 1000; ++i) {
      const double time = time_dist(generator);
      const auto target_point =
          make_with_random_values<tnsr::I<double, 3, Frame::Inertial>>(
              make_not_null(&generator), coord_dist, 1.);
      const auto expected = expected_values(target_point, time);
      std::vector<double> result{};
      interpolator.interpolate_to_point(make_not_null(&result), target_point,
                                        time);
      CHECK(result.size() == expected.size());
      CHECK(result[0] == approx(expected[0]).epsilon(tolerance));
      CHECK(result[1] == approx(expected[1]).epsilon(tolerance));
    }
  };
  check_interpolator_values(constructed_interpolator);
  constructed_interpolator.write_to_h5(serialized_interpolator_file,
                                       "/ModalSpacetimeInterpolator");
  const auto deserialized_interpolator =
      ModalSpacetimeInterpolator<3, Frame::Inertial>(
          serialized_interpolator_file, "/ModalSpacetimeInterpolator");
  check_interpolator_values(deserialized_interpolator);

  const auto [min_time, _] = constructed_interpolator.time_bounds();
  CHECK(min_time == approx(initial_time_large_step));
  std::vector<double> result{};
  const tnsr::I<double, 3, Frame::Inertial> target_point{{0.1, 0.2, 0.3}};
  const double negative_time = -1e-20;
  CHECK_THROWS_WITH(constructed_interpolator.interpolate_to_point(
                        make_not_null(&result), target_point, negative_time),
                    Catch::Matchers::ContainsSubstring(
                        "lies outside the available data interval"));

  const double between_subfiles_times =
      (initial_time_small_step + initial_time_large_step) / 2.0;
  CHECK_THROWS_WITH(
      constructed_interpolator.interpolate_to_point(
          make_not_null(&result), target_point, between_subfiles_times),
      Catch::Matchers::ContainsSubstring(
          "lies outside the available data interval"));

  if (file_system::check_if_file_exists(h5_file_name_1)) {
    file_system::rm(h5_file_name_1, true);
  }
  if (file_system::check_if_file_exists(h5_file_name_2)) {
    file_system::rm(h5_file_name_2, true);
  }
  if (file_system::check_if_file_exists(serialized_interpolator_file)) {
    file_system::rm(serialized_interpolator_file, true);
  }
}

}  // namespace spectre::Exporter
