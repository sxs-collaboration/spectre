// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "IO/Exporter/Exporter.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace spectre::Exporter {

SPECTRE_TEST_CASE("Unit.IO.Exporter", "[Unit]") {
#ifdef _OPENMP
  // Disable OpenMP multithreading since multiple unit tests may run in parallel
  omp_set_num_threads(1);
#endif
  {
    INFO("Bundled volume data files");
    const auto interpolated_data = interpolate_to_points<3>(
        unit_test_src_path() + "/Visualization/Python/VolTestData*.h5",
        "element_data", ObservationStep{0}, {"Psi", "Phi_x", "Phi_y", "Phi_z"},
        {{{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}}});
    const auto& psi = interpolated_data[0];
    CHECK(psi[0] == approx(-0.07059806932542323));
    CHECK(psi[1] == approx(0.7869554122196492));
    CHECK(psi[2] == approx(0.9876185584100299));
    const auto& phi_y = interpolated_data[2];
    CHECK(phi_y[0] == approx(1.0569673471948728));
    CHECK(phi_y[1] == approx(0.6741524090220188));
    CHECK(phi_y[2] == approx(0.2629752479142838));
  }
  {
    INFO("Single-precision volume data");
    const domain::creators::Rectangle domain_creator{
        {{-1., -1.}}, {{1., 1.}}, {{0, 0}}, {{4, 4}}, {{false, false}}};
    const auto domain = domain_creator.create_domain();
    const ElementId<2> element_id{0};
    const Mesh<2> mesh{4, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    // Manufacture linear data so interpolation is exact
    const auto xi = logical_coordinates(mesh);
    const DataVector psi = 1.0 + get<0>(xi) + 2. * get<1>(xi);
    // Convert to single precision
    std::vector<float> psi_float(psi.size());
    for (size_t i = 0; i < psi.size(); ++i) {
      psi_float[i] = static_cast<float>(psi[i]);
    }
    // Write to file
    const std::string h5_file_name{"Unit.IO.Exporter.Float.h5"};
    if (file_system::check_if_file_exists(h5_file_name)) {
      file_system::rm(h5_file_name, true);
    }
    {  // scope to open and close file
      h5::H5File<h5::AccessType::ReadWrite> h5_file(h5_file_name);
      auto& volfile = h5_file.insert<h5::VolumeData>("/VolumeData", 0);
      volfile.write_volume_data(
          123, 0.,
          {ElementVolumeData{element_id,
                             {TensorComponent{"Psi", std::move(psi_float)}},
                             mesh}},
          serialize(domain));
    }
    const auto interpolated_data =
        interpolate_to_points<2>(h5_file_name, "/VolumeData",
                                 ObservationId{123}, {"Psi"}, {{{0.}, {0.}}});
    // Compare to single precision
    Approx custom_approx =
        Approx::custom()
            .epsilon(10. * std::numeric_limits<float>::epsilon())
            .scale(1.0);
    CHECK(interpolated_data[0][0] == custom_approx(1.));
    // Delete the test file
    if (file_system::check_if_file_exists(h5_file_name)) {
      file_system::rm(h5_file_name, true);
    }
  }
  {
    INFO("Extrapolation into BBH excisions");
    using Object = domain::creators::BinaryCompactObject<false>::Object;
    const domain::creators::BinaryCompactObject domain_creator{
        Object{1., 4., 8., true, true},
        Object{0.8, 2.5, -6., true, true},
        std::array<double, 2>{{0., 0.}},
        60.,
        300.,
        1.0,
        0_st,
        6_st,
        true,
        domain::CoordinateMaps::Distribution::Projective,
        domain::CoordinateMaps::Distribution::Inverse,
        120.};
    const auto domain = domain_creator.create_domain();
    const auto functions_of_time = domain_creator.functions_of_time();
    const double time = 1.0;
    const auto element_ids =
        initial_element_ids(domain_creator.initial_refinement_levels());
    const Mesh<3> mesh{18, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    const auto xi = logical_coordinates(mesh);
    std::vector<ElementVolumeData> element_volume_data{};
    const auto func = [](const auto& x) {
      auto x1 = x;
      get<0>(x1) -= 8;
      auto x2 = x;
      get<0>(x2) += 6;
      const auto r1 = get(magnitude(x1));
      const auto r2 = get(magnitude(x2));
      return blaze::evaluate(exp(-square(r1)) + exp(-square(r2)));
    };
    for (const auto& element_id : element_ids) {
      ElementMap<3, Frame::Inertial> element_map{
          element_id, domain.blocks()[element_id.block_id()]};
      const auto x = element_map(xi, time, functions_of_time);
      DataVector psi = func(x);
      element_volume_data.push_back(ElementVolumeData{
          element_id, {TensorComponent{"Psi", std::move(psi)}}, mesh});
    }
    // Write to file
    const std::string h5_file_name{"Unit.IO.Exporter.Excision.h5"};
    if (file_system::check_if_file_exists(h5_file_name)) {
      file_system::rm(h5_file_name, true);
    }
    {  // scope to open and close file
      h5::H5File<h5::AccessType::ReadWrite> h5_file(h5_file_name);
      auto& volfile = h5_file.insert<h5::VolumeData>("/VolumeData", 0);
      volfile.write_volume_data(123, time, element_volume_data,
                                serialize(domain),
                                serialize(functions_of_time));
    }
    // Interpolate
    tnsr::I<DataVector, 3> target_points{{{{8.5, 8.6, 8.7, 8.9, 9., 10.},
                                           {0., 0., 0., 0., 0., 0.},
                                           {0., 0., 0., 0., 0., 0.}}}};
    const size_t num_target_points = get<0>(target_points).size();
    std::array<std::vector<double>, 3> target_points_array{};
    for (size_t d = 0; d < 3; ++d) {
      gsl::at(target_points_array, d).resize(num_target_points);
      for (size_t i = 0; i < num_target_points; ++i) {
        gsl::at(target_points_array, d)[i] = target_points.get(d)[i];
      }
    }
    const auto interpolated_data = interpolate_to_points<3>(
        h5_file_name, "/VolumeData", ObservationId{123}, {"Psi"},
        target_points_array, true);
    CHECK(interpolated_data.size() == 1);
    CHECK(interpolated_data[0].size() == num_target_points);
    // Check result
    DataVector psi_interpolated(num_target_points);
    DataVector psi_expected(num_target_points);
    for (size_t i = 0; i < num_target_points; ++i) {
      psi_interpolated[i] = interpolated_data[0][i];
      tnsr::I<double, 3> x_target{
          {{get<0>(target_points)[i], get<1>(target_points)[i],
            get<2>(target_points)[i]}}};
      psi_expected[i] = func(x_target);
    }
    // These points are extrapolated and therefore less precise
    Approx approx_extrapolated = Approx::custom().epsilon(1.e-1).scale(1.0);
    for (size_t i = 0; i < 4; ++i) {
      CHECK(psi_interpolated[i] == approx_extrapolated(psi_expected[i]));
    }
    // This point is exact
    CHECK(psi_interpolated[4] == approx(psi_expected[4]));
    // This point is interpolated
    Approx approx_interpolated = Approx::custom().epsilon(1.e-6).scale(1.0);
    CHECK(psi_interpolated[5] == approx_interpolated(psi_expected[5]));
    // Delete the test file
    if (file_system::check_if_file_exists(h5_file_name)) {
      file_system::rm(h5_file_name, true);
    }
  }
}

}  // namespace spectre::Exporter
