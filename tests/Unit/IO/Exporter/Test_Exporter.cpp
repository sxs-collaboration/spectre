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
#include "Domain/Creators/Rectangle.hpp"
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
}

}  // namespace spectre::Exporter
