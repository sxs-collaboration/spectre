// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "IO/Exporter/ModalTimeSeriesReader.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"

namespace spectre::Exporter {

namespace {

// Deterministic nodal data so the streamed modal coefficients can be compared
// against a direct transform of the same data
double nodal_value(const size_t element_index, const size_t point_index,
                   const double time) {
  return std::sin(0.3 * static_cast<double>(point_index) +
                  static_cast<double>(element_index)) *
             (1.0 + 0.1 * time) +
         time;
}

DataVector nodal_data(const size_t element_index, const size_t num_points,
                      const double time) {
  DataVector result(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    result[i] = nodal_value(element_index, i, time);
  }
  return result;
}

void write_3d_test_file(const std::string& filename,
                        const std::string& subfile_name,
                        const std::vector<std::pair<size_t, double>>& obs,
                        const std::vector<std::string>& grid_names,
                        const std::vector<size_t>& element_indices,
                        const Mesh<3>& mesh) {
  const size_t num_points = mesh.number_of_grid_points();
  h5::H5File<h5::AccessType::ReadWrite> h5file{filename, true};
  auto& volfile = h5file.insert<h5::VolumeData>(subfile_name, 0);
  for (const auto& [obs_id, time] : obs) {
    std::vector<ElementVolumeData> elements{};
    elements.reserve(grid_names.size());
    for (size_t i = 0; i < grid_names.size(); ++i) {
      const auto psi = nodal_data(element_indices[i], num_points, time);
      // Store the second component in single precision to test the
      // float codepath
      std::vector<float> phi(num_points);
      for (size_t p = 0; p < num_points; ++p) {
        phi[p] = static_cast<float>(2.0 * psi[p] + 1.0);
      }
      elements.push_back(ElementVolumeData{
          ElementId<3>(grid_names[i]),
          {TensorComponent{"Psi", psi}, TensorComponent{"Phi", std::move(phi)}},
          mesh});
    }
    volfile.write_volume_data(obs_id, time, elements);
  }
}

// Writes a 1D volume file where each observation can hold a different set of
// grids, to construct the various unsupported scenarios
void write_1d_test_file(
    const std::string& filename, const std::string& subfile_name,
    const std::vector<std::pair<size_t, double>>& obs,
    const std::vector<std::vector<std::pair<std::string, size_t>>>&
        grids_per_obs,
    const Spectral::Basis basis = Spectral::Basis::Legendre) {
  h5::H5File<h5::AccessType::ReadWrite> h5file{filename, true};
  auto& volfile = h5file.insert<h5::VolumeData>(subfile_name, 0);
  for (size_t obs_index = 0; obs_index < obs.size(); ++obs_index) {
    const auto& [obs_id, time] = obs[obs_index];
    std::vector<ElementVolumeData> elements{};
    for (const auto& [grid_name, num_points] : grids_per_obs[obs_index]) {
      const Mesh<1> mesh{num_points, basis, Spectral::Quadrature::GaussLobatto};
      elements.push_back(ElementVolumeData{
          ElementId<1>(grid_name),
          {TensorComponent{"Psi", DataVector{num_points, time}}},
          mesh});
    }
    volfile.write_volume_data(obs_id, time, elements);
  }
}

void test_streaming() {
  const std::string filename_1{"TestModalTimeSeriesReader3D_1.h5"};
  const std::string filename_2{"TestModalTimeSeriesReader3D_2.h5"};
  file_system::rm(filename_1, true);
  file_system::rm(filename_2, true);
  const std::string subfile_name{"/VolumeData"};

  const Mesh<3> mesh{3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const size_t num_points = mesh.number_of_grid_points();
  // Elements split across two files
  const std::vector<std::string> grid_names_1{"[B0,(L1I0,L0I0,L0I0)]",
                                              "[B0,(L1I1,L0I0,L0I0)]"};
  const std::vector<std::string> grid_names_2{"[B1,(L1I0,L0I0,L0I0)]",
                                              "[B1,(L1I1,L0I0,L0I0)]"};
  std::vector<std::pair<size_t, double>> obs{};
  const double start_time = 0.25;
  const double time_step = 0.5;
  const size_t num_observations = 10;
  for (size_t i = 0; i < num_observations; ++i) {
    obs.emplace_back(i, start_time + static_cast<double>(i) * time_step);
  }
  write_3d_test_file(filename_1, subfile_name, obs, grid_names_1, {0, 1}, mesh);
  write_3d_test_file(filename_2, subfile_name, obs, grid_names_2, {2, 3}, mesh);

  // Construct from a glob to test glob resolution
  ModalTimeSeriesReader<3> reader{
      std::string{"TestModalTimeSeriesReader3D_*.h5"},
      subfile_name,
      {"Psi", "Phi"}};
  CHECK(reader.num_observations() == num_observations);
  CHECK(reader.start_time() == approx(start_time));
  CHECK(reader.time_step() == approx(time_step));
  CHECK(reader.tensor_components() == std::vector<std::string>{"Psi", "Phi"});
  REQUIRE(reader.elements().size() == 4);
  // Elements are grouped by file, in the order they appear in the files
  for (size_t i = 0; i < 2; ++i) {
    CHECK(reader.elements()[i].first == ElementId<3>(grid_names_1[i]));
    CHECK(reader.elements()[i + 2].first == ElementId<3>(grid_names_2[i]));
    CHECK(reader.elements()[i].second == mesh);
    CHECK(reader.elements()[i + 2].second == mesh);
  }

  const auto element_index_for = [&grid_names_1,
                                  &grid_names_2](const ElementId<3>& id) {
    for (size_t i = 0; i < grid_names_1.size(); ++i) {
      if (ElementId<3>(grid_names_1[i]) == id) {
        return i;
      }
    }
    for (size_t i = 0; i < grid_names_2.size(); ++i) {
      if (ElementId<3>(grid_names_2[i]) == id) {
        return i + grid_names_1.size();
      }
    }
    ERROR("Unexpected element ID: " << id);
  };

  const auto check_series =
      [&mesh, &num_points, &obs, &element_index_for](
          const ElementId<3>& element_id,
          const ModalTimeSeriesReader<3>::Series& series) {
        const size_t element_index = element_index_for(element_id);
        REQUIRE(series.size() == 2);
        REQUIRE(series[0].size() == num_points);
        REQUIRE(series[0][0].size() == obs.size());
        // Compute the expected modal coefficients directly and compare
        ModalVector expected_modes(num_points);
        DataVector nodal(num_points);
        double max_deviation = 0.0;
        for (size_t obs_index = 0; obs_index < obs.size(); ++obs_index) {
          const double time = obs[obs_index].second;
          // Psi (written in double precision)
          nodal = nodal_data(element_index, num_points, time);
          to_modal_coefficients(make_not_null(&expected_modes), nodal, mesh);
          for (size_t mode = 0; mode < num_points; ++mode) {
            max_deviation = std::max(
                max_deviation,
                std::abs(series[0][mode][obs_index] - expected_modes[mode]));
          }
          // Phi (written in single precision)
          for (size_t p = 0; p < num_points; ++p) {
            nodal[p] =
                static_cast<double>(static_cast<float>(2.0 * nodal[p] + 1.0));
          }
          to_modal_coefficients(make_not_null(&expected_modes), nodal, mesh);
          for (size_t mode = 0; mode < num_points; ++mode) {
            max_deviation = std::max(
                max_deviation,
                std::abs(series[1][mode][obs_index] - expected_modes[mode]));
          }
        }
        CHECK(max_deviation == approx(0.0));
      };

  // [modal_time_series_reader_example]
  for (const auto& [element_id, element_mesh] : reader.elements()) {
    const auto series = reader.modal_time_series(element_id);
    // ... process the time series of this element ...
    // [modal_time_series_reader_example]
    CHECK(element_mesh == mesh);
    check_series(element_id, series);
  }

  // Out-of-order access across files re-reads the file metadata but returns
  // the same data
  const auto& last_element_id = reader.elements().back().first;
  check_series(last_element_id, reader.modal_time_series(last_element_id));
  const auto& first_element_id = reader.elements().front().first;
  check_series(first_element_id, reader.modal_time_series(first_element_id));

  // Unknown element
  CHECK_THROWS_WITH(
      reader.modal_time_series(ElementId<3>("[B5,(L0I0,L0I0,L0I0)]")),
      Catch::Matchers::ContainsSubstring("does not exist in the volume"));

  // Restrict the time interval
  const ModalTimeSeriesReader<3> restricted_reader{
      std::vector<std::string>{filename_1, filename_2},
      subfile_name,
      {"Psi"},
      1.0,
      3.9};
  // Times are 0.25, 0.75, ..., 4.75, so [1.0, 3.9] keeps 1.25, ..., 3.75
  CHECK(restricted_reader.num_observations() == 6);
  CHECK(restricted_reader.start_time() == approx(1.25));
  CHECK(restricted_reader.time_step() == approx(time_step));

  file_system::rm(filename_1, true);
  file_system::rm(filename_2, true);
}

void test_errors() {
  const std::string filename_1{"TestModalTimeSeriesReader1D_1.h5"};
  const std::string filename_2{"TestModalTimeSeriesReader1D_2.h5"};
  file_system::rm(filename_1, true);
  file_system::rm(filename_2, true);
  // All error scenarios share the same two H5 files, with one subfile per
  // scenario, because creating H5 files is slow enough to hit the test
  // timeout on slow file systems.
  const std::vector<std::string> psi{"Psi"};
  const auto make_reader =
      [&filename_1,
       &psi](const std::string& subfile_name) -> ModalTimeSeriesReader<1> {
    return {std::vector<std::string>{filename_1}, subfile_name, psi};
  };

  {
    // Non-uniform observation times
    write_1d_test_file(
        filename_1, "/NonUniformTimes", {{0, 0.0}, {1, 1.0}, {2, 2.5}},
        {{{"[B0,(L0I0)]", 4}}, {{"[B0,(L0I0)]", 4}}, {{"[B0,(L0I0)]", 4}}});
    CHECK_THROWS_WITH(
        make_reader("/NonUniformTimes"),
        Catch::Matchers::ContainsSubstring("must be uniformly spaced"));
  }
  {
    // Too few observations
    write_1d_test_file(filename_1, "/TooFewObservations", {{0, 0.0}},
                       {{{"[B0,(L0I0)]", 4}}});
    CHECK_THROWS_WITH(
        make_reader("/TooFewObservations"),
        Catch::Matchers::ContainsSubstring("At least 2 observations"));
  }
  {
    // Wrong dimension
    write_1d_test_file(filename_1, "/WrongDim", {{0, 0.0}, {1, 1.0}},
                       {{{"[B0,(L0I0)]", 4}}, {{"[B0,(L0I0)]", 4}}});
    CHECK_THROWS_WITH(
        (ModalTimeSeriesReader<2>{std::vector<std::string>{filename_1},
                                  "/WrongDim", psi}),
        Catch::Matchers::ContainsSubstring("Mismatched dimensions"));
  }
  {
    // Mismatched observation values between files
    write_1d_test_file(filename_1, "/MismatchedObs", {{0, 0.0}, {1, 1.0}},
                       {{{"[B0,(L0I0)]", 4}}, {{"[B0,(L0I0)]", 4}}});
    write_1d_test_file(filename_2, "/MismatchedObs",
                       {{0, 0.0}, {1, std::nextafter(1.0, 2.0)}},
                       {{{"[B1,(L0I0)]", 4}}, {{"[B1,(L0I0)]", 4}}});
    CHECK_THROWS_WITH(
        (ModalTimeSeriesReader<1>{
            std::vector<std::string>{filename_1, filename_2}, "/MismatchedObs",
            psi}),
        Catch::Matchers::ContainsSubstring("Mismatched observation value"));
  }
  {
    // Different numbers of observations across files
    write_1d_test_file(filename_1, "/MissingObs", {{0, 0.0}, {1, 1.0}},
                       {{{"[B0,(L0I0)]", 4}}, {{"[B0,(L0I0)]", 4}}});
    write_1d_test_file(filename_2, "/MissingObs", {{0, 0.0}},
                       {{{"[B1,(L0I0)]", 4}}});
    CHECK_THROWS_WITH((ModalTimeSeriesReader<1>{
                          std::vector<std::string>{filename_1, filename_2},
                          "/MissingObs", psi}),
                      Catch::Matchers::ContainsSubstring(
                          "must contain the same observations"));
    write_1d_test_file(filename_1, "/ExtraObs", {{0, 0.0}, {1, 1.0}},
                       {{{"[B0,(L0I0)]", 4}}, {{"[B0,(L0I0)]", 4}}});
    write_1d_test_file(
        filename_2, "/ExtraObs", {{0, 0.0}, {1, 1.0}, {2, 2.0}},
        {{{"[B1,(L0I0)]", 4}}, {{"[B1,(L0I0)]", 4}}, {{"[B1,(L0I0)]", 4}}});
    CHECK_THROWS_WITH((ModalTimeSeriesReader<1>{
                          std::vector<std::string>{filename_1, filename_2},
                          "/ExtraObs", psi}),
                      Catch::Matchers::ContainsSubstring(
                          "must contain the same observations"));
  }
  {
    // Mismatched dimension in second file
    write_1d_test_file(filename_1, "/MismatchedFileDim", {{0, 0.0}, {1, 1.0}},
                       {{{"[B0,(L0I0)]", 4}}, {{"[B0,(L0I0)]", 4}}});
    write_3d_test_file(filename_2, "/MismatchedFileDim", {{0, 0.0}, {1, 1.0}},
                       {"[B1,(L0I0,L0I0,L0I0)]"}, {0},
                       Mesh<3>{2, Spectral::Basis::Legendre,
                               Spectral::Quadrature::GaussLobatto});
    CHECK_THROWS_WITH(
        (ModalTimeSeriesReader<1>{
            std::vector<std::string>{filename_1, filename_2},
            "/MismatchedFileDim", psi}),
        Catch::Matchers::ContainsSubstring("Mismatched dimensions"));
  }
  {
    // Duplicate element across files
    write_1d_test_file(filename_1, "/DuplicateElement", {{0, 0.0}, {1, 1.0}},
                       {{{"[B0,(L0I0)]", 4}}, {{"[B0,(L0I0)]", 4}}});
    write_1d_test_file(filename_2, "/DuplicateElement", {{0, 0.0}, {1, 1.0}},
                       {{{"[B0,(L0I0)]", 4}}, {{"[B0,(L0I0)]", 4}}});
    CHECK_THROWS_WITH((ModalTimeSeriesReader<1>{
                          std::vector<std::string>{filename_1, filename_2},
                          "/DuplicateElement", psi}),
                      Catch::Matchers::ContainsSubstring(
                          "must reside in exactly one volume file"));
  }
  {
    // Non-Legendre basis
    write_1d_test_file(filename_1, "/NonLegendre", {{0, 0.0}, {1, 1.0}},
                       {{{"[B0,(L0I0)]", 4}}, {{"[B0,(L0I0)]", 4}}},
                       Spectral::Basis::Chebyshev);
    CHECK_THROWS_WITH(
        make_reader("/NonLegendre"),
        Catch::Matchers::ContainsSubstring("Only the Legendre basis"));
  }
  {
    // Different number of elements per observation (h-refinement or
    // migration)
    write_1d_test_file(
        filename_1, "/ChangingNumElements", {{0, 0.0}, {1, 1.0}},
        {{{"[B0,(L0I0)]", 4}}, {{"[B0,(L1I0)]", 4}, {"[B0,(L1I1)]", 4}}});
    CHECK_THROWS_WITH(make_reader("/ChangingNumElements")
                          .modal_time_series(ElementId<1>("[B0,(L1I0)]")),
                      Catch::Matchers::ContainsSubstring("are not supported"));
  }
  {
    // Same number of elements but an element is missing at an earlier
    // observation (migration between files)
    write_1d_test_file(filename_1, "/MigratedElement", {{0, 0.0}, {1, 1.0}},
                       {{{"[B0,(L1I0)]", 4}, {"[B1,(L0I0)]", 4}},
                        {{"[B0,(L1I0)]", 4}, {"[B0,(L1I1)]", 4}}});
    CHECK_THROWS_WITH(make_reader("/MigratedElement")
                          .modal_time_series(ElementId<1>("[B0,(L1I1)]")),
                      Catch::Matchers::ContainsSubstring("is missing in file"));
  }
  {
    // Mesh changes between observations (p-refinement)
    write_1d_test_file(filename_1, "/ChangingMesh", {{0, 0.0}, {1, 1.0}},
                       {{{"[B0,(L0I0)]", 5}}, {{"[B0,(L0I0)]", 4}}});
    CHECK_THROWS_WITH(
        make_reader("/ChangingMesh")
            .modal_time_series(ElementId<1>("[B0,(L0I0)]")),
        Catch::Matchers::ContainsSubstring("Mesh changes between"));
  }
  {
    // No files found
    CHECK_THROWS_WITH(
        (ModalTimeSeriesReader<1>{std::vector<std::string>{}, "/VolumeData",
                                  psi}),
        Catch::Matchers::ContainsSubstring("No volume files found"));
    // A glob that matches nothing already raises in file_system::glob
    CHECK_THROWS_WITH(
        (ModalTimeSeriesReader<1>{std::string{"NonexistentGlob_*.h5"},
                                  "/VolumeData", psi}),
        Catch::Matchers::ContainsSubstring("Unable to resolve glob"));
  }
  file_system::rm(filename_1, true);
  file_system::rm(filename_2, true);
}

}  // namespace

// [[TimeOut, 15]]
SPECTRE_TEST_CASE("Unit.IO.Exporter.ModalTimeSeriesReader", "[Unit]") {
  test_streaming();
  test_errors();
}

}  // namespace spectre::Exporter
