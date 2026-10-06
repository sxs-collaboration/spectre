// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "IO/Exporter/ModalSpacetimeInterpolator.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"
#include "NumericalAlgorithms/Spectral/Legendre.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace spectre::Exporter {

namespace {

constexpr size_t dim = 2;
using Interpolator = ModalSpacetimeInterpolator<dim, Frame::Inertial>;

enum class Resolution { Coarse, Fine };

double owned_mode_value(const size_t component, const size_t element_index,
                        const Index<dim>& mode, const double time) {
  const auto mode_number = static_cast<double>(mode[0] + 3 * mode[1]);
  if (component == 0) {
    return 1.0 + 0.2 * static_cast<double>(element_index) + 0.03 * mode_number +
           (0.1 + 0.002 * mode_number) * time;
  }
  return -0.5 - 0.1 * static_cast<double>(element_index) + 0.02 * mode_number +
         (0.07 - 0.001 * mode_number) * time;
}

ModalVector source_modes(const Mesh<dim>& mesh, const Resolution resolution,
                         const size_t component, const size_t element_index,
                         const double time) {
  ModalVector result(mesh.number_of_grid_points());
  for (size_t collapsed_mode = 0; collapsed_mode < result.size();
       ++collapsed_mode) {
    const auto mode = expanded_index<dim>(collapsed_mode, mesh.extents());
    const bool supplied_by_coarse_mesh = mode[0] < 2 and mode[1] < 3;
    if (resolution == Resolution::Coarse) {
      // An exactly zero mode is owned and dropped by the coarse subfile. The
      // fine subfile must not resurrect it.
      result[collapsed_mode] =
          mode == Index<dim>{1, 0}
              ? 0.0
              : owned_mode_value(component, element_index, mode, time);
    } else if (supplied_by_coarse_mesh) {
      // Make an ownership error obvious: all overlapping fine modes disagree
      // strongly with the coarse data.
      result[collapsed_mode] =
          100.0 + owned_mode_value(component, element_index, mode, time);
    } else {
      result[collapsed_mode] =
          owned_mode_value(component, element_index, mode, time);
    }
  }
  return result;
}

ModalVector expected_modes(const Mesh<dim>& final_mesh, const size_t component,
                           const size_t element_index, const double time) {
  ModalVector result(final_mesh.number_of_grid_points());
  for (size_t collapsed_mode = 0; collapsed_mode < result.size();
       ++collapsed_mode) {
    const auto mode = expanded_index<dim>(collapsed_mode, final_mesh.extents());
    result[collapsed_mode] =
        mode == Index<dim>{1, 0}
            ? 0.0
            : owned_mode_value(component, element_index, mode, time);
  }
  return result;
}

void write_subfile(
    const std::string& filename, const std::string& subfile_name,
    const std::vector<double>& times, const Domain<dim>& domain,
    const domain::FunctionsOfTimeMap& functions_of_time, const Mesh<dim>& mesh,
    const std::vector<std::pair<ElementId<dim>, size_t>>& elements,
    const Resolution resolution, const bool write_functions_of_time = true) {
  h5::H5File<h5::AccessType::ReadWrite> h5file{filename, true};
  auto& volfile = h5file.insert<h5::VolumeData>(subfile_name, 0);
  for (size_t observation_id = 0; observation_id < times.size();
       ++observation_id) {
    const double time = times[observation_id];
    std::vector<ElementVolumeData> element_data{};
    element_data.reserve(elements.size());
    for (const auto& [element_id, element_index] : elements) {
      const auto psi_modes =
          source_modes(mesh, resolution, 0, element_index, time);
      const auto phi_modes =
          source_modes(mesh, resolution, 1, element_index, time);
      element_data.push_back(ElementVolumeData{
          element_id,
          {TensorComponent{"Psi", to_nodal_coefficients(psi_modes, mesh)},
           TensorComponent{"Phi", to_nodal_coefficients(phi_modes, mesh)}},
          mesh});
    }
    const auto serialized_functions_of_time =
        write_functions_of_time
            ? std::optional<std::vector<char>>{serialize(functions_of_time)}
            : std::nullopt;
    volfile.write_volume_data(observation_id, time, element_data,
                              serialize(domain), serialized_functions_of_time,
                              serialized_functions_of_time);
  }
}

std::vector<double> uniform_times(const double start, const double step,
                                  const size_t size) {
  std::vector<double> result(size);
  for (size_t i = 0; i < size; ++i) {
    result[i] = start + static_cast<double>(i) * step;
  }
  return result;
}

void test_interpolation_and_ownership() {
  const std::string filename_1{
      "Unit.IO.Exporter.ModalSpacetimeInterpolator1.h5"};
  const std::string filename_2{
      "Unit.IO.Exporter.ModalSpacetimeInterpolator2.h5"};
  file_system::rm(filename_1, true);
  file_system::rm(filename_2, true);

  const std::array<double, dim> velocity{{0.25, -0.1}};
  const domain::creators::time_dependence::UniformTranslation<dim>
      time_dependence{0.0, velocity};
  const domain::creators::Rectangle domain_creator{{{-1.0, -2.0}},
                                                   {{1.0, 2.0}},
                                                   {{1, 0}},
                                                   {{4, 5}},
                                                   {{false, false}},
                                                   {},
                                                   time_dependence.get_clone()};
  const auto domain = domain_creator.create_domain();
  const auto functions_of_time = domain_creator.functions_of_time();
  const auto element_ids =
      initial_element_ids(domain_creator.initial_refinement_levels());
  REQUIRE(element_ids.size() == 2);

  const Mesh<dim> coarse_mesh{
      {{2, 3}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};
  const Mesh<dim> fine_mesh{
      {{4, 5}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};
  const auto coarse_times = uniform_times(0.0, 0.25, 17);
  const auto fine_times = uniform_times(0.125, 0.5, 9);
  const std::string coarse_subfile{"/Coarse"};
  const std::string fine_subfile{"/Fine"};

  // Swap element placement between the two subfiles. Each subfile reader must
  // independently determine which file owns an element.
  write_subfile(filename_1, coarse_subfile, coarse_times, domain,
                functions_of_time, coarse_mesh, {{element_ids[0], 0}},
                Resolution::Coarse);
  write_subfile(filename_2, coarse_subfile, coarse_times, domain,
                functions_of_time, coarse_mesh, {{element_ids[1], 1}},
                Resolution::Coarse);
  write_subfile(filename_1, fine_subfile, fine_times, domain, functions_of_time,
                fine_mesh, {{element_ids[1], 1}}, Resolution::Fine);
  write_subfile(filename_2, fine_subfile, fine_times, domain, functions_of_time,
                fine_mesh, {{element_ids[0], 0}}, Resolution::Fine);

  const Interpolator interpolator{
      std::vector<std::string>{filename_1, filename_2},
      {coarse_subfile, fine_subfile},
      {"Psi", "Phi"},
      std::nullopt,
      std::nullopt,
      std::nullopt,
      Verbosity::Silent};
  CHECK(interpolator.tensor_components() ==
        std::vector<std::string>{"Psi", "Phi"});
  CHECK(interpolator.time_bounds()[0] == approx(0.125));
  CHECK(interpolator.time_bounds()[1] == approx(4.0));

  const std::array<tnsr::I<double, dim, Frame::ElementLogical>, 2>
      logical_points{
          {tnsr::I<double, dim, Frame::ElementLogical>{{-0.35, 0.2}},
           tnsr::I<double, dim, Frame::ElementLogical>{{0.45, -0.6}}}};
  for (const double time : {0.125, 1.375, 4.0}) {
    CAPTURE(time);
    for (size_t element_index = 0; element_index < element_ids.size();
         ++element_index) {
      CAPTURE(element_index);
      const ElementMap<dim, Frame::Inertial> element_map{
          element_ids[element_index],
          domain.blocks()[element_ids[element_index].block_id()]};
      const auto inertial_point =
          element_map(logical_points[element_index], time, functions_of_time);
      std::vector<double> result{};
      interpolator.interpolate_to_point(make_not_null(&result), inertial_point,
                                        time);
      REQUIRE(result.size() == 2);
      for (size_t component = 0; component < result.size(); ++component) {
        const double expected = Spectral::evaluate_legendre_series<dim>(
            expected_modes(fine_mesh, component, element_index, time),
            fine_mesh, logical_points[element_index]);
        CHECK(result[component] ==
              approx(expected).epsilon(2.0e-12).scale(1.0));
      }
    }
  }

  std::vector<double> result{};
  const tnsr::I<double, dim, Frame::Inertial> interior_point{{0.0, 0.0}};
  CHECK_THROWS_WITH(interpolator.interpolate_to_point(make_not_null(&result),
                                                      interior_point, 0.0),
                    Catch::Matchers::ContainsSubstring(
                        "lies outside the available data interval"));
  CHECK_THROWS_WITH(interpolator.interpolate_to_point(
                        make_not_null(&result), interior_point,
                        std::numeric_limits<double>::quiet_NaN()),
                    Catch::Matchers::ContainsSubstring(
                        "lies outside the available data interval"));
  CHECK_THROWS_WITH(
      interpolator.interpolate_to_point(
          make_not_null(&result),
          tnsr::I<double, dim, Frame::Inertial>{{100.0, 100.0}}, 1.0),
      Catch::Matchers::ContainsSubstring("Point is not in any block"));

  const std::vector<std::string> filenames{filename_1, filename_2};
  CHECK_THROWS_WITH(
      (Interpolator{filenames, {}, {"Psi"}}),
      Catch::Matchers::ContainsSubstring("at least one volume subfile"));
  CHECK_THROWS_WITH(
      (Interpolator{filenames, {coarse_subfile}, {}}),
      Catch::Matchers::ContainsSubstring("at least one tensor component"));
  CHECK_THROWS_WITH(
      (Interpolator{filenames, {coarse_subfile, coarse_subfile}, {"Psi"}}),
      Catch::Matchers::ContainsSubstring("subfile names must be unique"));

  const Mesh<dim> decreasing_mesh{
      {{2, 2}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};
  const std::string decreasing_subfile{"/DecreasingMesh"};
  write_subfile(filename_1, decreasing_subfile, fine_times, domain,
                functions_of_time, decreasing_mesh, {{element_ids[0], 0}},
                Resolution::Fine);
  write_subfile(filename_2, decreasing_subfile, fine_times, domain,
                functions_of_time, decreasing_mesh, {{element_ids[1], 1}},
                Resolution::Fine);
  CHECK_THROWS_WITH(
      (Interpolator{filenames, {coarse_subfile, decreasing_subfile}, {"Psi"}}),
      Catch::Matchers::ContainsSubstring("nondecreasing extents"));

  const std::string different_elements_subfile{"/DifferentElements"};
  const ElementId<dim> refined_element{"[B0,(L2I0,L0I0)]"};
  write_subfile(filename_1, different_elements_subfile, fine_times, domain,
                functions_of_time, fine_mesh, {{element_ids[0], 0}},
                Resolution::Fine);
  write_subfile(filename_2, different_elements_subfile, fine_times, domain,
                functions_of_time, fine_mesh, {{refined_element, 1}},
                Resolution::Fine);
  CHECK_THROWS_WITH(
      (Interpolator{
          filenames, {coarse_subfile, different_elements_subfile}, {"Psi"}}),
      Catch::Matchers::ContainsSubstring(
          "All subfiles must contain the same elements"));

  const std::string no_global_functions_subfile{"/NoGlobalFunctions"};
  write_subfile(filename_1, no_global_functions_subfile, fine_times, domain,
                functions_of_time, fine_mesh, {{element_ids[0], 0}},
                Resolution::Fine, false);
  write_subfile(filename_2, no_global_functions_subfile, fine_times, domain,
                functions_of_time, fine_mesh, {{element_ids[1], 1}},
                Resolution::Fine, false);
  CHECK_THROWS_WITH(
      (Interpolator{filenames, {no_global_functions_subfile}, {"Psi"}}),
      Catch::Matchers::ContainsSubstring(
          "absent from the global functions of time"));

  const auto late_times = uniform_times(5.0, 0.5, 9);
  const std::string late_subfile{"/Late"};
  write_subfile(filename_1, late_subfile, late_times, domain, functions_of_time,
                fine_mesh, {{element_ids[0], 0}}, Resolution::Fine);
  write_subfile(filename_2, late_subfile, late_times, domain, functions_of_time,
                fine_mesh, {{element_ids[1], 1}}, Resolution::Fine);
  CHECK_THROWS_WITH(
      (Interpolator{filenames, {coarse_subfile, late_subfile}, {"Psi"}}),
      Catch::Matchers::ContainsSubstring("do not have a common time interval"));

  {
    INFO("Explicit error tolerance");
    const auto check_at_time = [&](const Interpolator& local_interpolator,
                                   const double time,
                                   const double expected_scale) {
      CAPTURE(time);
      for (size_t element_index = 0; element_index < element_ids.size();
           ++element_index) {
        CAPTURE(element_index);
        const ElementMap<dim, Frame::Inertial> element_map{
            element_ids[element_index],
            domain.blocks()[element_ids[element_index].block_id()]};
        const auto inertial_point =
            element_map(logical_points[element_index], time, functions_of_time);
        std::vector<double> local_result{};
        local_interpolator.interpolate_to_point(make_not_null(&local_result),
                                                inertial_point, time);
        REQUIRE(local_result.size() == 2);
        for (size_t component = 0; component < local_result.size();
             ++component) {
          const double expected =
              expected_scale *
              Spectral::evaluate_legendre_series<dim>(
                  expected_modes(fine_mesh, component, element_index, time),
                  fine_mesh, logical_points[element_index]);
          CHECK(local_result[component] ==
                approx(expected).epsilon(2.0e-12).scale(1.0));
        }
      }
    };
    // A tight tolerance retains all modes and reproduces the data.
    const Interpolator tight_interpolator{
        filenames,        {coarse_subfile, fine_subfile},
        {"Psi", "Phi"},   std::nullopt,
        std::nullopt,     1.0e-10,
        Verbosity::Silent};
    check_at_time(tight_interpolator, 1.375, 1.0);
    // A tolerance above the amplitude of every mode drops all modes.
    const Interpolator loose_interpolator{
        filenames,        {coarse_subfile, fine_subfile},
        {"Psi", "Phi"},   std::nullopt,
        std::nullopt,     1.0e3,
        Verbosity::Silent};
    check_at_time(loose_interpolator, 1.375, 0.0);
    CHECK_THROWS_WITH((Interpolator{filenames,
                                    {coarse_subfile},
                                    {"Psi"},
                                    std::nullopt,
                                    std::nullopt,
                                    -1.0}),
                      Catch::Matchers::ContainsSubstring(
                          "error tolerance must be finite and non-negative"));
    CHECK_THROWS_WITH((Interpolator{filenames,
                                    {coarse_subfile},
                                    {"Psi"},
                                    std::nullopt,
                                    std::nullopt,
                                    std::numeric_limits<double>::quiet_NaN()}),
                      Catch::Matchers::ContainsSubstring(
                          "error tolerance must be finite and non-negative"));

    // Too few observations to estimate the tolerance, but an explicit
    // tolerance works.
    const auto short_times = uniform_times(0.0, 0.5, 6);
    const std::string short_subfile{"/Short"};
    write_subfile(filename_1, short_subfile, short_times, domain,
                  functions_of_time, coarse_mesh, {{element_ids[0], 0}},
                  Resolution::Coarse);
    write_subfile(filename_2, short_subfile, short_times, domain,
                  functions_of_time, coarse_mesh, {{element_ids[1], 1}},
                  Resolution::Coarse);
    CHECK_THROWS_WITH(
        (Interpolator{filenames, {short_subfile, fine_subfile}, {"Psi"}}),
        Catch::Matchers::ContainsSubstring("At least 9 observations"));
    const Interpolator short_interpolator{
        filenames,        {short_subfile, fine_subfile},
        {"Psi", "Phi"},   std::nullopt,
        std::nullopt,     1.0e-10,
        Verbosity::Silent};
    CHECK(short_interpolator.time_bounds()[0] == approx(0.125));
    CHECK(short_interpolator.time_bounds()[1] == approx(2.5));
    check_at_time(short_interpolator, 1.375, 1.0);
  }

  file_system::rm(filename_1, true);
  file_system::rm(filename_2, true);
}

}  // namespace

// [[TimeOut, 20]]
SPECTRE_TEST_CASE("Unit.IO.Exporter.ModalSpacetimeInterpolator", "[Unit]") {
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();
  test_interpolation_and_ownership();
}

}  // namespace spectre::Exporter
