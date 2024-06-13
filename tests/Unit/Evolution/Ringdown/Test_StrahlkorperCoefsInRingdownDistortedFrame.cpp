// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/TimeDependentOptions/Sphere.hpp"
#include "Domain/StrahlkorperTransformations.hpp"
#include "Evolution/Ringdown/StrahlkorperCoefsInRingdownDistortedFrame.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/IO/FillYlmLegendAndData.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrHorizon.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"

// [[TimeOut, 10]]
SPECTRE_TEST_CASE(
    "Unit.Evolution.Ringdown.StrahlkorperCoefsInRingdownDistortedFrame",
    "[Unit][Evolution]") {
  // Write a temporary H5 file with Strahlkorpers at different times, then
  // pass that file's path to strahlkorper_coefs_in_ringdown_distorted_frame().
  // First, if the temporary file exists, remove it
  const std::string horizons_file_name{"Unit.Evolution.Ringdown.SCoefsRDis.h5"};
  const std::string horizons_subfile_name{"/ObservationAhC__Ylm.dat"};
  if (file_system::check_if_file_exists(horizons_file_name)) {
    file_system::rm(horizons_file_name, true);
  }

  MAKE_GENERATOR(generator);

  // Start out with a Strahlkorper at rest in a grid frame, then map it
  // to the inertial frame. The shape map is an identity map (it's initialized
  // with Schwarzschild coefficients), so the grid->distorted map
  // here is the identity. But start in a grid frame to use the available
  // grid->distorted instantiation of strahlkorper_in_different_frame.
  constexpr const std::array<double, 3> expected_center{4.4, 5.5, 6.6};
  constexpr const size_t l_max = 12;
  constexpr const size_t m_max = 12;
  const auto kerr_horizon_radius = get(gr::Solutions::kerr_horizon_radius(
      ::ylm::Spherepack(l_max, m_max).theta_phi_points(), 1.0,
      {{0.0, 0.0, 0.8}}));
  const auto expected_strahlkorper = ylm::Strahlkorper<Frame::Grid>(
      l_max, m_max, kerr_horizon_radius, expected_center);

  // Make a set of times to evaluate the functions of time at
  static constexpr size_t number_of_times{9};
  const std::array<double, number_of_times> times{0.0, 0.5, 1.0, 1.5, 2.0,
                                                  2.5, 3.0, 3.5, 4.0};

  // Set match time to earliest time; must be earliest time, since
  // the grid->inertial map used below will not be valid at times earlier
  // than the match time
  const double match_time{0.0};

  // Next, set up a temporary domain (just to hold functions of time)
  // and some functions of time defining the ringdown Distorted->Inertial map.
  std::uniform_real_distribution<double> fot_dist{0.1, 0.5};
  const auto exp_func_and_2_derivs =
      make_with_random_values<std::array<double, 3>>(make_not_null(&generator),
                                                     make_not_null(&fot_dist));
  const auto exp_outer_bdry_func_and_2_derivs =
      make_with_random_values<std::array<double, 3>>(make_not_null(&generator),
                                                     make_not_null(&fot_dist));
  auto initial_unit_quaternion = make_with_random_values<std::array<double, 4>>(
      make_not_null(&generator), make_not_null(&fot_dist));
  const double initial_unit_quaternion_magnitude = sqrt(
      square(initial_unit_quaternion[0]) + square(initial_unit_quaternion[1]) +
      square(initial_unit_quaternion[2]) + square(initial_unit_quaternion[3]));
  for (size_t i = 0; i < 4; ++i) {
    gsl::at(initial_unit_quaternion, i) /= initial_unit_quaternion_magnitude;
  }
  const std::array<std::array<double, 4>, 3> rot_func_and_2_derivs{
      initial_unit_quaternion,
      make_with_random_values<std::array<double, 4>>(make_not_null(&generator),
                                                     make_not_null(&fot_dist)),
      make_with_random_values<std::array<double, 4>>(make_not_null(&generator),
                                                     make_not_null(&fot_dist))};

  std::uniform_real_distribution<double> settling_dist{0.5, 1.5};
  const double settling_timescale{settling_dist(generator)};

  const domain::creators::sphere::TimeDependentMapOptions::ShapeMapOptions
      shape_map_options{l_max, std::nullopt};
  const domain::creators::sphere::TimeDependentMapOptions::ExpansionMapOptions
      expansion_map_options{exp_func_and_2_derivs, settling_timescale,
                            exp_outer_bdry_func_and_2_derivs,
                            settling_timescale};
  const domain::creators::sphere::TimeDependentMapOptions::RotationMapOptions
      rotation_map_options{rot_func_and_2_derivs, settling_timescale};
  const domain::creators::sphere::TimeDependentMapOptions
      time_dependent_map_options{match_time, shape_map_options,
                                 rotation_map_options, expansion_map_options,
                                 std::nullopt};

  const domain::creators::Sphere domain_creator{
      0.01,
      100.0,
      // nullptr because no boundary condition
      domain::creators::Sphere::Excision{nullptr},
      static_cast<size_t>(0),
      static_cast<size_t>(5),
      false,
      std::nullopt,
      {50.0},
      domain::CoordinateMaps::Distribution::Linear,
      ShellWedges::All,
      time_dependent_map_options};
  const auto temporary_domain = domain_creator.create_domain();
  const auto functions_of_time = domain_creator.functions_of_time();

  // For each Strahlkorper, transform from distorted -> inertial using
  // strahlkorper_in_different_frame, then
  // get its inertial coefficients, and write them out to the h5 file
  std::vector<std::vector<double>> strahlkorper_ringdown_inertial_coefs{
      number_of_times};
  std::vector<std::string> legend{};
  ylm::Strahlkorper<Frame::Inertial> current_inertial_strahlkorper;
  for (size_t i = 0; i < number_of_times; ++i) {
    legend.resize(0);  // clear and reuse for next row of data
    strahlkorper_in_different_frame(
        make_not_null(&current_inertial_strahlkorper), expected_strahlkorper,
        temporary_domain, functions_of_time, gsl::at(times, i));
    ylm::fill_ylm_legend_and_data(
        make_not_null(&legend),
        make_not_null(&strahlkorper_ringdown_inertial_coefs[i]),
        current_inertial_strahlkorper, gsl::at(times, i), l_max);
  }
  {
    h5::H5File<h5::AccessType::ReadWrite> strahlkorper_file{horizons_file_name,
                                                            true};
    auto& coefs_file =
        strahlkorper_file.insert<h5::Dat>(horizons_subfile_name, legend, 4);
    coefs_file.append(strahlkorper_ringdown_inertial_coefs);
  }

  // Call strahlkorper_coefs_in_ringdown_distorted_frame()
  constexpr size_t times_to_retrieve{number_of_times - 2};
  const std::vector<DataVector> distorted_coefs =
      evolution::Ringdown::strahlkorper_coefs_in_ringdown_distorted_frame(
          horizons_file_name, horizons_subfile_name, times_to_retrieve,
          match_time, settling_timescale, exp_func_and_2_derivs,
          exp_outer_bdry_func_and_2_derivs, rot_func_and_2_derivs);

  // Checks
  // std::vector is the expected size
  const size_t times_retrieved = distorted_coefs.size();
  CHECK(times_retrieved == times_to_retrieve);

  // Check that the coefficients have the expected numerical values
  const auto& expected_coefs = expected_strahlkorper.coefficients();
  Approx custom_approx = Approx::custom().epsilon(1.0e-10).scale(1.0);
  for (size_t i = 0; i < times_retrieved; ++i) {
    const auto retrieved_coefs = gsl::at(distorted_coefs, i);
    CHECK_ITERABLE_CUSTOM_APPROX(expected_coefs, retrieved_coefs,
                                 custom_approx);
  }

  // Check that retrieved coefs are the expected size
  const size_t coefs_size_expected = expected_coefs.size();
  const size_t coefs_size_retrieved = distorted_coefs[0].size();
  CHECK(coefs_size_expected == coefs_size_retrieved);

  if (file_system::check_if_file_exists(horizons_file_name)) {
    file_system::rm(horizons_file_name, true);
  }
}
