// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <pup.h>
#include <random>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependentOptions/ExpansionMap.hpp"
#include "Domain/Creators/TimeDependentOptions/RotationMap.hpp"
#include "Domain/Creators/TimeDependentOptions/ShapeMap.hpp"
#include "Domain/Creators/TimeDependentOptions/Sphere.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/StrahlkorperTransformations.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Evolution/Ringdown/MinimumAhCExcisionRadius.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/ChangeCenterOfStrahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/IO/FillYlmLegendAndData.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "PointwiseFunctions/GeneralRelativity/KerrHorizon.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/Serialize.hpp"

// [[TimeOut, 45]]
SPECTRE_TEST_CASE("Unit.Evolution.Ringdown.MinimumAhCExcisionRadius",
                  "[Unit][Evolution]") {
  // Write a temporary H5 file with Strahlkorpers at different times, then
  // pass that file's path to strahlkorper_coefs_in_ringdown_distorted_frame().
  // First, if the temporary file exists, remove it
  const std::string inertial_horizons_file_name{
      "Unit.Evolution.Ringdown.InertialCoefs.h5"};
  const std::string inertial_horizons_subfile_name{"/ObservationAhC__Ylm.dat"};
  const std::string distorted_horizons_file_name{
      "Unit.Evolution.Ringdown.DisCoefs.h5"};
  const std::string distorted_horizons_subfile_name{"/DistortedAhC_Ylm.dat"};
  if (file_system::check_if_file_exists(inertial_horizons_file_name)) {
    file_system::rm(inertial_horizons_file_name, true);
  }
  if (file_system::check_if_file_exists(distorted_horizons_file_name)) {
    file_system::rm(distorted_horizons_file_name, true);
  }
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();

  MAKE_GENERATOR(generator);

  // Start out with a Strahlkorper at rest in a grid frame, then map it
  // to the inertial frame. The shape map is an identity map (it's initialized
  // with Schwarzschild coefficients), so the grid->distorted map
  // here is the identity. But start in a grid frame to use the available
  // grid->distorted instantiation of strahlkorper_in_different_frame.
  constexpr const size_t l_max = 12;
  constexpr const size_t m_max = 12;
  const auto kerr_horizon_radius = get(gr::Solutions::kerr_horizon_radius(
      ::ylm::Spherepack(l_max, m_max).theta_phi_points(), 1.0,
      {{0.0, 0.0, 0.8}}));
  auto expected_strahlkorper = ylm::Strahlkorper<Frame::Grid>(
      l_max, m_max, kerr_horizon_radius, std::array<double, 3>{1.0, 2.0, 3.0});
  auto distorted_strahlkorper = ylm::Strahlkorper<Frame::Distorted>(
      l_max, m_max, kerr_horizon_radius, std::array<double, 3>{1.0, 2.0, 3.0});

  // Make a set of times to evaluate the functions of time at
  static constexpr size_t number_of_times{9};
  const std::array<double, number_of_times> times{0.0, 0.2, 0.4, 0.6, 0.8,
                                                  1.0, 1.2, 1.4, 1.6};

  const double match_time{0.6};

  // Next, set up a temporary domain (just to hold functions of time)
  // and some functions of time defining the ringdown Distorted->Inertial map.
  std::uniform_real_distribution<double> fot_dist{0.1, 0.5};
  std::uniform_real_distribution<double> exp_dist{0.8, 1.};
  const auto exp_func_and_2_derivs =
      make_with_random_values<std::array<double, 3>>(make_not_null(&generator),
                                                     make_not_null(&exp_dist));
  const auto exp_outer_bdry_func_and_2_derivs =
      make_with_random_values<std::array<double, 3>>(make_not_null(&generator),
                                                     make_not_null(&exp_dist));
  auto initial_unit_quaternion = make_with_random_values<std::array<double, 4>>(
      make_not_null(&generator), make_not_null(&fot_dist));
  const double initial_unit_quaternion_magnitude = sqrt(
      square(initial_unit_quaternion[0]) + square(initial_unit_quaternion[1]) +
      square(initial_unit_quaternion[2]) + square(initial_unit_quaternion[3]));
  for (size_t i = 0; i < 4; ++i) {
    gsl::at(initial_unit_quaternion, i) /= initial_unit_quaternion_magnitude;
  }
  const std::vector<std::array<double, 4>> rot_func_and_2_derivs{
      initial_unit_quaternion,
      make_with_random_values<std::array<double, 4>>(make_not_null(&generator),
                                                     make_not_null(&fot_dist)),
      make_with_random_values<std::array<double, 4>>(make_not_null(&generator),
                                                     make_not_null(&fot_dist))};

  std::uniform_real_distribution<double> settling_dist{0.5, 1.5};
  const double settling_timescale{settling_dist(generator)};

  const domain::creators::time_dependent_options::ShapeMapOptions<
      false, domain::ObjectLabel::None>
      shape_map_options{l_max, std::nullopt};
  const domain::creators::time_dependent_options::ExpansionMapOptions<true>
      expansion_map_options{exp_func_and_2_derivs, settling_timescale,
                            exp_outer_bdry_func_and_2_derivs,
                            settling_timescale};
  const domain::creators::time_dependent_options::RotationMapOptions<true>
      rotation_map_options{rot_func_and_2_derivs, settling_timescale};
  const domain::creators::sphere::TimeDependentMapOptions
      time_dependent_map_options{times.at(0),          shape_map_options,
                                 rotation_map_options, expansion_map_options,
                                 std::nullopt,         true};

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
  std::vector<std::string> inertial_legend{};
  std::vector<std::string> distorted_legend{};
  ylm::Strahlkorper<Frame::Inertial> current_inertial_strahlkorper;
  for (size_t i = 0; i < number_of_times; ++i) {
    inertial_legend.resize(0);  // clear and reuse for next row of data
    strahlkorper_in_different_frame(
        make_not_null(&current_inertial_strahlkorper), expected_strahlkorper,
        temporary_domain, functions_of_time, gsl::at(times, i));
    ylm::change_expansion_center_of_strahlkorper_to_physical(
        make_not_null(&current_inertial_strahlkorper), 1e-8);
    ylm::fill_ylm_legend_and_data(
        make_not_null(&inertial_legend),
        make_not_null(&strahlkorper_ringdown_inertial_coefs[i]),
        current_inertial_strahlkorper, gsl::at(times, i), l_max);
  }
  std::vector<double> data{};
  distorted_legend.resize(0);
  ylm::fill_ylm_legend_and_data(make_not_null(&distorted_legend),
                                make_not_null(&data), distorted_strahlkorper,
                                match_time, l_max);
  {
    h5::H5File<h5::AccessType::ReadWrite> inertial_strahlkorper_file{
        inertial_horizons_file_name, true};
    auto& inertial_coefs_file = inertial_strahlkorper_file.insert<h5::Dat>(
        inertial_horizons_subfile_name, inertial_legend, 4);
    inertial_coefs_file.append(strahlkorper_ringdown_inertial_coefs);
    h5::H5File<h5::AccessType::ReadWrite> distorted_strahlkorper_file{
        distorted_horizons_file_name, true};
    auto& distorted_coefs_file = distorted_strahlkorper_file.insert<h5::Dat>(
        distorted_horizons_subfile_name, distorted_legend, 4);
    distorted_coefs_file.append(strahlkorper_ringdown_inertial_coefs);
  }

  // Next, set up a binary domain and make fake volume data where the functions
  // of time from the inspiral and domain can be extracted
  const domain::creators::time_dependent_options::RotationMapOptions<false>
      rotation_map_options_bco{std::array{0.0, 0.0, 0.0}};
  const domain::creators::time_dependent_options::ExpansionMapOptions<false>
      expansion_map_options_bco{{1.0, 1e-5, 0.0}, 50, -1.0e-6};
  const domain::creators::bco::TimeDependentMapOptions<false>
      time_dependent_map_options_bco{
          times.at(0),
          expansion_map_options_bco,
          rotation_map_options_bco,
          std::nullopt,
          std::nullopt,
          domain::creators::time_dependent_options::ShapeMapOptions<
              true, domain::ObjectLabel::A>{
              32_st, domain::creators::time_dependent_options::
                         KerrSchildFromBoyerLindquist{0.2, {0.0, 0.0, 0.0}}},
          domain::creators::time_dependent_options::ShapeMapOptions<
              true, domain::ObjectLabel::B>{
              32_st, domain::creators::time_dependent_options::
                         KerrSchildFromBoyerLindquist{0.4, {0.0, 0.0, 0.0}}},
          std::nullopt};

  using Object = domain::creators::BinaryCompactObject::Object;
  const domain::creators::BinaryCompactObject domain_creator_bco{
      Object{0.2, 0.5, 1.0, true, true},
      Object{0.1, 0.5, -1.5, true, true},
      std::array<double, 2>{{0.0, 0.0}},
      60.0,
      300.0,
      1.0,
      0_st,
      6_st,
      true,
      domain::CoordinateMaps::Distribution::Projective,
      std::vector<double>{},
      domain::CoordinateMaps::Distribution::Inverse,
      120.,
      false,
      false,
      time_dependent_map_options_bco};
  const auto domain_bco = domain_creator_bco.create_domain();
  const auto functions_of_time_bco = domain_creator_bco.functions_of_time();
  auto serialized_fots_bco = serialize(functions_of_time_bco);
  auto serialized_domain_bco = serialize(domain_bco);

  if (file_system::check_if_file_exists("BbhVolume0.h5")) {
    file_system::rm("BbhVolume0.h5", true);
  }
  h5::H5File<h5::AccessType::ReadWrite> h5_file{"BbhVolume0.h5", true};
  auto& vol_file = h5_file.insert<h5::VolumeData>("ForContinuation");

  // Write fake volume data
  for (size_t i = 0; i < times.size(); i++) {
    vol_file.write_volume_data(
        i, times.at(i),
        {ElementVolumeData{
            "MightyMorphin",
            {TensorComponent{"PowerRangers", DataVector{3, 0.0}}},
            {3},
            {Spectral::Basis::Legendre},
            {Spectral::Quadrature::GaussLobatto}}},
        serialized_domain_bco, serialized_fots_bco);
  }
  h5_file.close_current_object();

  const double ringdown_excision_radius =
      evolution::Ringdown::minimum_ahc_excision_radius(
          "BbhVolume0.h5", "ForContinuation", inertial_horizons_file_name,
          inertial_horizons_subfile_name, distorted_horizons_file_name,
          std::vector<std::string>{distorted_horizons_subfile_name}, match_time,
          settling_timescale, 0.2, 0.1, {1.0, 0.0, 0.0}, {-1.5, 0.0, 0.0},
          exp_func_and_2_derivs, exp_outer_bdry_func_and_2_derivs,
          rot_func_and_2_derivs, std::nullopt);

  // Checks
  CHECK(ringdown_excision_radius <
        current_inertial_strahlkorper.average_radius());

  if (file_system::check_if_file_exists(inertial_horizons_file_name)) {
    file_system::rm(inertial_horizons_file_name, true);
  }
  if (file_system::check_if_file_exists(distorted_horizons_file_name)) {
    file_system::rm(distorted_horizons_file_name, true);
  }
}
