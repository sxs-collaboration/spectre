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
#include "Domain/Creators/TimeDependentOptions/TranslationMap.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/StrahlkorperTransformations.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Evolution/Ringdown/StrahlkorperCoefsAndCenters.hpp"
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

// [[TimeOut, 20]]
SPECTRE_TEST_CASE("Unit.Evolution.Ringdown.StrahlkorperCoefsAndCenters",
                  "[Unit][Evolution]") {
  // Write a temporary H5 file with Strahlkorpers at different times, then
  // pass that file's path to strahlkorper_coefs_and_centers.
  // First, if the temporary file exists, remove it
  const std::string horizons_file_name{"Unit.Evolution.Ringdown.SCoefsRDis.h5"};
  const std::string horizons_subfile_name{"/ObservationAhC__Ylm.dat"};
  const std::string volume_file_name{"BbhVolume1.h5"};
  const std::string volume_file_subfile_name{"ForContinuation"};
  if (file_system::check_if_file_exists(horizons_file_name)) {
    file_system::rm(horizons_file_name, true);
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
      l_max, m_max, kerr_horizon_radius, std::array<double, 3>{4.4, 5.5, 6.6});

  // Make a set of times to evaluate the functions of time at
  static constexpr size_t number_of_times{9};
  const std::array<double, number_of_times> times{0.0, 0.2, 0.4, 0.6, 0.8,
                                                  1.0, 1.2, 1.4, 1.6};

  // Set match time to earliest time; must be earliest time, since
  // the grid->inertial map used below will not be valid at times earlier
  // than the match time
  const double match_time{1.6};

  // Next, set up a temporary domain (just to hold functions of time)
  // and some functions of time defining the ringdown Distorted->Inertial map.
  std::uniform_real_distribution<double> fot_dist{0.1, 0.5};
  std::uniform_real_distribution<double> exp_dist{0.6, 0.8};
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
  std::vector<std::string> legend{};
  ylm::Strahlkorper<Frame::Inertial> current_inertial_strahlkorper;
  for (size_t i = 0; i < number_of_times; ++i) {
    legend.resize(0);  // clear and reuse for next row of data
    strahlkorper_in_different_frame(
        make_not_null(&current_inertial_strahlkorper), expected_strahlkorper,
        temporary_domain, functions_of_time, gsl::at(times, i));
    ylm::change_expansion_center_of_strahlkorper_to_physical(
        make_not_null(&current_inertial_strahlkorper),
        evolution::Ringdown::expansion_center_tolerance);
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

  using Object = domain::creators::BinaryCompactObject<false>::Object;
  const domain::creators::BinaryCompactObject<false> domain_creator_bco{
      Object{0.1, 6., 8., true, true},
      Object{0.2, 6, -6., true, true},
      std::array<double, 2>{{0., 0.}},
      60.,
      300.,
      1.0,
      0_st,
      6_st,
      true,
      domain::CoordinateMaps::Distribution::Projective,
      std::vector<double>{},
      domain::CoordinateMaps::Distribution::Inverse,
      120.,
      time_dependent_map_options_bco};
  const auto domain_bco = domain_creator_bco.create_domain();
  const auto functions_of_time_bco = domain_creator_bco.functions_of_time();
  auto serialized_fots_bco = serialize(functions_of_time_bco);
  auto serialized_domain_bco = serialize(domain_bco);

  if (file_system::check_if_file_exists(volume_file_name)) {
    file_system::rm(volume_file_name, true);
  }
  h5::H5File<h5::AccessType::ReadWrite> h5_file{volume_file_name, true};
  auto& vol_file = h5_file.insert<h5::VolumeData>(volume_file_subfile_name);

  for (size_t i = 0; i < times.size(); i++) {
    vol_file.write_volume_data(
        i, times.at(i),
        {ElementVolumeData{
            "FakeElementName",
            {TensorComponent{"RandomTensor", DataVector{3, 0.0}}},
            {3},
            {Spectral::Basis::Legendre},
            {Spectral::Quadrature::GaussLobatto}}},
        serialized_domain_bco, serialized_fots_bco);
  }
  h5_file.close_current_object();

  // Call strahlkorper_coefs_and_centers()
  constexpr size_t times_to_retrieve{number_of_times - 2};
  const std::pair<std::vector<DataVector>, std::vector<std::array<double, 3>>>
      distorted_and_translation_coefs =
          evolution::Ringdown::strahlkorper_coefs_and_centers(
              volume_file_name, volume_file_subfile_name, horizons_file_name,
              horizons_subfile_name, times_to_retrieve, match_time,
              settling_timescale, exp_func_and_2_derivs,
              exp_outer_bdry_func_and_2_derivs, rot_func_and_2_derivs);

  // Checks
  // std::vector is the expected size
  const size_t times_retrieved = distorted_and_translation_coefs.first.size();
  CHECK(times_retrieved == times_to_retrieve);

  const auto distorted_coefs = distorted_and_translation_coefs.first;
  const auto translation_coefs = distorted_and_translation_coefs.second;

  // Check that retrieved coefs are the expected size
  const auto& expected_coefs = expected_strahlkorper.coefficients();
  const size_t coefs_size_expected = expected_coefs.size();
  const size_t coefs_size_retrieved = distorted_coefs[0].size();
  CHECK(coefs_size_expected == coefs_size_retrieved);
  CHECK(translation_coefs.size() == times_to_retrieve);

  if (file_system::check_if_file_exists(horizons_file_name)) {
    file_system::rm(horizons_file_name, true);
  }
  if (file_system::check_if_file_exists(volume_file_name)) {
    file_system::rm(volume_file_name, true);
  }
}
