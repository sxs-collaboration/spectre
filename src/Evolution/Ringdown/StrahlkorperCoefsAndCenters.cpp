// Distributed under the MIT License.
// See LICENSE.txt for details.
#include "Evolution/Ringdown/StrahlkorperCoefsAndCenters.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "DataStructures/Matrix.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordsToDifferentFrame.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/TimeDependentOptions/ExpansionMap.hpp"
#include "Domain/Creators/TimeDependentOptions/RotationMap.hpp"
#include "Domain/Creators/TimeDependentOptions/Sphere.hpp"
#include "Domain/Creators/TimeDependentOptions/TranslationMap.hpp"
#include "Domain/StrahlkorperTransformations.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Strahlkorper/ChangeCenterOfStrahlkorper.hpp"
#include "NumericalAlgorithms/Strahlkorper/IO/ReadSurfaceYlm.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace {
std::array<double, 3> extract_expansion_outer_boundary_func_and_2_derivs(
    const domain::FunctionsOfTimeMap& functions_of_time, const double time) {
  const auto function_of_time =
      functions_of_time.find("ExpansionOuterBoundary");
  if (function_of_time == functions_of_time.end()) {
    ERROR("ExpansionOuterBoundary function of time could not be found at time "
          << time << "M.");
  }
  const auto values = function_of_time->second->func_and_2_derivs(time);
  return std::array{values[0][0], values[1][0], values[2][0]};
}

std::vector<std::array<double, 4>> extract_rotation_func_and_2_derivs(
    const domain::FunctionsOfTimeMap& functions_of_time, const double time) {
  const auto function_of_time = functions_of_time.find("Rotation");
  if (function_of_time == functions_of_time.end()) {
    ERROR("Rotation function of time could not be found at "
          << "time " << time << "M.");
  }
  const auto values = function_of_time->second->func_and_2_derivs(time);
  std::vector<std::array<double, 4>> result(3);
  for (size_t deriv = 0; deriv < values.size(); ++deriv) {
    auto& result_deriv = result[deriv];
    for (size_t component = 0; component < 4; ++component) {
      gsl::at(result_deriv, component) = gsl::at(values, deriv)[component];
    }
  }
  return result;
}

std::array<std::array<double, 3>, 3> extract_translation_func_and_2_derivs(
    const domain::FunctionsOfTimeMap& functions_of_time, const double time) {
  const auto function_of_time = functions_of_time.find("Translation");
  if (function_of_time == functions_of_time.end()) {
    ERROR("Translation function of time could not be found at "
          << "time " << time << "M.");
  }
  const auto values = function_of_time->second->func_and_2_derivs(time);
  std::array<std::array<double, 3>, 3> result{};
  for (size_t deriv = 0; deriv < values.size(); ++deriv) {
    auto& result_deriv = gsl::at(result, deriv);
    for (size_t component = 0; component < 3; ++component) {
      gsl::at(result_deriv, component) = gsl::at(values, deriv)[component];
    }
  }
  return result;
}
}  // namespace

namespace evolution::Ringdown {
std::pair<std::vector<DataVector>, std::vector<std::array<double, 3>>>
strahlkorper_coefs_and_centers(
    const std::string& path_to_volume_data,
    const std::string& volume_subfile_name,
    const std::string& path_to_horizons_h5,
    const std::string& surface_subfile_name,
    const size_t requested_number_of_times_from_end, const double match_time,
    const double settling_timescale,
    const std::optional<std::array<double, 3>>& exp_func_and_2_derivs,
    const std::optional<std::array<double, 3>>&
        exp_outer_bdry_func_and_2_derivs,
    const std::optional<std::vector<std::array<double, 4>>>&
        rot_func_and_2_derivs,
    const std::optional<std::array<std::array<double, 3>, 3>>&
        trans_func_and_2_derivs) {
  // Read the AhC coefficients from the H5 file
  const std::vector<ylm::Strahlkorper<Frame::Inertial>>& ahc_inertial_h5 =
      ylm::read_surface_ylm<Frame::Inertial>(
          path_to_horizons_h5, surface_subfile_name,
          requested_number_of_times_from_end);
  std::vector<double> ahc_times{};
  {
    // Read the AhC times from the H5 file
    const h5::H5File<h5::AccessType::ReadOnly> ahc_h5_file{path_to_horizons_h5};
    const auto& dat = ahc_h5_file.get<h5::Dat>(surface_subfile_name);
    const Matrix& coefs_for_times = dat.get_data_subset(
        {0}, dat.get_dimensions()[0] - requested_number_of_times_from_end,
        ahc_inertial_h5.size());
    for (size_t i = 0; i < coefs_for_times.rows(); ++i) {
      ahc_times.push_back(coefs_for_times(i, 0));
    }
  }

  // Loop over the selected horizons, transforming each to the
  // temporary frame
  std::vector<DataVector> ahc_ringdown_distorted_coefs{};
  std::vector<std::array<double, 3>> ahc_inertial_centers{};
  // Here we transform the inertial strahlkorper into the temporary frame. In
  // order to do this, the inertial coords of the strahlkorper are mapped to the
  // logical frame to determine which block map to use, and then into the
  // temporary frame. This technically requires a shape map; however, we do not
  // yet know the shape map for the ringdown domain. Here, we use an identity
  // shape map instead because we are only concerned with the temporary frame,
  // not the correct ringdown grid frame. To avoid an unnecessary identity shape
  // map, we omit it in the domain above. This now makes the grid frame of this
  // temporary domain equivalent to the temporary frame. This is why we map the
  // strahlkorper into the "grid" frame instead of the "distorted" frame. It is
  // a simplification to avoid an unnecessary identity shape map.
  ylm::Strahlkorper<Frame::Grid> distorted_ahc;
  for (size_t i = 0; i < requested_number_of_times_from_end; ++i) {
    if (gsl::at(ahc_times, i) <= match_time) {
      const h5::H5File<h5::AccessType::ReadOnly> volume_file{
          path_to_volume_data};
      const auto& volume_data =
          volume_file.get<h5::VolumeData>(volume_subfile_name);

      const size_t obs_id_at_ahc_time =
          volume_data.find_observation_id(gsl::at(ahc_times, i), 1e-12);

      const auto serialized_inspiral_functions_of_time =
          volume_data.get_functions_of_time(obs_id_at_ahc_time);
      if (not serialized_inspiral_functions_of_time.has_value()) {
        ERROR(
            "No functions of time found in volume files at the specified "
            "time "
            << gsl::at(ahc_times, i) << "M.");
      }
      const auto inspiral_functions_of_time =
          deserialize<domain::FunctionsOfTimeMap>(
              serialized_inspiral_functions_of_time->data());

      const auto expansion_map_options_from_volume =
          exp_outer_bdry_func_and_2_derivs.has_value()
              ? domain::creators::time_dependent_options::ExpansionMapOptions<
                    true>{exp_func_and_2_derivs.value(), settling_timescale,
                          extract_expansion_outer_boundary_func_and_2_derivs(
                              inspiral_functions_of_time,
                              gsl::at(ahc_times, i)),
                          settling_timescale}
              : std::optional<domain::creators::time_dependent_options::
                                  ExpansionMapOptions<true>>{};
      const auto rotation_map_options_from_volume =
          rot_func_and_2_derivs.has_value()
              ? domain::creators::time_dependent_options::RotationMapOptions<
                    true>{extract_rotation_func_and_2_derivs(
                              inspiral_functions_of_time,
                              gsl::at(ahc_times, i)),
                          settling_timescale}
              : std::optional<domain::creators::time_dependent_options::
                                  RotationMapOptions<true>>{};
      const auto translation_map_options_from_volume =
          trans_func_and_2_derivs.has_value()
              ? domain::creators::time_dependent_options::TranslationMapOptions<
                    3>{extract_translation_func_and_2_derivs(
                    inspiral_functions_of_time, gsl::at(ahc_times, i))}
              : std::optional<domain::creators::time_dependent_options::
                                  TranslationMapOptions<3>>{};

      const domain::creators::sphere::TimeDependentMapOptions
          time_dependent_map_options{gsl::at(ahc_times, i),
                                     std::nullopt,
                                     rotation_map_options_from_volume,
                                     expansion_map_options_from_volume,
                                     translation_map_options_from_volume,
                                     true,
                                     std::nullopt};
      // We construct a temporary ringdown domain that will be used to transform
      // the common horizon from the inspiral inertial frame to the temporary
      // frame (ringdown intermediate frame in SpEC) for shape coefficients and
      // then from the temporary frame to the inertial frame for geometric
      // center positions for the ringdown translation map. We choose an inner
      // radius at machine precision so that every point on any Strahlkorper
      // transformed to this temporary ringdown domain will be mapped to a
      // block. It's assumed that the Strahlkorper has not translated beyond
      // 200M and has a radius < 200M in the inertial frame. The resolution on
      // this domain does not matter because the actual grid points in the
      // temporary ringdown domain are never used. We are only after how the
      // shape coefficients and center changes as you transform the Strahlkorper
      // to this temporary ringdown domain. The only member functions of the
      // domain that are used are the coordinate maps (and associated functions
      // of time) that map the "grid frame" of the temporary ringdown domain to
      // the inertial frame of the temporary ringdown domain. The "grid frame"
      // of the temporary ringdown domain is the inertial frame pushed through
      // the inverse-rotation-expansion-translation map, where "translation" is
      // the inspiral's translation map, and "rotation" and "expansion" are new
      // rotation and expansion maps that match the inspiral's rotation and
      // expansion at the outer boundary at the transition time, have some
      // smooth falloff in the interior, and have a "settle to constant" time
      // dependence. Thus the "grid frame" of the temporary domain is almost
      // (except for an uncorrected translation map) the distorted frame of the
      // ringdown. The "almost" is because in the "grid frame" of the temporary
      // ringdown domain, the center of the excision boundary moves, and is not
      // at the origin.
      const domain::creators::Sphere domain_creator_non_replay{
          // Inner radius and outer radius chosen so that every point on a
          // strahlkorper transformed to this domain will be mapped to a block.
          std::numeric_limits<double>::epsilon(),
          200.0,
          // nullptr because no boundary condition
          domain::creators::Sphere::Excision{nullptr},
          static_cast<size_t>(0),
          static_cast<size_t>(5),
          false,
          std::nullopt,
          // Radial partition used to separate Shell0/1 in the ringdown domain.
          // This value is what is currently used in the Bbh pipeline
          {50.0},
          domain::CoordinateMaps::Distribution::Linear,
          ShellWedges::All,
          time_dependent_map_options};
      const auto temporary_ringdown_domain_non_replay =
          domain_creator_non_replay.create_domain();
      const auto ringdown_functions_of_time_non_replay =
          domain_creator_non_replay.functions_of_time();
      strahlkorper_in_different_frame(
          make_not_null(&distorted_ahc), gsl::at(ahc_inertial_h5, i),
          temporary_ringdown_domain_non_replay,
          ringdown_functions_of_time_non_replay, gsl::at(ahc_times, i));
      // Relative tolerance is set to the value used in SpEC
      ylm::change_expansion_center_of_strahlkorper_to_physical(
          make_not_null(&distorted_ahc), expansion_center_tolerance);
      ahc_ringdown_distorted_coefs.push_back(distorted_ahc.coefficients());

      tnsr::I<DataVector, 3, ::Frame::Grid> grid_center_point{
          DataVector{1, 0.0}};
      grid_center_point[0] = distorted_ahc.expansion_center()[0];
      grid_center_point[1] = distorted_ahc.expansion_center()[1];
      grid_center_point[2] = distorted_ahc.expansion_center()[2];
      tnsr::I<DataVector, 3, ::Frame::Inertial> inertial_center_point{
          DataVector{1, 0.0}};
      // The center point is mapped to the inertial frame so that the geometric
      // center of AhC accounts for the inspiral's rotation, scaling and
      // translation. This ensures that the center of the excision is at the
      // correct location at the match time
      coords_to_different_frame(
          make_not_null(&inertial_center_point), grid_center_point,
          temporary_ringdown_domain_non_replay,
          ringdown_functions_of_time_non_replay, gsl::at(ahc_times, i));

      ahc_inertial_centers.push_back(std::array<double, 3>{
          get<0>(inertial_center_point)[0], get<1>(inertial_center_point)[0],
          get<2>(inertial_center_point)[0]});
    }
  }

  return std::pair{ahc_ringdown_distorted_coefs, ahc_inertial_centers};
}
}  // namespace evolution::Ringdown
