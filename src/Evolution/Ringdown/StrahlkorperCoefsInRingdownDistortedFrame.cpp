// Distributed under the MIT License.
// See LICENSE.txt for details.
#include "Evolution/Ringdown/StrahlkorperCoefsInRingdownDistortedFrame.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

#include "DataStructures/Matrix.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/TimeDependentOptions/Sphere.hpp"
#include "Domain/StrahlkorperTransformations.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/IO/ReadSurfaceYlm.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::Ringdown {
std::vector<DataVector> strahlkorper_coefs_in_ringdown_distorted_frame(
    const std::string& path_to_horizons_h5,
    const std::string& surface_subfile_name,
    const size_t requested_number_of_times_from_end, const double match_time,
    const double settling_timescale,
    const std::array<double, 3>& exp_func_and_2_derivs,
    const std::array<double, 3>& exp_outer_bdry_func_and_2_derivs,
    const std::array<std::array<double, 4>, 3>& rot_func_and_2_derivs) {
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
  // Create a time-dependent domain; only the the time-dependent map options
  // matter; the domain is just a spherical shell with inner and outer
  // radii chosen so any conceivable common horizon will fit between them.
  const domain::creators::sphere::TimeDependentMapOptions::ExpansionMapOptions
      expansion_map_options{exp_func_and_2_derivs, settling_timescale,
                            exp_outer_bdry_func_and_2_derivs,
                            settling_timescale};
  const domain::creators::sphere::TimeDependentMapOptions::RotationMapOptions
      rotation_map_options{rot_func_and_2_derivs, settling_timescale};
  const domain::creators::sphere::TimeDependentMapOptions
      time_dependent_map_options{match_time, std::nullopt, rotation_map_options,
                                 expansion_map_options, std::nullopt};
  const domain::creators::Sphere domain_creator{
      0.01,
      200.0,
      // nullptr because no boundary condition
      domain::creators::Sphere::Excision{nullptr},
      static_cast<size_t>(0),
      static_cast<size_t>(5),
      false,
      std::nullopt,
      {100.0},
      domain::CoordinateMaps::Distribution::Linear,
      ShellWedges::All,
      time_dependent_map_options};

  const auto temporary_domain = domain_creator.create_domain();
  const auto functions_of_time = domain_creator.functions_of_time();

  // Loop over the selected horizons, transforming each to the
  // ringdown distorted frame
  std::vector<DataVector> ahc_ringdown_distorted_coefs{};
  // Here we transform the inertial strahlkorper into the ringdown distorted
  // frame. In order to do this, the inertial coords of the strahlkorper are
  // mapped to the logical frame to determine which block map to use, and then
  // into the ringdown distorted frame. This technically requires a shape map,
  // however, at this time we do not yet know the shape map for the ringdown
  // domain so we can use an identity instead because we are only concerned with
  // the correct ringdown distorted frame, not the correct ringdown grid frame.
  // To avoid an unnecessary identity shape map, we omit it in the domain above.
  // This now makes the grid frame of this temporary domain equivalent to the
  // true ringdown distorted frame we are after. This is why we map the
  // strahlkorper into the "grid" frame instead of the "distorted" frame. It is
  // a simplification to avoid an unnecessary identity shape map.
  ylm::Strahlkorper<Frame::Grid> current_ahc;
  for (size_t i = 0; i < requested_number_of_times_from_end; ++i) {
    strahlkorper_in_different_frame(
        make_not_null(&current_ahc), gsl::at(ahc_inertial_h5, i),
        temporary_domain, functions_of_time, gsl::at(ahc_times, i));
    ahc_ringdown_distorted_coefs.push_back(current_ahc.coefficients());
  }

  return ahc_ringdown_distorted_coefs;
}
}  // namespace evolution::Ringdown
