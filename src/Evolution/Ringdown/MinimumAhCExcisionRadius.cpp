// Distributed under the MIT License.
// See LICENSE.txt for details.
#include "Evolution/Ringdown/MinimumAhCExcisionRadius.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordsToDifferentFrame.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/TimeDependentOptions/BinaryCompactObject.hpp"
#include "Domain/Creators/TimeDependentOptions/ExpansionMap.hpp"
#include "Domain/Creators/TimeDependentOptions/RotationMap.hpp"
#include "Domain/Creators/TimeDependentOptions/ShapeMap.hpp"
#include "Domain/Creators/TimeDependentOptions/Sphere.hpp"
#include "Domain/StrahlkorperTransformations.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/IO/ReadSurfaceYlm.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace {
void get_grid_point(
    const BlockLogicalCoords<3>& block_logical_coord,
    const Domain<3>& ringdown_domain,
    const domain::FunctionsOfTimeMap& ringdown_functions_of_time,
    const tnsr::I<double, 3, Frame::Inertial>& exc_inertial_frame_point,
    tnsr::I<double, 3, Frame::Grid>& exc_grid_frame_point, double match_time) {
  // If the point is mapped to a block then the ringdown excision radius
  // chosen does not enclose excisions A/B from the inspiral
  if (block_logical_coord.has_value()) {
    const auto& block_id_and_coords = block_logical_coord.value();
    const auto& block =
        ringdown_domain.blocks()[block_id_and_coords.id.get_index()];
    const auto& block_to_grid_map = block.moving_mesh_logical_to_grid_map();
    const auto grid_point = block_to_grid_map(
        block_id_and_coords.data, match_time, ringdown_functions_of_time);
    get<0>(exc_grid_frame_point) = grid_point[0];
    get<1>(exc_grid_frame_point) = grid_point[1];
    get<2>(exc_grid_frame_point) = grid_point[2];
  } else {
    // This point is inside the excision which is why it couldn't be
    // mapped to a block. Now we must use the shape map inverse to
    // determine where this point is.
    const auto inv_point = ringdown_domain.excision_spheres()
                               .at("ExcisionSphere")
                               .moving_mesh_grid_to_inertial_map()
                               .inverse(exc_inertial_frame_point, match_time,
                                        ringdown_functions_of_time);
    if (inv_point.has_value()) {
      get<0>(exc_grid_frame_point) = inv_point.value()[0];
      get<1>(exc_grid_frame_point) = inv_point.value()[1];
      get<2>(exc_grid_frame_point) = inv_point.value()[2];
    } else {
      ERROR("Map from Frame::Inertial to Frame::Grid is not invertible");
    }
  }
}
}  // namespace

namespace evolution::Ringdown {
double minimum_ahc_excision_radius(
    const std::string& path_to_volume_data,
    const std::string& volume_subfile_name,
    const std::string& path_to_horizons_h5,
    const std::string& surface_subfile_name,
    const std::string& path_to_AhC_distorted_h5,
    const std::vector<std::string>& AhC_distorted_subfile_names,
    double match_time, double settling_timescale, double excision_A_radius,
    double excision_B_radius, std::array<double, 3> excision_A_center,
    std::array<double, 3> excision_B_center,
    const std::optional<std::array<double, 3>>& exp_func_and_2_derivs,
    const std::optional<std::array<double, 3>>&
        exp_outer_bdry_func_and_2_derivs,
    const std::optional<std::vector<std::array<double, 4>>>&
        rot_func_and_2_derivs,
    const std::optional<std::array<std::array<double, 3>, 3>>&
        trans_func_and_2_derivs,
    double match_time_tol) {
  // Read the AhC coefficients from the H5 file
  const ylm::Strahlkorper<Frame::Inertial>& ahc_inertial_at_match_time =
      ylm::read_surface_ylm_single_time<Frame::Inertial>(
          path_to_horizons_h5, surface_subfile_name, match_time,
          match_time_tol);

  const h5::H5File<h5::AccessType::ReadOnly> volume_file{path_to_volume_data};
  const auto& volume_data =
      volume_file.get<h5::VolumeData>(volume_subfile_name);

  const size_t obs_id_at_match_time =
      volume_data.find_observation_id(match_time, match_time_tol);

  const auto serialized_inspiral_domain = volume_data.get_domain();
  if (not serialized_inspiral_domain.has_value()) {
    ERROR("No domain found in volume files at the specified match time.");
  }
  const auto inspiral_domain =
      deserialize<Domain<3>>(serialized_inspiral_domain->data());

  const auto serialized_inspiral_functions_of_time =
      volume_data.get_functions_of_time(obs_id_at_match_time);
  if (not serialized_inspiral_functions_of_time.has_value()) {
    ERROR("No functions of time found in volume files at the match time.");
  }
  const auto inspiral_functions_of_time =
      deserialize<domain::FunctionsOfTimeMap>(
          serialized_inspiral_functions_of_time->data());

  // Ringdown functions of time computed using
  // ComputeRingdownShapeAndTranslationFoT.py
  const auto shape_map_options =
      domain::creators::time_dependent_options::ShapeMapOptions<
          false, domain::ObjectLabel::None>{
          ahc_inertial_at_match_time.l_max(),
          domain::creators::time_dependent_options::YlmsFromFile{
              path_to_AhC_distorted_h5, AhC_distorted_subfile_names, match_time,
              match_time_tol, true, true}};
  const auto expansion_map_options =
      exp_func_and_2_derivs.has_value()
          ? domain::creators::time_dependent_options::ExpansionMapOptions<
                true>{exp_func_and_2_derivs.value(), settling_timescale,
                      exp_outer_bdry_func_and_2_derivs.value(),
                      settling_timescale}
          : std::optional<domain::creators::time_dependent_options::
                              ExpansionMapOptions<true>>{};
  const auto rotation_map_options =
      rot_func_and_2_derivs.has_value()
          ? domain::creators::time_dependent_options::RotationMapOptions<
                true>{rot_func_and_2_derivs.value(), settling_timescale}
          : std::optional<domain::creators::time_dependent_options::
                              RotationMapOptions<true>>{};
  const auto translation_map_options =
      trans_func_and_2_derivs.has_value()
          ? domain::creators::sphere::TimeDependentMapOptions::
                TranslationMapOptions{trans_func_and_2_derivs.value()}
          : std::optional<domain::creators::sphere::TimeDependentMapOptions::
                              TranslationMapOptions>{};

  const domain::creators::sphere::TimeDependentMapOptions
      ringdown_time_dependent_map_options{match_time,
                                          shape_map_options,
                                          rotation_map_options,
                                          expansion_map_options,
                                          translation_map_options,
                                          true,
                                          std::nullopt};

  const double ahc_average_radius = ahc_inertial_at_match_time.average_radius();
  const std::array<double, 3> ahc_ringdown_center =
      trans_func_and_2_derivs.has_value()
          ? trans_func_and_2_derivs.value()[0]
          : std::array<double, 3>{0.0, 0.0, 0.0};
  double ringdown_excision_factor = 0.94;
  // When constructing our domain in the inspiral, we explicitly choose the grid
  // frame centers of the excision boundaries so that the Newtonian coordinate
  // center of mass is at the origin. This allows us to use the formula below to
  // get the mass ratio. If we ever change how we construct our domain, then
  // this formula will no longer be true.
  const double mass_ratio = abs(excision_B_center[0] / excision_A_center[0]);
  const double eps = 1e-3 / mass_ratio;
  bool outer_converged = false;
  size_t current_outer_iteration = 0;
  const size_t max_iterations = 10;
  size_t current_l_max = 20;

  // The outer loop checks that multiple l_max values for the AhA/AhB excisions
  // fit inside the ringdown domain excision. The reason we iterate and check
  // for multiple l_max values is because a single high l_max value might not
  // sample the excision surfaces finely enough to yield an accurate
  // ringdown_excision_factor, especially for extremely distorted grids. So we
  // iterate over different l_max and demand convergence. This is more efficient
  // and robust than simply choosing a large l_max without iterating
  while (not outer_converged and current_outer_iteration < max_iterations) {
    // Section for constructing strahlkorpers for AhA/AhB
    const ylm::Strahlkorper<Frame::Grid> excision_a_inspiral_grid(
        current_l_max, excision_A_radius, excision_A_center);
    const ylm::Strahlkorper<Frame::Grid> excision_b_inspiral_grid(
        current_l_max, excision_B_radius, excision_B_center);

    const size_t points_size =
        get<0>(ylm::cartesian_coords(excision_a_inspiral_grid)).size();
    tnsr::I<DataVector, 3, Frame::Inertial> excision_a_inspiral_inertial_points{
        points_size};
    tnsr::I<DataVector, 3, Frame::Inertial> excision_b_inspiral_inertial_points{
        points_size};
    coords_to_different_frame(
        make_not_null(&excision_a_inspiral_inertial_points),
        ylm::cartesian_coords(excision_a_inspiral_grid), inspiral_domain,
        inspiral_functions_of_time, match_time);
    coords_to_different_frame(
        make_not_null(&excision_b_inspiral_inertial_points),
        ylm::cartesian_coords(excision_b_inspiral_grid), inspiral_domain,
        inspiral_functions_of_time, match_time);

    // Loop section finding the correct rmin_fac
    const double previous_outer_ringdown_excision_factor =
        ringdown_excision_factor;
    size_t current_inner_iteration = 0;
    bool inner_converged = false;

    // The inner loop changes the excision radius of the ringdown domain until
    // the previous ringdown radius and current are within a set tolerance
    while (not inner_converged and current_inner_iteration < max_iterations) {
      const double previous_inner_ringdown_excision_factor =
          ringdown_excision_factor;
      // This domain serves as a test domain for the ringdown using the
      // corrected FoTs generated by ComputeRingdownShapeAndTranslationFoT.py
      // which gives the shape map and corrected translation map that will be
      // used in the final ringdown. This domain is only used to transform the
      // excisions A/B from the inspiral to the ringdown grid frame to check if
      // the current excision radius chosen encloses excisions A/B, so the
      // resolution and outer radius of this test domain does not matter as this
      // domain will not be used to evolve anything.
      const domain::creators::Sphere ringdown_rmin_domain_creator{
          ahc_average_radius * ringdown_excision_factor,
          200.0,
          // nullptr because no boundary condition
          domain::creators::Sphere::Excision{nullptr},
          static_cast<size_t>(0),
          static_cast<size_t>(5),
          false,
          std::nullopt,
          {50.0},
          domain::CoordinateMaps::Distribution::Linear,
          ShellWedges::All,
          ringdown_time_dependent_map_options};

      const auto temporary_ringdown_rmin_domain =
          ringdown_rmin_domain_creator.create_domain();
      const auto ringdown_rmin_functions_of_time =
          ringdown_rmin_domain_creator.functions_of_time();

      const auto exc_a_block_logical = block_logical_coordinates(
          temporary_ringdown_rmin_domain, excision_a_inspiral_inertial_points,
          match_time, ringdown_rmin_functions_of_time);
      const auto exc_b_block_logical = block_logical_coordinates(
          temporary_ringdown_rmin_domain, excision_b_inspiral_inertial_points,
          match_time, ringdown_rmin_functions_of_time);

      const tnsr::I<DataVector, 3, Frame::Inertial>
          excision_a_ringdown_grid_points{points_size};
      const tnsr::I<DataVector, 3, Frame::Inertial>
          excision_b_ringdown_grid_points{points_size};

      double min_excision_radius = 0.0;
      tnsr::I<double, 3, Frame::Inertial> exc_a_inspiral_inertial_point{};
      tnsr::I<double, 3, Frame::Grid> exc_a_ringdown_grid_point{};
      tnsr::I<double, 3, Frame::Inertial> exc_b_inspiral_inertial_point{};
      tnsr::I<double, 3, Frame::Grid> exc_b_ringdown_grid_point{};
      for (size_t s = 0; s < get<0>(excision_a_inspiral_inertial_points).size();
           ++s) {
        get<0>(exc_a_inspiral_inertial_point) =
            get<0>(excision_a_inspiral_inertial_points)[s];
        get<1>(exc_a_inspiral_inertial_point) =
            get<1>(excision_a_inspiral_inertial_points)[s];
        get<2>(exc_a_inspiral_inertial_point) =
            get<2>(excision_a_inspiral_inertial_points)[s];
        get<0>(exc_b_inspiral_inertial_point) =
            get<0>(excision_b_inspiral_inertial_points)[s];
        get<1>(exc_b_inspiral_inertial_point) =
            get<1>(excision_b_inspiral_inertial_points)[s];
        get<2>(exc_b_inspiral_inertial_point) =
            get<2>(excision_b_inspiral_inertial_points)[s];

        get_grid_point(exc_a_block_logical[s], temporary_ringdown_rmin_domain,
                       ringdown_rmin_functions_of_time,
                       exc_a_inspiral_inertial_point, exc_a_ringdown_grid_point,
                       match_time);

        get_grid_point(exc_b_block_logical[s], temporary_ringdown_rmin_domain,
                       ringdown_rmin_functions_of_time,
                       exc_b_inspiral_inertial_point, exc_b_ringdown_grid_point,
                       match_time);

        const double excision_a_point_radius = sqrt(
            square(get<0>(exc_a_ringdown_grid_point) - ahc_ringdown_center[0]) +
            square(get<1>(exc_a_ringdown_grid_point) - ahc_ringdown_center[1]) +
            square(get<2>(exc_a_ringdown_grid_point) - ahc_ringdown_center[2]));
        const double excision_b_point_radius = sqrt(
            square(get<0>(exc_b_ringdown_grid_point) - ahc_ringdown_center[0]) +
            square(get<1>(exc_b_ringdown_grid_point) - ahc_ringdown_center[1]) +
            square(get<2>(exc_b_ringdown_grid_point) - ahc_ringdown_center[2]));
        min_excision_radius =
            std::max(min_excision_radius, excision_a_point_radius);
        min_excision_radius =
            std::max(min_excision_radius, excision_b_point_radius);
      }
      const double min_ringdown_excision_factor =
          min_excision_radius / ahc_average_radius;
      // This line changes the excision radius factor to be 3/4 of the way
      // between the minimum possible value to fit excisions A/B inside the
      // ringdown excision and the common horizon average radius
      ringdown_excision_factor =
          1.0 - 0.25 * (1.0 - min_ringdown_excision_factor);
      if (current_inner_iteration != 0 and
          abs(ringdown_excision_factor -
              previous_inner_ringdown_excision_factor) <= 0.5 * eps) {
        inner_converged = true;
      } else {
        current_inner_iteration++;
      }
    }
    if (current_outer_iteration != 0 and
        abs(ringdown_excision_factor -
            previous_outer_ringdown_excision_factor) <= 0.5 * eps) {
      outer_converged = true;
    }
    current_outer_iteration++;
    // Increment l max by 6 every iteration.
    current_l_max += 6;
    if (current_outer_iteration > max_iterations) {
      ERROR(
          "Max Iterations for finding a suitable excision radius exceeded. "
          "Going to sleep.");
    }
  }
  // This section makes a safe_ringdown_excision_factor by truncating any
  // significant figures past eps and then adds a bit of safety to the
  // safe_ringdown_excision_factor and compares to the original
  // ringdown_excision_factor. If it's too close, add more safety padding, then,
  // whichever is larger, choose that for the excision radius
  double safe_ringdown_excision_factor =
      static_cast<int>(ringdown_excision_factor / eps);
  safe_ringdown_excision_factor *= eps;
  safe_ringdown_excision_factor += eps;
  if (safe_ringdown_excision_factor - ringdown_excision_factor < 0.5 * eps) {
    safe_ringdown_excision_factor += eps;
  }
  const double ringdown_excision_radius =
      ringdown_excision_factor > safe_ringdown_excision_factor
          ? ahc_average_radius * ringdown_excision_factor
          : ahc_average_radius * safe_ringdown_excision_factor;

  // Checks for valid excision radius and potential issues
  if (ringdown_excision_radius > ahc_average_radius + eps) {
    ERROR("The excision radius chosen "
          << ringdown_excision_radius
          << " is larger than the radius of the common horizon "
          << ahc_average_radius
          << ". Either the common horizon does not actually contain the excised"
             " regions, or there is a problem with the functions of time used, "
             "or an error has occured in the excision radius finding routine.");
  } else if (ringdown_excision_radius > ahc_average_radius - 3 * eps) {
    ERROR(
        "The excision radius chosen is very close to the radius of the common "
        "horizon. Either the tolerances should be increased, or a later match "
        "time should be chosen to increase the distance between the excised "
        "regions and the common horizon.");
  }

  return ringdown_excision_radius;
}
}  // namespace evolution::Ringdown
