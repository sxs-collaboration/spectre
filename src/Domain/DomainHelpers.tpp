// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/DomainHelpers.hpp"

#include <cstddef>
#include <memory>
#include <vector>

#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/ShellType.hpp"
#include "Domain/CoordinateMaps/SphericalToCartesianPfaffian.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

template <typename SourceFrame, typename TargetFrame, typename... AppendMaps>
std::vector<
    std::unique_ptr<domain::CoordinateMapBase<SourceFrame, TargetFrame, 3>>>
spherical_shells_coordinate_maps(
    const double inner_radius, const double outer_radius,
    const double inner_sphericity, const double outer_sphericity,
    const bool use_equiangular_map,
    const std::optional<std::pair<double, std::array<double, 3>>>&
        offset_options,
    const bool use_half_wedges, const std::vector<double>& radial_partitioning,
    const std::vector<domain::CoordinateMaps::Distribution>&
        radial_distribution,
    const std::vector<domain::CoordinateMaps::ShellType>& shell_types,
    const ShellWedges which_wedges, const double opening_angle,
    const AppendMaps&... append_maps) {
  ASSERT(not use_half_wedges or which_wedges == ShellWedges::All,
         "If we are using half wedges we must also be using ShellWedges::All.");
  ASSERT(radial_distribution.size() == 1 + radial_partitioning.size(),
         "Specify a radial distribution for every spherical shell. You "
         "specified "
             << radial_distribution.size() << " items, but the domain has "
             << 1 + radial_partitioning.size() << " shells.");
  ASSERT(shell_types.size() == 1 + radial_partitioning.size(),
         "Specify a type for every spherical shell. You specified "
             << shell_types.size() << " items, but the domain has "
             << 1 + radial_partitioning.size() << " shells.");

  const auto wedge_orientations = orientations_for_sphere_wrappings();

  std::vector<
      std::unique_ptr<domain::CoordinateMapBase<SourceFrame, TargetFrame, 3>>>
      maps{};

  const size_t number_of_layers = 1 + radial_partitioning.size();
  double temp_inner_radius = inner_radius;
  double temp_inner_sphericity = inner_sphericity;
  if (offset_options.has_value()) {
    ERROR("TODO");
  } else {
    double temp_outer_radius{};
    for (size_t layer_i = 0; layer_i < number_of_layers; layer_i++) {
      // Determine inner and outer radius for this shell
      if (layer_i != radial_partitioning.size()) {
        temp_outer_radius = radial_partitioning.at(layer_i);
      } else {
        temp_outer_radius = outer_radius;
      }
      if (shell_types[layer_i] == domain::CoordinateMaps::ShellType::Cubed) {
        // Compose the shell of wedges (deformed cubes)
        using Wedge3DMap = domain::CoordinateMaps::Wedge<3>;
        using Halves = Wedge3DMap::WedgeHalves;
        std::vector<Wedge3DMap> wedges{};
        if (not use_half_wedges) {
          for (size_t face_j = which_wedge_index(which_wedges); face_j < 6;
               face_j++) {
            wedges.emplace_back(
                temp_inner_radius, temp_outer_radius, temp_inner_sphericity,
                outer_sphericity, gsl::at(wedge_orientations, face_j),
                use_equiangular_map, Halves::Both, radial_distribution[layer_i],
                std::array<double, 2>({{M_PI_2, M_PI_2}}));
          }
        } else {
          for (size_t i = 0; i < 4; i++) {
            wedges.emplace_back(
                temp_inner_radius, temp_outer_radius, temp_inner_sphericity,
                outer_sphericity, gsl::at(wedge_orientations, i),
                use_equiangular_map, Halves::LowerOnly,
                radial_distribution[layer_i],
                std::array<double, 2>({{opening_angle, M_PI_2}}));
            wedges.emplace_back(
                temp_inner_radius, temp_outer_radius, temp_inner_sphericity,
                outer_sphericity, gsl::at(wedge_orientations, i),
                use_equiangular_map, Halves::UpperOnly,
                radial_distribution[layer_i],
                std::array<double, 2>({{opening_angle, M_PI_2}}));
          }
          const double endcap_opening_angle = M_PI - opening_angle;
          const std::array<double, 2> endcap_opening_angles = {
              {endcap_opening_angle, endcap_opening_angle}};
          wedges.emplace_back(
              temp_inner_radius, temp_outer_radius, temp_inner_sphericity,
              outer_sphericity, gsl::at(wedge_orientations, 4),
              use_equiangular_map, Halves::Both, radial_distribution[layer_i],
              endcap_opening_angles, false);
          wedges.emplace_back(
              temp_inner_radius, temp_outer_radius, temp_inner_sphericity,
              outer_sphericity, gsl::at(wedge_orientations, 5),
              use_equiangular_map, Halves::Both, radial_distribution[layer_i],
              endcap_opening_angles, false);
        }
        auto wedge_maps =
            domain::make_vector_coordinate_map_base<SourceFrame, TargetFrame,
                                                    3>(std::move(wedges),
                                                       append_maps...);
        maps.insert(maps.end(), std::make_move_iterator(wedge_maps.begin()),
                    std::make_move_iterator(wedge_maps.end()));
      } else if (shell_types[layer_i] ==
                 domain::CoordinateMaps::ShellType::Spherical) {
        // Use a single spherical shell (with spherical harmonics in angular
        // directions)
        ASSERT(
            inner_sphericity == 1. and outer_sphericity == 1.,
            "Spherical shells only support inner and outer sphericity == 1.");
        domain::CoordinateMaps::ProductOf3Maps<
            domain::CoordinateMaps::Interval,
            domain::CoordinateMaps::Identity<1>,
            domain::CoordinateMaps::Identity<1>>
            logical_to_spherical_map{{-1., 1., temp_inner_radius, outer_radius,
                                      radial_distribution[layer_i], 0.},
                                     {},
                                     {}};
        maps.push_back(
            domain::make_coordinate_map_base<SourceFrame, TargetFrame>(
                std::move(logical_to_spherical_map),
                domain::CoordinateMaps::SphericalToCartesianPfaffian{},
                append_maps...));
      } else {
        ERROR("Unknown ShellType");
      }

      if (layer_i != radial_partitioning.size()) {
        temp_inner_radius = radial_partitioning.at(layer_i);
        temp_inner_sphericity = outer_sphericity;
      }
    } // for layer
  }   // if offset_options
  return maps;
}
