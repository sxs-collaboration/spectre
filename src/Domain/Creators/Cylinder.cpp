// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Cylinder.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Utilities/MakeArray.hpp"

namespace Frame {
struct Logical;
struct Inertial;
}  // namespace Frame

namespace domain::creators {
Cylinder::Cylinder(
    typename InnerRadius::type inner_radius,
    typename OuterRadius::type outer_radius,
    typename LowerBound::type lower_bound,
    typename UpperBound::type upper_bound,
    typename IsPeriodicInZ::type is_periodic_in_z,
    typename InitialRefinement::type initial_refinement,
    typename InitialGridPoints::type initial_number_of_grid_points,
    typename UseEquiangularMap::type use_equiangular_map,
    typename RadialPartitioning::type radial_partitioning,
    typename HeightPartitioning::type height_partitioning) noexcept
    // clang-tidy: trivially copyable
    : inner_radius_(std::move(inner_radius)),                // NOLINT
      outer_radius_(std::move(outer_radius)),                // NOLINT
      lower_bound_(std::move(lower_bound)),                  // NOLINT
      upper_bound_(std::move(upper_bound)),                  // NOLINT
      is_periodic_in_z_(std::move(is_periodic_in_z)),        // NOLINT
      initial_refinement_(                                   // NOLINT
          std::move(initial_refinement)),                    // NOLINT
      initial_number_of_grid_points_(                        // NOLINT
          std::move(initial_number_of_grid_points)),         // NOLINT
      use_equiangular_map_(use_equiangular_map),             // NOLINT
      radial_partitioning_(std::move(radial_partitioning)),  // NOLINT
      height_partitioning_(std::move(height_partitioning)) {}

Domain<3> Cylinder::create_domain() const noexcept {
  const size_t number_of_shells = 1 + radial_partitioning_.size();
  const size_t number_of_discs = 1 + height_partitioning_.size();
  std::vector<PairOfFaces> pairs_of_faces{};
  if (is_periodic_in_z_) {
    // connect faces of end caps in the periodic z-direction
    const size_t corners_per_layer = 4 * (number_of_shells + 1);
    const size_t num_corners = number_of_discs * corners_per_layer;
    PairOfFaces center{
        {0, 1, 2, 3},
        {num_corners + 0, num_corners + 1, num_corners + 2, num_corners + 3}};
    pairs_of_faces.push_back(std::move(center));
    for (size_t j = 0; j < number_of_shells; j++) {
      PairOfFaces east{{1 + 4 * j, 5 + 4 * j, 3 + 4 * j, 7 + 4 * j},
                       {num_corners + 4 * j + 1, num_corners + 4 * j + 5,
                        num_corners + 4 * j + 3, num_corners + 4 * j + 7}};
      PairOfFaces north{{3 + 4 * j, 7 + 4 * j, 2 + 4 * j, 6 + 4 * j},
                        {num_corners + 4 * j + 3, num_corners + 4 * j + 7,
                         num_corners + 4 * j + 2, num_corners + 4 * j + 6}};
      PairOfFaces west{{2 + 4 * j, 6 + 4 * j, 0 + 4 * j, 4 + 4 * j},
                       {num_corners + 4 * j + 2, num_corners + 4 * j + 6,
                        num_corners + 4 * j + 0, num_corners + 4 * j + 4}};
      PairOfFaces south{{0 + 4 * j, 4 + 4 * j, 1 + 4 * j, 5 + 4 * j},
                        {num_corners + 4 * j + 0, num_corners + 4 * j + 4,
                         num_corners + 4 * j + 1, num_corners + 4 * j + 5}};
      pairs_of_faces.push_back(std::move(east));
      pairs_of_faces.push_back(std::move(north));
      pairs_of_faces.push_back(std::move(west));
      pairs_of_faces.push_back(std::move(south));
    }
  }
  return Domain<3>{
      cyl_wedge_coordinate_maps<Frame::Inertial>(
          inner_radius_, outer_radius_, lower_bound_, upper_bound_,
          use_equiangular_map_, radial_partitioning_, height_partitioning_),
      corners_for_cylindrical_layered_domains(number_of_shells,
                                              number_of_discs),
      pairs_of_faces};
}

std::vector<std::array<size_t, 3>> Cylinder::initial_extents() const noexcept {
  std::vector<std::array<size_t, 3>> gridpoints_vector;
  for (size_t layer = 0; layer < 1 + height_partitioning_.size(); layer++) {
    gridpoints_vector.push_back({{initial_number_of_grid_points_.at(1),
                                  initial_number_of_grid_points_.at(1),
                                  initial_number_of_grid_points_.at(2)}});
    for (size_t shell = 0; shell < 1 + radial_partitioning_.size(); shell++) {
      for (size_t face = 0; face < 4; face++) {
        gridpoints_vector.push_back(initial_number_of_grid_points_);
      }
    }
  }
  return gridpoints_vector;
}

std::vector<std::array<size_t, 3>> Cylinder::initial_refinement_levels() const
    noexcept {
  return {(1 + 4 * (1 + radial_partitioning_.size())) *
              (1 + height_partitioning_.size()),
          make_array<3>(initial_refinement_)};
}
}  // namespace domain::creators
