// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Cylinder.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
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
    typename HeightPartitioning::type height_partitioning,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        boundary_condition,
    const Options::Context& context)
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
      height_partitioning_(std::move(height_partitioning)),
      boundary_condition_(std::move(boundary_condition)) {
  if (not radial_partitioning_.empty() and boundary_condition_ != nullptr) {
    PARSE_ERROR(
        context,
        "Currently do not support specifying boundary conditions and "
        "multiple radial partitionings. Support can be added if desired.");
  }
  if (not height_partitioning_.empty() and boundary_condition_ != nullptr) {
    PARSE_ERROR(
        context,
        "Currently do not support specifying boundary conditions and multiple "
        "height partitionings. The domain creator code to support this is "
        "written but untested. To enable, please add tests.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(boundary_condition_)) {
    PARSE_ERROR(context,
                "Periodic boundary conditions are not supported in the radial "
                "direction. If you need periodic boundary conditions along the "
                "axis of symmetry, use the is_periodic_in_z option.");
  }
  using domain::BoundaryConditions::is_none;
  if (is_none(boundary_condition_)) {
    PARSE_ERROR(
        context,
        "None boundary condition is not supported. If you would like an "
        "outflow boundary condition, you must use that.");
  }
}

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

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions_all_blocks{};
  if (boundary_condition_ != nullptr) {
    // Note: The first block in each disk is the central cube.
    boundary_conditions_all_blocks.resize((1 + 4 * number_of_shells) *
                                          number_of_discs);

    // Boundary conditions in z
    for (size_t block_id = 0; not is_periodic_in_z_ and
                              block_id < boundary_conditions_all_blocks.size();
         ++block_id) {
      if (block_id < (1 + number_of_shells * 4)) {
        boundary_conditions_all_blocks[block_id][Direction<3>::lower_zeta()] =
            boundary_condition_->get_clone();
      }
      if (block_id >=
          boundary_conditions_all_blocks.size() - (1 + number_of_shells * 4)) {
        boundary_conditions_all_blocks[block_id][Direction<3>::upper_zeta()] =
            boundary_condition_->get_clone();
      }
    }
    // Radial boundary conditions
    ASSERT(radial_partitioning_.empty(),
           "We currently do not support multiple radial partitionings with "
           "boundary conditions. Please add support if you need this feature.");
    for (size_t block_id = 1; block_id < boundary_conditions_all_blocks.size();
         ++block_id) {
      // clang-tidy thinks we can get division by zero on the modulus operator.
      // NOLINTNEXTLINE
      if (block_id % (1 + 4 * number_of_shells) == 0) {
        // skip the central cubes. With multiple radial partitionings the inner
        // radial wedges also need to be skipped.
        continue;
      }
      boundary_conditions_all_blocks[block_id][Direction<3>::upper_xi()] =
          boundary_condition_->get_clone();
    }
  }

  return Domain<3>{
      cyl_wedge_coordinate_maps<Frame::Inertial>(
          inner_radius_, outer_radius_, lower_bound_, upper_bound_,
          use_equiangular_map_, radial_partitioning_, height_partitioning_),
      corners_for_cylindrical_layered_domains(number_of_shells,
                                              number_of_discs),
      pairs_of_faces, std::move(boundary_conditions_all_blocks)};
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
