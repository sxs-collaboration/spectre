// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Cylinder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeArray.hpp"

namespace Frame {
struct Logical;
struct Inertial;
}  // namespace Frame

namespace domain::creators {
Cylinder::Cylinder(double inner_radius, double outer_radius, double lower_bound,
                   double upper_bound, bool is_periodic_in_z,
                   size_t initial_refinement,
                   std::array<size_t, 3> initial_number_of_grid_points,
                   bool use_equiangular_map,
                   std::vector<double> radial_partitioning,
                   std::vector<double> height_partitioning,
                   const Options::Context& context)
    : inner_radius_(inner_radius),
      outer_radius_(outer_radius),
      lower_bound_(lower_bound),
      upper_bound_(upper_bound),
      is_periodic_in_z_(is_periodic_in_z),
      initial_refinement_(initial_refinement),
      initial_number_of_grid_points_(initial_number_of_grid_points),
      use_equiangular_map_(use_equiangular_map),
      radial_partitioning_(std::move(radial_partitioning)),
      height_partitioning_(std::move(height_partitioning)) {
  if (inner_radius_ > outer_radius_) {
    PARSE_ERROR(context,
                "Inner radius must be smaller than outer radius, but inner "
                "radius is " +
                    std::to_string(inner_radius_) + " and outer radius is " +
                    std::to_string(outer_radius_) + ".");
  }
  if (lower_bound_ > upper_bound_) {
    PARSE_ERROR(context,
                "Lower bound must be smaller than upper bound, but lower "
                "bound is " +
                    std::to_string(lower_bound_) + " and upper bound is " +
                    std::to_string(upper_bound_) + ".");
  }
  if (not std::is_sorted(radial_partitioning_.begin(),
                         radial_partitioning_.end())) {
    PARSE_ERROR(context,
                "Specify radial partitioning in ascending order. Specified "
                "radial partitioning is: " +
                    get_output(radial_partitioning_));
  }
  if (not radial_partitioning_.empty()) {
    if (radial_partitioning_.front() <= inner_radius_) {
      PARSE_ERROR(
          context,
          "First radial partition must be larger than inner radius, but is: " +
              std::to_string(inner_radius_));
    }
    if (radial_partitioning_.back() >= outer_radius_) {
      PARSE_ERROR(
          context,
          "Last radial partition must be smaller than outer radius, but is: " +
              std::to_string(outer_radius_));
    }
  }
  if (not std::is_sorted(height_partitioning_.begin(),
                         height_partitioning_.end())) {
    PARSE_ERROR(context,
                "Specify height partitioning in ascending order. Specified "
                "height partitioning is: " +
                    get_output(height_partitioning_));
  }
  if (not height_partitioning_.empty()) {
    if (height_partitioning_.front() <= lower_bound_) {
      PARSE_ERROR(
          context,
          "First height partition must be larger than lower bound, but is: " +
              std::to_string(lower_bound_));
    }
    if (height_partitioning_.back() >= upper_bound_) {
      PARSE_ERROR(
          context,
          "Last height partition must be smaller than upper bound, but is: " +
              std::to_string(upper_bound_));
    }
  }
}

Cylinder::Cylinder(
    double inner_radius, double outer_radius, double lower_bound,
    double upper_bound,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        lower_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        upper_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        mantle_boundary_condition,
    size_t initial_refinement,
    std::array<size_t, 3> initial_number_of_grid_points,
    bool use_equiangular_map, std::vector<double> radial_partitioning,
    std::vector<double> height_partitioning, const Options::Context& context)
    : Cylinder(inner_radius, outer_radius, lower_bound, upper_bound, false,
               initial_refinement, initial_number_of_grid_points,
               use_equiangular_map, std::move(radial_partitioning),
               std::move(height_partitioning), context) {
  lower_boundary_condition_ = std::move(lower_boundary_condition);
  upper_boundary_condition_ = std::move(upper_boundary_condition);
  mantle_boundary_condition_ = std::move(mantle_boundary_condition);

  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(lower_boundary_condition_) xor
      is_periodic(upper_boundary_condition_)) {
    PARSE_ERROR(context,
                "Either both lower and upper boundary condition must be "
                "periodic, or neither.");
  }
  if (is_periodic(lower_boundary_condition_) and
      is_periodic(upper_boundary_condition_)) {
    is_periodic_in_z_ = true;
    lower_boundary_condition_ = nullptr;
    upper_boundary_condition_ = nullptr;
  }
  if (is_periodic(mantle_boundary_condition_)) {
    PARSE_ERROR(context,
                "A Cylinder can't have periodic boundary conditions in the "
                "radial direction.");
  }
  using domain::BoundaryConditions::is_none;
  if (is_none(lower_boundary_condition_) or
      is_none(upper_boundary_condition_) or
      is_none(mantle_boundary_condition_)) {
    PARSE_ERROR(
        context,
        "None boundary condition is not supported. If you would like an "
        "outflow boundary condition, you must use that.");
  }
  if (mantle_boundary_condition_ == nullptr or
      (not is_periodic_in_z_ and (lower_boundary_condition_ == nullptr or
                                  upper_boundary_condition_ == nullptr))) {
    PARSE_ERROR(
        context,
        "Boundary conditions must not be 'nullptr'. Use the other constructor "
        "to specify 'is_periodic_in_z' instead of boundary conditions.");
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
  if (mantle_boundary_condition_ != nullptr) {
    // Note: The first block in each disk is the central cube.
    boundary_conditions_all_blocks.resize((1 + 4 * number_of_shells) *
                                          number_of_discs);

    // Boundary conditions in z
    for (size_t block_id = 0; not is_periodic_in_z_ and
                              block_id < boundary_conditions_all_blocks.size();
         ++block_id) {
      if (block_id < (1 + number_of_shells * 4)) {
        boundary_conditions_all_blocks[block_id][Direction<3>::lower_zeta()] =
            lower_boundary_condition_->get_clone();
      }
      if (block_id >=
          boundary_conditions_all_blocks.size() - (1 + number_of_shells * 4)) {
        boundary_conditions_all_blocks[block_id][Direction<3>::upper_zeta()] =
            upper_boundary_condition_->get_clone();
      }
    }
    // Radial boundary conditions
    for (size_t block_id = 1 + 4 * (number_of_shells - 1);
         block_id < boundary_conditions_all_blocks.size(); ++block_id) {
      // clang-tidy thinks we can get division by zero on the modulus operator.
      // NOLINTNEXTLINE
      if (block_id % (1 + 4 * number_of_shells) == 0) {
        // skip the central cubes and the inner radial wedges
        block_id += 4 * (number_of_shells - 1);
        continue;
      }
      boundary_conditions_all_blocks[block_id][Direction<3>::upper_xi()] =
          mantle_boundary_condition_->get_clone();
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
