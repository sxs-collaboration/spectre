// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/CartoonSphere2D.hpp"

#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/ShellDistribution.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/HasBoundary.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Options/ParseError.hpp"

namespace Frame {
struct Inertial;
struct BlockLogical;
}  // namespace Frame

namespace domain::creators {

detail::CartoonSphere2DOptionsHelper::CartoonSphere2DOptionsHelper(
    double inner_radius, double outer_radius, size_t initial_angular_refinement,
    typename InitialRadialRefinement::type&& initial_radial_refinement,
    std::array<size_t, 2> initial_number_of_grid_points,
    std::vector<double> radial_partitioning, bool use_equiangular_map,
    std::variant<Excision, InnerSquare> interior,
    typename domain::creators::CartoonSphere2D::RadialDistribution::type
        radial_distribution,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
        time_dependence,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    Options::Context&& context)
    : inner_radius_(inner_radius),
      outer_radius_(outer_radius),
      initial_angular_refinement_(initial_angular_refinement),
      initial_radial_refinement_(std::move(initial_radial_refinement)),
      initial_number_of_grid_points_(std::move(initial_number_of_grid_points)),
      radial_partitioning_(std::move(radial_partitioning)),
      use_equiangular_map_(use_equiangular_map),
      interior_(std::move(interior)),
      radial_distribution_(std::move(radial_distribution)),
      time_dependence_(std::move(time_dependence)),
      outer_boundary_condition_(std::move(outer_boundary_condition)),
      context_(std::move(context)) {}

CartoonSphere2D::CartoonSphere2D(
    double inner_radius, double outer_radius, size_t initial_angular_refinement,
    const typename InitialRadialRefinement::type& initial_radial_refinement,
    std::array<size_t, 2> initial_number_of_grid_points,
    std::vector<double> radial_partitioning, bool use_equiangular_map,
    std::variant<Excision, InnerSquare> interior,
    const typename RadialDistribution::type& radial_distribution,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
        time_dependence,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        cartoon_boundary_condition,
    const Options::Context& context)
    : inner_radius_(inner_radius),
      outer_radius_(outer_radius),
      initial_angular_refinement_(std::move(initial_angular_refinement)),
      initial_number_of_grid_points_(std::move(initial_number_of_grid_points)),
      radial_partitioning_(std::move(radial_partitioning)),
      use_equiangular_map_(use_equiangular_map),
      interior_(std::move(interior)),
      fill_interior_(std::holds_alternative<InnerSquare>(interior_)),
      time_dependence_(std::move(time_dependence)),
      outer_boundary_condition_(std::move(outer_boundary_condition)),
      cartoon_boundary_condition_(std::move(cartoon_boundary_condition)) {
  if (time_dependence_ == nullptr) {
    time_dependence_ =
        std::make_unique<domain::creators::time_dependence::None<3>>();
  }
  if (inner_radius_ >= outer_radius_) {
    PARSE_ERROR(context,
                "Inner radius must be smaller than outer radius, but inner "
                "radius is " +
                    std::to_string(inner_radius_) + " and outer radius is " +
                    std::to_string(outer_radius_) + ".");
  }
  set_shell_distribution(
      make_not_null(&num_shells_), make_not_null(&radial_distributions_),
      radial_partitioning_, radial_distribution, inner_radius_, outer_radius_,
      "inner", "outer", context);

  // radial_distributions_[0] is the innermost shell
  if (fill_interior_ and std::get<InnerSquare>(interior_).sphericity != 1.0 and
      radial_distributions_[0] != CoordinateMaps::Distribution::Linear) {
    PARSE_ERROR(
        context,
        "Cannot have a non-linear radial distribution in the innermost shell "
        "when the sphericity of wedges is not 1.0, got "
            << radial_distributions_[0] << ". You need to excise the center.");
  }
  // num_shells_ is the number of wedge shells, always > 0
  num_blocks_ = 3 * num_shells_ + (fill_interior_ ? 1 : 0);
  if (std::holds_alternative<size_t>(initial_radial_refinement)) {
    const auto input_value = std::get<size_t>(initial_radial_refinement);
    initial_radial_refinement_ = std::vector<size_t>(num_blocks_, input_value);
  } else {
    const auto& input_vec =
        std::get<std::vector<size_t>>(initial_radial_refinement);
    if (input_vec.size() != num_shells_) {
      PARSE_ERROR(context,
                  "InitialRadialRefinement must be one larger than "
                  "RadialPartitioning (size="
                      << radial_partitioning_.size() << "), but has size "
                      << input_vec.size() << ".");
    }
    initial_radial_refinement_.reserve(num_blocks_);
    if (fill_interior_) {
      initial_radial_refinement_.push_back(input_vec[0]);
    }
    for (size_t i = 0; i < num_shells_; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        initial_radial_refinement_.push_back(input_vec[i]);
      }
    }
    // reversing due to block_id scheme going from outer to inner layers
    std::reverse(initial_radial_refinement_.begin(),
                 initial_radial_refinement_.end());
  }

  using domain::BoundaryConditions::is_none;
  if (is_none(outer_boundary_condition_) or
      (not fill_interior_ and
       is_none(std::get<Excision>(interior_).boundary_condition))) {
    PARSE_ERROR(
        context,
        "None boundary condition is not supported. If you would like an "
        "outflow-type boundary condition, you must use that.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(outer_boundary_condition_) or
      (not fill_interior_ and
       is_periodic(std::get<Excision>(interior_).boundary_condition))) {
    PARSE_ERROR(context,
                "Cannot have periodic boundary conditions on a 2D sphere.");
  }

  if (cartoon_boundary_condition_ == nullptr) {
    PARSE_ERROR(
        context,
        "CartoonSphere2D should only be used with systems that have a "
        "cartoon-style boundary condition, but none was provided. This "
        "means the system is not set up to use cartoon methods. Make sure your "
        "system has a boundary condition that inherits from MarkAsCartoon in "
        "its standard_boundary_conditions list.");
  }

  // Check if user mistakenly specified a cartoon BC as an external boundary
  if (outer_boundary_condition_ != nullptr and
      domain::BoundaryConditions::is_cartoon(outer_boundary_condition_)) {
    PARSE_ERROR(
        context,
        "Cartoon boundary conditions should not be specified as external "
        "boundary conditions. They are automatically applied to internal "
        "cartoon boundaries. Please choose a different boundary condition "
        "for the outer boundary.");
  }

  block_names_.reserve(num_blocks_);
  // naming and numbering goes from outer-most shell, starting from bottom and
  // going counterclockwise, going to inside neighboring shell. The "center"
  // half-circle follows same numbering with the half-square being the last
  // block
  const std::array<std::string, 3> wedge_directions{"_LowerY", "_UpperX",
                                                    "_UpperY"};

  for (size_t i = 0; i < num_shells_; ++i) {
    const std::string shell = "Shell" + std::to_string(i);
    block_groups_[shell];
    for (const std::string& direction : wedge_directions) {
      const std::string name = shell + direction;
      block_names_.emplace_back(name);
      block_groups_[shell].insert(name);
    }
  }
  if (fill_interior_) {
    const std::string shell = "Shell" + std::to_string(num_shells_ - 1);
    block_names_.emplace_back(shell + "_HalfSquare");
    block_groups_[shell].insert(shell + "_HalfSquare");
  }
}

Domain<3> CartoonSphere2D::create_domain() const {
  using Affine = CoordinateMaps::Affine;
  using Equiangular = CoordinateMaps::Equiangular;
  using Affine2D = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  using Equiangular2D =
      CoordinateMaps::ProductOf2Maps<Equiangular, Equiangular>;
  using Wedge2DMap = CoordinateMaps::Wedge<2>;
  using Identity1D = CoordinateMaps::Identity<1>;
  using Wedge3DPrism =
      domain::CoordinateMaps::ProductOf2Maps<Wedge2DMap, Identity1D>;
  using Rotation3D = CoordinateMaps::DiscreteRotation<3>;

  std::vector<Block<3>> blocks;
  blocks.reserve(num_blocks_);

  const double inner_square_sphericity =
      fill_interior_ ? std::get<InnerSquare>(interior_).sphericity
                     : std::numeric_limits<double>::signaling_NaN();
  if (fill_interior_ and inner_square_sphericity != 0.0) {
    ERROR("CartoonSphere2D cannot have a non-zero inner sphericity, "
          << "got " << inner_square_sphericity << ".");
  }

  const auto aligned = OrientationMap<3>::create_aligned();
  // quarter turn
  const OrientationMap<3> turn_ccw(std::array<Direction<3>, 3>{
      {Direction<3>::lower_eta(), Direction<3>::upper_xi(),
       Direction<3>::upper_zeta()}});
  const OrientationMap<3> half_turn(std::array<Direction<3>, 3>{
      {Direction<3>::lower_xi(), Direction<3>::lower_eta(),
       Direction<3>::upper_zeta()}});
  // quarter turn
  const OrientationMap<3> turn_cw(std::array<Direction<3>, 3>{
      {Direction<3>::upper_eta(), Direction<3>::lower_xi(),
       Direction<3>::upper_zeta()}});

  for (size_t i = 0; i < num_shells_; ++i) {
    const bool on_inner = i == num_shells_ - 1;
    const bool has_square = on_inner and fill_interior_;
    const double inner_radius =
        on_inner ? inner_radius_
                 : radial_partitioning_[radial_partitioning_.size() - 1 - i];
    const double outer_radius =
        i == 0 ? outer_radius_
               : radial_partitioning_[radial_partitioning_.size() - i];
    const double inner_sphericity = has_square ? inner_square_sphericity : 1.0;
    // radial_distributions_[0] is innermost; loop goes outer->inner
    const auto shell_distribution = radial_distributions_[num_shells_ - 1 - i];
    // this is a half-wedge, -y
    auto coord_maps = make_vector_coordinate_map_base<Frame::BlockLogical,
                                                      Frame::Inertial, 3>(
        std::vector<Rotation3D>{Rotation3D{turn_ccw}},
        Wedge3DPrism{
            Wedge2DMap{
                inner_radius, outer_radius, inner_sphericity, 1.0,
                OrientationMap<2>{std::array<Direction<2>, 2>{
                    {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
                use_equiangular_map_,
                domain::CoordinateMaps::Wedge<2>::WedgeHalves::UpperOnly,
                shell_distribution},
            Identity1D{}});
    // this is a full wedge, +x
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            Rotation3D{turn_cw},
            Wedge3DPrism{
                Wedge2DMap{
                    inner_radius, outer_radius, inner_sphericity, 1.0,
                    OrientationMap<2>{std::array<Direction<2>, 2>{
                        {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},
                    use_equiangular_map_,
                    domain::CoordinateMaps::Wedge<2>::WedgeHalves::Both,
                    shell_distribution},
                Identity1D{}}));
    // this is a half-wedge, +y
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            Rotation3D{turn_cw},
            Wedge3DPrism{
                Wedge2DMap{
                    inner_radius, outer_radius, inner_sphericity, 1.0,
                    OrientationMap<2>{std::array<Direction<2>, 2>{
                        {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},
                    use_equiangular_map_,
                    domain::CoordinateMaps::Wedge<2>::WedgeHalves::LowerOnly,
                    shell_distribution},
                Identity1D{}}));
    if (has_square) {
      if (use_equiangular_map_) {
        coord_maps.emplace_back(
            make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                CoordinateMaps::ProductOf2Maps<Equiangular2D, Identity1D>{
                    Equiangular2D{
                        // note the non-standard logical coordinate bounds, such
                        // that the eta-neighbor collocation points match the
                        // half-wedges'
                        Equiangular(-3.0, 1.0, -1.0 * inner_radius_ / sqrt(2.0),
                                    inner_radius_ / sqrt(2.0)),
                        Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(2.0),
                                    inner_radius_ / sqrt(2.0))},
                    Identity1D{}}));
      } else {
        coord_maps.emplace_back(
            make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                CoordinateMaps::ProductOf2Maps<Affine2D, Identity1D>{
                    Affine2D{Affine(-1.0, 1.0, 0.0, inner_radius_ / sqrt(2.0)),
                             Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(2.0),
                                    inner_radius_ / sqrt(2.0))},
                    Identity1D{}}));
      }
    }

    std::vector<DirectionMap<3, BlockNeighbors<3>>> neighbors{
        static_cast<size_t>(3 + (has_square ? 1 : 0))};

    // between shell blocks
    neighbors[0].emplace(std::pair(Direction<3>::upper_xi(),
                                   BlockNeighbors<3>(3 * i + 1, half_turn)));

    neighbors[1].emplace(std::pair(Direction<3>::upper_xi(),
                                   BlockNeighbors<3>(3 * i + 0, half_turn)));
    neighbors[1].emplace(std::pair(Direction<3>::lower_xi(),
                                   BlockNeighbors<3>(3 * i + 2, aligned)));

    neighbors[2].emplace(std::pair(Direction<3>::upper_xi(),
                                   BlockNeighbors<3>(3 * i + 1, aligned)));

    // to the +r direction (if i==0, has external boundary)
    if (i != 0) {
      neighbors[0].emplace(
          std::pair(Direction<3>::lower_eta(),
                    BlockNeighbors<3>((i - 1) * 3 + 0, aligned)));
      neighbors[1].emplace(
          std::pair(Direction<3>::upper_eta(),
                    BlockNeighbors<3>((i - 1) * 3 + 1, aligned)));
      neighbors[2].emplace(
          std::pair(Direction<3>::upper_eta(),
                    BlockNeighbors<3>((i - 1) * 3 + 2, aligned)));
    }

    // to the -r direction (if excise_center_, innermost has external boundary)
    if (i < num_shells_ - 1) {
      neighbors[0].emplace(
          std::pair(Direction<3>::upper_eta(),
                    BlockNeighbors<3>((i + 1) * 3 + 0, aligned)));
      neighbors[1].emplace(
          std::pair(Direction<3>::lower_eta(),
                    BlockNeighbors<3>((i + 1) * 3 + 1, aligned)));
      neighbors[2].emplace(
          std::pair(Direction<3>::lower_eta(),
                    BlockNeighbors<3>((i + 1) * 3 + 2, aligned)));
    } else if (has_square) {
      // on the center half-circle
      neighbors[0].emplace(std::pair(Direction<3>::upper_eta(),
                                     BlockNeighbors<3>(3 * i + 3, aligned)));

      neighbors[1].emplace(std::pair(Direction<3>::lower_eta(),
                                     BlockNeighbors<3>(3 * i + 3, turn_ccw)));

      neighbors[2].emplace(std::pair(Direction<3>::lower_eta(),
                                     BlockNeighbors<3>(3 * i + 3, aligned)));

      neighbors[3].emplace(std::pair(Direction<3>::lower_eta(),
                                     BlockNeighbors<3>(3 * i + 0, aligned)));
      neighbors[3].emplace(std::pair(Direction<3>::upper_xi(),
                                     BlockNeighbors<3>(3 * i + 1, turn_cw)));
      neighbors[3].emplace(std::pair(Direction<3>::upper_eta(),
                                     BlockNeighbors<3>(3 * i + 2, aligned)));
    }

    for (size_t j = 0; j < neighbors.size(); ++j) {
      const auto topology = j % 3 == 1
                                ? domain::topologies::cartoon_cylinder
                                : domain::topologies::cartoon_cylinder_inner;
      blocks.emplace_back(std::move(coord_maps.at(j)), 3 * i + j,
                          std::move(neighbors.at(j)),
                          block_names_.at(3 * i + j), topology);
    }
  }

  Domain<3> domain(std::move(blocks), {}, block_groups_);

  if (not time_dependence_->is_none()) {
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>>
        block_maps_grid_to_inertial =
            time_dependence_->block_maps_grid_to_inertial(num_blocks_);
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, 3>>>
        block_maps_grid_to_distorted =
            time_dependence_->block_maps_grid_to_distorted(num_blocks_);
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial, 3>>>
        block_maps_distorted_to_inertial =
            time_dependence_->block_maps_distorted_to_inertial(num_blocks_);
    for (size_t block_id = 0; block_id < num_blocks_; ++block_id) {
      domain.inject_time_dependent_map_for_block(
          block_id, std::move(block_maps_grid_to_inertial[block_id]),
          std::move(block_maps_grid_to_distorted[block_id]),
          std::move(block_maps_distorted_to_inertial[block_id]));
    }
  }
  return domain;
}

std::vector<DirectionMap<
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
CartoonSphere2D::external_boundary_conditions() const {
  if (outer_boundary_condition_ == nullptr) {
    return {};
  }
  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{num_blocks_};

  // outer rim, +r direction
  boundary_conditions[0][Direction<3>::lower_eta()] =
      outer_boundary_condition_->get_clone();
  boundary_conditions[1][Direction<3>::upper_eta()] =
      outer_boundary_condition_->get_clone();
  boundary_conditions[2][Direction<3>::upper_eta()] =
      outer_boundary_condition_->get_clone();

  // x=0 axis
  for (size_t i = 0; i < num_shells_; ++i) {
    boundary_conditions[3 * i + 0][Direction<3>::lower_xi()] =
        cartoon_boundary_condition_->get_clone();
    boundary_conditions[3 * i + 2][Direction<3>::lower_xi()] =
        cartoon_boundary_condition_->get_clone();
  }
  if (fill_interior_) {
    boundary_conditions[num_blocks_ - 1][Direction<3>::lower_xi()] =
        cartoon_boundary_condition_->get_clone();
  } else {
    const auto& excision_boundary_condition =
        std::get<Excision>(interior_).boundary_condition;
    boundary_conditions[(num_shells_ - 1) * 3 + 0][Direction<3>::upper_eta()] =
        excision_boundary_condition->get_clone();
    boundary_conditions[(num_shells_ - 1) * 3 + 1][Direction<3>::lower_eta()] =
        excision_boundary_condition->get_clone();
    boundary_conditions[(num_shells_ - 1) * 3 + 2][Direction<3>::lower_eta()] =
        excision_boundary_condition->get_clone();
  }
  return boundary_conditions;
}

std::vector<std::array<size_t, 3>> CartoonSphere2D::initial_extents() const {
  std::vector<std::array<size_t, 3>> extended;
  extended.reserve(num_blocks_);

  for (size_t i = 0; i < num_blocks_; ++i) {
    // data is read as [r, theta] but coordinates are [theta, r]
    extended.emplace_back(
        std::array<size_t, 3>{initial_number_of_grid_points_[1],
                              initial_number_of_grid_points_[0], 1});
  }
  if (fill_interior_) {
    // dealing with half-square, want identical extents, matching to wedge theta
    extended.back()[1] = extended.back()[0];
  }
  return extended;
}

std::vector<std::array<size_t, 3>> CartoonSphere2D::initial_refinement_levels()
    const {
  std::vector<std::array<size_t, 3>> extended;
  extended.reserve(num_blocks_);
  for (size_t i = 0; i < num_blocks_; ++i) {
    // the angular refinement is that of a full wedge; for the half-wedges at
    // the top and bottom of the domain, we must decrease the refinement so
    // that the "physical" refinement is uniform
    const size_t shift =
        (i % 3 == 0 or i % 3 == 2) and initial_angular_refinement_ != 0 ? 1 : 0;
    // Coordinates are [theta, r]
    extended.emplace_back(std::array<size_t, 3>{
        initial_angular_refinement_ - shift, initial_radial_refinement_[i], 0});
  }
  if (fill_interior_) {
    // dealing with half-square, want proportional refinement, matching to wedge
    // theta
    extended.back()[1] = initial_angular_refinement_;
    const size_t shift = extended.back()[1] != 0 ? 1 : 0;
    extended.back()[0] = extended.back()[1] - shift;
  }
  return extended;
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
CartoonSphere2D::functions_of_time(
    const std::unordered_map<std::string, double>& initial_expiration_times)
    const {
  if (time_dependence_->is_none()) {
    return {};
  } else {
    return time_dependence_->functions_of_time(initial_expiration_times);
  }
}
}  // namespace domain::creators
