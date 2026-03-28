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
#include "Domain/Structure/OrientationMap.hpp"
#include "Options/ParseError.hpp"

namespace Frame {
struct Inertial;
struct BlockLogical;
}  // namespace Frame

namespace domain::creators {
CartoonSphere2D::CartoonSphere2D(
    double inner_radius, double outer_radius,
    typename InitialRefinement::type&& initial_refinement,
    typename InitialGridPoints::type&& initial_number_of_grid_points,
    std::vector<double> radial_partitioning, bool use_equiangular_map,
    std::variant<Excision, InnerSquare> interior,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
        time_dependence,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        y_axis_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    const Options::Context& context)
    : inner_radius_(inner_radius),
      outer_radius_(outer_radius),
      radial_partitioning_(std::move(radial_partitioning)),
      use_equiangular_map_(use_equiangular_map),
      interior_(std::move(interior)),
      fill_interior_(std::holds_alternative<InnerSquare>(interior_)),
      time_dependence_(std::move(time_dependence)),
      y_axis_boundary_condition_(std::move(y_axis_boundary_condition)),
      outer_boundary_condition_(std::move(outer_boundary_condition)) {
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

  std::vector<domain::CoordinateMaps::Distribution> dummy_vec;
  const auto dummy_distribution = CoordinateMaps::Distribution::Linear;
  // using this for checks of input values
  set_shell_distribution(make_not_null(&num_shells_), make_not_null(&dummy_vec),
                         radial_partitioning_, dummy_distribution,
                         inner_radius_, outer_radius_, "inner", "outer",
                         context);
  // num_shells_ is the number of wedge shells, always > 0
  num_blocks_ = 3 * num_shells_ + (fill_interior_ ? 1 : 0);

  const auto check_possible_list_instantiation =
      [this, &context](std::variant<std::array<size_t, 2>,
                                    std::vector<std::array<size_t, 2>>>
                           input_param,
                       const std::string& name) {
        if (std::holds_alternative<std::array<size_t, 2>>(input_param)) {
          const auto input_value = std::get<std::array<size_t, 2>>(input_param);
          return std::vector<std::array<size_t, 2>>(num_blocks_, input_value);
        } else {
          const auto& input_vec =
              std::get<std::vector<std::array<size_t, 2>>>(input_param);
          if (input_vec.size() != num_shells_) {
            PARSE_ERROR(
                context,
                name << " must be one larger than RadialPartitioning (size="
                     << radial_partitioning_.size() << "), but has size "
                     << input_vec.size() << ".");
          }
          std::vector<std::array<size_t, 2>> extended;
          extended.reserve(num_blocks_);
          if (fill_interior_) {
            extended.push_back(input_vec[0]);
          }
          for (size_t i = 0; i < num_shells_; ++i) {
            for (size_t j = 0; j < 3; ++j) {
              extended.push_back(input_vec[i]);
            }
          }
          // reversing due to block_id scheme going from outer to inner layers
          std::reverse(extended.begin(), extended.end());
          return extended;
        }
      };
  initial_refinement_ = check_possible_list_instantiation(
      std::move(initial_refinement), "InitialRefinement");
  initial_number_of_grid_points_ = check_possible_list_instantiation(
      std::move(initial_number_of_grid_points), "InitialGridPoints");

  if ((y_axis_boundary_condition_ == nullptr) !=
      (outer_boundary_condition_ == nullptr)) {
    PARSE_ERROR(context,
                "Must specify either both inner and outer boundary conditions "
                "or neither.");
  }
  using domain::BoundaryConditions::is_none;
  if (is_none(y_axis_boundary_condition_) or
      is_none(outer_boundary_condition_) or
      (not fill_interior_ and
       is_none(std::get<Excision>(interior_).boundary_condition))) {
    PARSE_ERROR(
        context,
        "None boundary condition is not supported. If you would like an "
        "outflow-type boundary condition, you must use that.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(y_axis_boundary_condition_) or
      is_periodic(outer_boundary_condition_) or
      (not fill_interior_ and
       is_periodic(std::get<Excision>(interior_).boundary_condition))) {
    PARSE_ERROR(context,
                "Cannot have periodic boundary conditions on a 2D sphere.");
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
  if (context != Options::Context{}) {
    // Run create_domain for non-default contexts to validate the constructed
    // domain.
    (void)create_domain(context);
  }
}

Domain<3> CartoonSphere2D::create_domain(
    const Options::Context& /*context*/) const {
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
                domain::CoordinateMaps::Wedge<2>::WedgeHalves::UpperOnly},
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
                    use_equiangular_map_},
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
                    domain::CoordinateMaps::Wedge<2>::WedgeHalves::LowerOnly},
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
      blocks.emplace_back(
          std::move(coord_maps.at(j)), 3 * i + j, std::move(neighbors.at(j)),
          block_names_.at(3 * i + j), domain::topologies::cartoon_cylinder);
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
        y_axis_boundary_condition_->get_clone();
    boundary_conditions[3 * i + 2][Direction<3>::lower_xi()] =
        y_axis_boundary_condition_->get_clone();
  }
  if (fill_interior_) {
    boundary_conditions[num_blocks_ - 1][Direction<3>::lower_xi()] =
        y_axis_boundary_condition_->get_clone();
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

Domain<3> CartoonSphere2D::create_domain() const {
  return create_domain(Options::Context{});
}

std::vector<std::array<size_t, 3>> CartoonSphere2D::initial_extents() const {
  std::vector<std::array<size_t, 3>> extended;
  extended.reserve(num_blocks_);

  std::transform(initial_number_of_grid_points_.begin(),
                 initial_number_of_grid_points_.end(),
                 std::back_inserter(extended),
                 [](const std::array<size_t, 2>& arr) -> std::array<size_t, 3> {
                   // data is read as [r, theta] but coordinates are [theta, r]
                   return {arr[1], arr[0], 1};
                 });
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

  size_t n = 0;
  std::transform(
      initial_refinement_.begin(), initial_refinement_.end(),
      std::back_inserter(extended),
      [&n](const std::array<size_t, 2>& arr) -> std::array<size_t, 3> {
        const size_t shift = (n % 3 == 0 or n % 3 == 2) and arr[1] != 0 ? 1 : 0;
        ++n;
        // data is read as [r, theta] but coordinates are [theta, r]
        return {arr[1] - shift, arr[0], 0};
      });
  if (fill_interior_) {
    // dealing with half-square, want proportional refinement, matching to wedge
    // theta
    extended.back()[1] = initial_refinement_.back()[1];
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
