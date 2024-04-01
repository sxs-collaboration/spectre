// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Rectilinear.hpp"

#include <array>
#include <memory>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"
#include "Options/ParseError.hpp"

namespace Frame {
struct BlockLogical;
struct Inertial;
}  // namespace Frame

namespace domain::creators {

template <size_t Dim>
Rectilinear<Dim>::Rectilinear(
    std::array<double, Dim> lower_bounds, std::array<double, Dim> upper_bounds,
    std::array<size_t, Dim> initial_refinement_levels,
    std::array<size_t, Dim> initial_num_points,
    std::array<bool, Dim> is_periodic,
    std::array<CoordinateMaps::DistributionAndSingularityPosition, Dim>
        distributions,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<Dim>>
        time_dependence,
    const Options::Context& context)
    : lower_bounds_(lower_bounds),
      upper_bounds_(upper_bounds),
      distributions_(distributions),
      is_periodic_(is_periodic),
      initial_refinement_levels_(initial_refinement_levels),
      initial_num_points_(initial_num_points),
      time_dependence_(std::move(time_dependence)) {
  if (time_dependence_ == nullptr) {
    time_dependence_ =
        std::make_unique<domain::creators::time_dependence::None<Dim>>();
  }
  for (size_t d = 0; d < Dim; ++d) {
    if (gsl::at(lower_bounds_, d) >= gsl::at(upper_bounds_, d)) {
      PARSE_ERROR(context,
                  "Lower bound ("
                      << gsl::at(lower_bounds_, d)
                      << ") must be strictly smaller than upper bound ("
                      << gsl::at(upper_bounds_, d) << ") in dimension " << d
                      << ".");
    }
    const auto singularity_pos =
        gsl::at(distributions_, d).singularity_position;
    if (singularity_pos.has_value() and
        *singularity_pos >= gsl::at(lower_bounds_, d) and
        *singularity_pos <= gsl::at(upper_bounds_, d)) {
      PARSE_ERROR(context, "The 'SingularityPosition' ("
                               << *singularity_pos
                               << ") falls inside the domain ["
                               << gsl::at(lower_bounds_, d) << ", "
                               << gsl::at(upper_bounds_, d) << "].");
    }
  }
}

template <size_t Dim>
Rectilinear<Dim>::Rectilinear(
    std::array<double, Dim> lower_bounds, std::array<double, Dim> upper_bounds,
    std::array<size_t, Dim> initial_refinement_levels,
    std::array<size_t, Dim> initial_num_points,
    std::array<
        std::array<
            std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>, 2>,
        Dim>
        boundary_conditions,
    std::array<CoordinateMaps::DistributionAndSingularityPosition, Dim>
        distributions,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<Dim>>
        time_dependence,
    const Options::Context& context)
    : Rectilinear(lower_bounds, upper_bounds, initial_refinement_levels,
                  initial_num_points, make_array<Dim>(false), distributions,
                  std::move(time_dependence), context) {
  boundary_conditions_ = std::move(boundary_conditions);
  using domain::BoundaryConditions::is_none;
  using domain::BoundaryConditions::is_periodic;
  for (size_t d = 0; d < Dim; ++d) {
    const auto& [lower_bc, upper_bc] = gsl::at(boundary_conditions_, d);
    ASSERT(lower_bc != nullptr and upper_bc != nullptr,
           "None of the boundary conditions can be nullptr.");
    if (is_none(lower_bc) or is_none(upper_bc)) {
      PARSE_ERROR(
          context,
          "None boundary condition is not supported. If you would like an "
          "outflow-type boundary condition, you must use that.");
    }
    if (is_periodic(lower_bc) != is_periodic(upper_bc)) {
      PARSE_ERROR(context,
                  "Periodic boundary conditions must be applied for both "
                  "upper and lower direction in a dimension.");
    }
    if (is_periodic(lower_bc) and is_periodic(upper_bc)) {
      gsl::at(is_periodic_, d) = true;
    }
  }
}

template <size_t Dim>
Domain<Dim> Rectilinear<Dim>::create_domain() const {
  // Handle periodicity by identifying faces
  std::vector<PairOfFaces> identifications{};
  if constexpr (Dim == 1) {
    if (is_periodic_[0]) {
      identifications.push_back({{0}, {1}});
    }
  } else if constexpr (Dim == 2) {
    if (is_periodic_[0]) {
      identifications.push_back({{0, 2}, {1, 3}});
    }
    if (is_periodic_[1]) {
      identifications.push_back({{0, 1}, {2, 3}});
    }
  } else {
    if (is_periodic_[0]) {
      identifications.push_back({{0, 4, 2, 6}, {1, 5, 3, 7}});
    }
    if (is_periodic_[1]) {
      identifications.push_back({{0, 1, 4, 5}, {2, 3, 6, 7}});
    }
    if (is_periodic_[2]) {
      identifications.push_back({{0, 1, 2, 3}, {4, 5, 6, 7}});
    }
  }

  auto block_map = [this]() {
    if constexpr (Dim == 1) {
      return Interval{-1.,
                      1.,
                      lower_bounds_[0],
                      upper_bounds_[0],
                      distributions_[0].distribution,
                      distributions_[0].singularity_position};
    } else if constexpr (Dim == 2) {
      return Interval2D{Interval{-1., 1., lower_bounds_[0], upper_bounds_[0],
                                 distributions_[0].distribution,
                                 distributions_[0].singularity_position},
                        Interval{-1., 1., lower_bounds_[1], upper_bounds_[1],
                                 distributions_[1].distribution,
                                 distributions_[1].singularity_position}};
    } else {
      return Interval3D{Interval{-1., 1., lower_bounds_[0], upper_bounds_[0],
                                 distributions_[0].distribution,
                                 distributions_[0].singularity_position},
                        Interval{-1., 1., lower_bounds_[1], upper_bounds_[1],
                                 distributions_[1].distribution,
                                 distributions_[1].singularity_position},
                        Interval{-1., 1., lower_bounds_[2], upper_bounds_[2],
                                 distributions_[2].distribution,
                                 distributions_[2].singularity_position}};
    }
  }();

  std::array<size_t, two_to_the(Dim)> block_corners{};
  std::iota(block_corners.begin(), block_corners.end(), 0_st);

  Domain<Dim> domain{
      make_vector_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          std::move(block_map)),
      {std::move(block_corners)},
      std::move(identifications),
      {},
      block_names_};

  if (not time_dependence_->is_none()) {
    domain.inject_time_dependent_map_for_block(
        0, std::move(time_dependence_->block_maps_grid_to_inertial(1)[0]),
        std::move(time_dependence_->block_maps_grid_to_distorted(1)[0]),
        std::move(time_dependence_->block_maps_distorted_to_inertial(1)[0]));
  }
  return domain;
}

template <size_t Dim>
std::vector<DirectionMap<
    Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
Rectilinear<Dim>::external_boundary_conditions() const {
  if (boundary_conditions_[0][0] == nullptr) {
#ifdef SPECTRE_DEBUG
    for (size_t d = 0; d < Dim; ++d) {
      ASSERT(gsl::at(boundary_conditions_, d)[0] == nullptr and
                 gsl::at(boundary_conditions_, d)[1] == nullptr,
             "Boundary conditions must be set for all directions or none.");
    }
#endif  // SPECTRE_DEBUG
    return {};
  }
  std::vector<DirectionMap<
      Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{1};
  for (size_t d = 0; d < Dim; ++d) {
    if (not gsl::at(is_periodic_, d)) {
      const auto& [lower_bc, upper_bc] = gsl::at(boundary_conditions_, d);
      boundary_conditions[0][Direction<Dim>{d, Side::Lower}] =
          lower_bc->get_clone();
      boundary_conditions[0][Direction<Dim>{d, Side::Upper}] =
          upper_bc->get_clone();
    }
  }
  return boundary_conditions;
}

template <size_t Dim>
std::vector<std::array<size_t, Dim>> Rectilinear<Dim>::initial_extents() const {
  return {initial_num_points_};
}

template <size_t Dim>
std::vector<std::array<size_t, Dim>>
Rectilinear<Dim>::initial_refinement_levels() const {
  return {initial_refinement_levels_};
}

template <size_t Dim>
std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
Rectilinear<Dim>::functions_of_time(
    const std::unordered_map<std::string, double>& initial_expiration_times)
    const {
  if (time_dependence_->is_none()) {
    return {};
  } else {
    return time_dependence_->functions_of_time(initial_expiration_times);
  }
}

template class Rectilinear<1>;
template class Rectilinear<2>;
template class Rectilinear<3>;

}  // namespace domain::creators
