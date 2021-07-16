// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace CoordinateMaps {
class Affine;
class Equiangular;
template <typename Map1, typename Map2>
class ProductOf2Maps;
template <size_t Dim>
class Wedge;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
/// \endcond

namespace domain {
namespace creators {
/// Create a 2D Domain in the shape of a disk from a square surrounded by four
/// wedges.
class Disk : public DomainCreator<2> {
 public:
  using maps_list =
      tmpl::list<domain::CoordinateMap<
                     Frame::BlockLogical, Frame::Inertial,
                     CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                                    CoordinateMaps::Affine>>,
                 domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                       CoordinateMaps::ProductOf2Maps<
                                           CoordinateMaps::Equiangular,
                                           CoordinateMaps::Equiangular>>,
                 domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                       CoordinateMaps::Wedge<2>>>;

  struct InnerRadius {
    using type = double;
    static constexpr Options::String help = {
        "Radius of the circle circumscribing the inner square."};
  };

  struct OuterRadius {
    using type = double;
    static constexpr Options::String help = {"Radius of the Disk."};
  };

  struct InitialRefinement {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial refinement level in each dimension."};
  };

  struct InitialGridPoints {
    using type = std::array<size_t, 2>;
    static constexpr Options::String help = {
        "Initial number of grid points in [r,theta]."};
  };

  struct UseEquiangularMap {
    using type = bool;
    static constexpr Options::String help = {
        "Use equiangular instead of equidistant coordinates."};
  };

  template <typename BoundaryConditionsBase>
  struct BoundaryCondition {
    static std::string name() noexcept { return "BoundaryCondition"; }
    static constexpr Options::String help =
        "The boundary condition to impose on all sides.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  using basic_options = tmpl::list<InnerRadius, OuterRadius, InitialRefinement,
                                   InitialGridPoints, UseEquiangularMap>;

  template <typename Metavariables>
  using options = tmpl::conditional_t<
      domain::BoundaryConditions::has_boundary_conditions_base_v<
          typename Metavariables::system>,
      tmpl::push_back<
          basic_options,
          BoundaryCondition<
              domain::BoundaryConditions::get_boundary_conditions_base<
                  typename Metavariables::system>>>,
      basic_options>;

  static constexpr Options::String help{
      "Creates a 2D Disk with five Blocks.\n"
      "Only one refinement level for both dimensions is currently supported.\n"
      "The number of gridpoints in each dimension can be set independently.\n"
      "The number of gridpoints along the dimensions of the square is equal\n"
      "to the number of gridpoints along the angular dimension of the wedges.\n"
      "Equiangular coordinates give better gridpoint spacings in the angular\n"
      "direction, while equidistant coordinates give better gridpoint\n"
      "spacings in the center block."};

  Disk(typename InnerRadius::type inner_radius,
       typename OuterRadius::type outer_radius,
       typename InitialRefinement::type initial_refinement,
       typename InitialGridPoints::type initial_number_of_grid_points,
       typename UseEquiangularMap::type use_equiangular_map,
       std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
           boundary_condition = nullptr,
       const Options::Context& context = {});

  Disk() = default;
  Disk(const Disk&) = delete;
  Disk(Disk&&) noexcept = default;
  Disk& operator=(const Disk&) = delete;
  Disk& operator=(Disk&&) noexcept = default;
  ~Disk() noexcept override = default;

  Domain<2> create_domain() const noexcept override;

  std::vector<std::array<size_t, 2>> initial_extents() const noexcept override;

  std::vector<std::array<size_t, 2>> initial_refinement_levels() const
      noexcept override;

 private:
  typename InnerRadius::type inner_radius_{};
  typename OuterRadius::type outer_radius_{};
  typename InitialRefinement::type initial_refinement_{};
  typename InitialGridPoints::type initial_number_of_grid_points_{};
  typename UseEquiangularMap::type use_equiangular_map_{false};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      boundary_condition_;
};
}  // namespace creators
}  // namespace domain
