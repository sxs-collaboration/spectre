// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <vector>

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
template <typename Map1, typename Map2, typename Map3>
class ProductOf3Maps;
class Wedge2D;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
/// \endcond

namespace domain::creators {
/// Create a 3D Domain in the shape of a cylinder where the cross-section
/// is a square surrounded by four two-dimensional wedges (see Wedge2D).
///
/// The outer shell can be split into sub-shells and the cylinder itself split
/// into disks, although this is not the case by default.
/// The block numbering starts at the inner square and goes clockwise, starting
/// with the eastern wedge, through consecutive shells, then repeats this
/// pattern for all layers bottom to top.
///
/// \image html Cylinder.png "The Cylinder Domain."
class Cylinder : public DomainCreator<3> {
 public:
  using maps_list = tmpl::list<
      domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                            CoordinateMaps::ProductOf3Maps<
                                CoordinateMaps::Affine, CoordinateMaps::Affine,
                                CoordinateMaps::Affine>>,
      domain::CoordinateMap<
          Frame::Logical, Frame::Inertial,
          CoordinateMaps::ProductOf3Maps<CoordinateMaps::Equiangular,
                                         CoordinateMaps::Equiangular,
                                         CoordinateMaps::Affine>>,
      domain::CoordinateMap<
          Frame::Logical, Frame::Inertial,
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::Wedge2D,
                                         CoordinateMaps::Affine>>>;

  struct InnerRadius {
    using type = double;
    static constexpr Options::String help = {
        "Radius of the circle circumscribing the inner square."};
  };

  struct OuterRadius {
    using type = double;
    static constexpr Options::String help = {"Radius of the cylinder."};
  };

  struct LowerBound {
    using type = double;
    static constexpr Options::String help = {
        "z-coordinate of the base of the cylinder."};
  };

  struct UpperBound {
    using type = double;
    static constexpr Options::String help = {
        "z-coordinate of the top of the cylinder."};
  };

  struct IsPeriodicInZ {
    using type = bool;
    static constexpr Options::String help = {
        "True if periodic in the cylindrical z direction."};
  };

  struct InitialRefinement {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial refinement level in each dimension."};
  };

  struct InitialGridPoints {
    using type = std::array<size_t, 3>;
    static constexpr Options::String help = {
        "Initial number of grid points in [r, theta, z]."};
  };

  struct UseEquiangularMap {
    using type = bool;
    static constexpr Options::String help = {
        "Use equiangular instead of equidistant coordinates."};
  };

  struct RadialPartitioning {
    using type = std::vector<double>;
    static constexpr Options::String help = {
        "Radial coordinates of the boundaries splitting the outer shell "
        "between InnerRadius and OuterRadius."};
  };

  struct HeightPartitioning {
    using type = std::vector<double>;
    static constexpr Options::String help = {
        "z-coordinates of the boundaries splitting the domain into discs "
        "between LowerBound and UpperBound."};
  };

  using options =
      tmpl::list<InnerRadius, OuterRadius, LowerBound, UpperBound,
                 IsPeriodicInZ, InitialRefinement, InitialGridPoints,
                 UseEquiangularMap, RadialPartitioning, HeightPartitioning>;

  static constexpr Options::String help{
      "Creates a right circular Cylinder with a square prism surrounded by \n"
      "wedges. \n"
      "The cylinder can be partitioned radially into multiple cylindrical \n"
      "shells as well as partitioned along the cylinder's height into \n"
      "multiple disks. Including this partitioning, the number of Blocks is \n"
      "given by (1 + 4*(1+n_s)) * (1+n_z), where n_s is the \n"
      "length of RadialPartitioning and n_z the length of \n"
      "HeightPartitioning.\n"
      "The circularity of the wedge changes from 0 to 1 within the first \n"
      "shell. The partitionings are empty by default.\n"
      "Only one refinement level for all dimensions is currently supported.\n"
      "The number of gridpoints in each dimension can be set independently, \n"
      "except for x and y in the square where they are equal to the number \n"
      "of gridpoints along the angular dimension of the wedges.\n"
      "Equiangular coordinates give better gridpoint spacings in the angular\n"
      "direction, while equidistant coordinates give better gridpoint\n"
      "spacings in the center block. This Domain uses equidistant coordinates\n"
      "by default. The boundary conditions are set to be periodic along the\n"
      "cylindrical z-axis by default."};

  Cylinder(typename InnerRadius::type inner_radius,
           typename OuterRadius::type outer_radius,
           typename LowerBound::type lower_bound,
           typename UpperBound::type upper_bound,
           typename IsPeriodicInZ::type is_periodic_in_z,
           typename InitialRefinement::type initial_refinement,
           typename InitialGridPoints::type initial_number_of_grid_points,
           typename UseEquiangularMap::type use_equiangular_map,
           typename RadialPartitioning::type radial_partitioning =
               std::vector<double>{},
           typename HeightPartitioning::type height_partitioning =
               std::vector<double>{}) noexcept;

  Cylinder() = default;
  Cylinder(const Cylinder&) = delete;
  Cylinder(Cylinder&&) noexcept = default;
  Cylinder& operator=(const Cylinder&) = delete;
  Cylinder& operator=(Cylinder&&) noexcept = default;
  ~Cylinder() noexcept override = default;

  Domain<3> create_domain() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_extents() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const
      noexcept override;

 private:
  typename InnerRadius::type inner_radius_{};
  typename OuterRadius::type outer_radius_{};
  typename LowerBound::type lower_bound_{};
  typename UpperBound::type upper_bound_{};
  typename IsPeriodicInZ::type is_periodic_in_z_{true};
  typename InitialRefinement::type initial_refinement_{};
  typename InitialGridPoints::type initial_number_of_grid_points_{};
  typename UseEquiangularMap::type use_equiangular_map_{false};
  typename RadialPartitioning::type radial_partitioning_{std::vector<double>{}};
  typename HeightPartitioning::type height_partitioning_{std::vector<double>{}};
};
}  // namespace domain::creators
