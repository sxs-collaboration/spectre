// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace CoordinateMaps {
class Affine;
template <size_t Dim>
class Identity;
class Interval;
class PolarToCartesian;
template <typename Map1, typename Map2>
class ProductOf2Maps;
template <typename Map1, typename Map2, typename Map3>
class ProductOf3Maps;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
/// \endcond

namespace domain::creators {
/*!
 * \brief Create a 3D filled cylinder domain with radial partitioning using a
 * B2/I1 filled cylinder at the center and Fourier hollow cylinders surrounding
 * it.
 */
class AngularCylinder final : public DomainCreator<3> {
 public:
  using maps_list = tmpl::list<domain::CoordinateMap<
      Frame::BlockLogical, Frame::Inertial,
      domain::CoordinateMaps::ProductOf3Maps<
          domain::CoordinateMaps::Affine, domain::CoordinateMaps::Identity<1>,
          domain::CoordinateMaps::Interval>,
      domain::CoordinateMaps::ProductOf2Maps<
          domain::CoordinateMaps::PolarToCartesian,
          domain::CoordinateMaps::Identity<1>>>>;

  /*!
   * \brief Radius of the cylinder's outer edge
   */
  struct OuterRadius {
    using type = double;
    static constexpr Options::String help = {"Radius of the cylinder."};
  };

  /*!
   * \brief Lower z-coordinate of the cylinder's base
   */
  struct LowerZBound {
    using type = double;
    static constexpr Options::String help = {
        "z-coordinate of the base of the cylinder."};
  };

  /*!
   * \brief Upper z-coordinate of the cylinder's top
   */
  struct UpperZBound {
    using type = double;
    static constexpr Options::String help = {
        "z-coordinate of the top of the cylinder."};
  };

  /*!
   * \brief Whether the cylinder is periodic in the z direction
   */
  struct IsPeriodicInZ {
    using type = bool;
    static constexpr Options::String help = {
        "True if periodic in the cylindrical z direction."};
  };

  /*!
   * \brief Radial coordinates of the boundaries splitting elements
   */
  struct RadialPartitioning {
    using type = std::vector<double>;
    static constexpr Options::String help = {
        "Radial coordinates of the boundaries splitting the inner cylinder and "
        "potential hollow cylinder shells"};
  };

  /*!
   * \brief Z-coordinates of the boundaries splitting the domain into layers
   */
  struct PartitioningInZ {
    using type = std::vector<double>;
    static constexpr Options::String help = {
        "z-coordinates of the boundaries splitting the domain into layers "
        "between LowerZBound and UpperZBound."};
  };

  /*!
   * \brief Initial number of \f$\theta\f$ gridpoints for filled cylinder. The
   * number for \f$r\f$ is accordingly set to match spectral space sizes. This
   * is enforced to be odd for numerical stability.
   */
  struct InitialCylinderThetaGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of grid points in theta (r is accordingly set). It "
        "must be an odd number for stability."};
  };

  /*!
   * \brief Initial number of z gridpoints for the cylinder
   */
  struct InitialCylinderZGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of grid points in z"};
  };

  /*!
   * \brief Initial number of \f$[r, \theta, z]\f$ gridpoints for hollow
   * cylinders. Can be one triplet which is applied to all shells, or each can
   * be specified. The theta component must be odd for numerical stability.
   */
  struct InitialHollowCylinderGridPoints {
    using type =
        std::variant<std::array<size_t, 3>, std::vector<std::array<size_t, 3>>>;
    static constexpr Options::String help = {
        "Initial number of grid points in [r, theta, z] for each hollow "
        "cylinder. If one triplet is given, it will be applied to all blocks, "
        "otherwise each hollow cylinder can be specified. The theta component "
        "must be odd for better numerical stability."};
  };

  /*!
   * \brief Grid point distribution along the z-axis for each layer
   */
  struct DistributionInZ {
    using type = std::vector<domain::CoordinateMaps::Distribution>;
    static constexpr Options::String help = {
        "Select the distribution of grid points along the z-axis in each "
        "layer. The lowermost layer must have a 'Linear' distribution, because "
        "both a 'Logarithmic' and 'Inverse' distribution places its "
        "singularity at 'LowerZBound'. The 'PartitioningInZ' determines the "
        "number of layers."};
    static size_t lower_bound_on_size() { return 1; }
  };

  /*!
   * \brief Initial refinement levels in the z direction
   */
  struct InitialRefinementInZ {
    using type = std::variant<size_t, std::vector<size_t>>;
    static constexpr Options::String help = {
        "Initial refinement level in the z-direction. Can be either a single "
        "value applied to all layers, or a vector with one value per layer. "
        "The number of layers is determined by 'PartitioningInZ'."};
  };

  /*!
   * \brief Time dependence of the domain
   */
  struct TimeDependence {
    using type =
        std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>;
    static constexpr Options::String help = {
        "The time dependence of the moving mesh domain. Specify `None` for no "
        "time dependent maps."};
  };

  /*!
   * \brief Boundary conditions group
   */
  struct BoundaryConditions {
    static constexpr Options::String help =
        "Options for the boundary conditions";
  };

  /*!
   * \brief Boundary condition on the lower base
   */
  template <typename BoundaryConditionsBase>
  struct LowerZBoundaryCondition {
    using group = BoundaryConditions;
    static std::string name() { return "LowerZ"; }
    static constexpr Options::String help =
        "The boundary condition to be imposed on the lower base of the "
        "cylinder, i.e. at the `LowerZBound` in the z-direction.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  /*!
   * \brief Boundary condition on the upper base
   */
  template <typename BoundaryConditionsBase>
  struct UpperZBoundaryCondition {
    using group = BoundaryConditions;
    static std::string name() { return "UpperZ"; }
    static constexpr Options::String help =
        "The boundary condition to be imposed on the upper base of the "
        "cylinder, i.e. at the `UpperZBound` in the z-direction.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  /*!
   * \brief Boundary condition on the radial boundary
   */
  template <typename BoundaryConditionsBase>
  struct MantleBoundaryCondition {
    using group = BoundaryConditions;
    static std::string name() { return "Mantle"; }
    static constexpr Options::String help =
        "The boundary condition to be imposed on the mantle of the "
        "cylinder, i.e. at the `OuterRadius` in the radial direction.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  using basic_options =
      tmpl::list<OuterRadius, LowerZBound, UpperZBound, RadialPartitioning,
                 PartitioningInZ, InitialCylinderThetaGridPoints,
                 InitialCylinderZGridPoints, InitialHollowCylinderGridPoints,
                 DistributionInZ, InitialRefinementInZ, TimeDependence>;

  template <typename Metavariables>
  using options = tmpl::append<
      basic_options,
      tmpl::conditional_t<
          domain::BoundaryConditions::has_boundary_conditions_base_v<
              typename Metavariables::system>,
          tmpl::list<
              LowerZBoundaryCondition<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>>,
              UpperZBoundaryCondition<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>>,
              MantleBoundaryCondition<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>>>,
          tmpl::list<IsPeriodicInZ>>>;

  static constexpr Options::String help{
      "Creates a cylinder using a Zernike basis radially, Fourier in the "
      "angular direction, and I1 in the z direction"};

  AngularCylinder(
      typename OuterRadius::type outer_radius,
      typename LowerZBound::type lower_z_bound,
      typename UpperZBound::type upper_z_bound,
      typename RadialPartitioning::type radial_partitioning,
      typename PartitioningInZ::type partitioning_in_z,
      typename InitialCylinderThetaGridPoints::type
          initial_cylinder_theta_grid_points,
      typename InitialCylinderZGridPoints::type initial_cylinder_z_grid_points,
      typename InitialHollowCylinderGridPoints::type
          initial_hollow_cylinder_grid_points,
      typename DistributionInZ::type distribution_in_z,
      typename InitialRefinementInZ::type initial_refinement_in_z,
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
          time_dependence = nullptr,
      bool is_periodic_in_z = false,
      const Options::Context& context = {});

  AngularCylinder(
      typename OuterRadius::type outer_radius,
      typename LowerZBound::type lower_z_bound,
      typename UpperZBound::type upper_z_bound,
      typename RadialPartitioning::type radial_partitioning,
      typename PartitioningInZ::type partitioning_in_z,
      typename InitialCylinderThetaGridPoints::type
          initial_cylinder_theta_grid_points,
      typename InitialCylinderZGridPoints::type initial_cylinder_z_grid_points,
      typename InitialHollowCylinderGridPoints::type
          initial_hollow_cylinder_grid_points,
      typename DistributionInZ::type distribution_in_z,
      typename InitialRefinementInZ::type initial_refinement_in_z,
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
          time_dependence = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          lower_z_boundary_condition = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          upper_z_boundary_condition = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          mantle_boundary_condition = nullptr,
      const Options::Context& context = {});

  AngularCylinder() = default;
  AngularCylinder(const AngularCylinder&) = delete;
  AngularCylinder(AngularCylinder&&) = default;
  AngularCylinder& operator=(const AngularCylinder&) = delete;
  AngularCylinder& operator=(AngularCylinder&&) = default;
  ~AngularCylinder() override = default;

  Domain<3> create_domain() const override;

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override;

  std::vector<std::array<size_t, 3>> initial_extents() const override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const override;

  std::vector<std::string> block_names() const override { return block_names_; }

  std::unordered_map<std::string, std::unordered_set<std::string>>
  block_groups() const override {
    return block_groups_;
  }

 private:
  Domain<3> build_domain(const Options::Context& context) const;
  Domain<3> domain_{};
  typename OuterRadius::type outer_radius_{};
  typename LowerZBound::type lower_z_bound_{};
  typename UpperZBound::type upper_z_bound_{};
  typename IsPeriodicInZ::type is_periodic_in_z_{};
  typename RadialPartitioning::type radial_partitioning_{};
  typename PartitioningInZ::type partitioning_in_z_{};
  typename InitialCylinderThetaGridPoints::type
      initial_cylinder_theta_grid_points_{};
  typename InitialCylinderZGridPoints::type initial_cylinder_z_grid_points_{};
  std::vector<std::array<size_t, 3>> initial_hollow_cylinder_grid_points_{};
  DistributionInZ::type distribution_in_z_{};
  std::vector<size_t> initial_refinement_in_z_{};
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dependence_ = nullptr;
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      lower_z_boundary_condition_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      upper_z_boundary_condition_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      mantle_boundary_condition_{};
  std::vector<std::string> block_names_;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups_;
  size_t num_blocks_{};
};
}  // namespace domain::creators
