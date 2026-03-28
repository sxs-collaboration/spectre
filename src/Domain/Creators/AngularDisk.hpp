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
class PolarToCartesian;
template <typename Map1, typename Map2>
class ProductOf2Maps;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
/// \endcond

namespace domain::creators {
/*!
 * \brief Create a 2D filled disk domain with radial partitioning using a B2
 * filled disk at the center and Fourier hollow disks surrounding it.
 */
class AngularDisk final : public DomainCreator<2> {
 public:
  using maps_list = tmpl::list<domain::CoordinateMap<
      Frame::BlockLogical, Frame::Inertial,
      domain::CoordinateMaps::ProductOf2Maps<
          domain::CoordinateMaps::Affine, domain::CoordinateMaps::Identity<1>>,
      domain::CoordinateMaps::PolarToCartesian>>;

  /*!
   * \brief Radius of the disk's outer edge
   */
  struct OuterRadius {
    using type = double;
    static constexpr Options::String help = {"Radius of the Disk."};
  };

  /*!
   * \brief Radial coordinates of the boundaries splitting elements
   */
  struct RadialPartitioning {
    using type = std::vector<double>;
    static constexpr Options::String help = {
        "Radial coordinates of the boundaries splitting the inner disk and "
        "potential annulus shells"};
  };

  /*!
   * \brief Initial number of \f$\theta\f$ gridpoints for filled disk. The
   * number for \f$r\f$ is accordingly set to match spectral space sizes. This
   * is enforced to be odd for numerical stability.
   */
  struct InitialDiskThetaGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of grid points in theta (r is accordingly set). This "
        "value must be odd for better numerical stability."};
  };

  /*!
   * \brief Initial number of \f$[r, \theta]\f$ gridpoints. Can be one pair
   * which is applied to all shells, or each can be specified.
   */
  struct InitialAnnulusGridPoints {
    using type =
        std::variant<std::array<size_t, 2>, std::vector<std::array<size_t, 2>>>;
    static constexpr Options::String help = {
        "Initial number of grid points in [r, theta] for each annulus. If one "
        "pair is given, it will be applied to all blocks, otherwise each "
        "annulus can be specified."};
  };

  /*!
   * \brief Boundary condition to impose on outer side
   */
  template <typename BoundaryConditionsBase>
  struct BoundaryCondition {
    static std::string name() { return "BoundaryCondition"; }
    static constexpr Options::String help =
        "The boundary condition to impose on outer side.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  /*!
   * \brief Time dependence of the domain
   */
  struct TimeDependence {
    using type =
        std::unique_ptr<domain::creators::time_dependence::TimeDependence<2>>;
    static constexpr Options::String help = {
        "The time dependence of the moving mesh domain. Specify `None` for no "
        "time dependent maps."};
  };

  using basic_options =
      tmpl::list<OuterRadius, RadialPartitioning, InitialDiskThetaGridPoints,
                 InitialAnnulusGridPoints, TimeDependence>;

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
      "Creates a disk using a Zernike basis radially and Fourier in the "
      "angular direction"};

  AngularDisk(
      typename OuterRadius::type outer_radius,
      typename RadialPartitioning::type radial_partitioning,
      typename InitialDiskThetaGridPoints::type initial_disk_grid_points,
      typename InitialAnnulusGridPoints::type initial_annulus_grid_points,
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<2>>
          time_dependence = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          boundary_condition = nullptr,
      const Options::Context& context = {});

  AngularDisk() = default;
  AngularDisk(const AngularDisk&) = delete;
  AngularDisk(AngularDisk&&) = default;
  AngularDisk& operator=(const AngularDisk&) = delete;
  AngularDisk& operator=(AngularDisk&&) = default;
  ~AngularDisk() override = default;

  Domain<2> create_domain() const override;

  std::vector<DirectionMap<
      2, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override;

  std::vector<std::array<size_t, 2>> initial_extents() const override;

  std::vector<std::array<size_t, 2>> initial_refinement_levels() const override;

  std::vector<std::string> block_names() const override { return block_names_; }

  std::unordered_map<std::string, std::unordered_set<std::string>>
  block_groups() const override {
    return block_groups_;
  }

 private:
  Domain<2> build_domain(const Options::Context& context) const;
  typename OuterRadius::type outer_radius_{};
  typename RadialPartitioning::type radial_partitioning_{};
  typename InitialDiskThetaGridPoints::type initial_disk_grid_points_{};
  std::vector<std::array<size_t, 2>> initial_annulus_grid_points_{};
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<2>>
      time_dependence_ = nullptr;
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      boundary_condition_;
  std::vector<std::string> block_names_;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups_;
  size_t num_blocks_{};
};
}  // namespace domain::creators
