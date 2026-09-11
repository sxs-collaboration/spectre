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
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
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
 * \brief An axisymmetric Cartoon domain covering the half-plane \f$x \ge 0\f$
 * with annular blocks that use a half-Fourier basis in the azimuthal
 * direction.
 *
 * \details This is the polar-coordinate counterpart of
 * `domain::creators::CartoonSphere2D`. Where `CartoonSphere2D` tiles the
 * half-plane with three Legendre wedges per shell, this creator uses a
 * single annular block per shell whose azimuthal direction is spanned by the
 * `Spectral::Basis::HalfFourier` basis.
 *
 * To match the symmetry to allowed solution, \f$\phi\f$ sweeps counterclockwise
 * from the \f$-y\f$ axis, through \f$+x\f$, to the \f$+y\f$ axis. The
 * symmetry axis \f$x = 0\f$ is not an external boundary: it is handled
 * internally by the parity of the half-Fourier basis. See
 * `Spectral::make_component_parity_array` and `Spectral::HalfFourier`.
 *
 * \note The origin must be excised as the radial topology is `I1`.
 */
class AngularCartoonSphere2D : public DomainCreator<3> {
 public:
  using maps_list = tmpl::list<domain::CoordinateMap<
      Frame::BlockLogical, Frame::Inertial,
      domain::CoordinateMaps::ProductOf3Maps<
          domain::CoordinateMaps::Interval, domain::CoordinateMaps::Identity<1>,
          domain::CoordinateMaps::Identity<1>>,
      domain::CoordinateMaps::ProductOf2Maps<
          domain::CoordinateMaps::PolarToCartesian,
          domain::CoordinateMaps::Identity<1>>,
      domain::CoordinateMaps::ProductOf2Maps<
          domain::CoordinateMaps::DiscreteRotation<2>,
          domain::CoordinateMaps::Identity<1>>>>;

  /// \brief Radius of the excised region at the center of the domain
  struct InnerRadius {
    using type = double;
    static constexpr Options::String help = {
        "Radius of the excised region at the center of the domain. Must be "
        "positive."};
  };

  /// \brief Radius of the outer edge of the domain
  struct OuterRadius {
    using type = double;
    static constexpr Options::String help = {
        "Radius of the outer edge of the domain."};
  };

  /// \brief Radial coordinates of the boundaries splitting the shells
  struct RadialPartitioning {
    using type = std::vector<double>;
    static constexpr Options::String help = {
        "Radial coordinates of the boundaries splitting the domain into "
        "concentric shells. Must be strictly increasing and lie strictly "
        "between InnerRadius and OuterRadius."};
  };

  /// \brief Distribution of grid points in each radial block
  struct RadialDistribution {
    using type = std::vector<domain::CoordinateMaps::Distribution>;
    static constexpr Options::String help = {
        "Distribution of grid points in the radial direction for each "
        "radial block. Must have one entry per radial block."};
  };

  /*!
   * \brief Initial number of \f$[r, \phi]\f$ grid points in each shell
   *
   * \details Can be a single pair that is applied to every shell, or one pair
   * per shell.
   */
  struct InitialGridPoints {
    using type =
        std::variant<std::array<size_t, 2>, std::vector<std::array<size_t, 2>>>;
    static constexpr Options::String help = {
        "Initial number of grid points in [r, phi] for each shell. If one "
        "pair is given it is applied to all shells, otherwise specify one "
        "pair per shell."};
  };

  /// \brief Initial refinement level in the radial direction of each shell
  struct InitialRefinementInR {
    using type = std::vector<size_t>;
    static constexpr Options::String help = {
        "Initial refinement level in the radial direction for each radial "
        "block. Must have one entry per radial block. The azimuthal and "
        "Cartoon directions cannot be h-refined."};
  };

  /// \brief Time dependence of the domain
  struct TimeDependence {
    using type =
        std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>;
    static constexpr Options::String help = {
        "The time dependence of the moving mesh domain. Specify `None` for no "
        "time dependent maps."};
  };

  /// \brief Boundary conditions group
  struct BoundaryConditions {
    static constexpr Options::String help =
        "Options for the boundary conditions";
  };

  /// \brief Boundary condition at the inner (excision) radius
  template <typename BoundaryConditionsBase>
  struct InnerBoundaryCondition {
    using group = BoundaryConditions;
    static std::string name() { return "InnerBoundary"; }
    static constexpr Options::String help =
        "The boundary condition imposed at the inner (excision) radius.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  /// \brief Boundary condition at the outer radius
  template <typename BoundaryConditionsBase>
  struct OuterBoundaryCondition {
    using group = BoundaryConditions;
    static std::string name() { return "OuterBoundary"; }
    static constexpr Options::String help =
        "The boundary condition imposed at the outer radius.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  using basic_options = tmpl::list<InnerRadius, OuterRadius, RadialPartitioning,
                                   RadialDistribution, InitialGridPoints,
                                   InitialRefinementInR, TimeDependence>;

  template <typename Metavariables>
  using options = tmpl::conditional_t<
      domain::BoundaryConditions::has_boundary_conditions_base_v<
          typename Metavariables::system>,
      tmpl::push_back<
          basic_options,
          InnerBoundaryCondition<
              domain::BoundaryConditions::get_boundary_conditions_base<
                  typename Metavariables::system>>,
          OuterBoundaryCondition<
              domain::BoundaryConditions::get_boundary_conditions_base<
                  typename Metavariables::system>>>,
      basic_options>;

  static constexpr Options::String help{
      "An axisymmetric Cartoon domain covering the half-plane x >= 0 with "
      "annular blocks that use a half-Fourier basis in the azimuthal "
      "direction."};

  AngularCartoonSphere2D(
      typename InnerRadius::type inner_radius,
      typename OuterRadius::type outer_radius,
      typename RadialPartitioning::type radial_partitioning,
      typename RadialDistribution::type radial_distribution,
      typename InitialGridPoints::type initial_grid_points,
      typename InitialRefinementInR::type initial_refinement_in_r,
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
          time_dependence = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          inner_boundary_condition = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          outer_boundary_condition = nullptr,
      const Options::Context& context = {});

  AngularCartoonSphere2D() = default;
  AngularCartoonSphere2D(const AngularCartoonSphere2D&) = delete;
  AngularCartoonSphere2D(AngularCartoonSphere2D&&) = default;
  AngularCartoonSphere2D& operator=(const AngularCartoonSphere2D&) = delete;
  AngularCartoonSphere2D& operator=(AngularCartoonSphere2D&&) = default;
  ~AngularCartoonSphere2D() override = default;

  Domain<3> create_domain() const override;

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override;

  std::vector<std::array<size_t, 3>> initial_extents() const override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const override;

  /// Block names are `Shell0`, `Shell1`, ...
  std::vector<std::string> block_names() const override { return block_names_; }

  /// Block groups are `Shells`.
  std::unordered_map<std::string, std::unordered_set<std::string>>
  block_groups() const override {
    return block_groups_;
  }

  auto functions_of_time(const std::unordered_map<std::string, double>&
                             initial_expiration_times = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

 private:
  typename InnerRadius::type inner_radius_{};
  typename OuterRadius::type outer_radius_{};
  typename RadialPartitioning::type radial_partitioning_;
  typename RadialDistribution::type radial_distribution_;
  std::vector<std::array<size_t, 2>> initial_grid_points_;
  std::vector<size_t> initial_refinement_in_r_;
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dependence_ = nullptr;
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      inner_boundary_condition_;
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      outer_boundary_condition_;
  std::vector<std::string> block_names_;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups_;
  size_t num_blocks_{};
};
}  // namespace domain::creators
