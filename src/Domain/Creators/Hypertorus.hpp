// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t VolumeDim>
class Domain;
namespace domain {
namespace CoordinateMaps {
class Affine;
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

/// \brief Create a domain consisting of a single Block with the topology of a
/// hypertorus in `Dim` dimensions.
///
/// \details The domain will have topology S1 (i.e. is periodic) in each
/// dimension.  Therefore the domain has no external boundaries.  The domain
/// cannot be refined, and is intended for using DG with a Fourier basis in
/// each dimension.  If you want to h-refine a periodic domain, use one of
/// the rectilinear domain creators (i.e. Interval, Rectangle, or Brick).
template <size_t Dim>
class Hypertorus : public DomainCreator<Dim> {
 private:
  static_assert(Dim == 1 or Dim == 2 or Dim == 3,
                "Hypertorus domain is only implemented in 1, 2, or 3 "
                "dimensions.");

  using Affine = CoordinateMaps::Affine;
  using Affine2D = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

 public:
  using maps_list = tmpl::list<domain::CoordinateMap<
      Frame::BlockLogical, Frame::Inertial,
      tmpl::conditional_t<Dim == 1, Affine,
                          tmpl::conditional_t<Dim == 2, Affine2D, Affine3D>>>>;

  static std::string name() {
    if constexpr (Dim == 1) {
      return "PeriodicInterval";
    } else if constexpr (Dim == 2) {
      return "PeriodicRectangle";
    } else {
      return "PeriodicBrick";
    }
  }

  /// Lower coordinate bound in each dimension
  struct LowerBound {
    using type = std::array<double, Dim>;
    static constexpr Options::String help = {"Lower bound in each dimension."};
  };

  /// Upper coordinate bound in each dimension
  struct UpperBound {
    using type = std::array<double, Dim>;
    static constexpr Options::String help = {"Upper bound in each dimension."};
  };

  /// \brief Initial maximum mode number \f$M\f$ retained in the Fourier basis
  /// in each dimension
  ///
  /// \details The number of grid points will be \f$2M + 1\f$.
  struct InitialMaximumModeNumber {
    using type = std::array<size_t, Dim>;
    static constexpr Options::String help = {
        "Initial value of M, the maximum retained mode in the Fourier basis."};
  };

  /// Time dependence for the domain. Specify `None` for no time dependent maps
  struct TimeDependence {
    using type =
        std::unique_ptr<domain::creators::time_dependence::TimeDependence<Dim>>;
    static constexpr Options::String help = {
        "The time dependence of the moving mesh domain."};
  };

  template <typename Metavariables>
  using options = tmpl::list<LowerBound, UpperBound, InitialMaximumModeNumber,
                             TimeDependence>;

  static constexpr Options::String help{"A periodic rectilinear domain."};

  Hypertorus(
      const std::array<double, Dim>& lower_bounds,
      const std::array<double, Dim>& upper_bounds,
      const std::array<size_t, Dim>& initial_max_modes,
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<Dim>>
          time_dependence = nullptr,
      const Options::Context& context = {});

  Hypertorus() = default;
  Hypertorus(const Hypertorus&) = delete;
  Hypertorus(Hypertorus&&) = default;
  Hypertorus& operator=(const Hypertorus&) = delete;
  Hypertorus& operator=(Hypertorus&&) = default;
  ~Hypertorus() override = default;

  Domain<Dim> create_domain() const override;

  std::vector<DirectionMap<
      Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override;

  std::vector<std::array<size_t, Dim>> initial_extents() const override;

  std::vector<std::array<size_t, Dim>> initial_refinement_levels()
      const override;

  auto functions_of_time(const std::unordered_map<std::string, double>&
                             initial_expiration_times = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

  std::vector<std::string> block_names() const override { return block_names_; }

  std::unordered_map<std::string, std::unordered_set<std::string>>
  block_groups() const override {
    return {{name(), {name()}}};
  }

 private:
  std::array<double, Dim> lower_bounds_{};
  std::array<double, Dim> upper_bounds_{};
  std::array<size_t, Dim> initial_num_points_{};
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<Dim>>
      time_dependence_;
  inline static const std::vector<std::string> block_names_{name()};
};

using PeriodicInterval = Hypertorus<1>;
using PeriodicRectangle = Hypertorus<2>;
using PeriodicBrick = Hypertorus<3>;

}  // namespace domain::creators
