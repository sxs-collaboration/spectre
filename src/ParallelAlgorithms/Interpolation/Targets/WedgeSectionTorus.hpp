// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/ComputeTargetPoints.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
namespace intrp {
namespace Tags {
template <typename TemporalId>
struct TemporalIds;
}  // namespace Tags
}  // namespace intrp
/// \endcond

namespace intrp {

namespace OptionHolders {
/// \brief A solid torus of points, useful, e.g., when measuring data from an
/// accretion disc.
///
/// The torus's cross section (e.g., a cut at \f$\phi=0\f$) is a wedge-like
/// shape bounded by \f$r_{\text{min}} \le r \le r_{\text{max}}\f$ and
/// \f$\theta_{\text{min}} \le \theta \le \theta_{\text{max}}\f$.
///
/// The grid points are located on surfaces of constant \f$r\f$, \f$\theta\f$,
/// and \f$\phi\f$. There are `NumberRadialPoints` points in the radial
/// direction between `MinRadius` and `MaxRadius` (including these endpoints);
/// `NumberThetaPoints` points in the \f$\theta\f$ direction between `MinTheta`
/// and `MaxTheta` (including these endpoints); `NumberPhiPoints` points in the
/// \f$\phi\f$ direction (with one point always at \f$\phi=0\f$).
///
/// By default, the points follow a Legendre Gauss-Lobatto distribution in the
/// \f$r\f$ and \f$\theta\f$ directions, and a uniform distribution in the
/// \f$\phi\f$ direction. The distribution in the \f$r\f$ (and/or \f$\theta\f$)
/// direction can be made uniform using the `UniformRadialGrid` (and/or
/// `UniformThetaGrid`) option.
///
/// The `target_points` form a 3D mesh ordered with \f$r\f$ varying fastest,
/// then \f$\theta\f$, and finally \f$\phi\f$ varying slowest.
///
/// \note Input coordinates (radii, angles) are interpreted in the
/// `Frame::Inertial`
struct WedgeSectionTorus {
  struct MinRadius {
    using type = double;
    static constexpr Options::String help = {"Inner radius of torus"};
    static type lower_bound() { return 0.0; }
  };
  struct MaxRadius {
    using type = double;
    static constexpr Options::String help = {"Outer radius of torus"};
    static type lower_bound() { return 0.0; }
  };
  struct MinTheta {
    using type = double;
    static constexpr Options::String help = {"Angle of top of wedge (radians)"};
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return M_PI; }
  };
  struct MaxTheta {
    using type = double;
    static constexpr Options::String help = {
        "Angle of bottom of wedge (radians)"};
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return M_PI; }
  };
  struct NumberRadialPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Number of radial points, including endpoints"};
    static type lower_bound() { return 2; }
  };
  struct NumberThetaPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Number of theta points, including endpoints"};
    static type lower_bound() { return 2; }
  };
  struct NumberPhiPoints {
    using type = size_t;
    static constexpr Options::String help = {"Number of phi points"};
    static type lower_bound() { return 1; }
  };
  struct UniformRadialGrid {
    using type = bool;
    static constexpr Options::String help = {"Use uniform radial grid"};
  };
  struct UniformThetaGrid {
    using type = bool;
    static constexpr Options::String help = {"Use uniform theta grid"};
  };

  using options =
      tmpl::list<MinRadius, MaxRadius, MinTheta, MaxTheta, NumberRadialPoints,
                 NumberThetaPoints, NumberPhiPoints, UniformRadialGrid,
                 UniformThetaGrid>;
  static constexpr Options::String help = {
      "A torus extending from MinRadius to MaxRadius in r, MinTheta to MaxTheta"
      " in theta, and 2pi in phi."};

  WedgeSectionTorus(double min_radius_in, double max_radius_in,
                    double min_theta_in, double max_theta_in,
                    size_t number_of_radial_points_in,
                    size_t number_of_theta_points_in,
                    size_t number_of_phi_points_in,
                    bool use_uniform_radial_grid_in,
                    bool use_uniform_theta_grid_in,
                    const Options::Context& context = {});

  WedgeSectionTorus() = default;
  WedgeSectionTorus(const WedgeSectionTorus& /*rhs*/) = default;
  WedgeSectionTorus& operator=(const WedgeSectionTorus& /*rhs*/) = delete;
  WedgeSectionTorus(WedgeSectionTorus&& /*rhs*/) = default;
  WedgeSectionTorus& operator=(WedgeSectionTorus&& /*rhs*/) = default;
  ~WedgeSectionTorus() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  double min_radius;
  double max_radius;
  double min_theta;
  double max_theta;
  size_t number_of_radial_points;
  size_t number_of_theta_points;
  size_t number_of_phi_points;
  bool use_uniform_radial_grid;
  bool use_uniform_theta_grid;
};

bool operator==(const WedgeSectionTorus& lhs, const WedgeSectionTorus& rhs);
bool operator!=(const WedgeSectionTorus& lhs, const WedgeSectionTorus& rhs);

}  // namespace OptionHolders

namespace OptionTags {
template <typename InterpolationTargetTag>
struct WedgeSectionTorus {
  using type = OptionHolders::WedgeSectionTorus;
  static constexpr Options::String help{
      "Options for interpolation onto Kerr horizon."};
  static std::string name() {
    return pretty_type::name<InterpolationTargetTag>();
  }
  using group = InterpolationTargets;
};
}  // namespace OptionTags

namespace Tags {
template <typename InterpolationTargetTag>
struct WedgeSectionTorus : db::SimpleTag {
  using type = OptionHolders::WedgeSectionTorus;
  using option_tags =
      tmpl::list<OptionTags::WedgeSectionTorus<InterpolationTargetTag>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& option) { return option; }
};
}  // namespace Tags

namespace TargetPoints {
/// \brief Computes points in a wedge-sectioned torus.
///
/// Conforms to the intrp::protocols::ComputeTargetPoints protocol
///
/// For requirements on InterpolationTargetTag, see
/// intrp::protocols::InterpolationTargetTag
template <typename InterpolationTargetTag>
struct WedgeSectionTorus
    : tt::ConformsTo<intrp::protocols::ComputeTargetPoints> {
  using const_global_cache_tags =
      tmpl::list<Tags::WedgeSectionTorus<InterpolationTargetTag>>;
  using is_sequential = std::false_type;
  using frame = Frame::Inertial;

  template <typename Metavariables, typename DbTags>
  static tnsr::I<DataVector, 3, Frame::Inertial> points(
      const db::DataBox<DbTags>& box,
      const tmpl::type_<Metavariables>& /*meta*/) {
    const auto& options =
        get<Tags::WedgeSectionTorus<InterpolationTargetTag>>(box);

    // Compute locations of constant r/theta/phi surfaces
    const size_t num_radial = options.number_of_radial_points;
    const DataVector radii_1d = [&num_radial, &options]() {
      DataVector result(num_radial);
      if (options.use_uniform_radial_grid) {
        // uniform point distribution
        const double coefficient =
            (options.max_radius - options.min_radius) / (num_radial - 1.0);
        for (size_t r = 0; r < num_radial; ++r) {
          result[r] = options.min_radius + coefficient * r;
        }
      } else {
        // radial Legendre-Gauss-Lobatto distribution via linear rescaling
        const double mean = 0.5 * (options.max_radius + options.min_radius);
        const double diff = 0.5 * (options.max_radius - options.min_radius);
        result =
            mean + diff * Spectral::collocation_points<
                              SpatialDiscretization::Basis::Legendre,
                              SpatialDiscretization::Quadrature::GaussLobatto>(
                              num_radial);
      }
      return result;
    }();
    const size_t num_theta = options.number_of_theta_points;
    const DataVector thetas_1d = [&num_theta, &options]() {
      DataVector result(num_theta);
      if (options.use_uniform_theta_grid) {
        // uniform point distribution
        const double coefficient =
            (options.max_theta - options.min_theta) / (num_theta - 1.0);
        for (size_t theta = 0; theta < num_theta; ++theta) {
          result[theta] = options.min_theta + coefficient * theta;
        }
      } else {
        // Legendre-Gauss-Lobatto point distribution (in theta)
        const double mean = 0.5 * (options.max_theta + options.min_theta);
        const double diff = 0.5 * (options.max_theta - options.min_theta);
        result =
            mean + diff * Spectral::collocation_points<
                              SpatialDiscretization::Basis::Legendre,
                              SpatialDiscretization::Quadrature::GaussLobatto>(
                              num_theta);
      }
      return result;
    }();
    const size_t num_phi = options.number_of_phi_points;
    const DataVector phis_1d = [&num_phi]() {
      DataVector result(num_phi);
      // We do NOT want a grid point at phi = 2pi, as this would duplicate the
      // phi = 0 data. So, divide by num_phi rather than (n-1) as elsewhere.
      const double coefficient = 2.0 * M_PI / num_phi;
      for (size_t phi = 0; phi < num_phi; ++phi) {
        result[phi] = coefficient * phi;
      }
      return result;
    }();

    // Take tensor product to get full 3D r/theta/phi points
    const size_t num_total = num_radial * num_theta * num_phi;
    DataVector radii(num_total);
    DataVector thetas(num_total);
    DataVector phis(num_total);
    for (size_t phi = 0; phi < num_phi; ++phi) {
      for (size_t theta = 0; theta < num_theta; ++theta) {
        for (size_t r = 0; r < num_radial; ++r) {
          const size_t i =
              r + theta * num_radial + phi * num_theta * num_radial;
          radii[i] = radii_1d[r];
          thetas[i] = thetas_1d[theta];
          phis[i] = phis_1d[phi];
        }
      }
    }

    // Compute x/y/z coordinates
    // Note: theta measured from +z axis, phi measured from +x axis
    tnsr::I<DataVector, 3, Frame::Inertial> target_points(num_total);
    get<0>(target_points) = radii * sin(thetas) * cos(phis);
    get<1>(target_points) = radii * sin(thetas) * sin(phis);
    get<2>(target_points) = radii * cos(thetas);

    return target_points;
  }

  template <typename Metavariables, typename DbTags, typename TemporalId>
  static tnsr::I<DataVector, 3, Frame::Inertial> points(
      const db::DataBox<DbTags>& box, const tmpl::type_<Metavariables>& meta,
      const TemporalId& /*temporal_id*/) {
    return points(box, meta);
  }
};

}  // namespace TargetPoints
}  // namespace intrp
