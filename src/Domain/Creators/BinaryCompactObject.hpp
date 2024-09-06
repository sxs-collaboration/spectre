// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/TimeDependentOptions/BinaryCompactObject.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace CoordinateMaps {
class Affine;
class Equiangular;
template <size_t VolumeDim>
class Identity;
template <typename Map1, typename Map2>
class ProductOf2Maps;
template <typename Map1, typename Map2, typename Map3>
class ProductOf3Maps;
template <size_t Dim>
class Wedge;
template <size_t VolumeDim>
class DiscreteRotation;
class Frustum;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;

namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime
}  // namespace domain

namespace Frame {
struct Grid;
struct Distorted;
struct Inertial;
struct BlockLogical;
}  // namespace Frame
/// \endcond

namespace domain {
namespace creators {
namespace bco {
/*!
 * \brief Create a set of centers of objects for the binary domains.
 *
 * \details Will add the following centers to the set:
 *
 * - Center: The origin
 * - CenterA: Center of object A
 * - CenterB: Center of object B
 *
 * \return Object required by the DomainCreator%s
 */
std::unordered_map<std::string, tnsr::I<double, 3, Frame::Grid>>
create_grid_anchors(const std::array<double, 3>& center_a,
                    const std::array<double, 3>& center_b);
}  // namespace bco

/*!
 * \ingroup ComputationalDomainGroup
 *
 * \brief A general domain for two compact objects.
 *
 * \image html binary_compact_object_domain.png "A BHNS domain."
 *
 * Creates a 3D Domain that represents a binary compact object solution. The
 * Domain consists of 4 or 5 nested layers of blocks; these layers are, working
 * from the interior toward the exterior:
 *
 * - **Object A/B interior**: (optional) The block at the center of each
 *   compact object, if not excised. If present, this block is a cube. If
 *   excised, the hole left by its absence is spherical.
 * - **Object A/B shell**: The 6 blocks that resolve each individual compact
 *   object. This layer has a spherical outer boundary - if the corresponding
 *   interior block exists, then the layer is a cube-to-sphere transition; if
 *   the interior block is excised, then the layer is a spherical shell.
 * - **Object A/B cube**: The 6 blocks that surround each object with a cube.
 *   Around each compact object, this layer transitions from a sphere to a cube.
 * - **Envelope**: The 10 blocks that transition from the two inner cubes to a
 *   sphere centered at the origin.
 * - **Outer shell**: The 10 blocks that form an outer shell centered at the
 *   origin, consisting of 2 endcap Wedges on the +x and -x axes, and 8 half
 *   Wedges along the yz plane. This layer is spherical, so a logarithmic map
 *   can optionally be used in this layer. This allows the domain to extend to
 *   large radial distances from the compact objects. This layer can be
 *   h-refined radially, creating a layer of multiple concentric spherical
 *   shells.
 *
 * \par Notes:
 *
 * - Object A is located to the right of the origin (along the positive x-axis)
 *   and Object B is located to the left of the origin.
 * - This domain offers some grid anchors. See
 *   `domain::creators::bco::create_grid_anchors` for which ones are offered.
 * - "Cutting plane" refers to the plane along which the domain divides into two
 *   hemispheres. The cutting plane always intersects the x-axis at the origin.
 * - The x-coordinate locations of the two objects should be chosen such that
 *   the center of mass is located at x=0.
 * - The cubes are first constructed at the origin. Then, they are translated
 *   left/right by their Object's x-coordinate and offset depending on the cube
 *   length.
 * - The CubeScale option describes how to scale the length of the cube
 *   surrounding object A/B. It must be greater than or equal to 1.0 with 1.0
 *   meaning the side length of the cube is the initial physical separation
 *   between the two objects. If CubeScale is greater than 1.0, the centers of
 *   the two objects will be offset relative to the centers of the cubes.
 * - Alternatively, one can replace the inner shell and cube blocks of each
 *   object with a single cartesian cube. This is less efficient, but allows
 *   testing of methods only coded on cartesian grids.
 *
 * \par Time dependence:
 * The following time-dependent maps are applied:
 *
 * - A piecewise `Expansion`, a `Rotation` and a piecewise `Translation` is
 * applied to all blocks from the Grid to the Inertial frame. However, if there
 * is a shape map in the block (defined below), then the expansion, rotation,
 * and translation maps go from the Distorted to the Inertial frame.
 * - If an object is excised, then the corresponding shell has a
 *   `Shape` map. The shape map goes from the Grid to the Distorted frame.
 *
 * All time dependent maps are optional to specify. To include a map, specify
 * its options. Otherwise specify `None` for that map. You can also turn off
 * time dependent maps all together by specifying `None` for the
 * `TimeDependentMaps` option. See
 * `domain::creators::bco::TimeDependentMapOptions`. This class must pass a
 * template parameter of `false` to
 * `domain::creators::bco::TimeDependentMapOptions`.
 *
 * The `UseWorldtube` template parameter is set to false by default. When set to
 * true, some of the functions of time will be `IntegratedFunctionOfTime` used
 * to control the orbit of the worldtube.
 */
template <bool UseWorldtube = false>
class BinaryCompactObject : public DomainCreator<3> {
 private:
  // Time-independent maps
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Identity2D = CoordinateMaps::Identity<2>;
  // The Translation type is no longer needed, but it is kept here for backwards
  // compatibility with old domains.
  using Translation = CoordinateMaps::ProductOf2Maps<Affine, Identity2D>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular3D =
      CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Equiangular>;

 public:
  using maps_list = tmpl::flatten<tmpl::list<
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial, Affine3D>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            Equiangular3D>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial, Affine3D,
                            Translation>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            CoordinateMaps::DiscreteRotation<3>, Affine3D>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            Equiangular3D>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial, Equiangular3D,
                            Translation>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            CoordinateMaps::Frustum>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            CoordinateMaps::Wedge<3>>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            CoordinateMaps::Wedge<3>, Translation>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial, Affine3D,
                            Affine3D>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial, Equiangular3D,
                            Affine3D>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            CoordinateMaps::Wedge<3>, Affine3D>,
      bco::TimeDependentMapOptions<false>::maps_list>>;

  /// Options for an excision region in the domain
  struct Excision {
    static constexpr Options::String help = {
        "Excise the interior of the object, leaving a spherical hole in its "
        "absence."};
    template <typename BoundaryConditionsBase>
    struct BoundaryCondition {
      static std::string name() { return "ExciseWithBoundaryCondition"; }
      using type = std::unique_ptr<BoundaryConditionsBase>;
      static constexpr Options::String help = {
          "The boundary condition to impose on the excision surface."};
    };
    template <typename Metavariables>
    using options = tmpl::list<BoundaryCondition<
        domain::BoundaryConditions::get_boundary_conditions_base<
            typename Metavariables::system>>>;
    Excision() = default;
    // NOLINTNEXTLINE(google-explicit-constructor)
    Excision(std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
                 boundary_condition_in)
        : boundary_condition(std::move(boundary_condition_in)) {}
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        boundary_condition;
  };

  /// Options for one of the two objects in the binary domain
  struct Object {
    static constexpr Options::String help = {
        "Options for an object in a binary domain."};
    struct InnerRadius {
      using type = double;
      static constexpr Options::String help = {
          "Inner coordinate radius of Layer 1."};
      static double lower_bound() { return 0.; }
    };
    struct OuterRadius {
      using type = double;
      static constexpr Options::String help = {
          "Outer coordinate radius of Layer 1"};
      static double lower_bound() { return 0.; }
    };
    struct XCoord {
      using type = double;
      static constexpr Options::String help = {"x-coordinate of center."};
    };
    struct Interior {
      using type = Options::Auto<Excision>;
      static constexpr Options::String help = {
          "Specify 'ExciseWithBoundaryCondition' and a boundary condition to "
          "excise Layer 0, leaving a spherical hole in its absence, or set to "
          "'Auto' to fill the interior."};
    };
    struct ExciseInterior {
      using type = bool;
      static constexpr Options::String help = {
          "Excise Layer 0, leaving a spherical hole in its absence."};
    };
    struct UseLogarithmicMap {
      using type = bool;
      static constexpr Options::String help = {
          "Use a logarithmically spaced radial grid in the part of Layer 1 "
          "enveloping the object (requires the interior is excised)"};
    };
    template <typename Metavariables>
    using options = tmpl::list<
        InnerRadius, OuterRadius, XCoord,
        tmpl::conditional_t<
            domain::BoundaryConditions::has_boundary_conditions_base_v<
                typename Metavariables::system>,
            Interior, ExciseInterior>,
        UseLogarithmicMap>;
    Object() = default;
    Object(double local_inner_radius, double local_outer_radius,
           double local_x_coord, std::optional<Excision> interior,
           bool local_use_logarithmic_map)
        : inner_radius(local_inner_radius),
          outer_radius(local_outer_radius),
          x_coord(local_x_coord),
          inner_boundary_condition(
              interior.has_value()
                  ? std::make_optional(std::move(interior->boundary_condition))
                  : std::nullopt),
          use_logarithmic_map(local_use_logarithmic_map) {}
    Object(double local_inner_radius, double local_outer_radius,
           double local_x_coord, bool local_excise_interior,
           bool local_use_logarithmic_map)
        : inner_radius(local_inner_radius),
          outer_radius(local_outer_radius),
          x_coord(local_x_coord),
          inner_boundary_condition(
              local_excise_interior
                  ? std::optional<std::unique_ptr<
                        domain::BoundaryConditions::BoundaryCondition>>{nullptr}
                  : std::nullopt),
          use_logarithmic_map(local_use_logarithmic_map) {}

    /// Whether or not the object should be excised from the domain, leaving a
    /// spherical hole. When this is true, `inner_boundary_condition` is
    /// guaranteed to hold a value (though it might be a `nullptr` if we are not
    /// working with boundary conditions).
    bool is_excised() const;

    double inner_radius;
    double outer_radius;
    double x_coord;
    std::optional<
        std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>
        inner_boundary_condition;
    bool use_logarithmic_map;
  };

  // Simpler version of an object: a single cube centered on (xCoord,0,0)
  struct CartesianCubeAtXCoord {
    static constexpr Options::String help = {
        "Options to set a single cube at a location on the x-axis"};
    struct XCoord {
      static std::string name() { return "CartesianCubeAtXCoord"; }
      using type = double;
      static constexpr Options::String help = {"x-coordinate of center."};
    };
    using options = tmpl::list<XCoord>;
    CartesianCubeAtXCoord() = default;
    // NOLINTNEXTLINE(google-explicit-constructor)
    CartesianCubeAtXCoord(const double x_coord_in) : x_coord(x_coord_in) {}
    bool is_excised() const { return false; }
    double x_coord;
  };

  struct ObjectA {
    using type = std::variant<Object, CartesianCubeAtXCoord>;
    static constexpr Options::String help = {
        "Options for the object to the right of the origin (along the positive "
        "x-axis)."};
  };

  struct ObjectB {
    using type = std::variant<Object, CartesianCubeAtXCoord>;
    static constexpr Options::String help = {
        "Options for the object to the left of the origin (along the negative "
        "x-axis)."};
  };

  struct CenterOfMassOffset {
    using type = std::array<double, 2>;
    static constexpr Options::String help = {
        "Offset in the y and z axes applied to both object A and B in order to "
        "control the center of mass. This moves the location of the two objects"
        " in the grid frame but keeps the Envelope and OuterShell centered on "
        "the origin in the grid frame."};
  };

  struct Envelope {
    static constexpr Options::String help = {
        "Options for the sphere enveloping the two objects."};
  };

  struct EnvelopeRadius {
    using group = Envelope;
    static std::string name() { return "Radius"; }
    using type = double;
    static constexpr Options::String help = {
        "Radius of the sphere enveloping the two objects."};
  };

  struct OuterShell {
    static constexpr Options::String help = {
        "Options for the outer spherical shell."};
  };

  struct OuterRadius {
    using group = OuterShell;
    static std::string name() { return "Radius"; }
    using type = double;
    static constexpr Options::String help = {"Radius of the entire domain."};
  };

  struct OpeningAngle {
    using group = OuterShell;
    static std::string name() { return "OpeningAngle"; }
    using type = double;
    static constexpr Options::String help = {
        "The combined opening angle of the two half wedges of the outer shell"
        " in degrees. A value of 120.0 partitions the x-y and x-z slices of the"
        " outer shell into six Blocks of equal angular size."};
  };

  struct CubeScale {
    using type = double;
    static constexpr Options::String help = {
        "Specify the desired cube scale that must be greater than or equal to "
        "1.0. The initial separation is multiplied by this cube scale to "
        "produce larger cubes around each object which is desirable when "
        "closer to merger."};
    static double lower_bound() { return 1.0; }
  };

  struct InitialRefinement {
    using type =
        std::variant<size_t, std::array<size_t, 3>,
                     std::vector<std::array<size_t, 3>>,
                     std::unordered_map<std::string, std::array<size_t, 3>>>;
    static constexpr Options::String help = {
        "Initial refinement level in each block of the domain. See main help "
        "text for details."};
  };

  struct InitialGridPoints {
    using type =
        std::variant<size_t, std::array<size_t, 3>,
                     std::vector<std::array<size_t, 3>>,
                     std::unordered_map<std::string, std::array<size_t, 3>>>;
    static constexpr Options::String help = {
        "Initial number of grid points in the elements of each block of the "
        "domain. See main help text for details."};
  };

  struct UseEquiangularMap {
    using type = bool;
    static constexpr Options::String help = {
        "Distribute grid points equiangularly."};
    static bool suggested_value() { return true; }
  };

  struct RadialDistributionEnvelope {
    using group = Envelope;
    static std::string name() { return "RadialDistribution"; }
    using type = CoordinateMaps::Distribution;
    static constexpr Options::String help = {
        "The distribution of radial grid points in the envelope, the layer "
        "made of ten bulged Frustums."};
  };

  struct RadialDistributionOuterShell {
    using group = OuterShell;
    static std::string name() { return "RadialDistribution"; }
    using type = CoordinateMaps::Distribution;
    static constexpr Options::String help = {
        "The distribution of radial grid points in Layer 5, the outer "
        "spherical shell that covers the wave zone."};
  };

  template <typename BoundaryConditionsBase>
  struct OuterBoundaryCondition {
    using group = OuterShell;
    static std::string name() { return "BoundaryCondition"; }
    static constexpr Options::String help =
        "Options for the outer boundary conditions.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  // This is for optional time dependent maps
  struct TimeDependentMaps {
    using type = Options::Auto<bco::TimeDependentMapOptions<false>,
                               Options::AutoLabel::None>;
    static constexpr Options::String help =
        bco::TimeDependentMapOptions<false>::help;
  };

  template <typename Metavariables>
  using options = tmpl::append<
      tmpl::list<ObjectA, ObjectB, CenterOfMassOffset, EnvelopeRadius,
                 OuterRadius, CubeScale, InitialRefinement, InitialGridPoints,
                 UseEquiangularMap, RadialDistributionEnvelope,
                 RadialDistributionOuterShell, OpeningAngle, TimeDependentMaps>,
      tmpl::conditional_t<
          domain::BoundaryConditions::has_boundary_conditions_base_v<
              typename Metavariables::system>,
          tmpl::list<OuterBoundaryCondition<
              domain::BoundaryConditions::get_boundary_conditions_base<
                  typename Metavariables::system>>>,
          tmpl::list<>>>;

  static constexpr Options::String help{
      "A general domain for two compact objects. Each object is represented by "
      "a cube along the x-axis. Object A is located on the right and Object B "
      "is located on the left. Their locations should be chosen such that "
      "their center of mass is located at the origin."
      "The interior of each object can have a spherical excision to "
      "represent a black hole."
      "\n"
      "The two objects are enveloped by a sphere centered at the origin, "
      "and by an outer shell that can transition to large outer radii."
      "\n"
      "Both the InitialRefinement and the InitialGridPoints can be one of "
      "the following:\n"
      "  - A single number: Uniform refinement in all blocks and "
      "dimensions\n"
      "  - Three numbers: Refinement in [polar, azimuthal, radial] direction "
      "in all blocks\n"
      "  - A map from block names or groups to three numbers: Per-block "
      "refinement in [polar, azimuthal, radial] direction\n"
      "  - A list, with [polar, azimuthal, radial] refinement for each block\n"
      "\n"
      "The domain can rotate around the "
      "z-axis and expand/compress radially. The two objects can each have a "
      "spherical distortion (shape map)."};

  BinaryCompactObject(
      typename ObjectA::type object_A, typename ObjectB::type object_B,
      std::array<double, 2> center_of_mass_offset, double envelope_radius,
      double outer_radius, double cube_scale,
      const typename InitialRefinement::type& initial_refinement,
      const typename InitialGridPoints::type& initial_number_of_grid_points,
      bool use_equiangular_map = true,
      CoordinateMaps::Distribution radial_distribution_envelope =
          CoordinateMaps::Distribution::Projective,
      CoordinateMaps::Distribution radial_distribution_outer_shell =
          CoordinateMaps::Distribution::Linear,
      double opening_angle_in_degrees = 90.0,
      std::optional<bco::TimeDependentMapOptions<false>>
          time_dependent_options = std::nullopt,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          outer_boundary_condition = nullptr,
      const Options::Context& context = {});

  BinaryCompactObject() = default;
  BinaryCompactObject(const BinaryCompactObject&) = delete;
  BinaryCompactObject(BinaryCompactObject&&) = default;
  BinaryCompactObject& operator=(const BinaryCompactObject&) = delete;
  BinaryCompactObject& operator=(BinaryCompactObject&&) = default;
  ~BinaryCompactObject() override = default;

  Domain<3> create_domain() const override;

  std::unordered_map<std::string, tnsr::I<double, 3, Frame::Grid>>
  grid_anchors() const override {
    return grid_anchors_;
  }

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override;

  std::vector<std::array<size_t, 3>> initial_extents() const override {
    return initial_number_of_grid_points_;
  }

  std::vector<std::array<size_t, 3>> initial_refinement_levels()
      const override {
    return initial_refinement_;
  }

  std::vector<std::string> block_names() const override { return block_names_; }

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
  typename ObjectA::type object_A_{};
  typename ObjectB::type object_B_{};
  std::array<double, 2> center_of_mass_offset_{};
  double envelope_radius_ = std::numeric_limits<double>::signaling_NaN();
  double outer_radius_ = std::numeric_limits<double>::signaling_NaN();
  std::vector<std::array<size_t, 3>> initial_refinement_{};
  std::vector<std::array<size_t, 3>> initial_number_of_grid_points_{};
  bool use_equiangular_map_ = true;
  CoordinateMaps::Distribution radial_distribution_envelope_ =
      CoordinateMaps::Distribution::Projective;
  CoordinateMaps::Distribution radial_distribution_outer_shell_ =
      CoordinateMaps::Distribution::Linear;
  double translation_{};
  double length_inner_cube_{};
  double length_outer_cube_{};
  size_t number_of_blocks_{};
  size_t first_outer_shell_block_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      outer_boundary_condition_;
  std::vector<std::string> block_names_{};
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups_{};
  std::unordered_map<std::string, tnsr::I<double, 3, Frame::Grid>>
      grid_anchors_{};
  double offset_x_coord_a_{};
  double offset_x_coord_b_{};

  // Variables to handle std::variant on Object A and B
  double x_coord_a_{};
  double x_coord_b_{};
  bool is_excised_a_ = false;
  bool is_excised_b_ = false;
  bool use_single_block_a_ = false;
  bool use_single_block_b_ = false;
  std::optional<bco::TimeDependentMapOptions<false>> time_dependent_options_{};
  double opening_angle_ = std::numeric_limits<double>::signaling_NaN();
};
}  // namespace creators
}  // namespace domain
