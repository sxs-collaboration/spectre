// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>

#include "ApparentHorizons/Tags.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Options/Options.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"

/// \cond
class DataVector;
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
/// A spherical surface.
///
/// \details The parameter `Lmax` controls the number of collocation points on
/// the spherical surface equal to `(l_max + 1) * (2 * l_max + 1)`
struct Sphere {
  struct Lmax {
    using type = size_t;
    static constexpr Options::String help = {
        "The number of collocation points on the sphere will be equal to "
        "`(l_max + 1) * (2 * l_max + 1)`"};
  };
  struct Center {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {"Center of the sphere"};
  };
  struct Radius {
    using type = double;
    static constexpr Options::String help = {"Radius of the sphere"};
    static constexpr double lower_bound() { return 0.; }
  };
  using options = tmpl::list<Lmax, Center, Radius>;
  static constexpr Options::String help = {"A spherical surface."};
  Sphere(const size_t l_max_in, const std::array<double, 3> center_in,
         const double radius_in);

  Sphere() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  size_t l_max{0};
  std::array<double, 3> center{std::numeric_limits<double>::signaling_NaN()};
  double radius{std::numeric_limits<double>::signaling_NaN()};
};

bool operator==(const Sphere& lhs, const Sphere& rhs);
bool operator!=(const Sphere& lhs, const Sphere& rhs);

}  // namespace OptionHolders

namespace OptionTags {
template <typename InterpolationTargetTag>
struct Sphere {
  using type = OptionHolders::Sphere;
  static constexpr Options::String help{
      "Options for interpolation onto a sphere."};
  static std::string name() { return Options::name<InterpolationTargetTag>(); }
  using group = InterpolationTargets;
};
}  // namespace OptionTags

namespace Tags {
template <typename InterpolationTargetTag>
struct Sphere : db::SimpleTag {
  using type = OptionHolders::Sphere;
  using option_tags = tmpl::list<OptionTags::Sphere<InterpolationTargetTag>>;

  static constexpr bool pass_metavariables = false;
  static OptionHolders::Sphere create_from_options(
      const OptionHolders::Sphere& option) {
    return option;
  }
};
}  // namespace Tags

namespace TargetPoints {
/// \brief Computes points on a spherical surface.
///
/// For requirements on InterpolationTargetTag, see InterpolationTarget
template <typename InterpolationTargetTag, typename Frame>
struct Sphere {
  using const_global_cache_tags =
      tmpl::list<Tags::Sphere<InterpolationTargetTag>>;
  using is_sequential = std::false_type;
  using frame = Frame;

  using simple_tags = typename StrahlkorperTags::items_tags<Frame>;
  using compute_tags = typename StrahlkorperTags::compute_items_tags<Frame>;

  template <typename DbTags, typename Metavariables>
  static void initialize(const gsl::not_null<db::DataBox<DbTags>*> box,
                         const Parallel::GlobalCache<Metavariables>& cache) {
    const auto& sphere =
        Parallel::get<Tags::Sphere<InterpolationTargetTag>>(cache);
    const size_t l_max = sphere.l_max;
    // Make a spherical strahlkorper
    ::Strahlkorper<Frame> strahlkorper(
        l_max, l_max, DataVector{(l_max + 1) * (2 * l_max + 1), sphere.radius},
        sphere.center);
    Initialization::mutate_assign<simple_tags>(box, std::move(strahlkorper));
  }

  template <typename Metavariables, typename DbTags, typename TemporalId>
  static tnsr::I<DataVector, 3, Frame> points(
      const db::DataBox<DbTags>& box,
      const tmpl::type_<Metavariables>& /*meta*/,
      const TemporalId& /*temporal_id*/) {
    return db::get<StrahlkorperTags::CartesianCoords<Frame>>(box);
  }
  template <typename Metavariables, typename DbTags>
  static tnsr::I<DataVector, 3, Frame> points(
      const db::DataBox<DbTags>& box,
      const tmpl::type_<Metavariables>& /*meta*/) {
    return db::get<StrahlkorperTags::CartesianCoords<Frame>>(box);
  }
};

}  // namespace TargetPoints
}  // namespace intrp
