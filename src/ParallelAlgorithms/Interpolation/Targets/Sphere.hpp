// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <set>
#include <variant>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Transpose.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/ComputeTargetPoints.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/AngularOrdering.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"

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
/// A series of concentric spherical surfaces.
///
/// \details The parameter `LMax` sets the number of collocation points on
/// each spherical surface equal to `(l_max + 1) * (2 * l_max + 1)`. The
/// parameter `AngularOrdering` encodes the collocation ordering. For example,
/// the apparent horizon finder relies on spherepack routines that require
/// `Strahlkorper` for `AngularOrdering`, and using these surfaces for a CCE
/// worldtube requires `Cce` for `AngularOrdering`.
struct Sphere {
  struct LMax {
    using type = size_t;
    static constexpr Options::String help = {
        "The number of collocation points on each sphere will be equal to "
        "`(l_max + 1) * (2 * l_max + 1)`"};
  };
  struct Center {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {"Center of every sphere"};
  };
  struct Radius {
    using type = std::variant<double, std::vector<double>>;
    static constexpr Options::String help = {"Radius of the sphere(s)"};
  };
  struct AngularOrdering {
    using type = intrp::AngularOrdering;
    static constexpr Options::String help = {
        "Chooses theta,phi ordering in 2d array"};
  };
  using options = tmpl::list<LMax, Center, Radius, AngularOrdering>;
  static constexpr Options::String help = {
      "An arbitrary number of spherical surface."};
  Sphere(const size_t l_max_in, const std::array<double, 3> center_in,
         const typename Radius::type& radius_in,
         intrp::AngularOrdering angular_ordering_in,
         const Options::Context& context = {});

  Sphere() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  size_t l_max{0};
  std::array<double, 3> center{std::numeric_limits<double>::signaling_NaN()};
  std::set<double> radii;
  intrp::AngularOrdering angular_ordering;
};

bool operator==(const Sphere& lhs, const Sphere& rhs);
bool operator!=(const Sphere& lhs, const Sphere& rhs);

}  // namespace OptionHolders

namespace OptionTags {
template <typename InterpolationTargetTag>
struct Sphere {
  using type = OptionHolders::Sphere;
  static constexpr Options::String help{
      "Options for interpolation onto a sphere(s)."};
  static std::string name() {
    return pretty_type::name<InterpolationTargetTag>();
  }
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

template <typename Frame>
struct AllCoords : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Frame>;
};
}  // namespace Tags

namespace TargetPoints {
/// \brief Computes points on spherical surfaces.
///
/// Conforms to the intrp::protocols::ComputeTargetPoints protocol
///
/// For requirements on InterpolationTargetTag, see
/// intrp::protocols::InterpolationTargetTag
template <typename InterpolationTargetTag, typename Frame>
struct Sphere : tt::ConformsTo<intrp::protocols::ComputeTargetPoints> {
  using const_global_cache_tags =
      tmpl::list<Tags::Sphere<InterpolationTargetTag>>;
  using is_sequential = std::false_type;
  using frame = Frame;

  using simple_tags =
      tmpl::list<StrahlkorperTags::Strahlkorper<Frame>, Tags::AllCoords<Frame>>;
  using compute_tags = typename StrahlkorperTags::compute_items_tags<Frame>;

  template <typename DbTags, typename Metavariables>
  static void initialize(const gsl::not_null<db::DataBox<DbTags>*> box,
                         const Parallel::GlobalCache<Metavariables>& cache) {
    const auto& sphere =
        Parallel::get<Tags::Sphere<InterpolationTargetTag>>(cache);
    const size_t l_max = sphere.l_max;
    const auto& radii = sphere.radii;

    // Total number of points is number of points for one sphere times the
    // number of spheres we use.
    const size_t num_points = radii.size() * (l_max + 1) * (2 * l_max + 1);

    tnsr::I<DataVector, 3, Frame> all_coords{num_points};

    size_t index = 0;
    for (const double radius : radii) {
      ::Strahlkorper<Frame> strahlkorper(
          l_max, l_max, DataVector{(l_max + 1) * (2 * l_max + 1), radius},
          sphere.center);

      db::mutate<StrahlkorperTags::Strahlkorper<Frame>>(
          [&strahlkorper](
              const gsl::not_null<::Strahlkorper<Frame>*> local_strahlkorper) {
            *local_strahlkorper = std::move(strahlkorper);
          },
          box);

      // This copy is ok because it's just in initialization
      auto coords = db::get<StrahlkorperTags::CartesianCoords<Frame>>(*box);

      // If the angular ordering is Strahlkorper then we don't have to do
      // anything to the coords because they are already in the right order
      if (sphere.angular_ordering == intrp::AngularOrdering::Cce) {
        const auto physical_extents =
            strahlkorper.ylm_spherepack().physical_extents();
        auto transposed_coords =
            tnsr::I<DataVector, 3, Frame>(get<0>(coords).size());

        for (size_t i = 0; i < 3; ++i) {
          transpose(make_not_null(&transposed_coords.get(i)), coords.get(i),
                    physical_extents[0], physical_extents[1]);
        }
        coords = std::move(transposed_coords);
      }

      const size_t tmp_index = index;
      for (size_t i = 0; i < 3; ++i) {
        for (size_t local_index = 0; local_index < coords.get(i).size();
             local_index++, index++) {
          all_coords.get(i)[index] = coords.get(i)[local_index];
        }
        if (i != 2) {
          index = tmp_index;
        }
      }
    }

    // If this fails, there is a bug. Can't really test it
    // LCOV_EXCL_START
    ASSERT(index == all_coords.get(0).size(),
           "Didn't initialize points of Sphere target correctly. index = "
               << index << " all_coords.size() = " << all_coords.get(0).size());
    // LCOV_EXCL_STOP

    Initialization::mutate_assign<tmpl::list<Tags::AllCoords<Frame>>>(
        box, std::move(all_coords));
  }

  template <typename Metavariables, typename DbTags, typename TemporalId>
  static tnsr::I<DataVector, 3, Frame> points(
      const db::DataBox<DbTags>& box,
      const tmpl::type_<Metavariables>& metavariables_v,
      const TemporalId& /*temporal_id*/) {
    return points(box, metavariables_v);
  }
  template <typename Metavariables, typename DbTags>
  static tnsr::I<DataVector, 3, Frame> points(
      const db::DataBox<DbTags>& box,
      const tmpl::type_<Metavariables>& /*meta*/) {
    return db::get<Tags::AllCoords<Frame>>(box);
  }
};
}  // namespace TargetPoints
}  // namespace intrp
