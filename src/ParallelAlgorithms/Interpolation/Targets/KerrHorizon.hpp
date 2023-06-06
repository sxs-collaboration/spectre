// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Transpose.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/ComputeTargetPoints.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/AngularOrdering.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrHorizon.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

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
/// A surface that conforms to the horizon of a Kerr black hole in
/// Kerr-Schild coordinates.
///
/// \details In addition to the parameters for the Kerr black hole, this holder
/// contains the `LMax` which encodes the angular resolution of the spherical
/// harmonic basis and `AngularOrdering` which encodes the collocation
/// ordering.
struct KerrHorizon {
  struct LMax {
    using type = size_t;
    static constexpr Options::String help = {
        "KerrHorizon is expanded in Ylms up to l=LMax"};
  };
  struct Center {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {"Center of black hole"};
  };
  struct Mass {
    using type = double;
    static constexpr Options::String help = {"Mass of black hole"};
  };
  struct DimensionlessSpin {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Dimensionless spin of black hole"};
  };
  struct AngularOrdering {
    using type = intrp::AngularOrdering;
    static constexpr Options::String help = {
        "Chooses theta,phi ordering in 2d array"};
  };
  using options =
      tmpl::list<LMax, Center, Mass, DimensionlessSpin, AngularOrdering>;
  static constexpr Options::String help = {
      "A Strahlkorper conforming to the horizon (in Kerr-Schild coordinates)"
      " of a Kerr black hole with a specified center, mass, and spin."};

  KerrHorizon(size_t l_max_in, std::array<double, 3> center_in, double mass_in,
              std::array<double, 3> dimensionless_spin_in,
              intrp::AngularOrdering angular_ordering_in,
              const Options::Context& context = {});

  KerrHorizon() = default;
  KerrHorizon(const KerrHorizon& /*rhs*/) = default;
  KerrHorizon& operator=(const KerrHorizon& /*rhs*/) = delete;
  KerrHorizon(KerrHorizon&& /*rhs*/) = default;
  KerrHorizon& operator=(KerrHorizon&& /*rhs*/) = default;
  ~KerrHorizon() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  size_t l_max{};
  std::array<double, 3> center{};
  double mass{};
  std::array<double, 3> dimensionless_spin{};
  intrp::AngularOrdering angular_ordering;
};

bool operator==(const KerrHorizon& lhs, const KerrHorizon& rhs);
bool operator!=(const KerrHorizon& lhs, const KerrHorizon& rhs);

}  // namespace OptionHolders

namespace OptionTags {
template <typename InterpolationTargetTag>
struct KerrHorizon {
  using type = OptionHolders::KerrHorizon;
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
struct KerrHorizon : db::SimpleTag {
  using type = OptionHolders::KerrHorizon;
  using option_tags =
      tmpl::list<OptionTags::KerrHorizon<InterpolationTargetTag>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& option) { return option; }
};
}  // namespace Tags

namespace TargetPoints {
/// \brief Computes points on a Kerr horizon.
///
/// The points are such that they conform to the horizon of a Kerr
/// black hole (in Kerr-Schild coordinates) with given center, mass,
/// and dimensionless spin, as specified in the options.
///
/// \note The returned points are not actually on the horizon;
/// instead, they are collocation points of a Strahlkorper whose
/// spectral representation matches the horizon shape up to order
/// LMax, where LMax is the spherical-harmonic order specified in the
/// options.  As LMax increases, the returned points converge to the
/// horizon.
///
/// Conforms to the intrp::protocols::ComputeTargetPoints protocol
///
/// For requirements on InterpolationTargetTag, see
/// intrp::protocols::InterpolationTargetTag
template <typename InterpolationTargetTag, typename Frame>
struct KerrHorizon : tt::ConformsTo<intrp::protocols::ComputeTargetPoints> {
  using const_global_cache_tags =
      tmpl::list<Tags::KerrHorizon<InterpolationTargetTag>>;
  using is_sequential = std::false_type;
  using frame = Frame;

  using simple_tags = typename StrahlkorperTags::items_tags<Frame>;
  using compute_tags = typename StrahlkorperTags::compute_items_tags<Frame>;

  template <typename DbTags, typename Metavariables>
  static void initialize(const gsl::not_null<db::DataBox<DbTags>*> box,
                         const Parallel::GlobalCache<Metavariables>& cache) {
    const auto& kerr_horizon =
        Parallel::get<Tags::KerrHorizon<InterpolationTargetTag>>(cache);

    // Make a Strahlkorper with the correct shape.
    ::Strahlkorper<Frame> strahlkorper(
        kerr_horizon.l_max, kerr_horizon.l_max,
        get(gr::Solutions::kerr_horizon_radius(
            ::ylm::Spherepack(kerr_horizon.l_max, kerr_horizon.l_max)
                .theta_phi_points(),
            kerr_horizon.mass, kerr_horizon.dimensionless_spin)),
        kerr_horizon.center);
    Initialization::mutate_assign<simple_tags>(box, std::move(strahlkorper));
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
    const auto& kerr_horizon =
        db::get<Tags::KerrHorizon<InterpolationTargetTag>>(box);
    if (kerr_horizon.angular_ordering == intrp::AngularOrdering::Strahlkorper) {
      return db::get<StrahlkorperTags::CartesianCoords<Frame>>(box);
    } else {
      const auto& strahlkorper =
          db::get<StrahlkorperTags::Strahlkorper<Frame>>(box);
      const auto& coords =
          db::get<StrahlkorperTags::CartesianCoords<Frame>>(box);
      const auto physical_extents =
          strahlkorper.ylm_spherepack().physical_extents();
      auto transposed_coords =
          tnsr::I<DataVector, 3, Frame>(get<0>(coords).size());
      for (size_t i = 0; i < 3; ++i) {
        transpose(make_not_null(&transposed_coords.get(i)), coords.get(i),
                  physical_extents[0], physical_extents[1]);
      }
      return transposed_coords;
    }
  }
};

}  // namespace TargetPoints
}  // namespace intrp
