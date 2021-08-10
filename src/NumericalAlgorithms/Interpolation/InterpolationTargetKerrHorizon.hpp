// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "ApparentHorizons/YlmSpherepack.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Transpose.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrHorizon.hpp"
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
/// contains the `Lmax` which encodes the angular resolution of the spherical
/// harmonic basis and `ThetaVariesFastest` which encodes the collocation
/// ordering. For example, the apparent horizon finder relies on spherepack
/// routines that require `true` for `ThetaVariesFastest`, and using this
/// surface for a CCE worldtube requires `false` for `ThetaVariesFastest`.
struct KerrHorizon {
  struct Lmax {
    using type = size_t;
    static constexpr Options::String help = {
        "KerrHorizon is expanded in Ylms up to l=Lmax"};
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
  struct ThetaVariesFastest {
    using type = bool;
    static constexpr Options::String help = {
        "Chooses theta,phi ordering in 2d array"};
  };
  using options =
      tmpl::list<Lmax, Center, Mass, DimensionlessSpin, ThetaVariesFastest>;
  static constexpr Options::String help = {
      "A Strahlkorper conforming to the horizon (in Kerr-Schild coordinates)"
      " of a Kerr black hole with a specified center, mass, and spin."};

  KerrHorizon(size_t l_max_in, std::array<double, 3> center_in, double mass_in,
              std::array<double, 3> dimensionless_spin_in,
              bool theta_varies_fastest_in = true,
              const Options::Context& context = {});

  KerrHorizon() = default;
  KerrHorizon(const KerrHorizon& /*rhs*/) = default;
  KerrHorizon& operator=(const KerrHorizon& /*rhs*/) = delete;
  KerrHorizon(KerrHorizon&& /*rhs*/) noexcept = default;
  KerrHorizon& operator=(KerrHorizon&& /*rhs*/) noexcept = default;
  ~KerrHorizon() = default;

  // clang-tidy non-const reference pointer.
  void pup(PUP::er& p) noexcept;  // NOLINT

  size_t l_max{};
  std::array<double, 3> center{};
  double mass{};
  std::array<double, 3> dimensionless_spin{};
  bool theta_varies_fastest_memory_layout{};
};

bool operator==(const KerrHorizon& lhs, const KerrHorizon& rhs) noexcept;
bool operator!=(const KerrHorizon& lhs, const KerrHorizon& rhs) noexcept;

}  // namespace OptionHolders

namespace OptionTags {
template <typename InterpolationTargetTag>
struct KerrHorizon {
  using type = OptionHolders::KerrHorizon;
  static constexpr Options::String help{
      "Options for interpolation onto Kerr horizon."};
  static std::string name() noexcept {
    return Options::name<InterpolationTargetTag>();
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
  static type create_from_options(const type& option) noexcept {
    return option;
  }
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
/// Lmax, where Lmax is the spherical-harmonic order specified in the
/// options.  As Lmax increases, the returned points converge to the
/// horizon.
///
/// For requirements on InterpolationTargetTag, see InterpolationTarget
template <typename InterpolationTargetTag, typename Frame>
struct KerrHorizon {
  using const_global_cache_tags =
      tmpl::list<Tags::KerrHorizon<InterpolationTargetTag>>;
  using initialization_tags =
      tmpl::append<StrahlkorperTags::items_tags<Frame>,
                   StrahlkorperTags::compute_items_tags<Frame>>;
  using is_sequential = std::false_type;
  using frame = Frame;

  using simple_tags = typename StrahlkorperTags::items_tags<Frame>;
  using compute_tags = typename StrahlkorperTags::compute_items_tags<Frame>;

  template <typename DbTags, typename Metavariables>
  static void initialize(
      const gsl::not_null<db::DataBox<DbTags>*> box,
      const Parallel::GlobalCache<Metavariables>& cache) noexcept {
    const auto& kerr_horizon =
        Parallel::get<Tags::KerrHorizon<InterpolationTargetTag>>(cache);

    // Make a Strahlkorper with the correct shape.
    ::Strahlkorper<Frame> strahlkorper(
        kerr_horizon.l_max, kerr_horizon.l_max,
        get(gr::Solutions::kerr_horizon_radius(
            ::YlmSpherepack(kerr_horizon.l_max, kerr_horizon.l_max)
                .theta_phi_points(),
            kerr_horizon.mass, kerr_horizon.dimensionless_spin)),
        kerr_horizon.center);
    Initialization::mutate_assign<simple_tags>(box, std::move(strahlkorper));
  }

  template <typename Metavariables, typename DbTags, typename TemporalId>
  static tnsr::I<DataVector, 3, Frame> points(
      const db::DataBox<DbTags>& box,
      const tmpl::type_<Metavariables>& metavariables_v,
      const TemporalId& /*temporal_id*/) noexcept {
    return points(box, metavariables_v);
  }
  template <typename Metavariables, typename DbTags>
  static tnsr::I<DataVector, 3, Frame> points(
      const db::DataBox<DbTags>& box,
      const tmpl::type_<Metavariables>& /*meta*/) noexcept {
    const auto& kerr_horizon =
        db::get<Tags::KerrHorizon<InterpolationTargetTag>>(box);
    if (kerr_horizon.theta_varies_fastest_memory_layout) {
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
