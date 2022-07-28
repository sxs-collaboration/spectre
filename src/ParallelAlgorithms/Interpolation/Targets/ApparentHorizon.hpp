// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "ApparentHorizons/FastFlow.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Variables.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/ComputeTargetPoints.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
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
namespace Tags {
struct Verbosity;
}  // namespace Tags
/// \endcond

namespace intrp {

namespace OptionHolders {
/// Options for finding an apparent horizon.
template <typename Frame>
struct ApparentHorizon {
  /// See Strahlkorper for suboptions.
  struct InitialGuess {
    static constexpr Options::String help = {"Initial guess"};
    using type = Strahlkorper<Frame>;
  };
  /// See ::FastFlow for suboptions.
  struct FastFlow {
    static constexpr Options::String help = {"FastFlow options"};
    using type = ::FastFlow;
  };
  struct Verbosity {
    static constexpr Options::String help = {"Verbosity"};
    using type = ::Verbosity;
  };
  using options = tmpl::list<InitialGuess, FastFlow, Verbosity>;
  static constexpr Options::String help = {
      "Provide an initial guess for the apparent horizon surface\n"
      "(Strahlkorper) and apparent-horizon-finding-algorithm (FastFlow)\n"
      "options."};

  ApparentHorizon(Strahlkorper<Frame> initial_guess_in, ::FastFlow fast_flow_in,
                  ::Verbosity verbosity_in);

  ApparentHorizon() = default;
  ApparentHorizon(const ApparentHorizon& /*rhs*/) = default;
  ApparentHorizon& operator=(const ApparentHorizon& /*rhs*/) = delete;
  ApparentHorizon(ApparentHorizon&& /*rhs*/) = default;
  ApparentHorizon& operator=(ApparentHorizon&& /*rhs*/) = default;
  ~ApparentHorizon() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  Strahlkorper<Frame> initial_guess{};
  ::FastFlow fast_flow{};
  ::Verbosity verbosity{::Verbosity::Quiet};
};

template <typename Frame>
bool operator==(const ApparentHorizon<Frame>& lhs,
                const ApparentHorizon<Frame>& rhs);
template <typename Frame>
bool operator!=(const ApparentHorizon<Frame>& lhs,
                const ApparentHorizon<Frame>& rhs);

}  // namespace OptionHolders

namespace OptionTags {
struct ApparentHorizons {
  static constexpr Options::String help{"Options for apparent horizon finders"};
};

template <typename InterpolationTargetTag, typename Frame>
struct ApparentHorizon {
  using type = OptionHolders::ApparentHorizon<Frame>;
  static constexpr Options::String help{
      "Options for interpolation onto apparent horizon."};
  static std::string name() {
    return pretty_type::name<InterpolationTargetTag>();
  }
  using group = ApparentHorizons;
};
}  // namespace OptionTags

namespace Tags {
template <typename InterpolationTargetTag, typename Frame>
struct ApparentHorizon : db::SimpleTag {
  using type = OptionHolders::ApparentHorizon<Frame>;
  using option_tags =
      tmpl::list<OptionTags::ApparentHorizon<InterpolationTargetTag, Frame>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& option) { return option; }
};
}  // namespace Tags

namespace TargetPoints {
/// \brief Computes points on a trial apparent horizon`.
///
/// This differs from `KerrHorizon` in the following ways:
/// - It supplies points on a prolonged Strahlkorper, at a higher resolution
///   than the Strahlkorper in the DataBox, as needed for horizon finding.
/// - It uses a `FastFlow` in the DataBox.
/// - It has different options (including those for `FastFlow`).
///
/// Conforms to the intrp::protocols::ComputeTargetPoints protocol
///
/// For requirements on InterpolationTargetTag, see
/// intrp::protocols::InterpolationTargetTag
template <typename InterpolationTargetTag, typename Frame>
struct ApparentHorizon : tt::ConformsTo<intrp::protocols::ComputeTargetPoints> {
  using const_global_cache_tags =
      tmpl::list<Tags::ApparentHorizon<InterpolationTargetTag, Frame>>;
  using is_sequential = std::true_type;
  using frame = Frame;

  using common_tags =
      tmpl::push_back<StrahlkorperTags::items_tags<Frame>, ::ah::Tags::FastFlow,
                      logging::Tags::Verbosity<InterpolationTargetTag>,
                      ::ah::Tags::PreviousStrahlkorpers<Frame>>;
  using simple_tags = tmpl::append<
      common_tags,
      tmpl::conditional_t<
          std::is_same_v<Frame, ::Frame::Inertial>, tmpl::list<>,
          tmpl::list<StrahlkorperTags::Strahlkorper<::Frame::Inertial>>>>;
  using compute_tags = typename StrahlkorperTags::compute_items_tags<Frame>;

  template <typename DbTags, typename Metavariables>
  static void initialize(const gsl::not_null<db::DataBox<DbTags>*> box,
                         const Parallel::GlobalCache<Metavariables>& cache) {
    const auto& options =
        Parallel::get<Tags::ApparentHorizon<InterpolationTargetTag, Frame>>(
            cache);

    // Put Strahlkorper and its ComputeItems, FastFlow, and verbosity
    // into a new DataBox.  The first element of PreviousStrahlkorpers
    // is initialized to (time=NaN, strahlkorper=options.initial_guess).
    // The NaN is a sentinel value which indicates that the
    // PreviousStrahlkorper has not been computed but is instead the
    // supplied initial guess.
    //
    // Note that if frame is not inertial,
    // StrahlkorperTags::Strahlkorper<::Frame::Inertial> is already
    // default initialized so there is no need to do anything special
    // here for StrahlkorperTags::Strahlkorper<::Frame::Inertial>.
    Initialization::mutate_assign<common_tags>(
        box, options.initial_guess, options.fast_flow, options.verbosity,
        std::deque<std::pair<double, ::Strahlkorper<Frame>>>{std::make_pair(
            std::numeric_limits<double>::signaling_NaN(),
            options.initial_guess)});
  }

  template <typename Metavariables, typename DbTags, typename TemporalId>
  static tnsr::I<DataVector, 3, Frame> points(
      const db::DataBox<DbTags>& box,
      const tmpl::type_<Metavariables>& /*meta*/,
      const TemporalId& /*temporal_id*/) {
    const auto& fast_flow = db::get<::ah::Tags::FastFlow>(box);
    const auto& strahlkorper =
        db::get<StrahlkorperTags::Strahlkorper<Frame>>(box);

    const size_t L_mesh = fast_flow.current_l_mesh(strahlkorper);
    const auto prolonged_strahlkorper =
        Strahlkorper<Frame>(L_mesh, L_mesh, strahlkorper);

    Variables<tmpl::list<::Tags::Tempi<0, 2, ::Frame::Spherical<Frame>>,
                         ::Tags::Tempi<1, 3, Frame>, ::Tags::TempScalar<2>>>
        temp_buffer(prolonged_strahlkorper.ylm_spherepack().physical_size());

    auto& theta_phi =
        get<::Tags::Tempi<0, 2, ::Frame::Spherical<Frame>>>(temp_buffer);
    auto& r_hat = get<::Tags::Tempi<1, 3, Frame>>(temp_buffer);
    auto& radius = get<::Tags::TempScalar<2>>(temp_buffer);
    StrahlkorperTags::ThetaPhiCompute<Frame>::function(
        make_not_null(&theta_phi), prolonged_strahlkorper);
    StrahlkorperTags::RhatCompute<Frame>::function(make_not_null(&r_hat),
                                                   theta_phi);
    StrahlkorperTags::RadiusCompute<Frame>::function(make_not_null(&radius),
                                                     prolonged_strahlkorper);

    tnsr::I<DataVector, 3, Frame> prolonged_coords{};
    StrahlkorperTags::CartesianCoordsCompute<Frame>::function(
        make_not_null(&prolonged_coords), prolonged_strahlkorper, radius,
        r_hat);

    return prolonged_coords;
  }
};

}  // namespace TargetPoints
}  // namespace intrp
