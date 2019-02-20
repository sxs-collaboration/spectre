// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Interpolation/InterpolationTargetApparentHorizon.hpp"

#include <algorithm>

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace intrp {
namespace OptionHolders {
template <typename Frame>
ApparentHorizon<Frame>::ApparentHorizon(Strahlkorper<Frame> initial_guess_in,
                                        ::FastFlow fast_flow_in,
                                        Verbosity verbosity_in) noexcept
    : initial_guess(std::move(initial_guess_in)),
      fast_flow(std::move(fast_flow_in)),    // NOLINT
      verbosity(std::move(verbosity_in)) {}  // NOLINT
// clang-tidy std::move of trivially copyable type.

template <typename Frame>
void ApparentHorizon<Frame>::pup(PUP::er& p) noexcept {
  p | initial_guess;
  p | fast_flow;
  p | verbosity;
}

template <typename Frame>
bool operator==(const ApparentHorizon<Frame>& lhs,
                const ApparentHorizon<Frame>& rhs) noexcept {
  return lhs.initial_guess == rhs.initial_guess and
         lhs.fast_flow == rhs.fast_flow and lhs.verbosity == rhs.verbosity;
}

template <typename Frame>
bool operator!=(const ApparentHorizon<Frame>& lhs,
                const ApparentHorizon<Frame>& rhs) noexcept {
  return not(lhs == rhs);
}

// So far instantiate for only one frame. But once we
// have control systems working, we will need this for at
// least two frames.
template struct ApparentHorizon<Frame::Inertial>;
template bool operator==(const ApparentHorizon<Frame::Inertial>& lhs,
                         const ApparentHorizon<Frame::Inertial>& rhs) noexcept;
template bool operator!=(const ApparentHorizon<Frame::Inertial>& lhs,
                         const ApparentHorizon<Frame::Inertial>& rhs) noexcept;

}  // namespace OptionHolders
}  // namespace intrp
