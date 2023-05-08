// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

/// \cond
template <typename Frame>
class Strahlkorper;
template <size_t Dim>
class Domain;
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

/// \brief Transforms a Strahlkorper from SrcFrame to DestFrame.
///
/// The destination Strahlkorper has the same l_max() and m_max() as the
/// source Strahlkorper.
///
/// \note Because the Blocks inside the Domain allow access to
/// maps only between a selected subset of frames, we cannot use
/// strahlkorper_in_different_frame to map between arbitrary frames;
/// allowing strahlkorper_in_different_frame to work on more frames
/// requires adding member functions to Block.
///
/// \note strahlkorper_in_different_frame is currently instantiated
/// only for SrcFrame=Grid and DestFrame=Inertial.  In particular, we
/// intentionally do not instantiate strahlkorper_in_different_frame
/// for SrcFrame=Grid and DestFrame=Distorted, because our
/// Grid->Distorted maps preserve centers and angles, so for
/// Grid->Distorted one should use
/// strahlkorper_in_different_frame_aligned because it is more
/// efficient.
template <typename SrcFrame, typename DestFrame>
void strahlkorper_in_different_frame(
    gsl::not_null<Strahlkorper<DestFrame>*> dest_strahlkorper,
    const Strahlkorper<SrcFrame>& src_strahlkorper, const Domain<3>& domain,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    double time);

/// \brief Transforms a Strahlkorper from SrcFrame to DestFrame, for easy maps.
///
/// This is a simplified version of strahlkorper_in_different_frame
/// with the following two assumptions: First, the map leaves the
/// center of the Strahlkorper invariant. Second, for every point on
/// the Strahlkorper, the map leaves the angles about the center
/// (i.e. the quantity
/// \f$(x^i-c^i)/\sqrt{(x^0-c^0)^2+(x^1-c^1)^2+(x^2-c^2)^2}\f$)
/// invariant. If those assumptions are not satisfied, then
/// strahlkorper_in_different_frame_aligned will give the wrong answer
/// and you must use strahlkorper_in_different_frame.  Those
/// assumptions are satisfied for the shape and size maps but not for
/// general maps.
///
/// \note strahlkorper_in_different_frame_aligned is instantiated only
/// for SrcFrame=Grid and DestFrame=Distorted or Inertial. This
/// doesn't mean that strahlkorper_in_different_frame_aligned is
/// always correct for Grid->Inertial; the correctness depends on
/// whether the Grid->Inertial map satisfies the above assumptions,
/// which it might for simple cases (like single BHs) but it won't for
/// more complicated cases (like BBHs).  To add different frames to
/// strahlkorper_in_different_frame_aligned, one must add `if
/// constexpr`s to call the appropriate member functions of Block.
template <typename SrcFrame, typename DestFrame>
void strahlkorper_in_different_frame_aligned(
    gsl::not_null<Strahlkorper<DestFrame>*> dest_strahlkorper,
    const Strahlkorper<SrcFrame>& src_strahlkorper, const Domain<3>& domain,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    double time);
