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
/// Note that because the Blocks inside the Domain allow access to
/// maps only between a selected subset of frames, we cannot use
/// strahlkorper_in_different_frame to map between arbitrary frames;
/// allowing strahlkorper_in_different_frame to work on more frames
/// requires adding member functions to Block.
template <typename SrcFrame, typename DestFrame>
void strahlkorper_in_different_frame(
    gsl::not_null<Strahlkorper<DestFrame>*> dest_strahlkorper,
    const Strahlkorper<SrcFrame>& src_strahlkorper, const Domain<3>& domain,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    double time);
