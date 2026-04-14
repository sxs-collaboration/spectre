// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"

/*!
 * \brief Functionality for evolving a ringdown following a compact-binary
 * merger.
 */
namespace evolution::Ringdown {
/*!
 * \brief This function is used to transition from inspiral to ringdown. It
 * reads inertial frame common horizon Strahlkorper coefs from a file and
 * returns the Strahlkorper's temporary frame coefs and the inertial frame
 * geometric centers at multiple times which will be used to initialize the
 * shape and translation function of time for the ringdown.
 *
 * \details Reads common horizon Strahlkorpers (assumed to be in the inertial
 * frame) from a file, then transforms them to a temporary ringdown domain
 * defined by the expansion, rotation, and translation maps from the inspiral
 * specified by `exp_func_and_2_derivs`, `exp_outer_bdry_func_and_2_derivs`,
 * `rot_func_and_2_derivs`, and `trans_func_and_2_derivs`. The expansion and
 * rotation functions of time from the inspiral are the same as the ringdown
 * frame's expansion and rotation maps at the given `match_time`, but later
 * settle to constant values by the given `settling_timescale`. The translation
 * function of time supplied from the inspiral is not the translation map we'll
 * use in the final ringdown, it is only used to correctly map the common
 * horizon's geometric center so that we can make the corrected translation map
 * for the final ringdown. How to correct the translation map is explained in
 * detail below. A shape map is not specified, because we do not yet know the
 * shape coefficients of the common horizon for the ringdown. There are 4 frames
 * we transition between to find the shape coefficients and correct the
 * translation map. The frames are as follows:
 *
 * $\textbf{Inertial frame}$: This is the same for ringdown and inspiral, by
 * definition. It's the frame where we measure the waveforms, and it's the frame
 * in which we define the tensor components (e.g. the 'x' and 'y' in 'g_xy').
 *
 * $\textbf{Ringdown Grid Frame}$: This is the frame where the ringdown excision
 * boundary is spherical and where the ringdown grid lives.
 *
 * $\textbf{Temporary frame}$: The inertial frame pushed through the inverse
 * translation-rotation-expansion map, where "translation" is the inspiral's
 * translation map, and "rotation" and "expansion" are new rotation and
 * expansion maps that match the inspiral's rotation and expansion at the outer
 * boundary at the match time, have some smooth falloff in the interior, and
 * have a "settle to constant" time dependence. This frame is almost the
 * ringdown distorted frame, except for an uncorrected translation map which is
 * explained in detail below.
 *
 * $\textbf{Ringdown distorted frame}$: This is the ringdown grid frame pushed
 * through the (final) forward distortion map. Or equivalently, this is the
 * inertial frame pushed through the inverse translation-rotation-expansion map,
 * where here "translation" means the final corrected translation map, not the
 * inspiral's translation map, and "rotation" and "expansion" are the same as in
 * the temporary frame.
 *
 * We get the shape coefficients by transforming the common horizon to the
 * temporary frame which is almost the ringdown distorted frame except for the
 * uncorrected translation map. The translation map is corrected by transforming
 * the common horizon from the temporary frame to the inertial frame and saving
 * the geometric center points at multiple times. We then take those geometric
 * center points and build the corrected translation function of time. This is
 * done because the center of the Strahlkorper in this temporary frame is NOT
 * the origin (this is the case whether or not you say Recenter=true when
 * transforming the Strahlkorper), but the distortion map does its distortion
 * about the origin.
 * Possible ways to account for this:
 *
 * 1) Put a translation map before the distortion map.
 *
 * 2) Change the center of the distortion map.  (then we need to live with
 * this during the ringdown).
 *
 * 3) Correct the current translation map so that the excision boundary maps to
 * the correct place.
 *
 * We choose 3). Section 6 of \cite Hemberger2012jz explains in more
 * detail of how we initialize the shape/translation map, but idea is that in
 * Eq. (104), the horizon can be written as
 * \f{equation}{
 * x^{\bar{i}}_{AH} = x^{\bar{i}}_{AHc} + n^{\bar{i}} \sum_{lm}S_{lm}
 * Y_{lm}(\theta,\phi)
 * \f}
 * where $x^{\bar{i}}_{AH}$ is the common horizon Strahklorper in the temporary
 * frame, $x^{\bar{i}}_{AHc}$ is the expansion center of the Strahlkorper in the
 * temporary frame, which is time-dependent and not zero, $n^{\bar{i}}$ is the
 * direction unit vector in the ($\theta,\phi$) direction relative to
 * $x^{\bar{i}}_{AHc}$, $\bar{i}$ is the index corresponding to the
 * temporary frame, and $S_{lm}$ are the Strahlkorper coefficients.
 * Now,
 * \f{equation}{
 * x^i = T0^i + M^i_{\bar{i}} x^{\bar{i}}
 * \f}
 * where $T0^i$ is the inspiral translation map, and $M^i_{\bar{i}}$ is
 * scaling+rotation. Thus,
 * \f{equation}{
 * x^i_AH = T0^i + M^i_{\bar{i}} x^{\bar{i}}_{AH}
 *        = T0^i + M^i_{\bar{i}} x^{\bar{i}}_{AHc} + M^i_{\bar{i}} n^{\bar{i}}
 *          \sum_{lm} S_{lm} Y_{lm}
 * \f}
 * Therefore if you define a new translation map $T^i$ as in Eq. (107)
 * (corrected translation map that will be used in the final ringdown)
 * $T^i = T0^i + M^i_{\bar{i}} x^{\bar{i}}_{AHc}$ (that is, you define $T^i$ to
 * be the same as $x^i_{AHc}$), then you can rewrite the relationship as
 * \f{equation}{
 * x^{i}_{AH} = T^i + M^i_{\bar{i}} \sum_{lm} S_{lm} Y_{lm} n^{\bar{i}}
 * \f}
 * Therefore we use a new map $x^i = T^i + M^i_{\bar{i}} x^{\tilde{i}}$ where
 * $\tilde{i}$ refers to the final ringdown distorted frame, $x^{\tilde{i}}$ is
 * a new coordinate where $x^{\tilde{i}}_{AHc} = 0$ and the coefficients of the
 * AH in the $x^{\tilde{i}}$ frame can be used unchanged (except for a minus
 * sign) in the distortion map that connects $x^{i_{grid}}$ and $x^{\tilde{i}}$.
 * This new map for $x^i$ ensures that the excision boundary is mapped to the
 * correct position.
 * \note Only temporary frame common horizon coefs and inertial frame geometric
 * center points within `requested_number_of_times_from_end` times from the
 * final time are returned. The functions of time are computed by
 * ComputeRingdownShapeAndTranslationFoT.py using the output from this function
 */
std::pair<std::vector<DataVector>, std::vector<std::array<double, 3>>>
strahlkorper_coefs_and_centers(
    const std::string& path_to_volume_data,
    const std::string& volume_subfile_name,
    const std::string& path_to_horizons_h5,
    const std::string& surface_subfile_name,
    size_t requested_number_of_times_from_end, double match_time,
    double settling_timescale,
    const std::optional<std::array<double, 3>>& exp_func_and_2_derivs =
        std::nullopt,
    const std::optional<std::array<double, 3>>&
        exp_outer_bdry_func_and_2_derivs = std::nullopt,
    const std::optional<std::vector<std::array<double, 4>>>&
        rot_func_and_2_derivs = std::nullopt,
    const std::optional<std::array<std::array<double, 3>, 3>>&
        trans_func_and_2_derivs = std::nullopt);

// Hardcoded value used in SpEC
constexpr double expansion_center_tolerance = 1e-8;
}  // namespace evolution::Ringdown
