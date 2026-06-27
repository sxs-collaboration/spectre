// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"

namespace evolution::Ringdown {

/*!
 * \brief This function finds a safe ringdown excision radius for starting
 * the ringdown of a common horizon from a binary inspiral. It does this by
 * finding a radius that will enclose every point on strahlkorpers that
 * represent excisions A/B from the inspiral at the match time. It does this by
 * taking excisions A/B from the inertial frame and transforming them to the
 * ringdown grid frame. It iterates over all the points from excisions A/B in
 * the ringdown-grid-frame until it finds the point that's the farthest from the
 * geometric center of the ringdown excision. This is now the minimum radius we
 * can choose to enclose the excisions A/B from the inspiral. This is not the
 * ideal radius however, we then choose a radius that's 3/4 of the way between
 * the radius of the common horizon at the match time and the minimum radius
 * that encloses excisions A/B from the inspiral and check for multiple l_max
 * values of excisions A/B to ensure they fit and the ringdown can start.

 * \details It does this by constructing inspiral-grid-frame AhA/AhB excision
 * strahlkorpers from the radii and centers passed to this function. It then
 * maps those excisions to the inertial-frame using the inspiral domain
 * and functions of time from the inspiral volume data and subfile supplied. We
 * then construct a test ringdown domain that has all the corrected functions of
 * time from ComputeRingdownShapeAndTranslationFoT.py and an initial guess for
 * the inner radius. More details are outlined below.
 * \param path_to_volume_data The full path to the volume data containing the
 * domain and functions of time
 * \param volume_subfile_name Subfile containing volume data output from the
 * inspiral
 * \param path_to_horizons_h5 Path to h5 file containing horizon data for AhA/B
 * \param surface_subfile_name Subfile containing horizon data for AhA/B
 * \param path_to_AhC_distorted_h5 Path to h5 file containing ringdown shape
 * coefficients computed using ComputeRingdownShapeAndTranslationFoT.py
 * \param AhC_distorted_subfile_names Subfiles in the h5 file containing
 * shape coefficients
 * \param match_time The time to match the functions of time
 * \param settling_timescale Timescale at which the functions of time settle to
 * constant values
 * \param excision_A_radius The radius of excision A from the inspiral grid
 * frame
 * \param excision_B_radius The radius of excision B from the inspiral grid
 * frame
 * \param excision_A_center The center of excision A from the inspiral grid
 * frame
 * \param excision_B_center The center of excision B from the inspiral grid
 * frame
 * \param exp_func_and_2_derivs Expansion function of time from the inspiral
 * \param exp_outer_bdry_func_and_2_derivs Outer boundary expansion function of
 * time from the inspiral
 * \param rot_func_and_2_derivs Rotation function of time from the inspiral
 * \param trans_func_and_2_derivs The corrected translation function of time
 * computed by ComputeRingdownShapeAndTranslationFoT.py
 * \param match_time_tol The difference allowed between the match time requested
 * and the time found in h5 files
 *
 * Using the corrected functions of time and our initial guess for the ringdown
 * inner radius (the excision radius), excisions A/B are transformed from the
 * inertial frame to the ringdown grid frame. The inner radius is iterated upon
 * using 2 main loops, the outer loop which changes the L_max of the excisions
 * A/B being transformed from the inspiral and the inner loop that changes the
 * excision radius used in the ringdown. The general flow for these loops is as
 * follows:
 *
 * Let $ahc\_average\_radius$ be the average radius of the common horizon in the
 * inertial frame and since have strahlkorper data from the inertial frame AhC
 * we set $ahc\_average\_radius$ so that lambda00=0 at the match time.
 *
 * Let $ringdown\_excision\_radius$ be the radius of (spherical) excision
 * boundary in ringdown grid frame.
 *
 * Let $ringdown\_excision\_factor$ be the factor we multiply the
 * $ahc\_average\_radius$ by to get the $ringdown\_excision\_radius$.
 *
 * The goal is to find a good $ringdown\_excision\_factor$
 *
 * 1) Choose an initial guess for the $ringdown\_excision\_factor$. We choose
 * this to be 0.94 by default.
 *
 * 2) Choose an initial guess for current_l_max, the angular resolution of
 * spherical Strahlkorpers that match excision A and B (not horizons) in the
 * inspiral grid frame. We choose current_l_max=20 intially.
 *
 * 3) Construct Strahlkorpers in the Inspiral grid frame that correspond to
 * excisions A/B, the excision regions (not the horizons) of the two individual
 * holes. These are spherical in the inspiral grid frame, and have
 * L=current_l_max.
 *
 * 4) Map the Strahlkorper excisions A/B to the inertial frame. Now they are not
 * spherical.
 *
 * 4a) Using the current $ringdown\_excision\_factor$, construct a test domain
 * to map excisions A/B from the inertial frame to the ringdown grid frame. This
 * domain's functions of time should contain the scaling and rotation functions
 * of time from the inspiral that settle to const and the shape and corrected
 * translation function of time output by
 * ComputeRingdownShapeAndTranslationFoT.py
 *
 * 4b) Map the Strahlkorper excisions A/B to the Ringdown grid frame. They are
 * still not spherical.
 *
 * 4c) Loop through all the points on Ringdown grid excisions A/B, and find the
 * point that is the maximum distance from the origin. Call that maximum
 * distance $min\_excision\_radius$.
 *
 * 4d) A Ringdown grid sphere of radius $min\_excision\_radius$ centered at the
 * origin will (barely) enclose both excisions A/B, so choose
 * $minimum\_ringdown\_excision\_factor = \frac{min\_excision\_radius}
 * {ahc\_average\_radius}$ as the minimum possible $ringdown\_excision\_factor$.
 * And then choose the new value of $ringdown\_excision\_factor$ to be
 * \f{equation}{
 * ringdown\_excision\_factor=1-0.25*(1-minimum\_ringdown\_excision\_factor)
 * \f}
 * Which puts the excision radius 3/4 of the way between the
 * $ahc\_average\_radius$ and the $min\_excision\_radius$
 *
 * 4e) If this is the first inner iteration, or if
 * $|ringdown\_excision\_factor-previous\_ringdown\_excision\_factor|>eps$, then
 * set $previous\_ringdown\_excision\_factor=ringdown\_excision\_factor$ and
 * goto 3a, where 3a is repeated using the new $ringdown\_excision\_factor$. We
 * choose $eps = \frac{10^{-3}}{q}$ where q is an approximation of the mass
 * ratio. The reason this is iterative is because the new
 * $ringdown\_excision\_factor$ in 3d depends on the ringdown_excision_factor
 * used when constructing the test domain.
 *
 * 5) If we get here, then inner iteration has converged.  But that iteration
 * depends on the $current\_l\_max$ used to construct excisions A/B in step 2.
 * If this is the first outer iteration, or if
 * $|ringdown\_excision\_factor-previous\_ringdown\_excision\_factor|>eps$, then
 * set $previous\_ringdown\_excision\_factor=ringdown\_excision\_factor$, and
 * increment $current\_l\_max$ by some increment (which we choose to be 6), and
 * go back to 2.
 *
 * 6) If we get here, the outer iteration is converged and now we have a final
 * $ringdown\_excision\_factor$.
 *
 * \note This implementation does not do everything SpEC does yet, it is
 * currently missing rescaling the shape coefficients held in
 * 'path_to_AhC_distorted_h5' by $excision\_radius / average\_ahc\_radius$ every
 * time it constructs a test ringdown domain. This step helps the shape of the
 * excision match the shape of the apparent horizon.
 */
double minimum_ahc_excision_radius(
    const std::string& path_to_volume_data,
    const std::string& volume_subfile_name,
    const std::string& path_to_horizons_h5,
    const std::string& surface_subfile_name,
    const std::string& path_to_AhC_distorted_h5,
    const std::vector<std::string>& AhC_distorted_subfile_names,
    double match_time, double settling_timescale, double excision_A_radius,
    double excision_B_radius, std::array<double, 3> excision_A_center,
    std::array<double, 3> excision_B_center,
    const std::optional<std::array<double, 3>>& exp_func_and_2_derivs =
        std::nullopt,
    const std::optional<std::array<double, 3>>&
        exp_outer_bdry_func_and_2_derivs = std::nullopt,
    const std::optional<std::vector<std::array<double, 4>>>&
        rot_func_and_2_derivs = std::nullopt,
    const std::optional<std::array<std::array<double, 3>, 3>>&
        trans_func_and_2_derivs = std::nullopt,
    double match_time_tol = 1e-12);
}  // namespace evolution::Ringdown
