// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <deque>

#include "DataStructures/LinkedMessageId.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "Utilities/Gsl.hpp"

template <size_t VolumeDim>
class Domain;
class FastFlow;
namespace ylm {
template <typename Frame>
class Strahlkorper;
}  // namespace ylm

namespace ah {
/*!
 * \brief Compute the target points for the current (next) iteration.
 *
 * \details Returns whether the computation of the target points was successful
 * or if there are some points outside the domain. This function will try
 * reinterpolating \p max_interpolation_retries times before an error occurs.
 * For the zeroth fast flow iteration, this will increase the $l=0,m=0$
 * coefficient (i.e. the size) by 50%. For other iterations, the coefficients
 * become
 * \begin{equation}
 * S^{\mathrm{new}}_{lm} = \frac{1}{2}\left(S^{\mathrm{previous}}_{lm} +
 * S^{\mathrm{failed}}_{lm}\right)
 * \end{equation}
 * where $S^{\mathrm{failed}}_{lm}$ are the coefficients of the failed
 * interpolation and $S^{\mathrm{previous}}_{lm}$ are the coefficients from the
 * previous successful iteration.
 */
template <typename Fr>
bool set_current_iteration_coords(
    gsl::not_null<ah::Storage::Iteration<Fr>*> current_iteration,
    const LinkedMessageId<double>& time, const FastFlow& fast_flow,
    const ylm::Strahlkorper<Fr>& initial_guess,
    const ylm::Strahlkorper<Fr>& previous_iteration_surface,
    const std::deque<ah::Storage::PreviousSurface<Fr>>& previous_surfaces,
    size_t max_interpolation_retries, const Domain<3>& domain,
    const domain::FunctionsOfTimeMap& functions_of_time);
}  // namespace ah
