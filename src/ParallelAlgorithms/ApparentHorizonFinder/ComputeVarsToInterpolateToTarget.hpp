// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "Utilities/Gsl.hpp"

namespace ah {
/*!
 * \brief Compute the `ah::vars_to_interpolate_to_target` for a given
 * \p element_id from the `ah::source_vars` in that element.
 */
template <typename Fr>
void compute_vars_to_interpolate_to_target(
    gsl::not_null<Variables<ah::vars_to_interpolate_to_target<3, Fr>>*>
        target_vars,
    const tnsr::aa<DataVector, 3>& spacetime_metric,
    const tnsr::aa<DataVector, 3>& pi, const tnsr::iaa<DataVector, 3>& phi,
    const tnsr::ijaa<DataVector, 3>& deriv_phi,
    const LinkedMessageId<double>& time, const Domain<3>& domain,
    const Mesh<3>& mesh, const ElementId<3>& element_id,
    const domain::FunctionsOfTimeMap& functions_of_time);
}  // namespace ah
