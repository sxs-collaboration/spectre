// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "PointwiseFunctions/AnalyticData/ScalarTensor/ScalarField/Inverser.hpp"
#include "PointwiseFunctions/AnalyticData/ScalarTensor/ScalarField/SuperposedInverser.hpp"
#include "PointwiseFunctions/AnalyticData/ScalarTensor/ScalarField/Zero.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarTensor::AnalyticData::ScalarField {

/*!
 * \brief List of all the initial guesses for the scalar field.
 */
template <size_t Dim>
using all_initial_guesses =
    tmpl::list<Zero<Dim>, Inverser<Dim>, SuperposedInverser<Dim>>;
}  // namespace ScalarTensor::AnalyticData::ScalarField
