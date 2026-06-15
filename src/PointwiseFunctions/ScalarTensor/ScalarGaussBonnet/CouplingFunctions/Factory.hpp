// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet/CouplingFunctions/Exponential.hpp"
#include "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet/CouplingFunctions/QuarticPolynomial.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarTensor::sgb::CouplingFunctions {
/*!
 * \brief Typelist of all implemented coupling functions for
 * Einstein-scalar-Gauss-Bonnet gravity.
 */
using all_coupling_functions = tmpl::list<Exponential, QuarticPolynomial>;
}  // namespace ScalarTensor::sgb::CouplingFunctions
