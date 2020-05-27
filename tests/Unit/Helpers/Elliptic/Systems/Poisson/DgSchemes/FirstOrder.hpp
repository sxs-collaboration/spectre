// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Matrix.hpp"

/// \cond
template <size_t Dim>
struct DomainCreator;
/// \endcond

namespace TestHelpers::Poisson::dg {

template <size_t Dim>
Matrix first_order_operator_matrix(const DomainCreator<Dim>& domain_creator,
                                   double penalty_parameter);

}  // namespace TestHelpers::Poisson::dg
