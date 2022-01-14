// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/CubicCrystal.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "Utilities/TMPL.hpp"

namespace Elasticity::ConstitutiveRelations {
template <size_t Dim>
using standard_constitutive_relations = tmpl::append<
    tmpl::list<IsotropicHomogeneous<Dim>>,
    tmpl::conditional_t<Dim == 3, tmpl::list<CubicCrystal>, tmpl::list<>>>;
}  // namespace Elasticity::ConstitutiveRelations
