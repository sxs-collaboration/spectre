// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/Creators/AlignedLattice.hpp"
#include "Domain/Creators/AngularCylinder.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/CartoonCylinder.hpp"
#include "Domain/Creators/CartoonSphere1D.hpp"
#include "Domain/Creators/CartoonSphere2D.hpp"
#include "Domain/Creators/Cylinder.hpp"
#include "Domain/Creators/CylindricalBinaryCompactObject.hpp"
#include "Domain/Creators/Factory.hpp"
#include "Domain/Creators/FrustalCloak.hpp"
#include "Domain/Creators/NonconformingSphericalShells.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Creators/RotatedBricks.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/SphericalShells.hpp"
#include "Utilities/TMPL.hpp"

namespace DomainCreators_detail {
template <>
struct domain_creators<3> {
  using type =
      tmpl::list<domain::creators::AlignedLattice<3>,
                 domain::creators::AngularCylinder,
                 domain::creators::BinaryCompactObject<false>,
                 domain::creators::Brick, domain::creators::CartoonCylinder,
                 domain::creators::CartoonSphere1D,
                 domain::creators::CartoonSphere2D, domain::creators::Cylinder,
                 domain::creators::CylindricalBinaryCompactObject,
                 domain::creators::FrustalCloak,
                 domain::creators::NonconformingSphericalShells,
                 domain::creators::RotatedBricks, domain::creators::Sphere,
                 domain::creators::SphericalShells>;
};
}  // namespace DomainCreators_detail
