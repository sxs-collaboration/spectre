// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/Creators/AlignedLattice.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Cylinder.hpp"
#include "Domain/Creators/CylindricalBinaryCompactObject.hpp"
#include "Domain/Creators/Factory.hpp"
#include "Domain/Creators/FrustalCloak.hpp"
#include "Domain/Creators/RotatedBricks.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Utilities/TMPL.hpp"

namespace DomainCreators_detail {
template <>
struct domain_creators<3> {
  using type = tmpl::list<domain::creators::AlignedLattice<3>,
                          domain::creators::BinaryCompactObject,
                          domain::creators::Brick, domain::creators::Cylinder,
                          domain::creators::CylindricalBinaryCompactObject,
                          domain::creators::FrustalCloak,
                          domain::creators::RotatedBricks,
                          domain::creators::Shell, domain::creators::Sphere>;
};
}  // namespace DomainCreators_detail
