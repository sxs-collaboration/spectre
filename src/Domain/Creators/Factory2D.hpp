// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/Creators/AlignedLattice.hpp"
#include "Domain/Creators/Disk.hpp"
#include "Domain/Creators/Factory.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Creators/RotatedRectangles.hpp"
#include "Utilities/TMPL.hpp"

namespace DomainCreators_detail {
template <>
struct domain_creators<2> {
  using type = tmpl::list<domain::creators::AlignedLattice<2>,
                          domain::creators::Disk, domain::creators::Rectangle,
                          domain::creators::RotatedRectangles>;
};
}  // namespace DomainCreators_detail
