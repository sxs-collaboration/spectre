// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/Creators/AlignedLattice.hpp"
#include "Domain/Creators/Factory.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/RotatedIntervals.hpp"
#include "Utilities/TMPL.hpp"

namespace DomainCreators_detail {
template <>
struct domain_creators<1> {
  using type = tmpl::list<domain::creators::AlignedLattice<1>,
                          domain::creators::Interval,
                          domain::creators::RotatedIntervals>;
};
}  // namespace DomainCreators_detail
