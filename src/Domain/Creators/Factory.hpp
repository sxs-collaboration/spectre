// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

namespace DomainCreators_detail {
template <size_t Dim>
struct domain_creators;
}  // namespace DomainCreators_detail

template <size_t Dim>
using domain_creators =
    typename DomainCreators_detail::domain_creators<Dim>::type;
