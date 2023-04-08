// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines helper functions for working with boost.

#pragma once

#include <boost/rational.hpp>
#include <pup.h>

namespace PUP {
/// @{
/// \ingroup ParallelGroup
/// Serialization of boost::rational for Charm++
template <class T>
void pup(PUP::er& p, boost::rational<T>& var) {  // NOLINT
  if (p.isUnpacking()) {
    typename boost::rational<T>::int_type n, d;
    p | n;
    p | d;
    var.assign(n, d);
  } else {
    typename boost::rational<T>::int_type n = var.numerator();
    typename boost::rational<T>::int_type d = var.denominator();
    p | n;
    p | d;
  }
}

template <typename T>
inline void operator|(PUP::er& p, boost::rational<T>& var) {  // NOLINT
  pup(p, var);
}
/// @}
}  // namespace PUP
