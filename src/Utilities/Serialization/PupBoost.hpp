// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines helper functions for working with boost.

#pragma once

#include <boost/container/static_vector.hpp>
#include <boost/math/quaternion.hpp>
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

template <class T, size_t N>
void pup(PUP::er& p, boost::container::static_vector<T, N>& v) {
  auto size = v.size();
  p | size;
  v.resize(size);
  if (PUP::as_bytes<T>::value) {
    PUParray(p, v.data(), size);
  } else {
    for (auto& x : v) {
      p | x;
    }
  }
}

template <class T, size_t N>
void operator|(PUP::er& p, boost::container::static_vector<T, N>& v) {
  pup(p, v);
}

template <typename T>
void pup(PUP::er& p, boost::math::quaternion<T>& quaternion) {
  T component_1 = quaternion.R_component_1();
  T component_2 = quaternion.R_component_2();
  T component_3 = quaternion.R_component_3();
  T component_4 = quaternion.R_component_4();
  p | component_1;
  p | component_2;
  p | component_3;
  p | component_4;
  if (p.isUnpacking()) {
    quaternion = boost::math::quaternion<T>(component_1, component_2,
                                            component_3, component_4);
  }
}

template <typename T>
void operator|(PUP::er& p, boost::math::quaternion<T>& quaternion) {
  pup(p, quaternion);
}
}  // namespace PUP
