// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "Utilities/Gsl.hpp"

namespace hydro::primitive_sources {
template <class T>
using V = std::array<T, 3>;
template <class T>
using M = std::array<V<T>, 3>;
template <class T>
using Sources =
    std::array<T, 5>;  // E, contravariant spatial Sx,Sy,Sz, spatial trace
inline size_t size(const double& /*x*/) { return 1; }
inline size_t size(const DataVector& x) { return x.size(); }
inline double cell(double x, size_t /*i*/) { return x; }
inline double cell(const DataVector& x, size_t i) { return x[i]; }
template <class T>
T constant(const T& x, double a) {
  if constexpr (std::is_same_v<T, double>) {
    return a;
  } else {
    return DataVector(x.size(), a);
  }
}
inline double fma_value(double a, double b, double c) {
  return std::fma(a, b, c);
}
inline DataVector fma_value(const DataVector& a, const DataVector& b,
                            const DataVector& c) {
  DataVector out(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    out[i] = std::fma(a[i], b[i], c[i]);
  }
  return out;
}
template <class T>
T sub_product(const T& x, const T& a, const T& b) {
  const T na = -a;
  return fma_value(na, b, x);
}
template <class T>
T diff(const T& a, const T& b, const T& c, const T& d) {
  const T cd = c * d;
  const T minus_cd = -cd;
  const T minus_c = -c;
  const T hi = fma_value(a, b, minus_cd);
  const T lo = fma_value(minus_c, d, cd);
  return T(hi + lo);
}
template <class T>
V<T> cross(const V<T>& a, const V<T>& b) {
  return {diff(a[1], b[2], a[2], b[1]), diff(a[2], b[0], a[0], b[2]),
          diff(a[0], b[1], a[1], b[0])};
}
template <class T>
T dot(const V<T>& a, const V<T>& b) {
  return T((a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]);
}
template <class T>
void finite(const T& x, const char* name) {
  for (size_t i = 0; i < size(x); ++i) {
    if (not std::isfinite(cell(x, i))) {
      throw std::domain_error(std::string(name) + ": nonfinite cell " +
                              std::to_string(i));
    }
  }
}
template <class T>
void positive(const T& x, const char* name) {
  finite(x, name);
  for (size_t i = 0; i < size(x); ++i) {
    if (not(cell(x, i) > 0)) {
      throw std::domain_error(std::string(name) + ": nonpositive cell " +
                              std::to_string(i));
    }
  }
}
// Exact sign classification of the received binary64 values for epsilon < -1.
// Callers validate finite inputs first. Sign comparisons use exact rationals.
inline boost::multiprecision::cpp_rational dyadic(double x) {
  static_assert(std::numeric_limits<double>::digits == 53);
  int exponent = 0;
  const double fraction = std::frexp(x, &exponent);
  boost::multiprecision::cpp_rational q =
      static_cast<std::int64_t>(std::ldexp(fraction, 53));
  const int shift = std::abs(exponent - 53);
  // Build the exact power of two with fixed byte-aligned multiprecision
  // shifts. The remaining built-in shift is bounded to [0, 31].
  boost::multiprecision::cpp_int scale = std::uint64_t{1} << (shift % 32);
  for (int i = 0; i < shift / 32; ++i) {
    scale <<= 32;
  }
  if (exponent >= 53) {
    q *= scale;
  } else {
    q /= scale;
  }
  return q;
}
inline bool exact_enthalpy_nonnegative(double rho, double epsilon, double p) {
  // rho,p nonnegative => exact H=rho*(1+epsilon)+p >=0 on this fast branch.
  if (epsilon >= -1 or rho == 0) {
    return true;
  }
  const auto r = dyadic(rho);
  const auto e = dyadic(epsilon);
  const auto pressure = dyadic(p);
  const boost::multiprecision::cpp_rational enthalpy = r + r * e + pressure;
  // A normalized rational has a positive denominator, so only its numerator
  // determines the sign. Do not cross-multiply rational denominators with zero.
  return numerator(enthalpy) >= 0;
}
/*!
 * \brief Compute energy density, contravariant momentum density and spatial
 * stress trace jointly from ideal-MHD primitives in a physical spatial metric.
 *
 * The returned order is E, S^x, S^y, S^z, trace. Inputs are rho, specific
 * internal energy epsilon, pressure p, contravariant velocity v and magnetic
 * field B, and the covariant spatial metric g. T is double or DataVector.
 * The library prepares H = rho + rho * epsilon + p directly; no specific
 * enthalpy or independently prepared Lorentz factor is required.
 *
 * All components must be finite and have matching nonempty extents. The metric
 * must be symmetric with positive Cholesky pivots, and 0 <= g(v,v) < 1.
 * Thermodynamic requirements are rho >= 0, p >= 0, and nonnegative exact H for
 * the received inputs, with finite nonnegative computed H. Epsilon may be
 * negative. The epsilon < -1 sign check uses exact dyadic rationals so that
 * underflow cannot turn negative H into an accepted zero. This validates the
 * stated algebraic domain; it does not establish EOS consistency or impose
 * additional energy conditions.
 *
 * At rho = 0, p must be zero and epsilon must still be finite. Electromagnetic
 * sources retain ideal-MHD kinematics for the supplied velocity. This boundary
 * is not a prescription for arbitrary Maxwell vacuum data. No atmosphere floor
 * or other silent input correction is applied. One invalid vector lane rejects
 * the entire request.
 *
 * \throws std::invalid_argument for empty or mismatched component extents.
 * \throws std::domain_error for failed finiteness, geometry, velocity or
 * thermodynamic checks, including nonfinite prepared quantities or outputs.
 */
template <class T>
Sources<T> joint_sources(const T& rho, const T& epsilon, const T& p,
                         const V<T>& v, const V<T>& B, const M<T>& g) {
  static_assert(std::is_same_v<T, double> or std::is_same_v<T, DataVector>);
  const size_t n = size(rho);
  if (n == 0) {
    throw std::invalid_argument("empty batch");
  }
  auto check = [&](const T& x) {
    if (size(x) != n) {
      throw std::invalid_argument("component size mismatch");
    }
    finite(x, "input");
  };
  check(rho);
  check(epsilon);
  check(p);
  for (size_t i = 0; i < 3; ++i) {
    check(gsl::at(v, i));
    check(gsl::at(B, i));
    for (size_t j = 0; j < 3; ++j) {
      check(gsl::at(gsl::at(g, i), j));
    }
  }
  for (size_t k = 0; k < n; ++k) {
    if (not(cell(rho, k) >= 0) or cell(p, k) < 0 or
        (cell(rho, k) == 0 and cell(p, k) != 0)) {
      throw std::domain_error("thermodynamic domain");
    }
    if (not exact_enthalpy_nonnegative(cell(rho, k), cell(epsilon, k),
                                       cell(p, k))) {
      throw std::domain_error("exact negative total enthalpy density");
    }
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        if (cell(gsl::at(gsl::at(g, i), j), k) !=
            cell(gsl::at(gsl::at(g, j), i), k)) {
          throw std::domain_error("nonsymmetric metric");
        }
      }
    }
  }
  const T zero = constant(rho, 0);
  const T one = constant(rho, 1);
  M<T> U;
  for (auto& row : U) {
    for (auto& x : row) {
      x = zero;
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    T pivot = gsl::at(gsl::at(g, i), i);
    for (size_t k = 0; k < i; ++k) {
      pivot = sub_product(pivot, gsl::at(gsl::at(U, k), i),
                          gsl::at(gsl::at(U, k), i));
    }
    positive(pivot, "Cholesky pivot");
    using std::sqrt;
    gsl::at(gsl::at(U, i), i) = sqrt(pivot);
    positive(gsl::at(gsl::at(U, i), i), "Cholesky diagonal");
    for (size_t j = i + 1; j < 3; ++j) {
      T x = gsl::at(gsl::at(g, i), j);
      for (size_t k = 0; k < i; ++k) {
        x = sub_product(x, gsl::at(gsl::at(U, k), i),
                        gsl::at(gsl::at(U, k), j));
      }
      gsl::at(gsl::at(U, i), j) = x / gsl::at(gsl::at(U, i), i);
      finite(gsl::at(gsl::at(U, i), j), "Cholesky offdiagonal");
    }
  }
  const auto q = cross(v, B);
  V<T> bv{zero, zero, zero};
  V<T> vv{zero, zero, zero};
  V<T> y{zero, zero, zero};
  V<T> Bl{zero, zero, zero};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      gsl::at(bv, i) += gsl::at(gsl::at(U, i), j) * gsl::at(B, j);
      gsl::at(vv, i) += gsl::at(gsl::at(U, i), j) * gsl::at(v, j);
    }
    T rhs = gsl::at(q, i);
    for (size_t k = 0; k < i; ++k) {
      rhs = sub_product(rhs, gsl::at(gsl::at(U, k), i), gsl::at(y, k));
    }
    gsl::at(y, i) = rhs / gsl::at(gsl::at(U, i), i);
    for (size_t j = 0; j < 3; ++j) {
      gsl::at(Bl, i) += gsl::at(gsl::at(g, i), j) * gsl::at(B, j);
    }
  }
  const T volume = U[0][0] * U[1][1] * U[2][2];
  positive(volume, "metric volume");
  for (auto& x : y) {
    x *= volume;
  }
  const T B2 = dot(bv, bv);
  const T E2 = dot(y, y);
  const T v2 = dot(vv, vv);
  finite(B2, "B2");
  finite(E2, "E2");
  finite(v2, "v2");
  for (size_t k = 0; k < n; ++k) {
    if (cell(v2, k) < 0 or not(cell(v2, k) < 1)) {
      throw std::domain_error("velocity outside timelike domain");
    }
  }
  const T H = rho + rho * epsilon + p;
  const T w2 = one / (one - v2);
  const T Eem = (B2 + E2) * .5;
  finite(H, "enthalpy density");
  for (size_t k = 0; k < n; ++k) {
    if (cell(H, k) < 0) {
      throw std::domain_error("negative total enthalpy density");
    }
  }
  const auto S = cross(Bl, q);
  Sources<T> out{H * w2 - p + Eem, H * w2 * v[0] + S[0], H * w2 * v[1] + S[1],
                 H * w2 * v[2] + S[2], H * w2 * v2 + 3. * p + Eem};
  for (const auto& x : out) {
    finite(x, "source");
  }
  return out;
}
}  // namespace hydro::primitive_sources
