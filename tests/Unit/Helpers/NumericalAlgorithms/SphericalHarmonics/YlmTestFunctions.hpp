// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "Utilities/Gsl.hpp"

namespace YlmTestFunctions {

/*!
 * \brief Product of polynomials regular on the surface of a sphere
 *
 * \details Computes \f$ n_x^{k_x} n_y^{k_y} n_z^{k_z} \f$ where \f$n_x = \sin
 * \theta \cos \phi\f$, \f$n_y = \sin \theta \sin \phi\f$, and \f$n_z = \cos
 * \theta\f$.  The function and its first derivatives are exactly representable
 * by spherical harmonics of order \f$(L,M)\f$ if \f$L > k_x + k_y + k_z\f$ and
 * \f$M > k_x + k_y\f$.
 *
 */
class ProductOfPolynomials {
 public:
  ProductOfPolynomials(size_t pow_nx, size_t pow_ny, size_t pow_nz);
  DataVector operator()(const DataVector& theta, const DataVector& phi) const;
  template <typename Fr>
  DataVector operator()(const tnsr::I<DataVector, 2, Fr>& theta_and_phi) const {
    return operator()(get<0>(theta_and_phi), get<1>(theta_and_phi));
  }
  DataVector df_dth(const DataVector& theta, const DataVector& phi) const;
  template <typename Fr>
  DataVector df_dth(const tnsr::I<DataVector, 2, Fr>& theta_and_phi) const {
    return df_dth(get<0>(theta_and_phi), get<1>(theta_and_phi));
  }
  // This is the Pfaffiaan derivative (extra factor 1/sin(th))
  DataVector df_dph(const DataVector& theta, const DataVector& phi) const;
  template <typename Fr>
  DataVector df_dph(const tnsr::I<DataVector, 2, Fr>& theta_and_phi) const {
    return df_dph(get<0>(theta_and_phi), get<1>(theta_and_phi));
  }
  double definite_integral() const;

 private:
  size_t pow_nx_;
  size_t pow_ny_;
  size_t pow_nz_;
};

template <size_t l, int m>
class Ylm {
 public:
  Ylm(size_t n_r, size_t L, size_t M);
  DataVector f() const;
  DataVector df_dth() const;
  // This is the Pfaffiaan derivative (returns 1/sin(theta) df_dph)
  DataVector df_dph() const;
  DataVector modes() const;

 private:
  size_t n_r_;
  size_t size_of_modes_;
  double c_lm_;
  size_t n_pts_;
  DataVector theta_;
  DataVector phi_;
  size_t offset_c_lm_;
};

template <size_t l, int m>
Ylm<l, m>::Ylm(const size_t n_r, size_t L, size_t M)
    : n_r_{n_r},
      size_of_modes_{2 * n_r * (L + 1) * (M + 1)},
      c_lm_{m == 0 ? sqrt(2.0 / M_PI)
                   : (m < 0 ? -sqrt(1.0 / M_PI) : sqrt(1.0 / M_PI))} {
  ASSERT(L > l, "Cannot represent derivatives on mesh");
  ASSERT(static_cast<int>(M) >= std::abs(m),
         "Cannot represent derivatives on mesh");
  const Mesh<3> mesh{
      {n_r, L + 1, 2 * M + 1},
      {Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
       Spectral::Basis::SphericalHarmonic},
      {Spectral::Quadrature::Gauss, Spectral::Quadrature::Gauss,
       Spectral::Quadrature::Equiangular}};
  const auto xi = logical_coordinates(mesh);
  n_pts_ = mesh.number_of_grid_points();
  theta_ = xi.get(1);
  phi_ = xi.get(2);
  ylm::SpherepackIterator it(L, M, n_r);
  it.set(l, m);
  offset_c_lm_ = it();
}

template <size_t l, int m>
DataVector Ylm<l, m>::modes() const {
  DataVector result{size_of_modes_, 0.0};
  for (size_t k = 0; k < n_r_; ++k) {
    result[offset_c_lm_ + k] = c_lm_;
  }
  return result;
}

using SecondDeriv = ylm::Spherepack::SecondDeriv;

class ScalarFunctionWithDerivs {
 public:
  virtual ~ScalarFunctionWithDerivs() = default;
  ScalarFunctionWithDerivs() = default;
  ScalarFunctionWithDerivs(const ScalarFunctionWithDerivs&) = default;
  ScalarFunctionWithDerivs(ScalarFunctionWithDerivs&&) = default;
  ScalarFunctionWithDerivs& operator=(const ScalarFunctionWithDerivs&) =
      default;
  ScalarFunctionWithDerivs& operator=(ScalarFunctionWithDerivs&&) = default;
  virtual void func(gsl::not_null<DataVector*> u, size_t stride, size_t offset,
                    const std::vector<double>& thetas,
                    const std::vector<double>& phis) const = 0;
  virtual void dfunc(gsl::not_null<std::array<double*, 2>*> du, size_t stride,
                     size_t offset, const std::vector<double>& thetas,
                     const std::vector<double>& phis) const = 0;
  virtual void ddfunc(gsl::not_null<SecondDeriv*> ddu, size_t stride,
                      size_t offset, const std::vector<double>& thetas,
                      const std::vector<double>& phis) const = 0;
  virtual void scalar_laplacian(gsl::not_null<DataVector*> slap, size_t stride,
                                size_t offset,
                                const std::vector<double>& thetas,
                                const std::vector<double>& phis) const = 0;
  virtual double integral() const = 0;
};

class Y00 : public ScalarFunctionWithDerivs {
 public:
  ~Y00() override = default;
  Y00() = default;
  Y00(const Y00&) = default;
  Y00(Y00&&) = default;
  Y00& operator=(const Y00&) = default;
  Y00& operator=(Y00&&) = default;
  void func(gsl::not_null<DataVector*> u, size_t stride, size_t offset,
            const std::vector<double>& thetas,
            const std::vector<double>& phis) const override;
  void dfunc(gsl::not_null<std::array<double*, 2>*> du, size_t stride,
             size_t offset, const std::vector<double>& thetas,
             const std::vector<double>& phis) const override;
  void ddfunc(gsl::not_null<SecondDeriv*> ddu, size_t stride, size_t offset,
              const std::vector<double>& thetas,
              const std::vector<double>& phis) const override;
  void scalar_laplacian(gsl::not_null<DataVector*> slap, size_t stride,
                        size_t offset, const std::vector<double>& thetas,
                        const std::vector<double>& phis) const override;
  double integral() const override { return sqrt(4.0 * M_PI); }
};

class Y10 : public ScalarFunctionWithDerivs {
 public:
  ~Y10() override = default;
  Y10() = default;
  Y10(const Y10&) = default;
  Y10(Y10&&) = default;
  Y10& operator=(const Y10&) = default;
  Y10& operator=(Y10&&) = default;
  void func(gsl::not_null<DataVector*> u, size_t stride, size_t offset,
            const std::vector<double>& thetas,
            const std::vector<double>& phis) const override;
  void dfunc(gsl::not_null<std::array<double*, 2>*> du, size_t stride,
             size_t offset, const std::vector<double>& thetas,
             const std::vector<double>& phis) const override;
  void ddfunc(gsl::not_null<SecondDeriv*> ddu, size_t stride, size_t offset,
              const std::vector<double>& thetas,
              const std::vector<double>& phis) const override;
  void scalar_laplacian(gsl::not_null<DataVector*> slap, size_t stride,
                        size_t offset, const std::vector<double>& thetas,
                        const std::vector<double>& phis) const override;
  double integral() const override { return 0.0; }
};

// Im(Y11(theta,phi))
class Y11 : public ScalarFunctionWithDerivs {
 public:
  ~Y11() override = default;
  Y11() = default;
  Y11(const Y11&) = default;
  Y11(Y11&&) = default;
  Y11& operator=(const Y11&) = default;
  Y11& operator=(Y11&&) = default;
  void func(gsl::not_null<DataVector*> u, size_t stride, size_t offset,
            const std::vector<double>& thetas,
            const std::vector<double>& phis) const override;
  void dfunc(gsl::not_null<std::array<double*, 2>*> du, size_t stride,
             size_t offset, const std::vector<double>& thetas,
             const std::vector<double>& phis) const override;
  void ddfunc(gsl::not_null<SecondDeriv*> ddu, size_t stride, size_t offset,
              const std::vector<double>& thetas,
              const std::vector<double>& phis) const override;
  void scalar_laplacian(gsl::not_null<DataVector*> slap, size_t stride,
                        size_t offset, const std::vector<double>& thetas,
                        const std::vector<double>& phis) const override;
  double integral() const override { return 0.0; }
};

class SimpleScalarFunction {
 public:
  virtual ~SimpleScalarFunction() = default;
  SimpleScalarFunction() = default;
  SimpleScalarFunction(const SimpleScalarFunction&) = default;
  SimpleScalarFunction(SimpleScalarFunction&&) = default;
  SimpleScalarFunction& operator=(const SimpleScalarFunction&) = default;
  SimpleScalarFunction& operator=(SimpleScalarFunction&&) = default;
  virtual DataVector func(const std::vector<double>& thetas,
                          const std::vector<double>& phis) const = 0;
};

// Re Y(10,10) + Im Y(10,7) + Re Y(6,2)
class FuncA : public SimpleScalarFunction {
 public:
  ~FuncA() override = default;
  FuncA() = default;
  FuncA(const FuncA&) = default;
  FuncA(FuncA&&) = default;
  FuncA& operator=(const FuncA&) = default;
  FuncA& operator=(FuncA&&) = default;
  DataVector func(const std::vector<double>& thetas,
                  const std::vector<double>& phis) const override;
};

// Im Y(10,7)+ Re Y(6,2)
class FuncB : public SimpleScalarFunction {
 public:
  ~FuncB() override = default;
  FuncB() = default;
  FuncB(const FuncB&) = default;
  FuncB(FuncB&&) = default;
  FuncB& operator=(const FuncB&) = default;
  FuncB& operator=(FuncB&&) = default;
  DataVector func(const std::vector<double>& thetas,
                  const std::vector<double>& phis) const override;
};

// Re Y(6,2)
class FuncC : public SimpleScalarFunction {
 public:
  ~FuncC() override = default;
  FuncC() = default;
  FuncC(const FuncC&) = default;
  FuncC(FuncC&&) = default;
  FuncC& operator=(const FuncC&) = default;
  FuncC& operator=(FuncC&&) = default;
  DataVector func(const std::vector<double>& thetas,
                  const std::vector<double>& phis) const override;
};

}  // namespace YlmTestFunctions
