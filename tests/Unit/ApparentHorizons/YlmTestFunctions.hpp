// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ApparentHorizons/YlmSpherepack.hpp"

namespace YlmTestFunctions {

using SecondDeriv = YlmSpherepack::SecondDeriv;

class ScalarFunctionWithDerivs {
 public:
  virtual ~ScalarFunctionWithDerivs() = default;
  ScalarFunctionWithDerivs() = default;
  ScalarFunctionWithDerivs(const ScalarFunctionWithDerivs&) = default;
  ScalarFunctionWithDerivs(ScalarFunctionWithDerivs&&) noexcept = default;
  ScalarFunctionWithDerivs& operator=(const ScalarFunctionWithDerivs&) =
      default;
  ScalarFunctionWithDerivs& operator=(ScalarFunctionWithDerivs&&) noexcept =
      default;
  virtual void func(gsl::not_null<DataVector*> u, size_t stride, size_t offset,
                    const std::vector<double>& thetas,
                    const std::vector<double>& phis) const noexcept = 0;
  virtual void dfunc(gsl::not_null<std::array<double*, 2>*> du, size_t stride,
                     size_t offset, const std::vector<double>& thetas,
                     const std::vector<double>& phis) const noexcept = 0;
  virtual void ddfunc(gsl::not_null<SecondDeriv*> ddu, size_t stride,
                      size_t offset, const std::vector<double>& thetas,
                      const std::vector<double>& phis) const noexcept = 0;
  virtual void scalar_laplacian(gsl::not_null<DataVector*> slap, size_t stride,
                                size_t offset,
                                const std::vector<double>& thetas,
                                const std::vector<double>& phis) const
      noexcept = 0;
  virtual double integral() const noexcept = 0;
};

class Y00 : public ScalarFunctionWithDerivs {
 public:
  void func(gsl::not_null<DataVector*> u, size_t stride, size_t offset,
            const std::vector<double>& thetas,
            const std::vector<double>& phis) const noexcept override;
  void dfunc(gsl::not_null<std::array<double*, 2>*> du, size_t stride,
             size_t offset, const std::vector<double>& thetas,
             const std::vector<double>& phis) const noexcept override;
  void ddfunc(gsl::not_null<SecondDeriv*> ddu, size_t stride, size_t offset,
              const std::vector<double>& thetas,
              const std::vector<double>& phis) const noexcept override;
  void scalar_laplacian(gsl::not_null<DataVector*> slap, size_t stride,
                        size_t offset, const std::vector<double>& thetas,
                        const std::vector<double>& phis) const
      noexcept override;
  double integral() const noexcept override { return sqrt(4.0 * M_PI); }
};

class Y10 : public ScalarFunctionWithDerivs {
 public:
  void func(gsl::not_null<DataVector*> u, size_t stride, size_t offset,
            const std::vector<double>& thetas,
            const std::vector<double>& phis) const noexcept override;
  void dfunc(gsl::not_null<std::array<double*, 2>*> du, size_t stride,
             size_t offset, const std::vector<double>& thetas,
             const std::vector<double>& phis) const noexcept override;
  void ddfunc(gsl::not_null<SecondDeriv*> ddu, size_t stride, size_t offset,
              const std::vector<double>& thetas,
              const std::vector<double>& phis) const noexcept override;
  void scalar_laplacian(gsl::not_null<DataVector*> slap, size_t stride,
                        size_t offset, const std::vector<double>& thetas,
                        const std::vector<double>& phis) const
      noexcept override;
  double integral() const noexcept override { return 0.0; }
};

// Im(Y11(theta,phi))
class Y11 : public ScalarFunctionWithDerivs {
 public:
  void func(gsl::not_null<DataVector*> u, size_t stride, size_t offset,
            const std::vector<double>& thetas,
            const std::vector<double>& phis) const noexcept override;
  void dfunc(gsl::not_null<std::array<double*, 2>*> du, size_t stride,
             size_t offset, const std::vector<double>& thetas,
             const std::vector<double>& phis) const noexcept override;
  void ddfunc(gsl::not_null<SecondDeriv*> ddu, size_t stride, size_t offset,
              const std::vector<double>& thetas,
              const std::vector<double>& phis) const noexcept override;
  void scalar_laplacian(gsl::not_null<DataVector*> slap, size_t stride,
                        size_t offset, const std::vector<double>& thetas,
                        const std::vector<double>& phis) const
      noexcept override;
  double integral() const noexcept override { return 0.0; }
};

class SimpleScalarFunction {
 public:
  virtual ~SimpleScalarFunction() = default;
  SimpleScalarFunction() = default;
  SimpleScalarFunction(const SimpleScalarFunction&) = default;
  SimpleScalarFunction(SimpleScalarFunction&&) noexcept = default;
  SimpleScalarFunction& operator=(const SimpleScalarFunction&) = default;
  SimpleScalarFunction& operator=(SimpleScalarFunction&&) noexcept = default;
  virtual DataVector func(const std::vector<double>& thetas,
                          const std::vector<double>& phis) const noexcept = 0;
};

// Re Y(10,10) + Im Y(10,7) + Re Y(6,2)
class FuncA : public SimpleScalarFunction {
 public:
  FuncA() = default;
  DataVector func(const std::vector<double>& thetas,
                  const std::vector<double>& phis) const noexcept override;
};

// Im Y(10,7)+ Re Y(6,2)
class FuncB : public SimpleScalarFunction {
 public:
  FuncB() = default;
  DataVector func(const std::vector<double>& thetas,
                  const std::vector<double>& phis) const noexcept override;
};

// Re Y(6,2)
class FuncC : public SimpleScalarFunction {
 public:
  FuncC() = default;
  DataVector func(const std::vector<double>& thetas,
                  const std::vector<double>& phis) const noexcept override;
};

}  // namespace YlmTestFunctions
