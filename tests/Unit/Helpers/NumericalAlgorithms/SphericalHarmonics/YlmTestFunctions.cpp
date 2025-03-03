// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/NumericalAlgorithms/SphericalHarmonics/YlmTestFunctions.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace YlmTestFunctions {

ProductOfPolynomials::ProductOfPolynomials(const size_t pow_nx,
                                           const size_t pow_ny,
                                           const size_t pow_nz)
    : pow_nx_{pow_nx}, pow_ny_{pow_ny}, pow_nz_{pow_nz} {}

DataVector ProductOfPolynomials::operator()(const DataVector& theta,
                                            const DataVector& phi) const {
  return pow(sin(theta), pow_nx_ + pow_ny_) * pow(cos(theta), pow_nz_) *
         pow(cos(phi), pow_nx_) * pow(sin(phi), pow_ny_);
}

DataVector ProductOfPolynomials::df_dth(const DataVector& theta,
                                        const DataVector& phi) const {
  if (pow_nx_ + pow_ny_ + pow_nz_ == 0) {
    return DataVector{theta.size(), 0.0};
  }
  if (pow_nx_ + pow_ny_ == 0) {
    return -static_cast<double>(pow_nz_) * sin(theta) *
           pow(cos(theta), pow_nz_ - 1) * pow(cos(phi), pow_nx_) *
           pow(sin(phi), pow_ny_);
  }
  if (pow_nz_ == 0) {
    return static_cast<double>(pow_nx_ + pow_ny_) *
           pow(sin(theta), pow_nx_ + pow_ny_ - 1) * cos(theta) *
           pow(cos(phi), pow_nx_) * pow(sin(phi), pow_ny_);
  }
  return (static_cast<double>(pow_nx_ + pow_ny_) *
              pow(sin(theta), pow_nx_ + pow_ny_ - 1) *
              pow(cos(theta), pow_nz_ + 1) -
          static_cast<double>(pow_nz_) *
              pow(sin(theta), pow_nx_ + pow_ny_ + 1) *
              pow(cos(theta), pow_nz_ - 1)) *
         pow(cos(phi), pow_nx_) * pow(sin(phi), pow_ny_);
}

DataVector ProductOfPolynomials::df_dph(const DataVector& theta,
                                        const DataVector& phi) const {
  if (pow_nx_ + pow_ny_ == 0) {
    return DataVector{theta.size(), 0.0};
  }
  if (pow_nx_ == 0) {
    return static_cast<double>(pow_ny_) * pow(sin(theta), pow_nx_ + pow_ny_) *
           pow(cos(theta), pow_nz_) * cos(phi) * pow(sin(phi), pow_ny_ - 1);
  }
  if (pow_ny_ == 0) {
    return -static_cast<double>(pow_nx_) * pow(sin(theta), pow_nx_ + pow_ny_) *
           pow(cos(theta), pow_nz_) * pow(cos(phi), pow_nx_ - 1) * sin(phi);
  }
  return pow(sin(theta), pow_nx_ + pow_ny_) * pow(cos(theta), pow_nz_) *
         (static_cast<double>(pow_ny_) * pow(cos(phi), pow_nx_ + 1) *
              pow(sin(phi), pow_ny_ - 1) -
          static_cast<double>(pow_nx_) * *pow(cos(phi), pow_nx_ - 1) *
              pow(sin(phi), pow_ny_ + 1));
}

double ProductOfPolynomials::definite_integral() const {
  if ((pow_nx_ % 2 == 1) or (pow_ny_ % 2 == 1) or (pow_nz_ % 2 == 1)) {
    return 0.0;
  }
  return 4.0 * M_PI *
         static_cast<double>(falling_factorial(pow_nx_, pow_nx_ / 2)) *
         static_cast<double>(falling_factorial(pow_ny_, pow_ny_ / 2)) *
         static_cast<double>(falling_factorial(pow_nz_, pow_nz_ / 2)) /
         static_cast<double>(
             falling_factorial(pow_nx_ + pow_ny_ + pow_nz_ + 1,
                               (pow_nx_ + pow_ny_ + pow_nz_) / 2 + 1));
}

template <>
DataVector Ylm<0, 0>::f() const {
  return DataVector{n_pts_, 1.0 / sqrt(4.0 * M_PI)};
}

template <>
DataVector Ylm<0, 0>::df_dth() const {
  return DataVector{n_pts_, 0.0};
}

template <>
DataVector Ylm<0, 0>::df_dph() const {
  return DataVector{n_pts_, 0.0};
}

template <>
DataVector Ylm<1, 0>::f() const {
  return DataVector{sqrt(0.75 / M_PI) * cos(theta_)};
}

template <>
DataVector Ylm<1, 0>::df_dth() const {
  return DataVector{-sqrt(0.75 / M_PI) * sin(theta_)};
}

template <>
DataVector Ylm<1, 0>::df_dph() const {
  return DataVector{n_pts_, 0.0};
}

template <>
DataVector Ylm<1, 1>::f() const {
  return DataVector{sqrt(0.75 / M_PI) * sin(theta_) * cos(phi_)};
}

template <>
DataVector Ylm<1, 1>::df_dth() const {
  return DataVector{sqrt(0.75 / M_PI) * cos(theta_) * cos(phi_)};
}

template <>
DataVector Ylm<1, 1>::df_dph() const {
  return DataVector{-sqrt(0.75 / M_PI) * sin(phi_)};
}

template <>
DataVector Ylm<1, -1>::f() const {
  return DataVector{sqrt(0.75 / M_PI) * sin(theta_) * sin(phi_)};
}

template <>
DataVector Ylm<1, -1>::df_dth() const {
  return DataVector{sqrt(0.75 / M_PI) * cos(theta_) * sin(phi_)};
}

template <>
DataVector Ylm<1, -1>::df_dph() const {
  return DataVector{sqrt(0.75 / M_PI) * cos(phi_)};
}

template <>
DataVector Ylm<2, 0>::f() const {
  return DataVector{sqrt(1.25 / M_PI) * (1.5 * square(cos(theta_)) - 0.5)};
}

template <>
DataVector Ylm<2, 0>::df_dth() const {
  return DataVector{-3.0 * sqrt(1.25 / M_PI) * sin(theta_) * cos(theta_)};
}

template <>
DataVector Ylm<2, 0>::df_dph() const {
  return DataVector{n_pts_, 0.0};
}

template <>
DataVector Ylm<2, 1>::f() const {
  return DataVector{sqrt(3.75 / M_PI) * sin(theta_) * cos(theta_) * cos(phi_)};
}

template <>
DataVector Ylm<2, 1>::df_dth() const {
  return DataVector{sqrt(3.75 / M_PI) *
                    (square(cos(theta_)) - square(sin(theta_))) * cos(phi_)};
}

template <>
DataVector Ylm<2, 1>::df_dph() const {
  return DataVector{-sqrt(3.75 / M_PI) * cos(theta_) * sin(phi_)};
}

template <>
DataVector Ylm<2, -1>::f() const {
  return DataVector{sqrt(3.75 / M_PI) * sin(theta_) * cos(theta_) * sin(phi_)};
}

template <>
DataVector Ylm<2, -1>::df_dth() const {
  return DataVector{sqrt(3.75 / M_PI) *
                    (square(cos(theta_)) - square(sin(theta_))) * sin(phi_)};
}

template <>
DataVector Ylm<2, -1>::df_dph() const {
  return DataVector{sqrt(3.75 / M_PI) * cos(theta_) * cos(phi_)};
}

template <>
DataVector Ylm<2, 2>::f() const {
  return DataVector{0.25 * sqrt(15.0 / M_PI) * square(sin(theta_)) *
                    cos(2.0 * phi_)};
}

template <>
DataVector Ylm<2, 2>::df_dth() const {
  return DataVector{0.5 * sqrt(15.0 / M_PI) * sin(theta_) * cos(theta_) *
                    cos(2.0 * phi_)};
}

template <>
DataVector Ylm<2, 2>::df_dph() const {
  return DataVector{-0.5 * sqrt(15.0 / M_PI) * sin(theta_) * sin(2.0 * phi_)};
}

template <>
DataVector Ylm<2, -2>::f() const {
  return DataVector{0.25 * sqrt(15.0 / M_PI) * square(sin(theta_)) *
                    sin(2.0 * phi_)};
}

template <>
DataVector Ylm<2, -2>::df_dth() const {
  return DataVector{0.5 * sqrt(15.0 / M_PI) * sin(theta_) * cos(theta_) *
                    sin(2.0 * phi_)};
}

template <>
DataVector Ylm<2, -2>::df_dph() const {
  return DataVector{0.5 * sqrt(15.0 / M_PI) * sin(theta_) * cos(2.0 * phi_)};
}

void Y00::func(const gsl::not_null<DataVector*> u, const size_t stride,
               const size_t offset, const std::vector<double>& thetas,
               const std::vector<double>& phis) const {
  // Can't make inv_sqrt_4_pi constexpr because sqrt isn't constexpr.
  static const double inv_sqrt_4_pi = 0.5 / sqrt(M_PI);
  for (size_t j = 0, s = 0; j < phis.size(); ++j) {
    for (size_t i = 0; i < thetas.size(); ++i, ++s) {
      (*u)[s * stride + offset] = inv_sqrt_4_pi;
    }
  }
}

void Y00::dfunc(const gsl::not_null<std::array<double*, 2>*> du,
                const size_t stride, const size_t offset,
                const std::vector<double>& thetas,
                const std::vector<double>& phis) const {
  for (size_t d = 0; d < du->size(); ++d) {
    for (size_t j = 0, s = 0; j < phis.size(); ++j) {
      for (size_t i = 0; i < thetas.size(); ++i, ++s) {
        gsl::at(*du, d)[s * stride + offset] = 0.0;
      }
    }
  }
}

void Y00::ddfunc(const gsl::not_null<SecondDeriv*> ddu, const size_t stride,
                 const size_t offset, const std::vector<double>& thetas,
                 const std::vector<double>& phis) const {
  for (size_t d = 0; d < 2; ++d) {
    for (size_t e = 0; e < 2; ++e) {
      for (size_t j = 0, s = 0; j < phis.size(); ++j) {
        for (size_t i = 0; i < thetas.size(); ++i, ++s) {
          ddu->get(d, e)[s * stride + offset] = 0.0;
        }
      }
    }
  }
}

void Y00::scalar_laplacian(const gsl::not_null<DataVector*> slap,
                           const size_t stride, const size_t offset,
                           const std::vector<double>& thetas,
                           const std::vector<double>& phis) const {
  size_t s = 0;
  for (size_t j = 0; j < phis.size(); ++j) {
    for (size_t i = 0; i < thetas.size(); ++i, ++s) {
      (*slap)[s * stride + offset] = 0.0;
    }
  }
}

void Y10::func(const gsl::not_null<DataVector*> u, const size_t stride,
               const size_t offset, const std::vector<double>& thetas,
               const std::vector<double>& phis) const {
  const double amplitude = sqrt(3.0 / 4.0 / M_PI);
  for (size_t j = 0, s = 0; j < phis.size(); ++j) {
    for (size_t i = 0; i < thetas.size(); ++i, ++s) {
      (*u)[s * stride + offset] = cos(thetas[i]) * amplitude;
    }
  }
}
void Y10::dfunc(const gsl::not_null<std::array<double*, 2>*> du,
                const size_t stride, const size_t offset,
                const std::vector<double>& thetas,
                const std::vector<double>& phis) const {
  const double amplitude = sqrt(3.0 / 4.0 / M_PI);
  for (size_t j = 0, s = 0; j < phis.size(); ++j) {
    for (size_t i = 0; i < thetas.size(); ++i, ++s) {
      gsl::at(*du, 0)[s * stride + offset] =
          -sin(thetas[i]) * amplitude;             // d/dth
      gsl::at(*du, 1)[s * stride + offset] = 0.0;  // sin^-1 theta d/dph
    }
  }
}

void Y10::ddfunc(const gsl::not_null<SecondDeriv*> ddu, const size_t stride,
                 const size_t offset, const std::vector<double>& thetas,
                 const std::vector<double>& phis) const {
  const double amplitude = sqrt(3.0 / 4.0 / M_PI);
  for (size_t j = 0, s = 0; j < phis.size(); ++j) {
    for (size_t i = 0; i < thetas.size(); ++i, ++s) {
      ddu->get(0, 0)[s * stride + offset] = -cos(thetas[i]) * amplitude;
      ddu->get(1, 1)[s * stride + offset] = 0.0;
      ddu->get(1, 0)[s * stride + offset] = 0.0;
      ddu->get(0, 1)[s * stride + offset] = 0.0;
    }
  }
}

void Y10::scalar_laplacian(const gsl::not_null<DataVector*> slap,
                           const size_t stride, const size_t offset,
                           const std::vector<double>& thetas,
                           const std::vector<double>& phis) const {
  const double amplitude = sqrt(3.0 / 4.0 / M_PI);
  for (size_t j = 0, s = 0; j < phis.size(); ++j) {
    for (size_t i = 0; i < thetas.size(); ++i, ++s) {
      (*slap)[s * stride + offset] = -2.0 * cos(thetas[i]) * amplitude;
    }
  }
}

void Y11::func(const gsl::not_null<DataVector*> u, const size_t stride,
               const size_t offset, const std::vector<double>& thetas,
               const std::vector<double>& phis) const {
  const double amplitude = -sqrt(3.0 / 8.0 / M_PI);
  size_t s = 0;
  for (const auto& phi : phis) {
    for (size_t i = 0; i < thetas.size(); ++i, ++s) {
      (*u)[s * stride + offset] = sin(thetas[i]) * sin(phi) * amplitude;
    }
  }
}
void Y11::dfunc(const gsl::not_null<std::array<double*, 2>*> du,
                const size_t stride, const size_t offset,
                const std::vector<double>& thetas,
                const std::vector<double>& phis) const {
  const double amplitude = -sqrt(3.0 / 8.0 / M_PI);
  size_t s = 0;
  for (const auto& phi : phis) {
    for (size_t i = 0; i < thetas.size(); ++i, ++s) {
      // d/dth
      gsl::at(*du, 0)[s * stride + offset] =
          cos(thetas[i]) * sin(phi) * amplitude;
      // sin^-1(theta) d/dph
      gsl::at(*du, 1)[s * stride + offset] = cos(phi) * amplitude;
    }
  }
}

void Y11::ddfunc(const gsl::not_null<SecondDeriv*> ddu, const size_t stride,
                 const size_t offset, const std::vector<double>& thetas,
                 const std::vector<double>& phis) const {
  const double amplitude = -sqrt(3.0 / 8.0 / M_PI);
  size_t s = 0;
  for (const auto& phi : phis) {
    for (size_t i = 0; i < thetas.size(); ++i, ++s) {
      ddu->get(0, 0)[s * stride + offset] =
          -sin(thetas[i]) * sin(phi) * amplitude;
      ddu->get(1, 1)[s * stride + offset] =
          -sin(phi) / sin(thetas[i]) * amplitude;
      ddu->get(1, 0)[s * stride + offset] =
          cos(thetas[i]) * cos(phi) * amplitude / sin(thetas[i]);
      ddu->get(0, 1)[s * stride + offset] =
          ddu->get(1, 0)[s * stride + offset] -
          cos(thetas[i]) * cos(phi) * amplitude / sin(thetas[i]);
    }
  }
}

void Y11::scalar_laplacian(const gsl::not_null<DataVector*> slap,
                           const size_t stride, const size_t offset,
                           const std::vector<double>& thetas,
                           const std::vector<double>& phis) const {
  const double amplitude = -sqrt(3.0 / 8.0 / M_PI);
  size_t s = 0;
  for (const auto& phi : phis) {
    for (size_t i = 0; i < thetas.size(); ++i, ++s) {
      (*slap)[s * stride + offset] =
          amplitude * (sin(phi) *
                           (cos(thetas[i]) * cos(thetas[i]) -
                            sin(thetas[i]) * sin(thetas[i])) /
                           sin(thetas[i]) -
                       sin(phi) / sin(thetas[i]));
    }
  }
}

DataVector FuncA::func(const std::vector<double>& thetas,
                       const std::vector<double>& phis) const {
  DataVector u(thetas.size() * phis.size());
  size_t s = 0;
  for (const auto& phi : phis) {
    for (const auto& theta : thetas) {
      const double sin_theta = sin(theta);
      const double cos_theta = cos(theta);
      u[s] = sqrt(969969.0 / M_PI) * pow<10>(sin_theta) * cos(10.0 * phi) /
                 1024.0 -
             sqrt(85085.0 / M_PI) * (3.0 / 512.0) * pow<7>(sin_theta) *
                 (19.0 * cube(cos_theta) - 3.0 * cos_theta) * sin(7.0 * phi) +
             sqrt(1365.0 / M_PI) / 64.0 * square(sin_theta) * cos(2.0 * phi) *
                 (33.0 * pow<4>(cos_theta) - 18.0 * square(cos_theta) + 1.0);
      ++s;
    }
  }
  return u;
}

DataVector FuncB::func(const std::vector<double>& thetas,
                       const std::vector<double>& phis) const {
  DataVector u(thetas.size() * phis.size());
  size_t s = 0;
  for (const auto& phi : phis) {
    for (const auto& theta : thetas) {
      const double sin_theta = sin(theta);
      const double cos_theta = cos(theta);
      u[s] = -sqrt(85085.0 / M_PI) * (3.0 / 512.0) * pow<7>(sin_theta) *
                 (19.0 * cube(cos_theta) - 3.0 * cos_theta) * sin(7.0 * phi) +
             sqrt(1365.0 / M_PI) / 64.0 * square(sin_theta) * cos(2.0 * phi) *
                 (33.0 * pow<4>(cos_theta) - 18.0 * square(cos_theta) + 1.0);
      ++s;
    }
  }
  return u;
}

DataVector FuncC::func(const std::vector<double>& thetas,
                       const std::vector<double>& phis) const {
  DataVector u(thetas.size() * phis.size());
  size_t s = 0;
  for (const auto& phi : phis) {
    for (const auto& theta : thetas) {
      const double sin_theta = sin(theta);
      const double cos_theta = cos(theta);
      u[s] = sqrt(1365.0 / M_PI) / 64.0 * square(sin_theta) * cos(2.0 * phi) *
             (33.0 * pow<4>(cos_theta) - 18.0 * square(cos_theta) + 1.0);
      ++s;
    }
  }
  return u;
}

}  // namespace YlmTestFunctions
