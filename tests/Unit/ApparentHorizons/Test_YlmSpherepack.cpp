// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "ApparentHorizons/YlmSpherepack.hpp"
#include "ApparentHorizons/YlmSpherepackHelper.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "tests/Unit/ApparentHorizons/YlmTestFunctions.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {

using SecondDeriv = YlmSpherepack::SecondDeriv;

void test_prolong_restrict() {
  YlmSpherepack ylm_a(10, 10);

  const YlmTestFunctions::FuncA func_a{};
  const YlmTestFunctions::FuncB func_b{};
  const YlmTestFunctions::FuncC func_c{};

  const auto& theta = ylm_a.theta_points();
  const auto& phi = ylm_a.phi_points();

  const auto u_a = func_a.func(theta, phi);
  const auto u_b = func_b.func(theta, phi);
  const auto u_c = func_c.func(theta, phi);

  const auto u_coef_a = ylm_a.phys_to_spec(u_a);
  {
    YlmSpherepack ylm_b(10, 7);
    const auto u_coef_a2b = ylm_a.prolong_or_restrict(u_coef_a, ylm_b);
    const auto u_coef_a2b2a = ylm_b.prolong_or_restrict(u_coef_a2b, ylm_a);
    const auto u_b_test = ylm_a.spec_to_phys(u_coef_a2b2a);
    CHECK_ITERABLE_APPROX(u_b, u_b_test);
  }

  {
    YlmSpherepack ylm_c(6, 2);
    const auto u_coef_a2c = ylm_a.prolong_or_restrict(u_coef_a, ylm_c);
    const auto u_coef_a2c2a = ylm_c.prolong_or_restrict(u_coef_a2c, ylm_a);
    const auto u_c_test = ylm_a.spec_to_phys(u_coef_a2c2a);
    CHECK_ITERABLE_APPROX(u_c, u_c_test);
  }
}

void test_loop_over_offset(
    const size_t l_max, const size_t m_max, const size_t physical_stride,
    const YlmTestFunctions::ScalarFunctionWithDerivs& func) {
  YlmSpherepack ylm_spherepack(l_max, m_max);

  // Fill data vectors
  const size_t physical_size = ylm_spherepack.physical_size() * physical_stride;
  const size_t spectral_size = ylm_spherepack.spectral_size() * physical_stride;
  DataVector u(physical_size), u_spec(spectral_size);

  // Fill analytic solution
  const std::vector<double>& theta = ylm_spherepack.theta_points();
  const std::vector<double>& phi = ylm_spherepack.phi_points();
  for (size_t off = 0; off < physical_stride; ++off) {
    func.func(&u, physical_stride, off, theta, phi);
  }

  // Evaluate spectral coefficients of initial scalar function
  ylm_spherepack.phys_to_spec_all_offsets(u_spec.data(), u.data(),
                                          physical_stride);

  // Test whether phys_to_spec and spec_to_phys are inverses.
  {
    std::vector<double> u_test(physical_size);
    ylm_spherepack.spec_to_phys_all_offsets(u_test.data(), u_spec.data(),
                                            physical_stride);
    for (size_t s = 0; s < physical_size; ++s) {
      CHECK(u[s] == approx(u_test[s]));
    }
  }

  // Test simplified interface
  {
    auto u_spec_simple =
        ylm_spherepack.phys_to_spec_all_offsets(u, physical_stride);
    CHECK_ITERABLE_APPROX(u_spec, u_spec_simple);

    auto u_test =
        ylm_spherepack.spec_to_phys_all_offsets(u_spec_simple, physical_stride);
    CHECK_ITERABLE_APPROX(u, u_test);
  }

  // Test gradient
  {
    std::vector<std::vector<double>> duteststor(2), dustor(2), duSpecstor(2);
    for (size_t i = 0; i < 2; ++i) {
      dustor[i].resize(physical_size);
      duteststor[i].resize(physical_size);
      duSpecstor[i].resize(physical_size);
    }
    std::array<double*, 2> du({{dustor[0].data(), dustor[1].data()}});
    std::array<double*, 2> duSpec(
        {{duSpecstor[0].data(), duSpecstor[1].data()}});
    std::array<double*, 2> dutest(
        {{duteststor[0].data(), duteststor[1].data()}});

    // Fill analytic result
    for (size_t off = 0; off < physical_stride; ++off) {
      func.dfunc(&dutest, physical_stride, off, theta, phi);
    }

    // Differentiate
    ylm_spherepack.gradient_from_coefs_all_offsets(duSpec, u_spec.data(),
                                                   physical_stride);
    ylm_spherepack.gradient_all_offsets(du, u.data(), physical_stride);

    // Test vs analytic result
    for (size_t d = 0; d < 2; ++d) {
      for (size_t s = 0; s < physical_size; ++s) {
        CHECK(gsl::at(dutest, d)[s] == approx(gsl::at(du, d)[s]));
        CHECK(gsl::at(dutest, d)[s] == approx(gsl::at(duSpec, d)[s]));
      }
    }

    // Test simplified interface of gradient
    {
      auto du_simple = ylm_spherepack.gradient_all_offsets(u, physical_stride);
      for (size_t d = 0; d < 2; ++d) {
        for (size_t s = 0; s < physical_size; ++s) {
          CHECK(gsl::at(dutest, d)[s] == approx(du_simple.get(d)[s]));
        }
      }
      du_simple = ylm_spherepack.gradient_from_coefs_all_offsets(
          u_spec, physical_stride);
      for (size_t d = 0; d < 2; ++d) {
        for (size_t s = 0; s < physical_size; ++s) {
          CHECK(gsl::at(dutest, d)[s] == approx(du_simple.get(d)[s]));
        }
      }
    }
  }
}

void test_theta_phi_points(
    const size_t l_max, const size_t m_max,
    const YlmTestFunctions::ScalarFunctionWithDerivs& func) {
  YlmSpherepack ylm_spherepack(l_max, m_max);

  // Fill with analytic function
  const auto& theta = ylm_spherepack.theta_points();
  const auto& phi = ylm_spherepack.phi_points();
  DataVector u(ylm_spherepack.physical_size());
  func.func(&u, 1, 0, theta, phi);

  const auto& theta_phi = ylm_spherepack.theta_phi_points();
  DataVector u_test(ylm_spherepack.physical_size());
  // fill pointwise using offset
  for (size_t s = 0; s < ylm_spherepack.physical_size(); ++s) {
    func.func(&u_test, 1, s, {gsl::at(theta_phi, 0)[s]},
              {gsl::at(theta_phi, 1)[s]});
  }
  CHECK_ITERABLE_APPROX(u, u_test);
}

void test_phys_to_spec(const size_t l_max, const size_t m_max,
                       const size_t physical_stride,
                       const size_t spectral_stride,
                       const YlmTestFunctions::ScalarFunctionWithDerivs& func) {
  const size_t n_th = l_max + 1;
  const size_t n_ph = 2 * m_max + 1;
  const size_t physical_size = n_th * n_ph * physical_stride;
  const size_t spectral_size = 2 * (l_max + 1) * (m_max + 1) * spectral_stride;

  YlmSpherepack ylm_spherepack(l_max, m_max);
  CHECK(physical_size == ylm_spherepack.physical_size() * physical_stride);
  CHECK(spectral_size == ylm_spherepack.spectral_size() * spectral_stride);

  const auto& theta = ylm_spherepack.theta_points();
  const auto& phi = ylm_spherepack.phi_points();

  DataVector u(physical_size), u_spec(spectral_size);

  // Fill with analytic function
  func.func(&u, physical_stride, 0, theta, phi);

  // Evaluate spectral coefficients of initial scalar function
  ylm_spherepack.phys_to_spec(u_spec.data(), u.data(), physical_stride, 0,
                              spectral_stride, 0);

  // Test whether phys_to_spec and spec_to_phys are inverses.
  {
    std::vector<double> u_test(physical_size), u_spec_test(spectral_size);
    ylm_spherepack.phys_to_spec(u_spec_test.data(), u.data(), physical_stride,
                                0, spectral_stride, 0);
    ylm_spherepack.spec_to_phys(u_test.data(), u_spec.data(), spectral_stride,
                                0, physical_stride, 0);
    for (size_t s = 0; s < physical_size; s += physical_stride) {
      CHECK(u[s] == approx(u_test[s]));
    }
    for (size_t s = 0; s < spectral_size; s += spectral_stride) {
      CHECK(u_spec[s] == u_spec_test[s]);
    }
  }

  // Test simplified interface of phys_to_spec/spec_to_phys
  if (physical_stride == 1 and spectral_stride == 1) {
    auto u_spec_simple = ylm_spherepack.phys_to_spec(u);
    CHECK_ITERABLE_APPROX(u_spec, u_spec_simple);

    auto u_test = ylm_spherepack.spec_to_phys(u_spec_simple);
    CHECK_ITERABLE_APPROX(u, u_test);
  }
}

void test_gradient(const size_t l_max, const size_t m_max,
                   const size_t physical_stride, const size_t spectral_stride,
                   const YlmTestFunctions::ScalarFunctionWithDerivs& func) {
  YlmSpherepack ylm_spherepack(l_max, m_max);
  const size_t physical_size = ylm_spherepack.physical_size() * physical_stride;
  const size_t spectral_size = ylm_spherepack.spectral_size() * spectral_stride;

  const auto& theta = ylm_spherepack.theta_points();
  const auto& phi = ylm_spherepack.phi_points();

  DataVector u(physical_size), u_spec(spectral_size);

  // Fill with analytic function
  func.func(&u, physical_stride, 0, theta, phi);

  // Evaluate spectral coefficients of initial scalar function
  ylm_spherepack.phys_to_spec(u_spec.data(), u.data(), physical_stride, 0,
                              spectral_stride, 0);

  // Test gradient
  {
    std::vector<std::vector<double>> duteststor(2), dustor(2), duSpecstor(2);
    for (size_t i = 0; i < 2; ++i) {
      dustor[i].resize(physical_size);
      duteststor[i].resize(physical_size);
      duSpecstor[i].resize(physical_size);
    }
    std::array<double*, 2> du({{dustor[0].data(), dustor[1].data()}});
    std::array<double*, 2> duSpec(
        {{duSpecstor[0].data(), duSpecstor[1].data()}});
    std::array<double*, 2> dutest(
        {{duteststor[0].data(), duteststor[1].data()}});

    // Differentiate
    ylm_spherepack.gradient_from_coefs(duSpec, u_spec.data(), spectral_stride,
                                       0, physical_stride, 0);
    ylm_spherepack.gradient(du, u.data(), physical_stride, 0);

    // Test vs analytic result
    func.dfunc(&dutest, physical_stride, 0, theta, phi);
    for (size_t d = 0; d < 2; ++d) {
      for (size_t s = 0; s < physical_size; s += physical_stride) {
        CHECK(gsl::at(dutest, d)[s] == approx(gsl::at(du, d)[s]));
        CHECK(gsl::at(dutest, d)[s] == approx(gsl::at(duSpec, d)[s]));
      }
    }

    if (physical_stride == 1 && spectral_stride == 1) {
      // Without strides and offsets.
      ylm_spherepack.gradient_from_coefs(duSpec, u_spec.data());
      ylm_spherepack.gradient(du, u.data());
      for (size_t d = 0; d < 2; ++d) {
        for (size_t s = 0; s < physical_size; ++s) {
          CHECK(gsl::at(dutest, d)[s] == approx(gsl::at(du, d)[s]));
          CHECK(gsl::at(dutest, d)[s] == approx(gsl::at(duSpec, d)[s]));
        }
      }

      // Test simplified interface of gradient
      auto du_simple = ylm_spherepack.gradient(u);
      for (size_t d = 0; d < 2; ++d) {
        for (size_t s = 0; s < physical_size; ++s) {
          CHECK(gsl::at(dutest, d)[s] == approx(du_simple.get(d)[s]));
        }
      }
      du_simple = ylm_spherepack.gradient_from_coefs(u_spec);
      for (size_t d = 0; d < 2; ++d) {
        for (size_t s = 0; s < physical_size; ++s) {
          CHECK(gsl::at(dutest, d)[s] == approx(du_simple.get(d)[s]));
        }
      }
    } else {
      // Test simplified interface of gradient for non-unit stride
      auto du_simple = ylm_spherepack.gradient(u, physical_stride);
      for (size_t d = 0; d < 2; ++d) {
        for (size_t s = 0; s < ylm_spherepack.physical_size(); ++s) {
          CHECK(gsl::at(dutest, d)[s * physical_stride] ==
                approx(du_simple.get(d)[s]));
        }
      }
      du_simple = ylm_spherepack.gradient_from_coefs(u_spec, spectral_stride);
      for (size_t d = 0; d < 2; ++d) {
        for (size_t s = 0; s < ylm_spherepack.physical_size(); ++s) {
          CHECK(gsl::at(dutest, d)[s * physical_stride] ==
                approx(du_simple.get(d)[s]));
        }
      }
    }
  }
}

void test_second_derivative(
    const size_t l_max, const size_t m_max, const size_t physical_stride,
    const size_t spectral_stride,
    const YlmTestFunctions::ScalarFunctionWithDerivs& func) {
  YlmSpherepack ylm_spherepack(l_max, m_max);
  const size_t physical_size = ylm_spherepack.physical_size() * physical_stride;
  const size_t spectral_size = ylm_spherepack.spectral_size() * spectral_stride;

  const auto& theta = ylm_spherepack.theta_points();
  const auto& phi = ylm_spherepack.phi_points();

  DataVector u(physical_size), u_spec(spectral_size);

  // Fill with analytic function
  func.func(&u, physical_stride, 0, theta, phi);

  // Evaluate spectral coefficients of initial scalar function
  ylm_spherepack.phys_to_spec(u_spec.data(), u.data(), physical_stride, 0,
                              spectral_stride, 0);

  // Test second_derivative
  {
    SecondDeriv ddu(physical_size), ddutest(physical_size);

    std::vector<std::vector<double>> dustor(2);
    for (size_t i = 0; i < 2; ++i) {
      dustor[i].resize(physical_size);
    }
    std::array<double*, 2> du{{dustor[0].data(), dustor[1].data()}};

    // Differentiate
    ylm_spherepack.second_derivative(du, &ddu, u.data(), physical_stride, 0);

    // Test ylm_spherepack derivative against func analytical result
    func.ddfunc(&ddutest, physical_stride, 0, theta, phi);
    for (size_t i = 0; i < 2; ++i) {
      for (size_t j = 0; j < 2; ++j) {
        for (size_t s = 0; s < physical_size; s += physical_stride) {
          CHECK(ddutest.get(i, j)[s] == approx(ddu.get(i, j)[s]));
        }
      }
    }

    if (physical_stride == 1 && spectral_stride == 1) {
      ylm_spherepack.second_derivative(du, &ddu, u.data());
      for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
          CHECK_ITERABLE_APPROX(ddutest.get(i, j), ddu.get(i, j));
        }
      }

      // Test first_and_second_derivative
      auto deriv_test = ylm_spherepack.first_and_second_derivative(u);
      for (size_t i = 0; i < 2; ++i) {
        for (size_t s = 0; s < physical_size; ++s) {
          CHECK(std::get<0>(deriv_test).get(i)[s] == approx(gsl::at(du, i)[s]));
        }
        for (size_t j = 0; j < 2; ++j) {
          CHECK_ITERABLE_APPROX(std::get<1>(deriv_test).get(i, j),
                                ddu.get(i, j));
        }
      }
    }
  }
}

void test_scalar_laplacian(
    const size_t l_max, const size_t m_max, const size_t physical_stride,
    const size_t spectral_stride,
    const YlmTestFunctions::ScalarFunctionWithDerivs& func) {
  YlmSpherepack ylm_spherepack(l_max, m_max);
  const size_t physical_size = ylm_spherepack.physical_size() * physical_stride;
  const size_t spectral_size = ylm_spherepack.spectral_size() * spectral_stride;

  const auto& theta = ylm_spherepack.theta_points();
  const auto& phi = ylm_spherepack.phi_points();

  DataVector u(physical_size), u_spec(spectral_size);

  // Fill with analytic function
  func.func(&u, physical_stride, 0, theta, phi);

  // Evaluate spectral coefficients of initial scalar function
  ylm_spherepack.phys_to_spec(u_spec.data(), u.data(), physical_stride, 0,
                              spectral_stride, 0);

  // Test scalar_laplacian
  {
    DataVector slaptest(physical_size), slap(physical_size),
        slapSpec(physical_size);

    // Differentiate
    ylm_spherepack.scalar_laplacian(slap.data(), u.data(), physical_stride, 0);
    ylm_spherepack.scalar_laplacian_from_coefs(
        slapSpec.data(), u_spec.data(), spectral_stride, 0, physical_stride, 0);

    // Test ylm_spherepack derivative against func analytical result
    func.scalar_laplacian(&slaptest, physical_stride, 0, theta, phi);
    for (size_t s = 0; s < physical_size; s += physical_stride) {
      CHECK(slaptest[s] == approx(slap[s]));
      CHECK(slaptest[s] == approx(slapSpec[s]));
    }

    // Test the default arguments for stride and offset
    if (physical_stride == 1 && spectral_stride == 1) {
      ylm_spherepack.scalar_laplacian(slap.data(), u.data());
      ylm_spherepack.scalar_laplacian_from_coefs(slapSpec.data(),
                                                 u_spec.data());
      CHECK_ITERABLE_APPROX(slaptest, slap);
      CHECK_ITERABLE_APPROX(slaptest, slapSpec);

      // Test simplified interface of scalar_laplacian
      auto slap1 = ylm_spherepack.scalar_laplacian(u);
      auto slap2 = ylm_spherepack.scalar_laplacian_from_coefs(u_spec);
      CHECK_ITERABLE_APPROX(slaptest, slap1);
      CHECK_ITERABLE_APPROX(slaptest, slap2);
    }
  }
}

void test_interpolation(
    const size_t l_max, const size_t m_max, const size_t physical_stride,
    const size_t spectral_stride,
    const YlmTestFunctions::ScalarFunctionWithDerivs& func) {
  YlmSpherepack ylm_spherepack(l_max, m_max);
  const size_t physical_size = ylm_spherepack.physical_size() * physical_stride;
  const size_t spectral_size = ylm_spherepack.spectral_size() * spectral_stride;

  const auto& theta = ylm_spherepack.theta_points();
  const auto& phi = ylm_spherepack.phi_points();

  DataVector u(physical_size), u_spec(spectral_size);

  // Fill with analytic function
  func.func(&u, physical_stride, 0, theta, phi);

  // Evaluate spectral coefficients of initial scalar function
  ylm_spherepack.phys_to_spec(u_spec.data(), u.data(), physical_stride, 0,
                              spectral_stride, 0);

  // Test interpolation
  {
    // Choose random points
    std::vector<std::array<double, 2>> points;
    {
      std::uniform_real_distribution<double> ran(0.0, 1.0);
      std::mt19937 gen;
      for (int n = 0; n < 10; ++n) {
        const double th = (2.0 * ran(gen) - 1.0) * M_PI;
        const double ph = 2.0 * ran(gen) * M_PI;
        points.emplace_back(std::array<double, 2>{{th, ph}});

        // For the next point, increase ph by 2pi so it is out of range.
        // Should be equivalent to the first point.
        points.emplace_back(std::array<double, 2>{{th, ph + 2.0 * M_PI}});

        // For the next point, decrease ph by 2pi so it is out of range.
        // Should be equivalent to the first point.
        points.emplace_back(std::array<double, 2>{{th, ph - 2.0 * M_PI}});

        // For the next point, use negative theta so it is out of range,
        // and also add pi to phi.
        // Should be equivalent to the first point.
        points.emplace_back(std::array<double, 2>{{-th, ph + M_PI}});

        // For the next point, theta -> 2pi - theta so that theta is out of
        // range.  Also add pi to Phi.
        // Should be equivalent to the first point.
        points.emplace_back(
            std::array<double, 2>{{2.0 * M_PI - th, ph + M_PI}});
      }
    }

    // Get interp info
    auto interpolation_info = ylm_spherepack.set_up_interpolation_info(points);

    // Interpolate
    std::vector<double> uintPhys(interpolation_info.size()),
        uintSpec(interpolation_info.size());
    ylm_spherepack.interpolate(&uintPhys, u.data(), interpolation_info,
                               physical_stride, 0);
    ylm_spherepack.interpolate_from_coefs(&uintSpec, u_spec, interpolation_info,
                                          spectral_stride);

    // Test vs analytic solution
    DataVector uintanal(interpolation_info.size());
    for (size_t s = 0; s < uintanal.size(); ++s) {
      func.func(&uintanal, 1, s, {points[s][0]}, {points[s][1]});
      CHECK(uintanal[s] == approx(uintPhys[s]));
      CHECK(uintanal[s] == approx(uintSpec[s]));
    }

    // Test for angles out of range.
    for (size_t s = 0; s < uintanal.size() / 5; ++s) {
      // All answers should agree in each group of five, since the values
      // of all the out-of-range angles should represent the same point.
      CHECK(uintPhys[5 * s + 1] == approx(uintPhys[5 * s]));
      CHECK(uintPhys[5 * s + 2] == approx(uintPhys[5 * s]));
      CHECK(uintPhys[5 * s + 3] == approx(uintPhys[5 * s]));
      CHECK(uintPhys[5 * s + 4] == approx(uintPhys[5 * s]));
    }

    // Tests default values of stride and offset.
    if (physical_stride == 1 && spectral_stride == 1) {
      ylm_spherepack.interpolate(&uintPhys, u.data(), interpolation_info);
      for (size_t s = 0; s < uintanal.size(); ++s) {
        CHECK(uintanal[s] == approx(uintPhys[s]));
      }
    }

    // Test simplified interpolation interface
    if (physical_stride == 1) {
      auto test_interp = ylm_spherepack.interpolate(u, points);
      for (size_t s = 0; s < uintanal.size(); ++s) {
        CHECK(uintanal[s] == approx(test_interp[s]));
      }
    }
    if (spectral_stride == 1) {
      auto test_interp = ylm_spherepack.interpolate_from_coefs(u_spec, points);
      for (size_t s = 0; s < uintanal.size(); ++s) {
        CHECK(uintanal[s] == approx(test_interp[s]));
      }
    }
  }
}

void test_integral(const size_t l_max, const size_t m_max,
                   const size_t physical_stride, const size_t spectral_stride,
                   const YlmTestFunctions::ScalarFunctionWithDerivs& func) {
  YlmSpherepack ylm_spherepack(l_max, m_max);
  const size_t physical_size = ylm_spherepack.physical_size() * physical_stride;
  const size_t spectral_size = ylm_spherepack.spectral_size() * spectral_stride;

  const auto& theta = ylm_spherepack.theta_points();
  const auto& phi = ylm_spherepack.phi_points();

  DataVector u(physical_size), u_spec(spectral_size);

  // Fill with analytic function
  func.func(&u, physical_stride, 0, theta, phi);

  // Evaluate spectral coefficients of initial scalar function
  ylm_spherepack.phys_to_spec(u_spec.data(), u.data(), physical_stride, 0,
                              spectral_stride, 0);

  // Test integral
  if (physical_stride == 1) {
    const double integral1 = ylm_spherepack.definite_integral(u.data());
    const auto& weights = ylm_spherepack.integration_weights();
    double integral2 = 0.0;
    for (size_t s = 0; s < physical_size; ++s) {
      integral2 += u[s] * weights[s];
    }
    const double integral_test = func.integral();
    CHECK(integral1 == approx(integral_test));
    CHECK(integral2 == approx(integral_test));
  }

  // Test average
  if (spectral_stride == 1) {
    const auto avg = ylm_spherepack.average(u_spec);
    const double avg_test = func.integral() / (4.0 * M_PI);
    CHECK(avg == approx(avg_test));
  }

  // Test add_constant
  if (spectral_stride == 1) {
    const double value_to_add = 1.367;
    DataVector u_spec_plus_value = u_spec;
    ylm_spherepack.add_constant(&u_spec_plus_value, value_to_add);
    const auto avg = ylm_spherepack.average(u_spec_plus_value);
    const double avg_test = func.integral() / (4.0 * M_PI) + value_to_add;
    CHECK(avg == approx(avg_test));
  }
}

void test_YlmSpherepack(
    const size_t l_max, const size_t m_max, const size_t physical_stride,
    const size_t spectral_stride,
    const YlmTestFunctions::ScalarFunctionWithDerivs& func) {
  test_phys_to_spec(l_max, m_max, physical_stride, spectral_stride, func);
  test_gradient(l_max, m_max, physical_stride, spectral_stride, func);
  test_second_derivative(l_max, m_max, physical_stride, spectral_stride, func);
  test_scalar_laplacian(l_max, m_max, physical_stride, spectral_stride, func);
  test_interpolation(l_max, m_max, physical_stride, spectral_stride, func);
  test_integral(l_max, m_max, physical_stride, spectral_stride, func);
  if (physical_stride == 1) {
    test_theta_phi_points(l_max, m_max, func);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizons.YlmSpherepack",
                  "[ApparentHorizons][Unit]") {
  for (size_t l_max = 3; l_max < 5; ++l_max) {
    for (size_t m_max = 2; m_max <= l_max; ++m_max) {
      for (size_t physical_stride = 1; physical_stride <= 4;
           physical_stride += 3) {
        for (size_t spectral_stride = 1; spectral_stride <= 4;
             spectral_stride += 3) {
          test_YlmSpherepack(l_max, m_max, physical_stride, spectral_stride,
                             YlmTestFunctions::Y00());
          test_YlmSpherepack(l_max, m_max, physical_stride, spectral_stride,
                             YlmTestFunctions::Y10());
          test_YlmSpherepack(l_max, m_max, physical_stride, spectral_stride,
                             YlmTestFunctions::Y11());
        }
      }
    }
  }

  for (size_t l_max = 3; l_max < 5; ++l_max) {
    for (size_t m_max = 2; m_max <= l_max; ++m_max) {
      test_loop_over_offset(l_max, m_max, 3, YlmTestFunctions::Y00());
      test_loop_over_offset(l_max, m_max, 3, YlmTestFunctions::Y10());
      test_loop_over_offset(l_max, m_max, 3, YlmTestFunctions::Y11());
      test_loop_over_offset(l_max, m_max, 1, YlmTestFunctions::Y11());
    }
  }

  test_prolong_restrict();

  YlmSpherepack s(4, 4);
  test_copy_semantics(s);
  auto s_copy = s;
  test_move_semantics(std::move(s), s_copy, 6_st, 5_st);
}

// [[OutputRegex, Attempt to allocate more than 9 temps.]]
SPECTRE_TEST_CASE("Unit.ApparentHorizons.YlmSpherepackHelper.alloc_error",
                  "[ApparentHorizons][Unit]") {
  ERROR_TEST();
  YlmSpherepack_detail::MemoryPool pool;
  for (size_t i = 0; i < 10;
       ++i) {  // Only 9 temps, so the 10th iteration will fail.
    pool.get(5);
  }
}

// [[OutputRegex, Attempt to free temp that was never allocated.]]
SPECTRE_TEST_CASE("Unit.ApparentHorizons.YlmSpherepackHelper.free_error",
                  "[ApparentHorizons][Unit]") {
  ERROR_TEST();
  YlmSpherepack_detail::MemoryPool pool;
  std::vector<double> dum(1, 0.0);
  pool.free(dum);
}

// [[OutputRegex, Attempt to free temp that was never allocated.]]
SPECTRE_TEST_CASE("Unit.ApparentHorizons.YlmSpherepackHelper.free_empty_error",
                  "[ApparentHorizons][Unit]") {
  ERROR_TEST();
  YlmSpherepack_detail::MemoryPool pool;
  std::vector<double> dum;
  pool.free(dum);
}

// [[OutputRegex, Attempt to free temp that was never allocated.]]
SPECTRE_TEST_CASE("Unit.ApparentHorizons.YlmSpherepackHelper.free_ptr_error",
                  "[ApparentHorizons][Unit]") {
  ERROR_TEST();
  YlmSpherepack_detail::MemoryPool pool;
  std::vector<double> dum(1, 1.0);
  pool.free(dum.data());
}

// [[OutputRegex, Must use l_max>=2, not l_max=1]]
SPECTRE_TEST_CASE("Unit.ApparentHorizons.YlmSpherepack.too_few_theta_pts",
                  "[ApparentHorizons][Unit]") {
  ERROR_TEST();
  YlmSpherepack ylm(1, 1);
}

// [[OutputRegex, Must use m_max>=2, not m_max=1]]
SPECTRE_TEST_CASE("Unit.ApparentHorizons.YlmSpherepack.too_few_phi_pts",
                  "[ApparentHorizons][Unit]") {
  ERROR_TEST();
  YlmSpherepack ylm(2, 1);
}
