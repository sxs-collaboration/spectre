// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Identity.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/DuDtTempTags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TimeDerivative.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpacetimeDerivativeOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {
// Since repeatedly generating new numbers to compare to SpEC when the function
// signature of the GH time derivative changes is difficult, we use a reference
// implementation that we test against the Spectral Einstein Code (SpEC) to test
// against our RHS changes.
template <size_t Dim>
std::tuple<tnsr::aa<DataVector, Dim, Frame::Inertial>,
           tnsr::aa<DataVector, Dim, Frame::Inertial>,
           tnsr::iaa<DataVector, Dim, Frame::Inertial>>
gh_rhs_reference_impl(
    const tnsr::aa<DataVector, Dim>& spacetime_metric,
    const tnsr::aa<DataVector, Dim>& pi, const tnsr::iaa<DataVector, Dim>& phi,
    const tnsr::iaa<DataVector, Dim>& d_spacetime_metric,
    const tnsr::iaa<DataVector, Dim>& d_pi,
    const tnsr::ijaa<DataVector, Dim>& d_phi, const Scalar<DataVector>& gamma0,
    const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
    const tnsr::a<DataVector, Dim>& gauge_function,
    const tnsr::ab<DataVector, Dim>& spacetime_deriv_gauge_function,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
    const tnsr::II<DataVector, Dim>& inverse_spatial_metric,
    const tnsr::AA<DataVector, Dim>& inverse_spacetime_metric,
    const tnsr::a<DataVector, Dim>& trace_christoffel,
    const tnsr::abb<DataVector, Dim>& christoffel_first_kind,
    const tnsr::Abb<DataVector, Dim>& christoffel_second_kind,
    const tnsr::A<DataVector, Dim>& normal_spacetime_vector,
    const tnsr::a<DataVector, Dim>& normal_spacetime_one_form) {
  const size_t n_pts = shift.begin()->size();

  tnsr::aa<DataVector, Dim, Frame::Inertial> dt_spacetime_metric(n_pts);
  tnsr::aa<DataVector, Dim, Frame::Inertial> dt_pi(n_pts);
  tnsr::iaa<DataVector, Dim, Frame::Inertial> dt_phi(n_pts);

  const DataVector gamma12 = gamma1.get() * gamma2.get();

  tnsr::Iaa<DataVector, Dim> phi_1_up{DataVector(n_pts, 0.)};
  for (size_t m = 0; m < Dim; ++m) {
    for (size_t mu = 0; mu < Dim + 1; ++mu) {
      for (size_t n = 0; n < Dim; ++n) {
        for (size_t nu = mu; nu < Dim + 1; ++nu) {
          phi_1_up.get(m, mu, nu) +=
              inverse_spatial_metric.get(m, n) * phi.get(n, mu, nu);
        }
      }
    }
  }

  tnsr::abC<DataVector, Dim> phi_3_up{DataVector(n_pts, 0.)};
  for (size_t m = 0; m < Dim; ++m) {
    for (size_t nu = 0; nu < Dim + 1; ++nu) {
      for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
        for (size_t beta = 0; beta < Dim + 1; ++beta) {
          phi_3_up.get(m, nu, alpha) +=
              inverse_spacetime_metric.get(alpha, beta) * phi.get(m, nu, beta);
        }
      }
    }
  }

  tnsr::aB<DataVector, Dim> pi_2_up{DataVector(n_pts, 0.)};
  for (size_t nu = 0; nu < Dim + 1; ++nu) {
    for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
      for (size_t beta = 0; beta < Dim + 1; ++beta) {
        pi_2_up.get(nu, alpha) +=
            inverse_spacetime_metric.get(alpha, beta) * pi.get(nu, beta);
      }
    }
  }

  tnsr::abC<DataVector, Dim> christoffel_first_kind_3_up{DataVector(n_pts, 0.)};
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    for (size_t nu = 0; nu < Dim + 1; ++nu) {
      for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
        for (size_t beta = 0; beta < Dim + 1; ++beta) {
          christoffel_first_kind_3_up.get(mu, nu, alpha) +=
              inverse_spacetime_metric.get(alpha, beta) *
              christoffel_first_kind.get(mu, nu, beta);
        }
      }
    }
  }

  tnsr::a<DataVector, Dim> pi_dot_normal_spacetime_vector{
      DataVector(n_pts, 0.)};
  for (size_t nu = 0; nu < Dim + 1; ++nu) {
    for (size_t mu = 0; mu < Dim + 1; ++mu) {
      pi_dot_normal_spacetime_vector.get(mu) +=
          normal_spacetime_vector.get(nu) * pi.get(nu, mu);
    }
  }

  DataVector pi_contract_two_normal_spacetime_vectors{DataVector(n_pts, 0.)};
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    pi_contract_two_normal_spacetime_vectors +=
        normal_spacetime_vector.get(mu) *
        pi_dot_normal_spacetime_vector.get(mu);
  }

  tnsr::ia<DataVector, Dim> phi_dot_normal_spacetime_vector{
      DataVector(n_pts, 0.)};
  for (size_t n = 0; n < Dim; ++n) {
    for (size_t nu = 0; nu < Dim + 1; ++nu) {
      for (size_t mu = 0; mu < Dim + 1; ++mu) {
        phi_dot_normal_spacetime_vector.get(n, nu) +=
            normal_spacetime_vector.get(mu) * phi.get(n, mu, nu);
      }
    }
  }

  tnsr::a<DataVector, Dim> phi_contract_two_normal_spacetime_vectors{
      DataVector(n_pts, 0.)};
  for (size_t n = 0; n < Dim; ++n) {
    for (size_t mu = 0; mu < Dim + 1; ++mu) {
      phi_contract_two_normal_spacetime_vectors.get(n) +=
          normal_spacetime_vector.get(mu) *
          phi_dot_normal_spacetime_vector.get(n, mu);
    }
  }

  tnsr::iaa<DataVector, Dim> three_index_constraint{DataVector(n_pts, 0.)};
  for (size_t n = 0; n < Dim; ++n) {
    for (size_t mu = 0; mu < Dim + 1; ++mu) {
      for (size_t nu = mu; nu < Dim + 1; ++nu) {
        three_index_constraint.get(n, mu, nu) =
            d_spacetime_metric.get(n, mu, nu) - phi.get(n, mu, nu);
      }
    }
  }

  tnsr::a<DataVector, Dim> one_index_constraint{DataVector(n_pts, 0.)};
  for (size_t nu = 0; nu < Dim + 1; ++nu) {
    one_index_constraint.get(nu) =
        gauge_function.get(nu) + trace_christoffel.get(nu);
  }

  DataVector normal_dot_one_index_constraint{DataVector(n_pts, 0.)};
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    normal_dot_one_index_constraint +=
        normal_spacetime_vector.get(mu) * one_index_constraint.get(mu);
  }

  const DataVector gamma1p1 = 1.0 + gamma1.get();

  tnsr::aa<DataVector, Dim> shift_dot_three_index_constraint{
      DataVector(n_pts, 0.)};
  for (size_t m = 0; m < Dim; ++m) {
    for (size_t mu = 0; mu < Dim + 1; ++mu) {
      for (size_t nu = mu; nu < Dim + 1; ++nu) {
        shift_dot_three_index_constraint.get(mu, nu) +=
            shift.get(m) * three_index_constraint.get(m, mu, nu);
      }
    }
  }

  // Here are the actual equations

  // Equation for dt_spacetime_metric
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    for (size_t nu = mu; nu < Dim + 1; ++nu) {
      dt_spacetime_metric.get(mu, nu) = -lapse.get() * pi.get(mu, nu);
      dt_spacetime_metric.get(mu, nu) +=
          gamma1p1 * shift_dot_three_index_constraint.get(mu, nu);
      for (size_t m = 0; m < Dim; ++m) {
        dt_spacetime_metric.get(mu, nu) += shift.get(m) * phi.get(m, mu, nu);
      }
    }
  }

  // Equation for dt_pi
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    for (size_t nu = mu; nu < Dim + 1; ++nu) {
      dt_pi.get(mu, nu) =
          -spacetime_deriv_gauge_function.get(mu, nu) -
          spacetime_deriv_gauge_function.get(nu, mu) -
          0.5 * pi_contract_two_normal_spacetime_vectors * pi.get(mu, nu) +
          gamma0.get() * (normal_spacetime_one_form.get(mu) *
                              one_index_constraint.get(nu) +
                          normal_spacetime_one_form.get(nu) *
                              one_index_constraint.get(mu)) -
          gamma0.get() * spacetime_metric.get(mu, nu) *
              normal_dot_one_index_constraint;

      for (size_t delta = 0; delta < Dim + 1; ++delta) {
        dt_pi.get(mu, nu) += 2 * christoffel_second_kind.get(delta, mu, nu) *
                                 gauge_function.get(delta) -
                             2 * pi.get(mu, delta) * pi_2_up.get(nu, delta);

        for (size_t n = 0; n < Dim; ++n) {
          dt_pi.get(mu, nu) +=
              2 * phi_1_up.get(n, mu, delta) * phi_3_up.get(n, nu, delta);
        }

        for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
          dt_pi.get(mu, nu) -=
              2. * christoffel_first_kind_3_up.get(mu, alpha, delta) *
              christoffel_first_kind_3_up.get(nu, delta, alpha);
        }
      }

      for (size_t m = 0; m < Dim; ++m) {
        dt_pi.get(mu, nu) -=
            pi_dot_normal_spacetime_vector.get(m + 1) * phi_1_up.get(m, mu, nu);

        for (size_t n = 0; n < Dim; ++n) {
          dt_pi.get(mu, nu) -=
              inverse_spatial_metric.get(m, n) * d_phi.get(m, n, mu, nu);
        }
      }

      dt_pi.get(mu, nu) *= lapse.get();

      dt_pi.get(mu, nu) +=
          gamma12 * shift_dot_three_index_constraint.get(mu, nu);

      for (size_t m = 0; m < Dim; ++m) {
        // DualFrame term
        dt_pi.get(mu, nu) += shift.get(m) * d_pi.get(m, mu, nu);
      }
    }
  }

  // Equation for dt_phi
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t mu = 0; mu < Dim + 1; ++mu) {
      for (size_t nu = mu; nu < Dim + 1; ++nu) {
        dt_phi.get(i, mu, nu) =
            0.5 * pi.get(mu, nu) *
                phi_contract_two_normal_spacetime_vectors.get(i) -
            d_pi.get(i, mu, nu) +
            gamma2.get() * three_index_constraint.get(i, mu, nu);
        for (size_t n = 0; n < Dim; ++n) {
          dt_phi.get(i, mu, nu) +=
              phi_dot_normal_spacetime_vector.get(i, n + 1) *
              phi_1_up.get(n, mu, nu);
        }

        dt_phi.get(i, mu, nu) *= lapse.get();
        for (size_t m = 0; m < Dim; ++m) {
          dt_phi.get(i, mu, nu) += shift.get(m) * d_phi.get(m, i, mu, nu);
        }
      }
    }
  }
  return std::make_tuple(std::move(dt_spacetime_metric), std::move(dt_pi),
                         std::move(dt_phi));
}

// Creates the random numbers that match what was used when comparing the GH RHS
// to the Spectral Einstein Code (SpEC) implementation.
template <typename Tensor>
Tensor create_tensor_with_random_values(
    const size_t n_pts, const gsl::not_null<std::mt19937*> gen) {
  Tensor tensor(n_pts);
  for (auto& data : tensor) {
    for (size_t s = 0; s < data.size(); ++s) {
      data[s] = std::uniform_real_distribution<>(-10, 10)(*gen);
    }
  }
  return tensor;
}

void test_reference_impl_against_spec() noexcept {
  const size_t dim = 3;
  const size_t n_pts = 2;

  // This test compares the output of the GeneralizedHarmonic RHS to the output
  // computed using SpEC for the specific input tensors generated by this seed.
  std::mt19937 gen(1.);

  const auto psi = create_tensor_with_random_values<
      tnsr::aa<DataVector, dim, Frame::Inertial>>(n_pts, make_not_null(&gen));
  const auto pi = create_tensor_with_random_values<
      tnsr::aa<DataVector, dim, Frame::Inertial>>(n_pts, make_not_null(&gen));
  const auto phi = create_tensor_with_random_values<
      tnsr::iaa<DataVector, dim, Frame::Inertial>>(n_pts, make_not_null(&gen));
  const auto d_psi = create_tensor_with_random_values<
      tnsr::iaa<DataVector, dim, Frame::Inertial>>(n_pts, make_not_null(&gen));
  const auto d_pi = create_tensor_with_random_values<
      tnsr::iaa<DataVector, dim, Frame::Inertial>>(n_pts, make_not_null(&gen));
  const auto d_phi = create_tensor_with_random_values<
      tnsr::ijaa<DataVector, dim, Frame::Inertial>>(n_pts, make_not_null(&gen));
  const auto gauge_function = create_tensor_with_random_values<
      tnsr::a<DataVector, dim, Frame::Inertial>>(n_pts, make_not_null(&gen));
  const auto spacetime_deriv_gauge_function = create_tensor_with_random_values<
      tnsr::ab<DataVector, dim, Frame::Inertial>>(n_pts, make_not_null(&gen));
  const auto gamma0 = create_tensor_with_random_values<Scalar<DataVector>>(
      n_pts, make_not_null(&gen));
  const auto gamma1 = create_tensor_with_random_values<Scalar<DataVector>>(
      n_pts, make_not_null(&gen));
  const auto gamma2 = create_tensor_with_random_values<Scalar<DataVector>>(
      n_pts, make_not_null(&gen));
  const auto lapse = create_tensor_with_random_values<Scalar<DataVector>>(
      n_pts, make_not_null(&gen));
  const auto shift = create_tensor_with_random_values<
      tnsr::I<DataVector, dim, Frame::Inertial>>(n_pts, make_not_null(&gen));
  const auto upper_spatial_metric = create_tensor_with_random_values<
      tnsr::II<DataVector, dim, Frame::Inertial>>(n_pts, make_not_null(&gen));
  const auto upper_psi = create_tensor_with_random_values<
      tnsr::AA<DataVector, dim, Frame::Inertial>>(n_pts, make_not_null(&gen));
  const auto christoffel_first_kind = create_tensor_with_random_values<
      tnsr::abb<DataVector, dim, Frame::Inertial>>(n_pts, make_not_null(&gen));
  const auto christoffel_second_kind = create_tensor_with_random_values<
      tnsr::Abb<DataVector, dim, Frame::Inertial>>(n_pts, make_not_null(&gen));
  const auto trace_christoffel_first_kind = create_tensor_with_random_values<
      tnsr::a<DataVector, dim, Frame::Inertial>>(n_pts, make_not_null(&gen));
  const auto normal_one_form = create_tensor_with_random_values<
      tnsr::a<DataVector, dim, Frame::Inertial>>(n_pts, make_not_null(&gen));
  const auto normal_vector = create_tensor_with_random_values<
      tnsr::A<DataVector, dim, Frame::Inertial>>(n_pts, make_not_null(&gen));

  const auto [dt_psi, dt_pi, dt_phi] = gh_rhs_reference_impl(
      psi, pi, phi, d_psi, d_pi, d_phi, gamma0, gamma1, gamma2, gauge_function,
      spacetime_deriv_gauge_function, lapse, shift, upper_spatial_metric,
      upper_psi, trace_christoffel_first_kind, christoffel_first_kind,
      christoffel_second_kind, normal_vector, normal_one_form);

  CHECK(dt_psi.get(0, 0)[0] == approx(-488.874963261792004));
  CHECK(dt_psi.get(0, 0)[1] == approx(-81.510619345029468));
  CHECK(dt_psi.get(0, 1)[0] == approx(-250.047822240104125));
  CHECK(dt_psi.get(0, 1)[1] == approx(296.302836131049901));
  CHECK(dt_psi.get(0, 2)[0] == approx(-500.997780531678018));
  CHECK(dt_psi.get(0, 2)[1] == approx(466.785608622853886));
  CHECK(dt_psi.get(0, 3)[0] == approx(467.671854125188190));
  CHECK(dt_psi.get(0, 3)[1] == approx(495.703587199368314));
  CHECK(dt_psi.get(1, 1)[0] == approx(-1455.466926116829427));
  CHECK(dt_psi.get(1, 1)[1] == approx(537.232777225229825));
  CHECK(dt_psi.get(1, 2)[0] == approx(605.305755751280685));
  CHECK(dt_psi.get(1, 2)[1] == approx(-147.555027862059319));
  CHECK(dt_psi.get(1, 3)[0] == approx(272.588366899036828));
  CHECK(dt_psi.get(1, 3)[1] == approx(103.726287040730483));
  CHECK(dt_psi.get(2, 2)[0] == approx(-698.404763643118372));
  CHECK(dt_psi.get(2, 2)[1] == approx(-54.529887031214074));
  CHECK(dt_psi.get(2, 3)[0] == approx(685.972755990463384));
  CHECK(dt_psi.get(2, 3)[1] == approx(354.613491895308584));
  CHECK(dt_psi.get(3, 3)[0] == approx(-245.215344706651052));
  CHECK(dt_psi.get(3, 3)[1] == approx(-276.589507479549297));
  CHECK(dt_pi.get(0, 0)[0] == approx(-1994.203229187604393));
  CHECK(dt_pi.get(0, 0)[1] == approx(-525201.290706016356125));
  CHECK(dt_pi.get(0, 1)[0] == approx(6311.905865276433360));
  CHECK(dt_pi.get(0, 1)[1] == approx(-240190.350753725302638));
  CHECK(dt_pi.get(0, 2)[0] == approx(-7434.327351519351396));
  CHECK(dt_pi.get(0, 2)[1] == approx(-252008.364267619152088));
  CHECK(dt_pi.get(0, 3)[0] == approx(2838.324638613014940));
  CHECK(dt_pi.get(0, 3)[1] == approx(51236.973434082428867));
  CHECK(dt_pi.get(1, 1)[0] == approx(-354.079220430901557));
  CHECK(dt_pi.get(1, 1)[1] == approx(126094.565118705097120));
  CHECK(dt_pi.get(1, 2)[0] == approx(-5619.191809738355005));
  CHECK(dt_pi.get(1, 2)[1] == approx(-107407.264584976015612));
  CHECK(dt_pi.get(1, 3)[0] == approx(897.475897012644054));
  CHECK(dt_pi.get(1, 3)[1] == approx(89268.754206339028315));
  CHECK(dt_pi.get(2, 2)[0] == approx(-21157.862553074402967));
  CHECK(dt_pi.get(2, 2)[1] == approx(-94493.937608642052510));
  CHECK(dt_pi.get(2, 3)[0] == approx(-6057.203556770715295));
  CHECK(dt_pi.get(2, 3)[1] == approx(-153608.523542909941170));
  CHECK(dt_pi.get(3, 3)[0] == approx(-17537.832989340076892));
  CHECK(dt_pi.get(3, 3)[1] == approx(-60984.946935138112167));
  CHECK(dt_phi.get(0, 0, 0)[0] == approx(334.313398747739541));
  CHECK(dt_phi.get(0, 0, 0)[1] == approx(25514.306297616123629));
  CHECK(dt_phi.get(0, 0, 1)[0] == approx(2077.905394153287034));
  CHECK(dt_phi.get(0, 0, 1)[1] == approx(-23798.504669162444770));
  CHECK(dt_phi.get(0, 0, 2)[0] == approx(-1284.875746578147073));
  CHECK(dt_phi.get(0, 0, 2)[1] == approx(-19831.646993788559485));
  CHECK(dt_phi.get(0, 0, 3)[0] == approx(233.700445801117098));
  CHECK(dt_phi.get(0, 0, 3)[1] == approx(-34774.544349022733513));
  CHECK(dt_phi.get(0, 1, 1)[0] == approx(885.722636195821224));
  CHECK(dt_phi.get(0, 1, 1)[1] == approx(-33932.040869534444937));
  CHECK(dt_phi.get(0, 1, 2)[0] == approx(-650.957577518966218));
  CHECK(dt_phi.get(0, 1, 2)[1] == approx(2638.696086360518620));
  CHECK(dt_phi.get(0, 1, 3)[0] == approx(31.485299569095631));
  CHECK(dt_phi.get(0, 1, 3)[1] == approx(-3754.688091563506987));
  // This value is only accurate to within 1e-13 when built on release with GCC
  // 6.3
  Approx coarser_approx = Approx::custom().epsilon(1e-13);
  CHECK(dt_phi.get(0, 2, 2)[0] == coarser_approx(10.787542001317307));
  CHECK(dt_phi.get(0, 2, 2)[1] == approx(6388.265784499862093));
  CHECK(dt_phi.get(0, 2, 3)[0] == approx(-264.672203939199221));
  CHECK(dt_phi.get(0, 2, 3)[1] == approx(-14106.605264503405124));
  CHECK(dt_phi.get(0, 3, 3)[0] == approx(-186.705594181490596));
  CHECK(dt_phi.get(0, 3, 3)[1] == approx(37619.063591638048820));
  CHECK(dt_phi.get(1, 0, 0)[0] == approx(-562.600849624935677));
  CHECK(dt_phi.get(1, 0, 0)[1] == approx(26636.175386835606332));
  CHECK(dt_phi.get(1, 0, 1)[0] == approx(-7348.564617671276210));
  CHECK(dt_phi.get(1, 0, 1)[1] == approx(-10239.582656574073553));
  CHECK(dt_phi.get(1, 0, 2)[0] == approx(3735.515737741963676));
  CHECK(dt_phi.get(1, 0, 2)[1] == approx(-9541.966102859829334));
  CHECK(dt_phi.get(1, 0, 3)[0] == approx(-189.717363255539595));
  CHECK(dt_phi.get(1, 0, 3)[1] == approx(-38768.276299298151571));
  CHECK(dt_phi.get(1, 1, 1)[0] == approx(-3992.834118054164264));
  CHECK(dt_phi.get(1, 1, 1)[1] == approx(-27052.090043482003239));
  CHECK(dt_phi.get(1, 1, 2)[0] == approx(3861.313476781318514));
  CHECK(dt_phi.get(1, 1, 2)[1] == approx(9909.233790204545585));
  CHECK(dt_phi.get(1, 1, 3)[0] == approx(781.120763452604706));
  CHECK(dt_phi.get(1, 1, 3)[1] == approx(32796.160467693429382));
  CHECK(dt_phi.get(1, 2, 2)[0] == approx(-1182.382985415981693));
  CHECK(dt_phi.get(1, 2, 2)[1] == approx(-19585.499004859902925));
  CHECK(dt_phi.get(1, 2, 3)[0] == approx(2533.123508025803858));
  CHECK(dt_phi.get(1, 2, 3)[1] == approx(39702.781747730077768));
  CHECK(dt_phi.get(1, 3, 3)[0] == approx(1356.930486013502104));
  CHECK(dt_phi.get(1, 3, 3)[1] == approx(18614.741669036127860));
  CHECK(dt_phi.get(2, 0, 0)[0] == approx(-267.727784019130809));
  CHECK(dt_phi.get(2, 0, 0)[1] == approx(-34312.754364113869087));
  CHECK(dt_phi.get(2, 0, 1)[0] == approx(-3628.105306092249975));
  CHECK(dt_phi.get(2, 0, 1)[1] == approx(26021.001208727309859));
  CHECK(dt_phi.get(2, 0, 2)[0] == approx(1787.972861399919111));
  CHECK(dt_phi.get(2, 0, 2)[1] == approx(20245.691906571166328));
  CHECK(dt_phi.get(2, 0, 3)[0] == approx(-345.850954809957727));
  CHECK(dt_phi.get(2, 0, 3)[1] == approx(45631.496531139637227));
  CHECK(dt_phi.get(2, 1, 1)[0] == approx(-1936.213815715472037));
  CHECK(dt_phi.get(2, 1, 1)[1] == approx(39876.740245914304978));
  CHECK(dt_phi.get(2, 1, 2)[0] == approx(2047.144489678545369));
  CHECK(dt_phi.get(2, 1, 2)[1] == approx(-6162.040420810454634));
  CHECK(dt_phi.get(2, 1, 3)[0] == approx(490.866985829649764));
  CHECK(dt_phi.get(2, 1, 3)[1] == approx(-5369.075417735416522));
  CHECK(dt_phi.get(2, 2, 2)[0] == approx(-444.587399207923738));
  CHECK(dt_phi.get(2, 2, 2)[1] == approx(1205.339390995000485));
  CHECK(dt_phi.get(2, 2, 3)[0] == approx(1208.970864104782549));
  CHECK(dt_phi.get(2, 2, 3)[1] == approx(1422.401000428901625));
  CHECK(dt_phi.get(2, 3, 3)[0] == approx(1116.070338196526109));
  CHECK(dt_phi.get(2, 3, 3)[1] == approx(-42638.998279054998420));
}

template <size_t Dim, typename Generator>
void test_compute_dudt(const gsl::not_null<Generator*> generator) noexcept {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  using gh_tags_list = tmpl::list<gr::Tags::SpacetimeMetric<Dim>,
                                  GeneralizedHarmonic::Tags::Pi<Dim>,
                                  GeneralizedHarmonic::Tags::Phi<Dim>>;

  const size_t num_grid_points_1d = 3;
  const Mesh<Dim> mesh(num_grid_points_1d, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
  const DataVector used_for_size(mesh.number_of_grid_points());

  Variables<gh_tags_list> evolved_vars(mesh.number_of_grid_points());
  fill_with_random_values(make_not_null(&evolved_vars), generator,
                          make_not_null(&distribution));
  // In order to satisfy the physical requirements on the spacetime metric we
  // compute it from the helper functions that generate a physical lapse, shift,
  // and spatial metric.
  gr::spacetime_metric(
      make_not_null(&get<gr::Tags::SpacetimeMetric<Dim>>(evolved_vars)),
      TestHelpers::gr::random_lapse(generator, used_for_size),
      TestHelpers::gr::random_shift<Dim>(generator, used_for_size),
      TestHelpers::gr::random_spatial_metric<Dim>(generator, used_for_size));

  InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
      inv_jac{};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      if (i == j) {
        inv_jac.get(i, j) = DataVector(mesh.number_of_grid_points(), 1.0);
      } else {
        inv_jac.get(i, j) = DataVector(mesh.number_of_grid_points(), 0.0);
      }
    }
  }

  const auto partial_derivs =
      partial_derivatives<gh_tags_list>(evolved_vars, mesh, inv_jac);

  const auto& spacetime_metric =
      get<gr::Tags::SpacetimeMetric<Dim>>(evolved_vars);
  const auto& phi = get<GeneralizedHarmonic::Tags::Phi<Dim>>(evolved_vars);
  const auto& pi = get<GeneralizedHarmonic::Tags::Pi<Dim>>(evolved_vars);
  const auto& d_spacetime_metric =
      get<Tags::deriv<gr::Tags::SpacetimeMetric<Dim>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(partial_derivs);
  const auto& d_phi =
      get<Tags::deriv<GeneralizedHarmonic::Tags::Phi<Dim>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(partial_derivs);
  const auto& d_pi =
      get<Tags::deriv<GeneralizedHarmonic::Tags::Pi<Dim>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(partial_derivs);
  ;

  const auto gamma0 = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&distribution), used_for_size);
  const auto gamma1 = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&distribution), used_for_size);
  const auto gamma2 = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&distribution), used_for_size);
  const auto gauge_function = make_with_random_values<tnsr::a<DataVector, Dim>>(
      generator, make_not_null(&distribution), used_for_size);
  const auto spacetime_deriv_gauge_function =
      make_with_random_values<tnsr::ab<DataVector, Dim>>(
          generator, make_not_null(&distribution), used_for_size);

  // Quantities as input for reference RHS
  const auto spatial_metric = gr::spatial_metric(spacetime_metric);
  const auto inverse_spatial_metric_and_det = determinant_and_inverse<
      gr::Tags::DetSpatialMetric<DataVector>,
      gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>>(
      spatial_metric);
  const auto& upper_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>>(
          inverse_spatial_metric_and_det);
  const auto shift = gr::shift(spacetime_metric, upper_spatial_metric);
  const auto lapse = gr::lapse(shift, spacetime_metric);
  const auto upper_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, upper_spatial_metric);
  tnsr::abb<DataVector, Dim> da_spacetime_metric;
  GeneralizedHarmonic::spacetime_derivative_of_spacetime_metric(
      make_not_null(&da_spacetime_metric), lapse, shift, pi, phi);
  const auto christoffel_first_kind =
      gr::christoffel_first_kind(da_spacetime_metric);
  const auto christoffel_second_kind = raise_or_lower_first_index(
      christoffel_first_kind, upper_spacetime_metric);
  const auto trace_christoffel_first_kind =
      trace_last_indices(christoffel_first_kind, upper_spacetime_metric);
  const auto normal_vector = gr::spacetime_normal_vector(lapse, shift);
  const auto normal_one_form =
      gr::spacetime_normal_one_form<Dim, Frame::Inertial>(lapse);

  const auto [expected_dt_spacetime_metric, expected_dt_pi, expected_dt_phi] =
      gh_rhs_reference_impl(
          spacetime_metric, pi, phi, d_spacetime_metric, d_pi, d_phi, gamma0,
          gamma1, gamma2, gauge_function, spacetime_deriv_gauge_function, lapse,
          shift, upper_spatial_metric, upper_spacetime_metric,
          trace_christoffel_first_kind, christoffel_first_kind,
          christoffel_second_kind, normal_vector, normal_one_form);

  tnsr::aa<DataVector, Dim, Frame::Inertial> dt_spacetime_metric(
      mesh.number_of_grid_points());
  tnsr::aa<DataVector, Dim, Frame::Inertial> dt_pi(
      mesh.number_of_grid_points());
  tnsr::iaa<DataVector, Dim, Frame::Inertial> dt_phi(
      mesh.number_of_grid_points());

  Variables<tmpl::list<
      GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1,
      GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma2,
      GeneralizedHarmonic::Tags::GaugeH<Dim>,
      GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<Dim>,
      GeneralizedHarmonic::Tags::Gamma1Gamma2,
      GeneralizedHarmonic::Tags::PiTwoNormals,
      GeneralizedHarmonic::Tags::NormalDotOneIndexConstraint,
      GeneralizedHarmonic::Tags::Gamma1Plus1,
      GeneralizedHarmonic::Tags::PiOneNormal<Dim>,
      GeneralizedHarmonic::Tags::GaugeConstraint<Dim, Frame::Inertial>,
      GeneralizedHarmonic::Tags::PhiTwoNormals<Dim>,
      GeneralizedHarmonic::Tags::ShiftDotThreeIndexConstraint<Dim>,
      GeneralizedHarmonic::Tags::PhiOneNormal<Dim>,
      GeneralizedHarmonic::Tags::PiSecondIndexUp<Dim>,
      GeneralizedHarmonic::Tags::ThreeIndexConstraint<Dim, Frame::Inertial>,
      GeneralizedHarmonic::Tags::PhiFirstIndexUp<Dim>,
      GeneralizedHarmonic::Tags::PhiThirdIndexUp<Dim>,
      GeneralizedHarmonic::Tags::SpacetimeChristoffelFirstKindThirdIndexUp<Dim>,
      gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<Dim, Frame::Inertial, DataVector>,
      gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataVector>,
      gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>,
      gr::Tags::DetSpatialMetric<DataVector>,
      gr::Tags::InverseSpacetimeMetric<Dim, Frame::Inertial, DataVector>,
      gr::Tags::SpacetimeChristoffelFirstKind<Dim, Frame::Inertial, DataVector>,
      gr::Tags::SpacetimeChristoffelSecondKind<Dim, Frame::Inertial,
                                               DataVector>,
      gr::Tags::TraceSpacetimeChristoffelFirstKind<Dim, Frame::Inertial,
                                                   DataVector>,
      gr::Tags::SpacetimeNormalVector<Dim, Frame::Inertial, DataVector>,
      gr::Tags::SpacetimeNormalOneForm<Dim, Frame::Inertial, DataVector>,
      gr::Tags::DerivativesOfSpacetimeMetric<Dim, Frame::Inertial, DataVector>>>
      buffer(mesh.number_of_grid_points());

  GeneralizedHarmonic::TimeDerivative<Dim>::apply(
      make_not_null(&dt_spacetime_metric), make_not_null(&dt_pi),
      make_not_null(&dt_phi),
      make_not_null(
          &get<GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1>(
              buffer)),
      make_not_null(
          &get<GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma2>(
              buffer)),
      make_not_null(&get<GeneralizedHarmonic::Tags::GaugeH<Dim>>(buffer)),
      make_not_null(
          &get<GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<Dim>>(buffer)),
      make_not_null(&get<GeneralizedHarmonic::Tags::Gamma1Gamma2>(buffer)),
      make_not_null(&get<GeneralizedHarmonic::Tags::PiTwoNormals>(buffer)),
      make_not_null(
          &get<GeneralizedHarmonic::Tags::NormalDotOneIndexConstraint>(buffer)),
      make_not_null(&get<GeneralizedHarmonic::Tags::Gamma1Plus1>(buffer)),
      make_not_null(&get<GeneralizedHarmonic::Tags::PiOneNormal<Dim>>(buffer)),
      make_not_null(
          &get<
              GeneralizedHarmonic::Tags::GaugeConstraint<Dim, Frame::Inertial>>(
              buffer)),
      make_not_null(
          &get<GeneralizedHarmonic::Tags::PhiTwoNormals<Dim>>(buffer)),
      make_not_null(
          &get<GeneralizedHarmonic::Tags::ShiftDotThreeIndexConstraint<Dim>>(
              buffer)),
      make_not_null(&get<GeneralizedHarmonic::Tags::PhiOneNormal<Dim>>(buffer)),
      make_not_null(
          &get<GeneralizedHarmonic::Tags::PiSecondIndexUp<Dim>>(buffer)),
      make_not_null(&get<GeneralizedHarmonic::Tags::ThreeIndexConstraint<
                        Dim, Frame::Inertial>>(buffer)),
      make_not_null(
          &get<GeneralizedHarmonic::Tags::PhiFirstIndexUp<Dim>>(buffer)),
      make_not_null(
          &get<GeneralizedHarmonic::Tags::PhiThirdIndexUp<Dim>>(buffer)),
      make_not_null(
          &get<GeneralizedHarmonic::Tags::
                   SpacetimeChristoffelFirstKindThirdIndexUp<Dim>>(buffer)),
      make_not_null(&get<gr::Tags::Lapse<DataVector>>(buffer)),
      make_not_null(
          &get<gr::Tags::Shift<Dim, Frame::Inertial, DataVector>>(buffer)),
      make_not_null(
          &get<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataVector>>(
              buffer)),
      make_not_null(&get<gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial,
                                                        DataVector>>(buffer)),
      make_not_null(&get<gr::Tags::DetSpatialMetric<DataVector>>(buffer)),
      make_not_null(&get<gr::Tags::InverseSpacetimeMetric<Dim, Frame::Inertial,
                                                          DataVector>>(buffer)),
      make_not_null(
          &get<gr::Tags::SpacetimeChristoffelFirstKind<Dim, Frame::Inertial,
                                                       DataVector>>(buffer)),
      make_not_null(
          &get<gr::Tags::SpacetimeChristoffelSecondKind<Dim, Frame::Inertial,
                                                        DataVector>>(buffer)),
      make_not_null(&get<gr::Tags::TraceSpacetimeChristoffelFirstKind<
                        Dim, Frame::Inertial, DataVector>>(buffer)),
      make_not_null(&get<gr::Tags::SpacetimeNormalVector<Dim, Frame::Inertial,
                                                         DataVector>>(buffer)),
      make_not_null(&get<gr::Tags::SpacetimeNormalOneForm<Dim, Frame::Inertial,
                                                          DataVector>>(buffer)),
      make_not_null(
          &get<gr::Tags::DerivativesOfSpacetimeMetric<Dim, Frame::Inertial,
                                                      DataVector>>(buffer)),
      d_spacetime_metric, d_pi, d_phi, spacetime_metric, pi, phi, gamma0,
      gamma1, gamma2, gauge_function, spacetime_deriv_gauge_function);

  CHECK_ITERABLE_APPROX(
      get<GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1>(
          buffer),
      gamma1);
  CHECK_ITERABLE_APPROX(
      get<GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma2>(
          buffer),
      gamma2);
  CHECK_ITERABLE_APPROX(get<GeneralizedHarmonic::Tags::GaugeH<Dim>>(buffer),
                        gauge_function);
  CHECK_ITERABLE_APPROX(
      get<GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<Dim>>(buffer),
      spacetime_deriv_gauge_function);

  CHECK_ITERABLE_APPROX(expected_dt_spacetime_metric, dt_spacetime_metric);
  CHECK_ITERABLE_APPROX(expected_dt_pi, dt_pi);
  CHECK_ITERABLE_APPROX(expected_dt_phi, dt_phi);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.DuDt",
                  "[Unit][GeneralizedHarmonic]") {
  test_reference_impl_against_spec();

  MAKE_GENERATOR(generator);
  test_compute_dudt<1>(make_not_null(&generator));
  test_compute_dudt<2>(make_not_null(&generator));
  test_compute_dudt<3>(make_not_null(&generator));
}
