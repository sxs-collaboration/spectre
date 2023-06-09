// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphericalKerrSchild.hpp"

#include <cmath>  // IWYU pragma: keep
#include <numeric>
#include <ostream>
#include <typeinfo>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/CrossProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/ParseError.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/StdHelpers.hpp"

namespace gr::Solutions {

SphericalKerrSchild::SphericalKerrSchild(
    const double mass, SphericalKerrSchild::Spin::type dimensionless_spin,
    SphericalKerrSchild::Center::type center, const Options::Context& context)

    : mass_(mass),
      // NOLINTNEXTLINE(performance-move-const-arg)
      dimensionless_spin_(std::move(dimensionless_spin)),
      // NOLINTNEXTLINE(performance-move-const-arg)
      center_(std::move(center)) {
  const double spin_magnitude = magnitude(dimensionless_spin_);
  if (spin_magnitude > 1.) {
    PARSE_ERROR(context, "Spin magnitude must be < 1. Given spin: "
                             << dimensionless_spin_ << " with magnitude "
                             << spin_magnitude);
  }
  if (mass_ <= 0.) {
    PARSE_ERROR(context, "Mass must be > 0. Given mass: " << mass_);
  }
}

SphericalKerrSchild::SphericalKerrSchild(CkMigrateMessage* /*unused*/) {}

void SphericalKerrSchild::pup(PUP::er& p) {
  p | mass_;
  p | dimensionless_spin_;
  p | center_;
}

template <typename DataType, typename Frame>
SphericalKerrSchild::IntermediateComputer<
    DataType, Frame>::IntermediateComputer(const SphericalKerrSchild& solution,
                                           const tnsr::I<DataType, 3, Frame>& x)
    : solution_(solution), x_(x) {}

namespace {
auto dimensionful_spin(const SphericalKerrSchild& solution) {
  return solution.dimensionless_spin() * solution.mass();
}

auto spin_a_and_squared(const SphericalKerrSchild& solution) {
  const auto spin_a = dimensionful_spin(solution);
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
  return std::make_tuple(spin_a, a_squared);
}
}  // namespace

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_minus_center,
    const gsl::not_null<CachedBuffer*> /*cache*/,
    internal_tags::x_minus_center<DataType, Frame> /*meta*/) const {
  *x_minus_center = x_;
  for (size_t i = 0; i < 3; ++i) {
    x_minus_center->get(i) -= gsl::at(solution_.center(), i);
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> r_squared,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::r_squared<DataType> /*meta*/) const {
  const auto& x_minus_center =
      cache->get_var(*this, internal_tags::x_minus_center<DataType, Frame>{});

  get(*r_squared) = square(get<0>(x_minus_center)) +
                    square(get<1>(x_minus_center)) +
                    square(get<2>(x_minus_center));
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> r,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::r<DataType> /*meta*/) const {
  const auto& r_squared =
      get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));

  get(*r) = sqrt(r_squared);
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> rho,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::rho<DataType> /*meta*/) const {
  const auto& r_squared =
      get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));
  const auto [spin_a, a_squared] = spin_a_and_squared(solution_);

  get(*rho) = sqrt(r_squared + a_squared);
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> helper_matrix_F,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::helper_matrix_F<DataType, Frame> /*meta*/) const {
  const auto [spin_a, a_squared] = spin_a_and_squared(solution_);
  const auto& rho = get(cache->get_var(*this, internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      helper_matrix_F->get(i, j) = -1. / (rho * cube(r));
      if (i == j) {  // Kronecker delta
        helper_matrix_F->get(i, j) *=
            (a_squared - gsl::at(spin_a, i) * gsl::at(spin_a, j));
      } else {
        helper_matrix_F->get(i, j) *= -gsl::at(spin_a, i) * gsl::at(spin_a, j);
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> transformation_matrix_P,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::transformation_matrix_P<DataType, Frame> /*meta*/) const {
  const auto spin_a = dimensionful_spin(solution_);
  const auto& rho = get(cache->get_var(*this, internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      transformation_matrix_P->get(i, j) =
          -(gsl::at(spin_a, i) * gsl::at(spin_a, j)) / (r * (rho + r));
      if (i == j) {  // Kronecker delta
        transformation_matrix_P->get(i, j) += rho / r;
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> jacobian,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::jacobian<DataType, Frame> /*meta*/) const {
  const auto& x_minus_center =
      cache->get_var(*this, internal_tags::x_minus_center<DataType, Frame>{});
  const auto& transformation_matrix_P = cache->get_var(
      *this, internal_tags::transformation_matrix_P<DataType, Frame>{});
  const auto& helper_matrix_F =
      cache->get_var(*this, internal_tags::helper_matrix_F<DataType, Frame>{});

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      jacobian->get(j, i) = transformation_matrix_P.get(i, j);
      for (size_t k = 0; k < 3; ++k) {
        jacobian->get(j, i) += helper_matrix_F.get(i, k) *
                               x_minus_center.get(k) * x_minus_center.get(j);
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> helper_matrix_D,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::helper_matrix_D<DataType, Frame> /*meta*/) const {
  const auto [spin_a, a_squared] = spin_a_and_squared(solution_);
  const auto& rho = get(cache->get_var(*this, internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      helper_matrix_D->get(i, j) = 1. / (r * cube(rho));
      if (i == j) {  // Kronecker delta
        helper_matrix_D->get(i, j) *=
            (a_squared - gsl::at(spin_a, i) * gsl::at(spin_a, j));
      } else {
        helper_matrix_D->get(i, j) *= -gsl::at(spin_a, i) * gsl::at(spin_a, j);
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> helper_matrix_C,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::helper_matrix_C<DataType, Frame> /*meta*/) const {
  const auto& helper_matrix_F =
      cache->get_var(*this, internal_tags::helper_matrix_F<DataType, Frame>{});
  const auto& helper_matrix_D =
      cache->get_var(*this, internal_tags::helper_matrix_D<DataType, Frame>{});

  for (size_t i = 0; i < 3; ++i) {
    for (size_t m = 0; m < 3; ++m) {
      helper_matrix_C->get(i, m) =
          helper_matrix_D.get(i, m) - 3. * helper_matrix_F.get(i, m);
    }
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::iJk<DataType, 3, Frame>*> deriv_jacobian,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::deriv_jacobian<DataType, Frame> /*meta*/) const {
  const auto& helper_matrix_C =
      cache->get_var(*this, internal_tags::helper_matrix_C<DataType, Frame>{});
  const auto& helper_matrix_F =
      cache->get_var(*this, internal_tags::helper_matrix_F<DataType, Frame>{});
  const auto& x_minus_center =
      cache->get_var(*this, internal_tags::x_minus_center<DataType, Frame>{});
  const auto& r_squared =
      get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));

  for (size_t k = 0; k < 3; ++k) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t i = 0; i < 3; ++i) {
        deriv_jacobian->get(k, j, i) =
            helper_matrix_F.get(i, j) * x_minus_center.get(k) +
            helper_matrix_F.get(i, k) * x_minus_center.get(j);
        for (size_t m = 0; m < 3; ++m) {
          if (j == k) {  // Kronecker delta
            deriv_jacobian->get(k, j, i) +=
                helper_matrix_F.get(i, m) * x_minus_center.get(m);
          }
          deriv_jacobian->get(k, j, i) +=
              helper_matrix_C.get(i, m) * x_minus_center.get(k) *
              x_minus_center.get(m) * x_minus_center.get(j) / r_squared;
        }
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> transformation_matrix_Q,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::transformation_matrix_Q<DataType, Frame> /*meta*/) const {
  const auto spin_a = dimensionful_spin(solution_);
  const auto& rho = get(cache->get_var(*this, internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      transformation_matrix_Q->get(i, j) =
          (gsl::at(spin_a, i) * gsl::at(spin_a, j)) / ((rho + r) * rho);
      if (i == j) {  // Kronecker delta
        transformation_matrix_Q->get(i, j) += r / rho;
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> helper_matrix_G1,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::helper_matrix_G1<DataType, Frame> /*meta*/) const {
  const auto [spin_a, a_squared] = spin_a_and_squared(solution_);
  const auto& rho = get(cache->get_var(*this, internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));

  for (size_t i = 0; i < 3; ++i) {
    for (size_t m = 0; m < 3; ++m) {
      helper_matrix_G1->get(i, m) = 1. / (r * square(rho));
      if (i == m) {  // Kronecker delta
        helper_matrix_G1->get(i, m) *=
            (a_squared - gsl::at(spin_a, i) * gsl::at(spin_a, m));
      } else {
        helper_matrix_G1->get(i, m) *= -gsl::at(spin_a, i) * gsl::at(spin_a, m);
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> a_dot_x,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::a_dot_x<DataType> /*meta*/) const {
  const auto& x_minus_center =
      cache->get_var(*this, internal_tags::x_minus_center<DataType, Frame>{});
  const auto spin_a = dimensionful_spin(solution_);

  get(*a_dot_x) = spin_a[0] * get<0>(x_minus_center) +
                  spin_a[1] * get<1>(x_minus_center) +
                  spin_a[2] * get<2>(x_minus_center);
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> s_number,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::s_number<DataType> /*meta*/) const {
  const auto& r_squared =
      get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));
  const auto& a_dot_x =
      get(cache->get_var(*this, internal_tags::a_dot_x<DataType>{}));

  get(*s_number) = r_squared + square(a_dot_x) / r_squared;
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> helper_matrix_G2,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::helper_matrix_G2<DataType, Frame> /*meta*/) const {
  const auto& transformation_matrix_Q = cache->get_var(
      *this, internal_tags::transformation_matrix_Q<DataType, Frame>{});
  const auto& rho = get(cache->get_var(*this, internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
  const auto& s_number =
      get(cache->get_var(*this, internal_tags::s_number<DataType>{}));

  for (size_t n = 0; n < 3; ++n) {
    for (size_t j = 0; j < 3; ++j) {
      helper_matrix_G2->get(n, j) =
          square(rho) / (s_number * r) * transformation_matrix_Q.get(n, j);
    }
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> G1_dot_x,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::G1_dot_x<DataType, Frame> /*meta*/) const {
  const auto& x_minus_center =
      cache->get_var(*this, internal_tags::x_minus_center<DataType, Frame>{});
  const auto& helper_matrix_G1 =
      cache->get_var(*this, internal_tags::helper_matrix_G1<DataType, Frame>{});

  for (size_t i = 0; i < 3; ++i) {
    G1_dot_x->get(i) = helper_matrix_G1.get(i, 0) * x_minus_center.get(0);
    for (size_t m = 1; m < 3; ++m) {
      G1_dot_x->get(i) += helper_matrix_G1.get(i, m) * x_minus_center.get(m);
    }
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3, Frame>*> G2_dot_x,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::G2_dot_x<DataType, Frame> /*meta*/) const {
  const auto& x_minus_center =
      cache->get_var(*this, internal_tags::x_minus_center<DataType, Frame>{});
  const auto& helper_matrix_G2 =
      cache->get_var(*this, internal_tags::helper_matrix_G2<DataType, Frame>{});

  for (size_t j = 0; j < 3; ++j) {
    G2_dot_x->get(j) = helper_matrix_G2.get(0, j) * x_minus_center.get(0);
    for (size_t n = 1; n < 3; ++n) {
      G2_dot_x->get(j) += helper_matrix_G2.get(n, j) * x_minus_center.get(n);
    }
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> inv_jacobian,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::inv_jacobian<DataType, Frame> /*meta*/) const {
  const auto& transformation_matrix_Q = cache->get_var(
      *this, internal_tags::transformation_matrix_Q<DataType, Frame>{});
  const auto& G1_dot_x =
      cache->get_var(*this, internal_tags::G1_dot_x<DataType, Frame>{});
  const auto& G2_dot_x =
      cache->get_var(*this, internal_tags::G2_dot_x<DataType, Frame>{});

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      inv_jacobian->get(j, i) =
          transformation_matrix_Q.get(i, j) + G1_dot_x.get(i) * G2_dot_x.get(j);
    }
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> helper_matrix_E1,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::helper_matrix_E1<DataType, Frame> /*meta*/) const {
  const auto& rho = get(cache->get_var(*this, internal_tags::rho<DataType>{}));
  const auto& r_squared =
      get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));
  const auto [spin_a, a_squared] = spin_a_and_squared(solution_);

  for (size_t i = 0; i < 3; ++i) {
    for (size_t m = 0; m < 3; ++m) {
      helper_matrix_E1->get(i, m) =
          -(square(rho) + 2. * r_squared) / (r_squared * pow<4>(rho));
      if (i == m) {  // Kronecker delta
        helper_matrix_E1->get(i, m) *=
            (a_squared - gsl::at(spin_a, i) * gsl::at(spin_a, m));
      } else {
        helper_matrix_E1->get(i, m) *= -gsl::at(spin_a, i) * gsl::at(spin_a, m);
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> helper_matrix_E2,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::helper_matrix_E2<DataType, Frame> /*meta*/) const {
  const auto& rho = get(cache->get_var(*this, internal_tags::rho<DataType>{}));
  const auto [spin_a, a_squared] = spin_a_and_squared(solution_);
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
  const auto& a_dot_x =
      get(cache->get_var(*this, internal_tags::a_dot_x<DataType>{}));
  const auto& s_number =
      get(cache->get_var(*this, internal_tags::s_number<DataType>{}));
  const auto& helper_matrix_G2 =
      cache->get_var(*this, internal_tags::helper_matrix_G2<DataType, Frame>{});
  const auto& transformation_matrix_P = cache->get_var(
      *this, internal_tags::transformation_matrix_P<DataType, Frame>{});

  for (size_t n = 0; n < 3; ++n) {
    for (size_t j = 0; j < 3; ++j) {
      helper_matrix_E2->get(n, j) =
          (1. / s_number) * transformation_matrix_P.get(n, j);
      helper_matrix_E2->get(n, j) +=
          ((-a_squared / (square(rho) * r)) -
           (2. / s_number) * (r - (square(a_dot_x) / cube(r)))) *
          helper_matrix_G2.get(n, j);
    }
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::iJk<DataType, 3, Frame>*> deriv_inv_jacobian,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::deriv_inv_jacobian<DataType, Frame> /*meta*/) const {
  const auto spin_a = dimensionful_spin(solution_);
  const auto& helper_matrix_D =
      cache->get_var(*this, internal_tags::helper_matrix_D<DataType, Frame>{});
  const auto& helper_matrix_G1 =
      cache->get_var(*this, internal_tags::helper_matrix_G1<DataType, Frame>{});
  const auto& helper_matrix_G2 =
      cache->get_var(*this, internal_tags::helper_matrix_G2<DataType, Frame>{});
  const auto& helper_matrix_E1 =
      cache->get_var(*this, internal_tags::helper_matrix_E1<DataType, Frame>{});
  const auto& helper_matrix_E2 =
      cache->get_var(*this, internal_tags::helper_matrix_E2<DataType, Frame>{});
  const auto& x_minus_center =
      cache->get_var(*this, internal_tags::x_minus_center<DataType, Frame>{});
  const auto& a_dot_x =
      get(cache->get_var(*this, internal_tags::a_dot_x<DataType>{}));
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
  const auto& G1_dot_x =
      cache->get_var(*this, internal_tags::G1_dot_x<DataType, Frame>{});
  const auto& G2_dot_x =
      cache->get_var(*this, internal_tags::G2_dot_x<DataType, Frame>{});
  const auto& s_number =
      get(cache->get_var(*this, internal_tags::s_number<DataType>{}));

  for (size_t k = 0; k < 3; ++k) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t i = 0; i < 3; ++i) {
        deriv_inv_jacobian->get(k, j, i) =
            helper_matrix_D.get(i, j) * x_minus_center.get(k) +
            helper_matrix_G1.get(i, k) * G2_dot_x.get(j) +
            helper_matrix_G2.get(k, j) * G1_dot_x.get(i) -
            2. * a_dot_x * gsl::at(spin_a, k) * G1_dot_x.get(i) *
                G2_dot_x.get(j) / (s_number * square(r));
        for (size_t m = 0; m < 3; ++m) {
          deriv_inv_jacobian->get(k, j, i) +=
              (helper_matrix_E1.get(i, m) * x_minus_center.get(m) *
                   G2_dot_x.get(j) * x_minus_center.get(k) +
               G1_dot_x.get(i) * x_minus_center.get(k) * x_minus_center.get(m) *
                   helper_matrix_E2.get(m, j)) /
              r;
        }
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> H,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::H<DataType> /*meta*/) const {
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
  const auto& a_dot_x =
      get(cache->get_var(*this, internal_tags::a_dot_x<DataType>{}));

  get(*H) = solution_.mass() * cube(r) / (pow<4>(r) + square(a_dot_x));
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> kerr_schild_x,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::kerr_schild_x<DataType, Frame> /*meta*/) const {
  const auto& x_minus_center =
      cache->get_var(*this, internal_tags::x_minus_center<DataType, Frame>{});
  const auto& transformation_matrix_P = cache->get_var(
      *this, internal_tags::transformation_matrix_P<DataType, Frame>{});

  for (size_t i = 0; i < 3; ++i) {
    kerr_schild_x->get(i) =
        transformation_matrix_P.get(i, 0) * x_minus_center.get(0);
    for (size_t j = 1; j < 3; ++j) {
      kerr_schild_x->get(i) +=
          transformation_matrix_P.get(i, j) * x_minus_center.get(j);
    }
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> a_cross_x,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::a_cross_x<DataType, Frame> /*meta*/) const {
  const auto spin_a = dimensionful_spin(solution_);
  const tnsr::I<DataType, 3, Frame>& kerr_schild_x =
      cache->get_var(*this, internal_tags::kerr_schild_x<DataType, Frame>{});

  // temp_spin used to convert the spin from type array to type tnsr
  auto temp_spin = make_with_value<tnsr::I<DataType, 3, Frame>>(
      get_size(get_element(kerr_schild_x, 0)), 0.);

  for (size_t i = 0; i < 3; ++i) {
    temp_spin[i] = gsl::at(spin_a, i);
  }
  *a_cross_x = cross_product(temp_spin, kerr_schild_x);
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> kerr_schild_l,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::kerr_schild_l<DataType, Frame> /*meta*/) const {
  const auto spin_a = dimensionful_spin(solution_);
  const auto& a_dot_x =
      get(cache->get_var(*this, internal_tags::a_dot_x<DataType>{}));
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
  const auto& a_cross_x =
      cache->get_var(*this, internal_tags::a_cross_x<DataType, Frame>{});
  const auto& rho = get(cache->get_var(*this, internal_tags::rho<DataType>{}));
  const auto& kerr_schild_x =
      cache->get_var(*this, internal_tags::kerr_schild_x<DataType, Frame>{});

  const auto sqr_inv_rho = 1. / square(rho);

  for (int i = 0; i < 3; ++i) {
    kerr_schild_l->get(i) =
        sqr_inv_rho * (r * kerr_schild_x.get(i) +
                       a_dot_x * gsl::at(spin_a, i) / r - a_cross_x.get(i));
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::i<DataType, 4, Frame>*> l_lower,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::l_lower<DataType, Frame> /*meta*/) const {
  const auto& kerr_schild_l =
      cache->get_var(*this, internal_tags::kerr_schild_l<DataType, Frame>{});
  const auto& jacobian =
      cache->get_var(*this, internal_tags::jacobian<DataType, Frame>{});

  l_lower->get(0) = 1.;  // this is l_t

  for (size_t j = 0; j < 3; ++j) {
    l_lower->get(j + 1) = jacobian.get(j, 0) * kerr_schild_l.get(0);
    for (size_t i = 1; i < 3; ++i) {
      l_lower->get(j + 1) += jacobian.get(j, i) * kerr_schild_l.get(i);
    }
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 4, Frame>*> l_upper,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::l_upper<DataType, Frame> /*meta*/) const {
  const auto& kerr_schild_l =
      cache->get_var(*this, internal_tags::kerr_schild_l<DataType, Frame>{});
  const auto& inv_jacobian =
      cache->get_var(*this, internal_tags::inv_jacobian<DataType, Frame>{});

  l_upper->get(0) = -1.;  // this is l^t

  for (size_t j = 0; j < 3; ++j) {
    l_upper->get(j + 1) = inv_jacobian.get(0, j) * kerr_schild_l.get(0);
    for (size_t i = 1; i < 3; ++i) {
      l_upper->get(j + 1) += inv_jacobian.get(i, j) * kerr_schild_l.get(i);
    }
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> deriv_r,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::deriv_r<DataType, Frame> /*meta*/) const {
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
  const auto& a_dot_x =
      get(cache->get_var(*this, internal_tags::a_dot_x<DataType>{}));
  const auto& kerr_schild_x =
      cache->get_var(*this, internal_tags::kerr_schild_x<DataType, Frame>{});
  const auto& H = cache->get_var(*this, internal_tags::H<DataType>{});
  const auto spin_a = dimensionful_spin(solution_);

  // temp_mass used to convert mass from type double to Scalar
  const auto temp_mass = make_with_value<Scalar<DataType>>(
      get_size(get_element(kerr_schild_x, 0)), solution_.mass());
  const auto deriv_r_denom = get(H) / get(temp_mass);

  for (size_t i = 0; i < 3; ++i) {
    deriv_r->get(i) =
        deriv_r_denom *
        (kerr_schild_x.get(i) + a_dot_x * gsl::at(spin_a, i) / square(r));
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 4, Frame>*> deriv_H,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::deriv_H<DataType, Frame> /*meta*/) const {
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
  const auto& a_dot_x =
      get(cache->get_var(*this, internal_tags::a_dot_x<DataType>{}));
  const auto& H = cache->get_var(*this, internal_tags::H<DataType>{});
  const auto& jacobian =
      cache->get_var(*this, internal_tags::jacobian<DataType, Frame>{});
  const auto spin_a = dimensionful_spin(solution_);
  const auto& deriv_r =
      cache->get_var(*this, internal_tags::deriv_r<DataType, Frame>{});

  const auto H_denom = 1. / (pow<4>(r) + square(a_dot_x));
  const auto factor = 3. / r - 4. * cube(r) * H_denom;

  deriv_H->get(0) = 0.;  // set time component to 0
  for (size_t i = 0; i < 3; ++i) {
    deriv_H->get(i + 1) =
        get(H) *
        (factor * deriv_r.get(i) - 2. * H_denom * a_dot_x * gsl::at(spin_a, i));
  }

  // Explicitly copy because we modify components in loop below
  const auto deriv_H_x = deriv_H->get(1);
  const auto deriv_H_y = deriv_H->get(2);
  const auto deriv_H_z = deriv_H->get(3);

  for (size_t j = 0; j < 3; ++j) {
    deriv_H->get(j + 1) = jacobian.get(j, 0) * deriv_H_x +
                          jacobian.get(j, 1) * deriv_H_y +
                          jacobian.get(j, 2) * deriv_H_z;
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ij<DataType, 4, Frame>*> kerr_schild_deriv_l,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::kerr_schild_deriv_l<DataType, Frame> /*meta*/) const {
  const auto& kerr_schild_x =
      cache->get_var(*this, internal_tags::kerr_schild_x<DataType, Frame>{});
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
  const auto& a_dot_x =
      get(cache->get_var(*this, internal_tags::a_dot_x<DataType>{}));
  const auto& rho = get(cache->get_var(*this, internal_tags::rho<DataType>{}));
  const auto spin_a = dimensionful_spin(solution_);
  const auto& kerr_schild_l =
      cache->get_var(*this, internal_tags::kerr_schild_l<DataType, Frame>{});
  const auto& deriv_r =
      cache->get_var(*this, internal_tags::deriv_r<DataType, Frame>{});

  const auto sqr_inv_rho = 1. / square(rho);

  kerr_schild_deriv_l->get(0, 0) = 0.;  // set first time component to 0
  for (size_t i = 0; i < 3; ++i) {      // set remaining time components to 0
    kerr_schild_deriv_l->get(i + 1, 0) = 0.;
    kerr_schild_deriv_l->get(0, i + 1) = 0.;
    for (size_t j = 0; j < 3; ++j) {
      kerr_schild_deriv_l->get(j + 1, i + 1) =
          sqr_inv_rho * ((kerr_schild_x.get(i) - 2. * r * kerr_schild_l.get(i) -
                          a_dot_x * gsl::at(spin_a, i) / square(r)) *
                             deriv_r.get(j) +
                         gsl::at(spin_a, i) * gsl::at(spin_a, j) / r);
      if (i == j) {
        kerr_schild_deriv_l->get(j + 1, i + 1) += sqr_inv_rho * r;
      } else {  // add sqr_inv_rho*epsilon^ijk a_k
        size_t k = (j + 1) % 3;
        if (k == i) {  // j+1 = i (cyclic), so choose minus sign
          ++k;
          k %= 3;  // and set k to be neither i nor j
          kerr_schild_deriv_l->get(j + 1, i + 1) -=
              sqr_inv_rho * gsl::at(spin_a, k);
        } else {  // i+1 = j (cyclic), so choose plus sign
          kerr_schild_deriv_l->get(j + 1, i + 1) +=
              sqr_inv_rho * gsl::at(spin_a, k);
        }
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ij<DataType, 4, Frame>*> deriv_l,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::deriv_l<DataType, Frame> /*meta*/) const {
  const auto& kerr_schild_l =
      cache->get_var(*this, internal_tags::kerr_schild_l<DataType, Frame>{});
  const auto& jacobian =
      cache->get_var(*this, internal_tags::jacobian<DataType, Frame>{});
  const auto& deriv_jacobian =
      cache->get_var(*this, internal_tags::deriv_jacobian<DataType, Frame>{});
  const auto& kerr_schild_deriv_l = cache->get_var(
      *this, internal_tags::kerr_schild_deriv_l<DataType, Frame>{});

  deriv_l->get(0, 0) = 0.;          // set first time component to 0
  for (size_t i = 0; i < 3; ++i) {  // set remaining time components to 0
    deriv_l->get(i + 1, 0) = 0.;
    deriv_l->get(0, i + 1) = 0.;
    for (size_t j = 0; j < 3; ++j) {
      deriv_l->get(j + 1, i + 1) = 0.;
      for (size_t k = 0; k < 3; ++k) {
        for (size_t m = 0; m < 3; ++m) {
          deriv_l->get(j + 1, i + 1) += jacobian.get(i, k) *
                                        jacobian.get(j, m) *
                                        kerr_schild_deriv_l.get(m + 1, k + 1);
        }
        deriv_l->get(j + 1, i + 1) +=
            kerr_schild_l.get(k) * deriv_jacobian.get(j, i, k);
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> lapse_squared,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::lapse_squared<DataType> /*meta*/) const {
  const auto& H = get(cache->get_var(*this, internal_tags::H<DataType>{}));
  const auto& l_upper =
      cache->get_var(*this, internal_tags::l_upper<DataType, Frame>{});

  get(*lapse_squared) = 1. / (1. + 2. * square(l_upper.get(0)) * H);
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> lapse,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::Lapse<DataType> /*meta*/) const {
  const auto& lapse_squared =
      get(cache->get_var(*this, internal_tags::lapse_squared<DataType>{}));

  get(*lapse) = sqrt(lapse_squared);
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> deriv_lapse_multiplier,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::deriv_lapse_multiplier<DataType> /*meta*/) const {
  const auto& lapse = get(cache->get_var(*this, gr::Tags::Lapse<DataType>{}));
  const auto& lapse_squared =
      get(cache->get_var(*this, internal_tags::lapse_squared<DataType>{}));

  get(*deriv_lapse_multiplier) =
      -square(null_vector_0_) * lapse * lapse_squared;
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> shift_multiplier,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::shift_multiplier<DataType> /*meta*/) const {
  const auto& H = get(cache->get_var(*this, internal_tags::H<DataType>{}));
  const auto& lapse_squared =
      get(cache->get_var(*this, internal_tags::lapse_squared<DataType>{}));

  get(*shift_multiplier) = -2. * null_vector_0_ * H * lapse_squared;
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> shift,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::Shift<DataType, 3, Frame> /*meta*/) const {
  const auto& l_upper =
      cache->get_var(*this, internal_tags::l_upper<DataType, Frame>{});
  const auto& shift_multiplier =
      get(cache->get_var(*this, internal_tags::shift_multiplier<DataType>{}));

  for (size_t i = 0; i < 3; ++i) {
    shift->get(i) = shift_multiplier * l_upper.get(i + 1);
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::iJ<DataType, 3, Frame>*> deriv_shift,
    const gsl::not_null<CachedBuffer*> cache,
    DerivShift<DataType, Frame> /*meta*/) const {
  const auto& H = get(cache->get_var(*this, internal_tags::H<DataType>{}));
  const auto& l_upper =
      cache->get_var(*this, internal_tags::l_upper<DataType, Frame>{});
  const auto& l_lower =
      cache->get_var(*this, internal_tags::l_lower<DataType, Frame>{});
  const auto& lapse_squared =
      get(cache->get_var(*this, internal_tags::lapse_squared<DataType>{}));
  const auto& deriv_H =
      cache->get_var(*this, internal_tags::deriv_H<DataType, Frame>{});
  const auto& deriv_l =
      cache->get_var(*this, internal_tags::deriv_l<DataType, Frame>{});
  const auto& inv_jacobian =
      cache->get_var(*this, internal_tags::inv_jacobian<DataType, Frame>{});
  const auto& deriv_inv_jacobian = cache->get_var(
      *this, internal_tags::deriv_inv_jacobian<DataType, Frame>{});

  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) {
      deriv_shift->get(k, i) =
          4. * H * l_upper.get(0) * l_upper.get(i + 1) * square(lapse_squared) *
              (square(l_upper.get(0)) * deriv_H.get(k + 1) +
               2. * H * l_upper.get(0) * deriv_l.get(k + 1, 0)) -
          2. * lapse_squared *
              (l_upper.get(0) * l_upper.get(i + 1) * deriv_H.get(k + 1) +
               H * l_upper.get(i + 1) * deriv_l.get(k + 1, 0));
      for (int j = 0; j < 3; ++j) {
        for (int m = 0; m < 3; ++m) {
          deriv_shift->get(k, i) +=
              -2. * lapse_squared * H * l_upper.get(0) *
              (inv_jacobian.get(j, i) * inv_jacobian.get(j, m) *
                   deriv_l.get(k + 1, m + 1) +
               inv_jacobian.get(j, i) * l_lower.get(m + 1) *
                   deriv_inv_jacobian.get(k, j, m) +
               inv_jacobian.get(j, m) * l_lower.get(m + 1) *
                   deriv_inv_jacobian.get(k, j, i));
        }
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> spatial_metric,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::SpatialMetric<DataType, 3, Frame> /*meta*/) const {
  const auto& H = get(cache->get_var(*this, internal_tags::H<DataType>{}));
  const auto& l_lower =
      cache->get_var(*this, internal_tags::l_lower<DataType, Frame>{});
  const auto& jacobian =
      cache->get_var(*this, internal_tags::jacobian<DataType, Frame>{});

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {  // Symmetry
      spatial_metric->get(i, j) =
          2. * H * l_lower.get(i + 1) * l_lower.get(j + 1);
      for (size_t m = 0; m < 3; ++m) {
        spatial_metric->get(i, j) += jacobian.get(i, m) * jacobian.get(j, m);
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, 3, Frame>*> deriv_spatial_metric,
    const gsl::not_null<CachedBuffer*> cache,
    DerivSpatialMetric<DataType, Frame> /*meta*/) const {
  const auto& l_lower =
      cache->get_var(*this, internal_tags::l_lower<DataType, Frame>{});
  const auto& deriv_H =
      cache->get_var(*this, internal_tags::deriv_H<DataType, Frame>{});
  const auto& H = get(cache->get_var(*this, internal_tags::H<DataType>{}));
  const auto& deriv_l =
      cache->get_var(*this, internal_tags::deriv_l<DataType, Frame>{});
  const auto& jacobian =
      cache->get_var(*this, internal_tags::jacobian<DataType, Frame>{});
  const auto& deriv_jacobian =
      cache->get_var(*this, internal_tags::deriv_jacobian<DataType, Frame>{});

  for (int k = 0; k < 3; ++k) {
    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {  // Symmetry
        deriv_spatial_metric->get(k, i, j) =
            2. * l_lower.get(i + 1) * l_lower.get(j + 1) * deriv_H.get(k + 1) +
            2. * H *
                (l_lower.get(i + 1) * deriv_l.get(k + 1, j + 1) +
                 l_lower.get(j + 1) * deriv_l.get(k + 1, i + 1));
        for (int m = 0; m < 3; ++m) {
          deriv_spatial_metric->get(k, i, j) +=
              deriv_jacobian.get(k, i, m) * jacobian.get(j, m) +
              deriv_jacobian.get(k, j, m) * jacobian.get(i, m);
        }
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> dt_spatial_metric,
    const gsl::not_null<CachedBuffer*> /*cache*/,
    ::Tags::dt<gr::Tags::SpatialMetric<DataType, 3, Frame>> /*meta*/) const {
  std::fill(dt_spatial_metric->begin(), dt_spatial_metric->end(), 0.);
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::II<DataType, 3, Frame>*> inverse_spatial_metric,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::InverseSpatialMetric<DataType, 3, Frame> /*meta*/) const {
  const auto& H = get(cache->get_var(*this, internal_tags::H<DataType>{}));
  const auto& lapse_squared =
      get(cache->get_var(*this, internal_tags::lapse_squared<DataType>{}));
  const auto& l_upper =
      cache->get_var(*this, internal_tags::l_upper<DataType, Frame>{});
  const auto& inv_jacobian =
      cache->get_var(*this, internal_tags::inv_jacobian<DataType, Frame>{});
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {  // Symmetry
      inverse_spatial_metric->get(i, j) =
          -2. * H * lapse_squared * l_upper.get(i + 1) * l_upper.get(j + 1);
      for (size_t m = 0; m < 3; ++m) {
        inverse_spatial_metric->get(i, j) +=
            inv_jacobian.get(m, i) * inv_jacobian.get(m, j);
      }
    }
  }
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> extrinsic_curvature,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::ExtrinsicCurvature<DataType, 3, Frame> /*meta*/) const {
  gr::extrinsic_curvature(
      extrinsic_curvature, cache->get_var(*this, gr::Tags::Lapse<DataType>{}),
      cache->get_var(*this, gr::Tags::Shift<DataType, 3, Frame>{}),
      cache->get_var(*this, DerivShift<DataType, Frame>{}),
      cache->get_var(*this, gr::Tags::SpatialMetric<DataType, 3, Frame>{}),
      cache->get_var(*this,
                     ::Tags::dt<gr::Tags::SpatialMetric<DataType, 3, Frame>>{}),
      cache->get_var(*this, DerivSpatialMetric<DataType, Frame>{}));
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, 3, Frame>*>
        spatial_christoffel_first_kind,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::SpatialChristoffelFirstKind<DataType, 3, Frame> /*meta*/) const {
  const auto& d_spatial_metric =
      cache->get_var(*this, DerivSpatialMetric<DataType, Frame>{});
  gr::christoffel_first_kind<3, Frame, IndexType::Spatial, DataType>(
      spatial_christoffel_first_kind, d_spatial_metric);
}

template <typename DataType, typename Frame>
void SphericalKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ijj<DataType, 3, Frame>*>
        spatial_christoffel_second_kind,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::SpatialChristoffelSecondKind<DataType, 3, Frame> /*meta*/) const {
  const auto& spatial_christoffel_first_kind = cache->get_var(
      *this, gr::Tags::SpatialChristoffelFirstKind<DataType, 3, Frame>{});
  const auto& inverse_spatial_metric = cache->get_var(
      *this, gr::Tags::InverseSpatialMetric<DataType, 3, Frame>{});
  raise_or_lower_first_index<DataType, SpatialIndex<3, UpLo::Lo, Frame>,
                             SpatialIndex<3, UpLo::Lo, Frame>>(
      spatial_christoffel_second_kind, spatial_christoffel_first_kind,
      inverse_spatial_metric);
}

template <typename DataType, typename Frame>
tnsr::i<DataType, 3, Frame>
SphericalKerrSchild::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    DerivLapse<DataType, Frame> /*meta*/) {
  tnsr::i<DataType, 3, Frame> result{};
  const auto& deriv_H =
      get_var(computer, internal_tags::deriv_H<DataType, Frame>{});
  const auto& deriv_lapse_multiplier =
      get(get_var(computer, internal_tags::deriv_lapse_multiplier<DataType>{}));

  for (size_t i = 0; i < 3; ++i) {
    result.get(i) = deriv_lapse_multiplier * deriv_H.get(i + 1);
  }
  return result;
}

template <typename DataType, typename Frame>
Scalar<DataType>
SphericalKerrSchild::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    ::Tags::dt<gr::Tags::Lapse<DataType>> /*meta*/) {
  const auto& H = get(get_var(computer, internal_tags::H<DataType>{}));

  return make_with_value<Scalar<DataType>>(H, 0.);
}

template <typename DataType, typename Frame>
tnsr::I<DataType, 3, Frame>
SphericalKerrSchild::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    ::Tags::dt<gr::Tags::Shift<DataType, 3, Frame>> /*meta*/) {
  const auto& H = get(get_var(computer, internal_tags::H<DataType>()));

  return make_with_value<tnsr::I<DataType, 3, Frame>>(H, 0.);
}

template <typename DataType, typename Frame>
Scalar<DataType>
SphericalKerrSchild::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    gr::Tags::SqrtDetSpatialMetric<DataType> /*meta*/) {
  const auto& jacobian =
      get_var(computer, internal_tags::jacobian<DataType, Frame>{});

  return Scalar<DataType>(get(determinant(jacobian)) /
                          get(get_var(computer, gr::Tags::Lapse<DataType>{})));
}

template <typename DataType, typename Frame>
tnsr::i<DataType, 3, Frame>
SphericalKerrSchild::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    gr::Tags::DerivDetSpatialMetric<DataType, 3, Frame> /*meta*/) {
  const auto& deriv_H =
      get_var(computer, internal_tags::deriv_H<DataType, Frame>{});

  auto result =
      make_with_value<tnsr::i<DataType, 3, Frame>>(get<0>(deriv_H), 0.);
  for (size_t i = 0; i < 3; ++i) {
    result.get(i) = 2. * square(null_vector_0_) * deriv_H.get(i + 1);
  }
  return result;
}

template <typename DataType, typename Frame>
Scalar<DataType>
SphericalKerrSchild::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) {
  return trace(
      get_var(computer, gr::Tags::ExtrinsicCurvature<DataType, 3, Frame>{}),
      get_var(computer, gr::Tags::InverseSpatialMetric<DataType, 3, Frame>{}));
}

template <typename DataType, typename Frame>
tnsr::I<DataType, 3, Frame>
SphericalKerrSchild::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    gr::Tags::TraceSpatialChristoffelSecondKind<DataType, 3, Frame> /*meta*/) {
  const auto& inverse_spatial_metric =
      get_var(computer, gr::Tags::InverseSpatialMetric<DataType, 3, Frame>{});
  const auto& spatial_christoffel_second_kind = get_var(
      computer, gr::Tags::SpatialChristoffelSecondKind<DataType, 3, Frame>{});
  return trace_last_indices<DataType, SpatialIndex<3, UpLo::Up, Frame>,
                            SpatialIndex<3, UpLo::Lo, Frame>>(
      spatial_christoffel_second_kind, inverse_spatial_metric);
}

bool operator==(const SphericalKerrSchild& lhs,
                const SphericalKerrSchild& rhs) {
  return lhs.mass() == rhs.mass() and
         lhs.dimensionless_spin() == rhs.dimensionless_spin() and
         lhs.center() == rhs.center();
}

bool operator!=(const SphericalKerrSchild& lhs,
                const SphericalKerrSchild& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                            \
  template class SphericalKerrSchild::IntermediateVars<DTYPE(data),     \
                                                       FRAME(data)>;    \
  template class SphericalKerrSchild::IntermediateComputer<DTYPE(data), \
                                                           FRAME(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector, double),
                        (::Frame::Inertial, ::Frame::Grid))
#undef INSTANTIATE
#undef DTYPE
#undef FRAME
}  // namespace gr::Solutions
