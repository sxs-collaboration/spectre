// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/AnisotropicHomogeneous.hpp"

#include <array>
#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/DynamicVector.h>
#include <blaze/math/SymmetricMatrix.h>
#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Elasticity::ConstitutiveRelations {

namespace {
// Voigt ordering used here:
// 0: xx
// 1: yy
// 2: zz
// 3: yz
// 4: xz
// 5: xy
inline std::array<DataVector, 6> strain_to_voigt(
    const tnsr::ii<DataVector, 3>& strain) {
  return {get<0, 0>(strain),       get<1, 1>(strain),
          get<2, 2>(strain),       2.0 * get<1, 2>(strain),
          2.0 * get<0, 2>(strain), 2.0 * get<0, 1>(strain)};
}

inline void voigt_to_stress(
    const gsl::not_null<tnsr::II<DataVector, 3>*> stress,
    const std::array<DataVector, 6>& s) {
  get<0, 0>(*stress) = s[0];
  get<1, 1>(*stress) = s[1];
  get<2, 2>(*stress) = s[2];

  // tnsr::II stores symmetric components; set only lower-tri
  get<2, 1>(*stress) = s[3];  // yz
  get<2, 0>(*stress) = s[4];  // xz
  get<1, 0>(*stress) = s[5];  // xy
}
}  // namespace

AnisotropicHomogeneous::AnisotropicHomogeneous() {
  for (auto& row : stiffness_) {
    row.fill(std::numeric_limits<double>::signaling_NaN());
  }
}

AnisotropicHomogeneous::AnisotropicHomogeneous(
    std::array<std::array<double, 6>, 6> stiffness,
    const Options::Context& context)
    : stiffness_(std::move(stiffness)) {
  for (size_t i = 0; i < 6; ++i) {
    for (size_t j = i + 1; j < 6; ++j) {
      if (gsl::at(gsl::at(stiffness_, i), j) !=
          gsl::at(gsl::at(stiffness_, j), i)) {
        PARSE_ERROR(context,
                    "StiffnessMatrix must be symmetric in Voigt notation, but "
                        << "C[" << i << "][" << j
                        << "]=" << gsl::at(gsl::at(stiffness_, i), j)
                        << " and C[" << j << "][" << i
                        << "]=" << gsl::at(gsl::at(stiffness_, j), i));
      }
    }
  }
  // The strain energy density e^T C e / 2 must be positive for every nonzero
  // strain, i.e. all eigenvalues of the symmetric Voigt matrix C must be
  // positive. The threshold is relative to the largest eigenvalue so that it
  // is independent of the units of C and so that numerically singular
  // matrices are rejected deterministically. A non-positive eigenvalue
  // means the material is unstable; a positive eigenvalue below the
  // threshold means the matrix is numerically too ill-conditioned.
  blaze::SymmetricMatrix<blaze::DynamicMatrix<double>> voigt_matrix(6);
  for (size_t i = 0; i < 6; ++i) {
    for (size_t j = i; j < 6; ++j) {
      voigt_matrix(i, j) = gsl::at(gsl::at(stiffness_, i), j);
    }
  }
  blaze::DynamicVector<double> eigenvalues(6);
  blaze::eigen(voigt_matrix, eigenvalues);
  const double min_eigenvalue = blaze::min(eigenvalues);
  const double max_eigenvalue = blaze::max(eigenvalues);
  if (min_eigenvalue <= 0.) {
    PARSE_ERROR(context,
                "StiffnessMatrix must be positive definite, but its smallest "
                "eigenvalue is "
                    << min_eigenvalue << " (largest: " << max_eigenvalue
                    << "). The material would be unstable (non-positive "
                       "strain energy for some strain).");
  }
  if (not(min_eigenvalue > min_relative_eigenvalue * max_eigenvalue)) {
    PARSE_ERROR(
        context,
        "StiffnessMatrix is positive definite but too "
        "ill-conditioned to be accepted: its smallest eigenvalue "
            << min_eigenvalue << " is not above " << min_relative_eigenvalue
            << " times its largest eigenvalue " << max_eigenvalue << ".");
  }
}

std::unique_ptr<ConstitutiveRelation<3>> AnisotropicHomogeneous::get_clone()
    const {
  return std::make_unique<AnisotropicHomogeneous>(*this);
}

void AnisotropicHomogeneous::stress(
    const gsl::not_null<tnsr::II<DataVector, 3>*> stress,
    const tnsr::ii<DataVector, 3>& strain,
    const tnsr::I<DataVector, 3>& /*x*/) const {
  const auto e = strain_to_voigt(strain);

  auto s = make_with_value<std::array<DataVector, 6>>(strain, 0.0);

  for (size_t a = 0; a < 6; ++a) {
    for (size_t b = 0; b < 6; ++b) {
      gsl::at(s, a) += gsl::at(gsl::at(stiffness_, a), b) * gsl::at(e, b);
    }
    gsl::at(s, a) *= -1.0;  // T = -C e
  }

  voigt_to_stress(stress, s);
}

const std::array<std::array<double, 6>, 6>&
AnisotropicHomogeneous::stiffness_matrix() const {
  return stiffness_;
}

PUP::able::PUP_ID AnisotropicHomogeneous::my_PUP_ID = 0;

void AnisotropicHomogeneous::pup(PUP::er& p) { p | stiffness_; }

bool operator==(const AnisotropicHomogeneous& lhs,
                const AnisotropicHomogeneous& rhs) {
  return lhs.stiffness_matrix() == rhs.stiffness_matrix();
}

bool operator!=(const AnisotropicHomogeneous& lhs,
                const AnisotropicHomogeneous& rhs) {
  return not(lhs == rhs);
}

}  // namespace Elasticity::ConstitutiveRelations
