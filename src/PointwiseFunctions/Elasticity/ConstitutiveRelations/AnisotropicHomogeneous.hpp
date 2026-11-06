// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Elasticity::ConstitutiveRelations {

/*!
 * \brief A general homogeneous anisotropic linear elastic material in 3D
 *
 * \details The material is specified by a symmetric 6x6 stiffness matrix in
 * Voigt notation. The stress-strain relation is
 *
 * \f[
 * T^{ij} = -Y^{ijkl} S_{kl}
 * \f]
 *
 * where \f$S_{ij}=\nabla_{(i}u_{j)}\f$ is the (symmetric) strain tensor.
 *
 * We interpret the Voigt strain vector with engineering shear strains:
 *
 * \f[
 * e = [S_{xx}, S_{yy}, S_{zz}, 2S_{yz}, 2S_{xz}, 2S_{xy}]
 * \f]
 *
 * and the Voigt stress vector as:
 *
 * \f[
 * s = [T_{xx}, T_{yy}, T_{zz}, T_{yz}, T_{xz}, T_{xy}]
 * \f]
 *
 * Then \f$ s = -C e \f$, where \f$C\f$ is the 6x6 stiffness matrix.
 *
 * The matrix must be symmetric and positive definite, so that the strain
 * energy density \f$e^T C e / 2\f$ is positive for every nonzero strain. The
 * positive-definiteness check is relative: the smallest eigenvalue of \f$C\f$
 * must exceed `min_relative_eigenvalue` times its largest eigenvalue, so the
 * check does not depend on the units of \f$C\f$ and numerically singular
 * matrices are rejected deterministically. A matrix with a non-positive
 * eigenvalue is reported as unstable; a positive definite matrix below the
 * ratio is reported as too ill-conditioned. All entries must be finite.
 * All checks run in all build types.
 */
class AnisotropicHomogeneous : public ConstitutiveRelation<3> {
 public:
  static constexpr size_t volume_dim = 3;

  /// The smallest eigenvalue of the Voigt matrix must exceed this fraction of
  /// the largest eigenvalue for the matrix to count as positive definite.
  static constexpr double min_relative_eigenvalue = 1.e-12;

  struct StiffnessMatrix {
    using type = std::array<std::array<double, 6>, 6>;
    static constexpr Options::String help = {
        "Symmetric, positive definite 6x6 stiffness matrix in Voigt "
        "notation (smallest eigenvalue > 1e-12 (`min_relative_eigenvalue`) "
        "times the largest). Interpreted with engineering shear strains: "
        "[Sxx,Syy,Szz,2Syz,2Sxz,2Sxy]. Entries are in the same units of "
        "stress as the other materials (e.g. GPa in the input files)."};
  };

  using options = tmpl::list<StiffnessMatrix>;

  static constexpr Options::String help = {
      "A constitutive relation that describes a homogeneous, anisotropic "
      "material in terms of a symmetric 6x6 stiffness matrix in Voigt "
      "notation. The matrix entries are in the same units of stress as the "
      "other materials (e.g. GPa in the input files)."};

  AnisotropicHomogeneous() {
    for (auto& row : stiffness_) {
      row.fill(std::numeric_limits<double>::signaling_NaN());
    }
  }

  AnisotropicHomogeneous(const AnisotropicHomogeneous&) = default;
  AnisotropicHomogeneous& operator=(const AnisotropicHomogeneous&) = default;
  AnisotropicHomogeneous(AnisotropicHomogeneous&&) = default;
  AnisotropicHomogeneous& operator=(AnisotropicHomogeneous&&) = default;
  ~AnisotropicHomogeneous() override = default;

  /// The stiffness matrix must be symmetric and positive definite (see the
  /// class documentation), which is checked in all build types (not only in
  /// Debug builds).
  explicit AnisotropicHomogeneous(
      std::array<std::array<double, 6>, 6> stiffness,
      const Options::Context& context = {});

  std::unique_ptr<ConstitutiveRelation<3>> get_clone() const override;

  void stress(gsl::not_null<tnsr::II<DataVector, 3>*> stress,
              const tnsr::ii<DataVector, 3>& strain,
              const tnsr::I<DataVector, 3>& x) const override;

  const std::array<std::array<double, 6>, 6>& stiffness_matrix() const;

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) override;  // NOLINT

  explicit AnisotropicHomogeneous(CkMigrateMessage* /*unused*/) {}

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(ConstitutiveRelation<3>), AnisotropicHomogeneous);

 private:
  std::array<std::array<double, 6>, 6> stiffness_{};
};

bool operator==(const AnisotropicHomogeneous& lhs,
                const AnisotropicHomogeneous& rhs);
bool operator!=(const AnisotropicHomogeneous& lhs,
                const AnisotropicHomogeneous& rhs);

}  // namespace Elasticity::ConstitutiveRelations
