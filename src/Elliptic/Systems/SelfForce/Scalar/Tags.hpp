// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Structure/BlockGroups.hpp"

/// \cond
class ComplexDataVector;
class DataVector;
/// \endcond

/*!
 * \ingroup EllipticSystemsGroup
 * \brief Items related to solving the self force of a scalar charge on a Kerr
 * background.
 *
 * \see ScalarSelfForce::FirstOrderSystem
 */
namespace ScalarSelfForce {

namespace OptionTags {

struct OptionGroup {
  static std::string name() { return "ScalarSelfForce"; }
  static constexpr Options::String help =
      "Options for the scalar self-force system";
};

struct NullSlicingBlocks {
  using group = OptionGroup;
  using type = std::vector<std::string>;
  static constexpr Options::String help =
      "The blocks in which to use null slicing (vtu-slicing).";
};

}  // namespace OptionTags

/// Tags for the ScalarSelfForce system.
namespace Tags {

/*!
 * \brief The complex m-mode field $\Psi_m$.
 *
 * Defined by the m-mode decomposition of the complex scalar field $\Phi$
 * [Eq. (2.6) in \cite Osburn:2022bby ]:
 *
 * \begin{equation}
 * \Phi(t,r,\theta,\phi) = \frac{1}{r} \sum_{m=-\infty}^{\infty}
 *   \Psi_m(r,\theta) e^{im\Delta\phi(r)} e^{im(\phi - \Omega t)}
 * \end{equation}
 *
 * where $\Delta\phi(r) = \frac{a}{r_\plus - r_\minus}
 * \ln(\frac{r-r_\plus}{r-r_\minus})$.
 */
struct MMode : db::SimpleTag {
  using type = Scalar<ComplexDataVector>;
};

/*!
 * \brief The factors $\alpha^i$ multiplying the radial and angular
 * derivatives in the principal part of the equations.
 *
 * These are the factors $\alpha^i$ that define the principal part of the
 * equations and allow writing them in first-order flux form given by
 * \begin{equation}
 * -\partial_i F^i + \beta \Psi_m + \gamma_i \partial_i \Psi_m = S_m
 * \end{equation}
 * with the flux
 * \begin{equation}
 * F^i = \alpha^i \partial_i \Psi_m
 * \text{.}
 * \end{equation}
 * They are set by the analytic data class (see
 * `ScalarSelfForce::AnalyticData::CircularOrbit`).
 */
struct Alpha : db::SimpleTag {
  using type = tnsr::I<ComplexDataVector, 2>;
};

/*!
 * \brief The factor multiplying the non-derivative terms in the equations.
 *
 * This is the factor $\beta$ in the general form of the equations
 * \begin{equation}
 * -\partial_i F^i + \beta \Psi_m + \gamma_i \partial_i \Psi_m = S_m
 * \text{.}
 * \end{equation}
 * This factor is set by the analytic data class (see
 * `ScalarSelfForce::AnalyticData::CircularOrbit`).
 */
struct Beta : db::SimpleTag {
  using type = Scalar<ComplexDataVector>;
};

/*!
 * \brief The factor multiplying the first-derivative terms in the equations.
 *
 * This is the factor $\gamma_i$ in the general form of the equations
 * \begin{equation}
 * -\partial_i F^i + \beta \Psi_m + \gamma_i \partial_i \Psi_m = S_m
 * \text{.}
 * \end{equation}
 * This factor is set by the analytic data class (see
 * `ScalarSelfForce::AnalyticData::CircularOrbit`).
 */
struct Gamma : db::SimpleTag {
  using type = tnsr::i<ComplexDataVector, 2>;
};

/*!
 * \brief A flag indicating that we are solving for the regularized field in
 * this element.
 *
 * In elements at and around the scalar point charge we use the effective source
 * approach to split the m-mode field into a singular and a regular field
 * [Eq. (3.6) in \cite Osburn:2022bby ]:
 * \begin{equation}
 * \Psi_m = \Psi_m^\mathcal{P} + \Psi_m^\mathcal{R}
 * \text{.}
 * \end{equation}
 * In these elements, we solve for $\Psi_m^\mathcal{R}$ rather than $\Psi_m$.
 */
struct FieldIsRegularized : db::SimpleTag {
  using type = bool;
};

/*!
 * \brief The singular field $\Psi_m^\mathcal{P}$.
 *
 * Only defined where `FieldIsRegularized` is true.
 */
struct SingularField : db::SimpleTag {
  using type = Scalar<ComplexDataVector>;
};

/*!
 * \brief The Boyer-Lindquist radius $r$.
 *
 * Computed numerically from the tortoise radial coordinate $r_\star$
 * [see Eq. (2.12) in \cite Osburn:2022bby ].
 */
struct BoyerLindquistRadius : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief Blocks in which we use null slicing (vtu-slicing).
 */
template <size_t Dim>
struct NullSlicingBlocks : db::SimpleTag {
  using type = std::vector<size_t>;
  using option_tags = tmpl::list<OptionTags::NullSlicingBlocks,
                                 domain::OptionTags::DomainCreator<Dim>>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(
      const std::vector<std::string>& null_slicing_blocks,
      const std::unique_ptr<DomainCreator<Dim>>& domain_creator) {
    return domain::block_ids_from_names(null_slicing_blocks,
                                        domain_creator->block_names(),
                                        domain_creator->block_groups());
  }
};

/*!
 * \brief The hyperboloidal boost function $H(r_*)$.
 */
struct BoostFunction : db::SimpleTag {
  using type = Scalar<ComplexDataVector>;
};

/*!
 * \brief The radial derivative of the boost function, $H'(r_*)$.
 */
struct BoostFunctionDeriv : db::SimpleTag {
  using type = Scalar<ComplexDataVector>;
};

}  // namespace Tags
}  // namespace ScalarSelfForce
