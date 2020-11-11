// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"

namespace GeneralizedHarmonic::ConstraintDamping::Tags {
/*!
 * \brief Constraint dammping parameter \f$\gamma_0\f$ for the generalized
 * harmonic system (cf. \cite Lindblom2005qh).
 */
struct ConstraintGamma0 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief Constraint dammping parameter \f$\gamma_1\f$ for the generalized
 * harmonic system (cf. \cite Lindblom2005qh).
 */
struct ConstraintGamma1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief Constraint dammping parameter \f$\gamma_2\f$ for the generalized
 * harmonic system (cf. \cite Lindblom2005qh).
 */
struct ConstraintGamma2 : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace GeneralizedHarmonic::ConstraintDamping::Tags
