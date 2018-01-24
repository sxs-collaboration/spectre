// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"

class DataVector;

namespace GeneralizedHarmonic {
/*!
 * \brief Conjugate momentum to the spacetime metric.
 *
 * \details If \f$ \psi_{ab} \f$ is the spacetime metric, and \f$ N \f$ and
 * \f$ N^i \f$ are the lapse and shift respectively, then we define
 * \f$ \Pi_{ab} = -\frac{1}{N} ( \partial_t \psi_{ab} + N^{i} \phi_{iab} ) \f$
 * where \f$\phi_{iab}\f$ is the variable defined by the tag Phi.
 */
template <size_t Dim, typename Frame>
struct Pi : db::DataBoxTag {
  using type = tnsr::aa<DataVector, Dim, Frame>;
  static constexpr db::DataBoxString label = "Pi";
};

/*!
 * \brief Auxiliary variable which is analytically the spatial derivative of the
 * spacetime metric
 * \details If \f$\psi_{ab}\f$ is the spacetime metric then we define
 * \f$\phi_{iab} = \partial_i \psi_{ab}\f$
 */
template <size_t Dim, typename Frame>
struct Phi : db::DataBoxTag {
  using type = tnsr::iaa<DataVector, Dim, Frame>;
  static constexpr db::DataBoxString label = "Phi";
};

struct ConstraintGamma0 : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString label = "ConstraintGamma0";
};
struct ConstraintGamma1 : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString label = "ConstraintGamma1";
};
struct ConstraintGamma2 : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString label = "ConstraintGamma2";
};
template <size_t Dim, typename Frame>
struct GaugeH : db::DataBoxTag {
  using type = tnsr::a<DataVector, Dim, Frame>;
  static constexpr db::DataBoxString label = "GaugeH";
};
template <size_t Dim, typename Frame>
struct SpacetimeDerivGaugeH : db::DataBoxTag {
  using type = tnsr::ab<DataVector, Dim, Frame>;
  static constexpr db::DataBoxString label = "SpacetimeDerivGaugeH";
};
}  // namespace GeneralizedHarmonic
