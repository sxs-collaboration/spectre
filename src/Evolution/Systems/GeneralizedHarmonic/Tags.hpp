// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TagsDeclarations.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"

class DataVector;

namespace GeneralizedHarmonic {
namespace Tags {
/*!
 * \brief Conjugate momentum to the spacetime metric.
 *
 * \details If \f$ \psi_{ab} \f$ is the spacetime metric, and \f$ N \f$ and
 * \f$ N^i \f$ are the lapse and shift respectively, then we define
 * \f$ \Pi_{ab} = -\frac{1}{N} ( \partial_t \psi_{ab} + N^{i} \phi_{iab} ) \f$
 * where \f$\phi_{iab}\f$ is the variable defined by the tag Phi.
 */
template <size_t Dim, typename Frame>
struct Pi : db::SimpleTag {
  using type = tnsr::aa<DataVector, Dim, Frame>;
  static std::string name() noexcept { return "Pi"; }
};

/*!
 * \brief Auxiliary variable which is analytically the spatial derivative of the
 * spacetime metric
 * \details If \f$\psi_{ab}\f$ is the spacetime metric then we define
 * \f$\phi_{iab} = \partial_i \psi_{ab}\f$
 */
template <size_t Dim, typename Frame>
struct Phi : db::SimpleTag {
  using type = tnsr::iaa<DataVector, Dim, Frame>;
  static std::string name() noexcept { return "Phi"; }
};

struct ConstraintGamma0 : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "ConstraintGamma0"; }
};
struct ConstraintGamma1 : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "ConstraintGamma1"; }
};
struct ConstraintGamma2 : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "ConstraintGamma2"; }
};
template <size_t Dim, typename Frame>
struct GaugeH : db::SimpleTag {
  using type = tnsr::a<DataVector, Dim, Frame>;
  static std::string name() noexcept { return "GaugeH"; }
};
template <size_t Dim, typename Frame>
struct SpacetimeDerivGaugeH : db::SimpleTag {
  using type = tnsr::ab<DataVector, Dim, Frame>;
  static std::string name() noexcept { return "SpacetimeDerivGaugeH"; }
};

// @{
/// \ingroup GeneralizedHarmonicGroup
/// \brief Tags corresponding to the characteristic fields of the generalized
/// harmonic system.
///
/// \details For details on how these are defined and computed, see
/// CharacteristicSpeedsCompute
template <size_t Dim, typename Frame>
struct UPsi : db::SimpleTag {
  using type = tnsr::aa<DataVector, Dim, Frame>;
  static std::string name() noexcept { return "UPsi"; }
};
template <size_t Dim, typename Frame>
struct UZero : db::SimpleTag {
  using type = tnsr::iaa<DataVector, Dim, Frame>;
  static std::string name() noexcept { return "UZero"; }
};
template <size_t Dim, typename Frame>
struct UPlus : db::SimpleTag {
  using type = tnsr::aa<DataVector, Dim, Frame>;
  static std::string name() noexcept { return "UPlus"; }
};
template <size_t Dim, typename Frame>
struct UMinus : db::SimpleTag {
  using type = tnsr::aa<DataVector, Dim, Frame>;
  static std::string name() noexcept { return "UMinus"; }
};
// @}

template <size_t Dim, typename Frame>
struct CharacteristicSpeeds : db::SimpleTag {
  using type = std::array<DataVector, 4>;
  static std::string name() noexcept { return "CharacteristicSpeeds"; }
};

template <size_t Dim, typename Frame>
struct CharacteristicFields : db::SimpleTag {
  using type = Variables<tmpl::list<UPsi<Dim, Frame>, UZero<Dim, Frame>,
                                    UPlus<Dim, Frame>, UMinus<Dim, Frame>>>;
  static std::string name() noexcept { return "CharacteristicFields"; }
};

template <size_t Dim, typename Frame>
struct EvolvedFieldsFromCharacteristicFields : db::SimpleTag {
  using type =
      Variables<tmpl::list<gr::Tags::SpacetimeMetric<Dim, Frame, DataVector>,
                           Pi<Dim, Frame>, Phi<Dim, Frame>>>;
  static std::string name() noexcept {
    return "EvolvedFieldsFromCharacteristicFields";
  }
};
}  // namespace Tags
}  // namespace GeneralizedHarmonic
