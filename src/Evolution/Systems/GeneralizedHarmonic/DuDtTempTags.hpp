// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
/// \endcond

namespace GeneralizedHarmonic {
namespace Tags {
/// \f$\gamma_1 \gamma_2\f$ constraint damping product
struct Gamma1Gamma2 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// \f$\Pi_{ab}n^an^b\f$
struct PiTwoNormals : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// \f$n^a \mathcal{C}_a\f$
struct NormalDotOneIndexConstraint : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// \f$\gamma_1 + 1\f$
struct Gamma1Plus1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// \f$\Pi_{ab}n^a\f$
template <size_t Dim>
struct PiOneNormal : db::SimpleTag {
  using type = tnsr::a<DataVector, Dim, Frame::Inertial>;
};

/// \f$\Phi_{iab}n^an^b\f$
template <size_t Dim>
struct PhiTwoNormals : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
};

/// \f$\beta^i \mathcal{C}_{iab}\f$
template <size_t Dim>
struct ShiftDotThreeIndexConstraint : db::SimpleTag {
  using type = tnsr::aa<DataVector, Dim, Frame::Inertial>;
};

/// \f$\v^i_g \mathcal{C}_{iab}\f$
template <size_t Dim>
struct MeshVelocityDotThreeIndexConstraint : db::SimpleTag {
  using type = tnsr::aa<DataVector, Dim, Frame::Inertial>;
};

/// \f$\Phi_{iab}n^a\f$
template <size_t Dim>
struct PhiOneNormal : db::SimpleTag {
  using type = tnsr::ia<DataVector, Dim, Frame::Inertial>;
};

/// \f$\Pi_a{}^b\f$
template <size_t Dim>
struct PiSecondIndexUp : db::SimpleTag {
  using type = tnsr::aB<DataVector, Dim, Frame::Inertial>;
};

/// \f$\Phi^i{}_{ab}\f$
template <size_t Dim>
struct PhiFirstIndexUp : db::SimpleTag {
  using type = tnsr::Iaa<DataVector, Dim, Frame::Inertial>;
};

/// \f$\Phi_{ia}{}^b\f$
template <size_t Dim>
struct PhiThirdIndexUp : db::SimpleTag {
  using type = tnsr::iaB<DataVector, Dim, Frame::Inertial>;
};

/// \f$\Gamma_{ab}{}^c\f$
template <size_t Dim>
struct SpacetimeChristoffelFirstKindThirdIndexUp : db::SimpleTag {
  using type = tnsr::abC<DataVector, Dim, Frame::Inertial>;
};
}  // namespace Tags
}  // namespace GeneralizedHarmonic
