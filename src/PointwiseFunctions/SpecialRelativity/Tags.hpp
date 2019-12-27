// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/SpecialRelativity/TagsDeclarations.hpp"

/// \ingroup EvolutionSystemsGroup
/// \brief Items related to special relativity.
namespace sr {
/// %Tags for hydrodynamic systems.
namespace Tags {
/// The Lorentz factor \f$W = (1-v^iv_i)^{-1/2}\f$, where \f$v^i\f$ is
/// the spatial velocity.
template <typename DataType>
struct LorentzFactor : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "LorentzFactor"; }
};

/// The square of the Lorentz factor \f$W^2\f$.
template <typename DataType>
struct LorentzFactorSquared {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "LorentzFactorSquared"; }
};

/// The spatial velocity \f$v^i\f$ of the fluid,
/// where \f$v^i=u^i/W + \beta^i/\alpha\f$.
/// Here \f$u^i\f$ is the spatial part of the 4-velocity,
/// \f$W\f$ is the Lorentz factor, \f$\beta^i\f$ is the shift vector,
/// and \f$\alpha\f$ is the lapse function. Note that \f$v^i\f$ is raised
/// and lowered with the spatial metric.
template <typename DataType, size_t Dim, typename Fr>
struct SpatialVelocity : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Fr>;
  static std::string name() noexcept {
    return Frame::prefix<Fr>() + "SpatialVelocity";
  }
};

/// The spatial velocity one-form \f$v_i\f$, where \f$v_i\f$ is raised
/// and lowered with the spatial metric.
template <typename DataType, size_t Dim, typename Fr>
struct SpatialVelocityOneForm : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Fr>;
  static std::string name() noexcept {
    return Frame::prefix<Fr>() + "SpatialVelocityOneForm";
  }
};

/// The spatial velocity squared \f$v^2 = v_i v^i\f$.
template <typename DataType>
struct SpatialVelocitySquared : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "SpatialVelocitySquared"; }
};
}  // namespace Tags
}  // namespace sr
