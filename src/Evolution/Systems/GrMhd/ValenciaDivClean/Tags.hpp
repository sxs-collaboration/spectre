// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/TagsDeclarations.hpp"
#include "Options/Options.hpp"

class DataVector;

namespace OptionTags {
/// \brief The constraint damping parameter
struct DampingParameter {
  using type = double;
  static constexpr OptionString help{
      "constraint damping parameter for divergence cleaning"};
};
}  // namespace OptionTags

namespace grmhd {
namespace ValenciaDivClean {
/// %Tags for the Valencia formulation of the ideal GRMHD equations
/// with divergence cleaning.
namespace Tags {

/// The characteristic speeds
struct CharacteristicSpeeds : db::SimpleTag {
  using type = std::array<DataVector, 9>;
  static std::string name() noexcept { return "CharacteristicSpeeds"; }
};

/// The constraint damping parameter for divergence cleaning
struct ConstraintDampingParameter : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "ConstraintDampingParameter"; }
};

/// The densitized rest-mass density \f${\tilde D}\f$
struct TildeD : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "TildeD"; }
};

/// The densitized energy density \f${\tilde \tau}\f$
struct TildeTau : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "TildeTau"; }
};

/// The densitized momentum density \f${\tilde S_i}\f$
template <typename Fr>
struct TildeS : db::SimpleTag {
  using type = tnsr::i<DataVector, 3, Fr>;
  static std::string name() noexcept { return Frame::prefix<Fr>() + "TildeS"; }
};

/// The densitized magnetic field \f${\tilde B^i}\f$
template <typename Fr>
struct TildeB : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Fr>;
  static std::string name() noexcept { return Frame::prefix<Fr>() + "TildeB"; }
};

/// The densitized divergence-cleaning field \f${\tilde \Phi}\f$
struct TildePhi : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "TildePhi"; }
};

}  // namespace Tags
}  // namespace ValenciaDivClean
}  // namespace grmhd
