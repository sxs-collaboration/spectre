// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/TagsDeclarations.hpp"

/// \cond
class DataVector;
/// \endcond

namespace RelativisticEuler {
namespace Valencia {
/// %Tags for the Valencia formulation of the relativistic Euler system.
namespace Tags {

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
template <size_t Dim, typename Fr>
struct TildeS : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Fr>;
  static std::string name() noexcept { return Frame::prefix<Fr>() + "TildeS"; }
};

}  // namespace Tags
}  // namespace Valencia
}  // namespace RelativisticEuler
