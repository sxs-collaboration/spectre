// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

#include "Evolution/Systems/RadiationTransport/Tags.hpp"

class DataVector;

/// Namespace for all radiation transport algorithms
namespace RadiationTransport {
/// Namespace for the grey-M1 radiation transport scheme
namespace M1Grey {
/// %Tags for the evolution of neutrinos using a grey M1 scheme.
namespace Tags {

/// The densitized energy density of neutrinos of a given species
/// \f${\tilde E}\f$
template <typename Fr, class Species>
struct TildeE : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept {
    return Frame::prefix<Fr>() + "TildeE_" + neutrinos::get_name(Species{});
  }
};

/// The densitized momentum density of neutrinos of a given species
/// \f${\tilde F_i}\f$
template <typename Fr, class Species>
struct TildeF : db::SimpleTag {
  using type = tnsr::i<DataVector, 3, Fr>;
  static std::string name() noexcept {
    return Frame::prefix<Fr>() + "TildeF_" + neutrinos::get_name(Species{});
  }
};

}  // namespace Tags
}  // namespace M1Grey
}  // namespace RadiationTransport
