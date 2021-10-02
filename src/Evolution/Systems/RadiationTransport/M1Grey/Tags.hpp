// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

#include "Evolution/Systems/RadiationTransport/Tags.hpp"

class DataVector;

/// Namespace for all radiation transport algorithms
namespace RadiationTransport {
/// Namespace for the grey-M1 radiation transport scheme
namespace M1Grey {
/// %Tags for the evolution of neutrinos using a grey M1 scheme.
namespace Tags {

/// The characteristic speeds
struct CharacteristicSpeeds : db::SimpleTag {
  using type = std::array<DataVector, 4>;
};

/// The densitized energy density of neutrinos of a given species
/// \f${\tilde E}\f$
template <typename Fr, class Species>
struct TildeE : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() {
    return Frame::prefix<Fr>() + "TildeE_" + neutrinos::get_name(Species{});
  }
};

/// The densitized momentum density of neutrinos of a given species
/// \f${\tilde S_i}\f$
template <typename Fr, class Species>
struct TildeS : db::SimpleTag {
  using type = tnsr::i<DataVector, 3, Fr>;
  static std::string name() {
    return Frame::prefix<Fr>() + "TildeS_" + neutrinos::get_name(Species{});
  }
};

/// The densitized pressure tensor of neutrinos of a given species
/// \f${\tilde P^{ij}}\f$
/// computed from \f${\tilde E}\f$, \f${\tilde S_i}\f$ using the M1 closure
template <typename Fr, class Species>
struct TildeP : db::SimpleTag {
  using type = tnsr::II<DataVector, 3, Fr>;
  static std::string name() {
    return Frame::prefix<Fr>() + "TildeP_" + neutrinos::get_name(Species{});
  }
};

/// The upper index momentum density of a neutrino species.
///
/// This tag does not know the species of neutrinos being used.
/// \f${\tilde S^i}\f$
template <typename Fr>
struct TildeSVector : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Fr>;
  static std::string name() { return Frame::prefix<Fr>() + "TildeSVector"; }
};

/// The M1 closure factor of neutrinos of
/// a given species \f${\xi}\f$
template <class Species>
struct ClosureFactor : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() {
    return "ClosureFactor_" + neutrinos::get_name(Species{});
  }
};

/// The fluid-frame densitized energy density of neutrinos of
/// a given species \f${\tilde J}\f$
template <class Species>
struct TildeJ : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() {
    return "TildeJ_" + neutrinos::get_name(Species{});
  }
};

/// The normal component of the fluid-frame momentum density of neutrinos of
/// a given species \f${\tilde H}^a t_a\f$
template <class Species>
struct TildeHNormal : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() {
    return "TildeHNormal_" + neutrinos::get_name(Species{});
  }
};

/// The spatial components of the fluid-frame momentum density of neutrinos of
/// a given species \f${\tilde H}^a {\gamma}_{ia}\f$
template <typename Fr, class Species>
struct TildeHSpatial : db::SimpleTag {
  using type = tnsr::i<DataVector, 3, Fr>;
  static std::string name() {
    return Frame::prefix<Fr>() + "TildeHSpatial_" +
           neutrinos::get_name(Species{});
  }
};

/// The emissivity of the fluid for neutrinos of a given species.
///
/// By convention, this is the emitted energy per unit volume.
template <class Species>
struct GreyEmissivity : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() {
    return "GreyEmissivity_" + neutrinos::get_name(Species{});
  }
};

/// The absorption opacity of the fluid for neutrinos of a
/// given species.
///
/// As c=1, this is both the absorption per unit length
/// and per unit time.
template <class Species>
struct GreyAbsorptionOpacity : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() {
    return "GreyAbsorptionOpacity_" + neutrinos::get_name(Species{});
  }
};

/// The scattering opacity of the fluid for neutrinos of a
/// given species.
///
/// As c=1, this is both the absorption per unit length and per unit time.
template <class Species>
struct GreyScatteringOpacity : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() {
    return "GreyScatteringOpacity_" + neutrinos::get_name(Species{});
  }
};

/// The normal component of the source term coupling the M1 and hydro equations.
///
/// This is the source term for the evolution of \f${\tilde E}\f$ (by
/// convention, added with a positive sign to \f${\tilde E}\f$ and a negative
/// sign to the hydro variable \f${\tilde \tau}\f$)
template <class Species>
struct M1HydroCouplingNormal : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() {
    return "M1HydroCouplingNormal_" + neutrinos::get_name(Species{});
  }
};

/// The spatial components of source term coupling the M1 and hydro equations.
///
/// i.e. \f${\tilde G}^a \gamma_{ia}\f$.
/// This is the source term for the evolution of \f${\tilde S}_i\f$ (by
/// convention, added with a positive sign to the neutrino \f${\tilde S}_i\f$
/// and a negative sign to the hydro variable \f${\tilde S}_i\f$)
template <typename Fr, class Species>
struct M1HydroCouplingSpatial : db::SimpleTag {
  using type = tnsr::i<DataVector, 3, Fr>;
  static std::string name() {
    return Frame::prefix<Fr>() + "M1HydroCouplingSpatial_" +
           neutrinos::get_name(Species{});
  }
};

}  // namespace Tags
}  // namespace M1Grey
}  // namespace RadiationTransport
