// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean {
/// %Tags for the combined system of the Generalized Harmonic formulation for
/// the Einstein field equations and the Valencia GRMHD formulation.
namespace Tags {
namespace detail {
// A const reference to another tag, used for rerouting arguments in the
// combined system utilities
template <typename Tag, typename Type = db::const_item_type<Tag, tmpl::list<>>>
struct TemporaryReference {
  using tag = Tag;
  using type = const Type&;
};
}  // namespace detail

/// Represents the trace reversed stress-energy tensor of the matter in the MHD
/// sector of the GRMHD system
struct TraceReversedStressEnergy : db::SimpleTag {
  using type = tnsr::aa<DataVector, 3>;
};

/// Represents the stress-energy tensor of the matter in the MHD sector of the
/// GRMHD system
struct StressEnergy : db::SimpleTag {
  using type = tnsr::aa<DataVector, 3>;
};

/// The comoving magnetic field \f$b^\mu\f$
struct ComovingMagneticField : db::SimpleTag {
  using type = tnsr::A<DataVector, 3>;
};

/// The fluid four-velocity \f$u^\mu\f$
struct FourVelocity : db::SimpleTag {
  using type = tnsr::A<DataVector, 3>;
};

/// The down-index comoving magnetic field \f$b_\mu\f$
struct ComovingMagneticFieldOneForm : db::SimpleTag {
  using type = tnsr::a<DataVector, 3>;
};

/// The down-index four-velocity \f$u_\mu\f$
struct FourVelocityOneForm : db::SimpleTag {
  using type = tnsr::a<DataVector, 3>;
};

/// \brief Tags sent for GRMHD primitive variable reconstruction. All tags sent
/// are the reconstruciton+spacetime tags
using primitive_grmhd_reconstruction_tags =
    tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
               hydro::Tags::ElectronFraction<DataVector>,
               hydro::Tags::Pressure<DataVector>,
               hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>,
               hydro::Tags::MagneticField<DataVector, 3>,
               hydro::Tags::DivergenceCleaningField<DataVector>>;

/// \brief Tags sent for spacetime evolution. All tags sent are the
/// reconstruciton+spacetime tags
using spacetime_reconstruction_tags =
    tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
               gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>>;

/// \brief All tags sent for primitive reconstruction, both GRMHD and spacetime
/// evolution tags.
using primitive_grmhd_and_spacetime_reconstruction_tags =
    tmpl::append<primitive_grmhd_reconstruction_tags,
                 spacetime_reconstruction_tags>;
}  // namespace Tags
}  // namespace grmhd::GhValenciaDivClean
