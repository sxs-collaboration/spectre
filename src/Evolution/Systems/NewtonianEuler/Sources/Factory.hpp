// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/NewtonianEuler/Sources/LaneEmdenGravitationalField.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/NoSource.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/Source.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/UniformAcceleration.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/VortexPerturbation.hpp"

namespace NewtonianEuler::Sources {
/// All the available source terms.
template <size_t Dim>
using all_sources = tmpl::append<
    tmpl::conditional_t<
        Dim == 3, tmpl::list<LaneEmdenGravitationalField, VortexPerturbation>,
        tmpl::list<>>,
    tmpl::list<NoSource<Dim>, UniformAcceleration<Dim>>>;
}  // namespace NewtonianEuler::Sources
