// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

/// Storage and time integration for boundary-evolved fields: quantities held
/// and time-integrated only on an element's external boundary faces (a
/// per-face, pointwise ODE), driven by boundary
/// conditions that opt in via a `boundary_evolved_variables` type alias.
namespace evolution::dg::BoundaryEvolvedFields {}
