// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ParallelAlgorithms/RayTracer/RaySources/Camera.hpp"
#include "ParallelAlgorithms/RayTracer/RaySources/ParallelRays.hpp"
#include "Utilities/TMPL.hpp"

namespace ray_tracing {

using all_ray_sources = tmpl::list<Camera, ParallelRays>;

}  // namespace ray_tracing
