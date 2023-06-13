// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <csignal>

#include "Utilities/ErrorHandling/Error.hpp"

#ifdef __CUDA_ARCH__
// When building for Nvidia GPUs we need to disable the use of vector
// intrinsics.
#define BLAZE_USE_VECTORIZATION 0
#endif

#ifdef SPECTRE_DEBUG
#define BLAZE_THROW(EXCEPTION) ERROR(EXCEPTION.what())
#else  // SPECTRE_DEBUG
#define BLAZE_THROW(EXCEPTION)
#endif  // SPECTRE_DEBUG
