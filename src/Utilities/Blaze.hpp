// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Includes Blaze library with specific configs

#pragma once

#ifdef __GNUC__
#pragma GCC system_header
#endif

/// \cond

// Override cache size
//#define _BLAZE_SYSTEM_CACHESIZE_H_
// constexpr size_t cacheSize = 6291456UL;

// Override padding, streaming and kernel options
#define _BLAZE_SYSTEM_OPTIMIZATIONS_H_
namespace blaze {
constexpr bool usePadding = false;
constexpr bool useStreaming = true;
constexpr bool useOptimizedKernels = true;
}

// Override SMP configurations
#define _BLAZE_SYSTEM_SMP_H_
#define BLAZE_USE_SHARED_MEMORY_PARALLELIZATION 0
#define BLAZE_OPENMP_PARALLEL_MODE 0
#define BLAZE_CPP_THREADS_PARALLEL_MODE 0

// Disable MPI parallelization
#define _BLAZE_SYSTEM_MPI_H_
#define BLAZE_MPI_PARALLEL_MODE 0

// Disable HPX parallelization
#define BLAZE_HPX_PARALLEL_MODE 0
/// \endcond
