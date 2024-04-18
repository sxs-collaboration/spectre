/*
 * Distributed under the MIT License.
 * See LICENSE.txt for details.
 */

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
/*!
 * \brief Same as the POSIX strerror_r.
 *
 * \details C wrapper to avoid visibility issues caused by GNU.  The
 * POSIX-compliant function strerror_r is only defined by glibc if
 * _GNU_SOURCE is not defined, but libstdc++ doesn't work without
 * _GNU_SOURCE defined.
 */
int spectre_strerror_r(int errnum, char* buf, size_t buflen);
#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */
