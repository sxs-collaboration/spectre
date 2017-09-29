// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

/// \file
/// Provides a C++ interface to the Fortran SPHEREPACK library.

extern "C" {
void shagsi_(const int&, const int&, double*, const int&, double*, const int&,
             double*, const int&, gsl::not_null<int*>);
void shags_(const int&, const int&, const int&, const int&, const int&,
            const int&, const int&, const int&, const double*, const int&,
            const int&, double*, double*, const int&, const int&, const int&,
            const int&, const double*, const int&, double*, const int&,
            gsl::not_null<int*>);
void shsgsi_(const int&, const int&, double*, const int&, double*, const int&,
             double*, const int&, gsl::not_null<int*>);
void shsgs_(const int&, const int&, const int&, const int&, const int&,
            const int&, const int&, const int&, double*, const int&, const int&,
            const double*, const double*, const int&, const int&, const int&,
            const int&, const double*, const int&, double*, const int&,
            gsl::not_null<int*>);
void vhagsi_(const int&, const int&, double*, const int&, double*, const int&,
             gsl::not_null<int*>);
void vhags_(const int&, const int&, const int&, const int&, const int&,
            const int&, const int&, const int&, const double*, const double*,
            const int&, const int&, double*, double*, double*, double*,
            const int&, const int&, const double*, const int&, double*,
            const int&, gsl::not_null<int*>);
void vhsgsi_(const int&, const int&, double*, const int&, double*, const int&,
             gsl::not_null<int*>);
void vhsgs_(const int&, const int&, const int&, const int&, const int&,
            const int&, const int&, const int&, double*, double*, const int&,
            const int&, const double*, const double*, const double*,
            const double*, const int&, const int&, const double*, const int&,
            double*, const int&, gsl::not_null<int*>);
void slapgs_(const int&, const int&, const int&, const int&, const int&,
             const int&, const int&, const int&, double*, const int&,
             const int&, const double*, const double*, const int&, const int&,
             const double*, const int&, double*, const int&,
             gsl::not_null<int*>);
void gradgs_(const int&, const int&, const int&, const int&, const int&,
             const int&, const int&, const int&, double*, double*, const int&,
             const int&, const double*, const double*, const int&, const int&,
             const double*, const int&, double*, const int&,
             gsl::not_null<int*>);
void divgs_(const int&, const int&, const int&, const int&, const int&,
            const int&, const int&, const int&, double*, const int&, const int&,
            const double*, const double*, const int&, const int&, const double*,
            const int&, double*, const int&, gsl::not_null<int*>);
void vrtgs_(const int&, const int&, const int&, const int&, const int&,
            const int&, const int&, const int&, double*, const int&, const int&,
            const double*, const double*, const int&, const int&, const double*,
            const int&, double*, const int&, gsl::not_null<int*>);
void vtsgsi_(const int&, const int&, double*, const int&, double*, const int&,
             double*, const int&, gsl::not_null<int*>);
void vtsgs_(const int&, const int&, const int&, const int&, const int&,
            const int&, const int&, const int&, double*, double*, const int&,
            const int&, const double*, const double*, const double*,
            const double*, const int&, const int&, const double*, const int&,
            double*, const int&, gsl::not_null<int*>);
void alfk_(const int&, const int&, double*);
void lfin_(const int&, const double*, const int&, const int&, const int&,
           double*, const int&, double*);
void gaqd_(const int&, double*, double*, double*, const int&,
           gsl::not_null<int*>);
}
