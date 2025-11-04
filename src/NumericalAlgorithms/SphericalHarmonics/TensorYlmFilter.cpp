// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmFilter.hpp"

#include <blaze/math/CompressedMatrix.h>
#include <complex>
#include <optional>

#include "DataStructures/SimpleSparseMatrix.hpp"
#include "DataStructures/SparseMatrixFiller.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmHelpers.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/WignerThreeJ.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/Numeric.hpp"

namespace ylm::TensorYlm {

namespace {

// Inner loops of the rank-1 calculation.  The purpose of this
// function is so that there are not so many nested loops inside of
// the main function, making the main function and this function more
// readable.
void inner_loops_one(SparseMatrixFiller& filler, SpherepackIterator& iter_src,
                     SpherepackIterator& iter_dest, const size_t src_comp_index,
                     const size_t dest_comp_index, const size_t ell_max,
                     const int mdest, const int msrc,
                     const std::complex<double>& coefjp, WignerThreeJ& threej_j,
                     WignerThreeJ& threej_p) {
  const auto add_element = [&filler, &iter_src, &iter_dest, dest_comp_index,
                            src_comp_index](const double element) {
    const size_t indx_dest =
        iter_dest() + dest_comp_index * iter_dest.spherepack_array_size();
    const size_t indx_src =
        iter_src() + src_comp_index * iter_src.spherepack_array_size();
    filler.add(element, indx_dest, indx_src);
  };
  for (size_t ell = std::max(
           {static_cast<size_t>(mdest), threej_j.l1_min(), threej_p.l1_min()});
       ell <= std::min({ell_max, threej_j.l1_max(), threej_p.l1_max()});
       ++ell) {
    const std::complex<double> correction =
        threej_j(ell) * threej_p(ell) * coefjp;
    if (msrc > 0) {
      // Main term, i.e. first term in Eq. (18).
      if (correction.imag() == 0.0) {
        // ReRe
        iter_src.set(ell, static_cast<size_t>(msrc),
                     SpherepackIterator::CoefficientArray::a);
        iter_dest.set(ell, static_cast<size_t>(mdest),
                      SpherepackIterator::CoefficientArray::a);
        add_element(correction.real());

        // ImIm
        iter_src.set(ell, static_cast<size_t>(msrc),
                     SpherepackIterator::CoefficientArray::b);
        iter_dest.set(ell, static_cast<size_t>(mdest),
                      SpherepackIterator::CoefficientArray::b);
        add_element(correction.real());
      } else {
        // ReIm
        iter_src.set(ell, static_cast<size_t>(msrc),
                     SpherepackIterator::CoefficientArray::b);
        iter_dest.set(ell, static_cast<size_t>(mdest),
                      SpherepackIterator::CoefficientArray::a);
        add_element(-correction.imag());

        // ImRe
        iter_src.set(ell, static_cast<size_t>(msrc),
                     SpherepackIterator::CoefficientArray::a);
        iter_dest.set(ell, static_cast<size_t>(mdest),
                      SpherepackIterator::CoefficientArray::b);
        add_element(correction.imag());
      }
    } else {
      // Second term in Eq. (18).
      const double sign = (msrc % 2 == 0 ? 1.0 : -1.0);
      if (correction.imag() == 0.0) {
        // ReRe
        iter_src.set(ell, static_cast<size_t>(-msrc),
                     SpherepackIterator::CoefficientArray::a);
        iter_dest.set(ell, static_cast<size_t>(mdest),
                      SpherepackIterator::CoefficientArray::a);
        add_element(sign * correction.real());

        // ImIm
        iter_src.set(ell, static_cast<size_t>(-msrc),
                     SpherepackIterator::CoefficientArray::b);
        iter_dest.set(ell, static_cast<size_t>(mdest),
                      SpherepackIterator::CoefficientArray::b);
        add_element(-sign * correction.real());
      } else {
        // ReIm
        iter_src.set(ell, static_cast<size_t>(-msrc),
                     SpherepackIterator::CoefficientArray::b);
        iter_dest.set(ell, static_cast<size_t>(mdest),
                      SpherepackIterator::CoefficientArray::a);
        add_element(sign * correction.imag());

        // ImRe
        iter_src.set(ell, static_cast<size_t>(-msrc),
                     SpherepackIterator::CoefficientArray::a);
        iter_dest.set(ell, static_cast<size_t>(mdest),
                      SpherepackIterator::CoefficientArray::b);
        add_element(sign * correction.imag());
      }
    }
  }
}

// Inner loops of the rank-2 calculation.  The purpose of this
// function is so that there are not so many nested loops inside of
// the main function, making the main function and this function more
// readable.
void inner_loops_two(SparseMatrixFiller& filler, SpherepackIterator& iter_src,
                     SpherepackIterator& iter_dest, const size_t src_comp_index,
                     const size_t dest_comp_index, const size_t ell_max,
                     const size_t lprime, const int mprime, const size_t lbar,
                     const int mbar, const int mtilde, const double coeflbar,
                     const double coeflprime, WignerThreeJ& threej_lbar,
                     WignerThreeJ& threej_ltilde, const std::vector<int>& mbars,
                     const std::vector<int>& mtildes,
                     const std::array<helpers::BasisVector, 2>& src_bvs,
                     const std::array<helpers::BasisVector, 2>& dest_bvs,
                     const double symm_factor, const double sign_y,
                     std::vector<WignerThreeJ>& threej_pqs,
                     std::vector<WignerThreeJ>& threej_uvs) {
  const auto add_element = [&filler, &iter_src, &iter_dest, src_comp_index,
                            dest_comp_index](const double element) {
    const size_t indx_dest =
        iter_dest() + dest_comp_index * iter_dest.spherepack_array_size();
    const size_t indx_src =
        iter_src() + src_comp_index * iter_src.spherepack_array_size();
    filler.add(element, indx_dest, indx_src);
  };
  size_t mbar_indx = 0;
  const int m_dest = mprime + mbar;
  for (int p = -1; p <= 1; p += 2) {
    for (int q = -1; q <= 1; q += 2, ++mbar_indx) {
      if (mbar == mbars[mbar_indx]) {
        const double threej_pq_val = threej_pqs[mbar_indx](lbar);
        if (threej_pq_val != 0.0) {
          size_t mtilde_indx = 0;
          for (int u = -1; u <= 1; u += 2) {
            for (int v = -1; v <= 1; v += 2, ++mtilde_indx) {
              if (mtilde == mtildes[mtilde_indx]) {
                const double threej_uv_val = threej_uvs[mtilde_indx](lbar);
                if (threej_uv_val != 0.0) {
                  const std::complex<double> k_coefs =
                      helpers::bv_to_k(src_bvs[0], u) *
                      helpers::bv_to_k(dest_bvs[0], p) *
                      helpers::bv_to_k(src_bvs[1], v) *
                      helpers::bv_to_k(dest_bvs[1], q);
                  const int m_src = mprime - mtilde;
                  const std::complex<double> coef3j =
                      -coeflprime * coeflbar * sign_y * k_coefs;
                  for (size_t l_dest = std::max(
                           std::max(static_cast<size_t>(
                                        abs(static_cast<int>(lprime) -
                                            static_cast<int>(lbar))),
                                    static_cast<size_t>(abs(mprime + mbar))),
                           static_cast<size_t>(
                               std::max(abs(mtilde - mprime), m_dest)));
                       l_dest <= std::min(lprime + lbar, ell_max); ++l_dest) {
                    const double threej_lbar_val = threej_lbar(l_dest);
                    const double threej_ltilde_val = threej_ltilde(l_dest);
                    if (threej_lbar_val != 0.0 and threej_ltilde_val != 0.0) {
                      const double sign_lbar =
                          ((lprime + l_dest + lbar) % 2 == 0 ? 1.0 : -1.0);
                      // Corrects for permutation of columns in
                      // threej_lba
                      const std::complex<double> correction =
                          sign_lbar * threej_pq_val * threej_uv_val *
                          threej_lbar_val * threej_ltilde_val * symm_factor *
                          coef3j;
                      if (m_src > 0) {
                        // Main term, first term in Eq. (18)
                        if (correction.imag() == 0.0) {
                          // ReRe
                          iter_src.set(l_dest, static_cast<size_t>(m_src),
                                       SpherepackIterator::CoefficientArray::a);
                          iter_dest.set(
                              l_dest, static_cast<size_t>(m_dest),
                              SpherepackIterator::CoefficientArray::a);
                          add_element(correction.real());

                          // ImIm
                          iter_src.set(l_dest, static_cast<size_t>(m_src),
                                       SpherepackIterator::CoefficientArray::b);
                          iter_dest.set(
                              l_dest, static_cast<size_t>(m_dest),
                              SpherepackIterator::CoefficientArray::b);
                          add_element(correction.real());
                        } else {
                          // ReIm
                          iter_src.set(l_dest, static_cast<size_t>(m_src),
                                       SpherepackIterator::CoefficientArray::b);
                          iter_dest.set(
                              l_dest, static_cast<size_t>(m_dest),
                              SpherepackIterator::CoefficientArray::a);
                          add_element(-correction.imag());

                          // ImRe
                          iter_src.set(l_dest, static_cast<size_t>(m_src),
                                       SpherepackIterator::CoefficientArray::a);
                          iter_dest.set(
                              l_dest, static_cast<size_t>(m_dest),
                              SpherepackIterator::CoefficientArray::b);
                          add_element(correction.imag());
                        }
                      } else {
                        // Second term in Eq. (18)
                        const double sign = (m_src % 2 == 0 ? 1.0 : -1.0);
                        if (correction.imag() == 0.0) {
                          // ReRe
                          iter_src.set(l_dest, static_cast<size_t>(-m_src),
                                       SpherepackIterator::CoefficientArray::a);
                          iter_dest.set(
                              l_dest, static_cast<size_t>(m_dest),
                              SpherepackIterator::CoefficientArray::a);
                          add_element(sign * correction.real());

                          // ImIm
                          iter_src.set(l_dest, static_cast<size_t>(-m_src),
                                       SpherepackIterator::CoefficientArray::b);
                          iter_dest.set(
                              l_dest, static_cast<size_t>(m_dest),
                              SpherepackIterator::CoefficientArray::b);
                          add_element(-sign * correction.real());
                        } else {
                          // ReIm
                          iter_src.set(l_dest, static_cast<size_t>(-m_src),
                                       SpherepackIterator::CoefficientArray::b);
                          iter_dest.set(
                              l_dest, static_cast<size_t>(m_dest),
                              SpherepackIterator::CoefficientArray::a);
                          add_element(sign * correction.imag());

                          // ImRe
                          iter_src.set(l_dest, static_cast<size_t>(-m_src),
                                       SpherepackIterator::CoefficientArray::a);
                          iter_dest.set(
                              l_dest, static_cast<size_t>(m_dest),
                              SpherepackIterator::CoefficientArray::b);
                          add_element(sign * correction.imag());
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

// Inner loops of the rank-3 calculation.  The purpose of this
// function is so that there are not so many nested loops inside of
// the main function, making the main function and this function more
// readable.
template <typename Symm>
void inner_loops_three(
    SparseMatrixFiller& filler, SpherepackIterator& iter_src,
    SpherepackIterator& iter_dest, const size_t src_comp_index,
    const size_t dest_comp_index, const size_t ell_max, const size_t lprime,
    const double coeflprime, const int mprime, const size_t lhat,
    const int mhat, WignerThreeJ& threej_mhat, const int mcheck,
    WignerThreeJ& threej_mcheck, const size_t mbar_indx, const int p,
    const int q, const int r, const int mr, const std::vector<int>& mbars,
    const std::vector<int>& mtildes,
    const std::array<helpers::BasisVector, 3>& src_bvs,
    const std::array<helpers::BasisVector, 3>& dest_bvs,
    const size_t src_multiplicity, std::vector<WignerThreeJ>& threej_pqs,
    std::vector<WignerThreeJ>& threej_uvs,
    std::vector<std::optional<WignerThreeJ>>& threej_ws,
    std::vector<std::optional<WignerThreeJ>>& threej_rs, const double sign_y) {
  const auto add_element = [&filler, &iter_src, &iter_dest, src_comp_index,
                            dest_comp_index](const double element) {
    const size_t indx_dest =
        iter_dest() + dest_comp_index * iter_dest.spherepack_array_size();
    const size_t indx_src =
        iter_src() + src_comp_index * iter_src.spherepack_array_size();
    filler.add(element, indx_dest, indx_src);
  };
  const int m_dest = mprime + mcheck;
  size_t mtilde_indx = 0;
  for (int u = -1; u <= 1; u += 2) {
    for (int v = -1; v <= 1; v += 2, ++mtilde_indx) {
      for (size_t lbar = static_cast<size_t>(
               std::max(abs(mbars[mbar_indx]), abs(mtildes[mtilde_indx])));
           lbar <= 2; ++lbar) {
        const double symm_factor =
            helpers::get_symm_factor<Symm>(src_multiplicity, lbar);
        if (symm_factor != 0.0) {
          // The division inside the index of the following
          // quantities is integer division.  Note that
          // q,v,r,w,v are always odd.
          const double threej_pq =
              // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
              threej_pqs[static_cast<size_t>((q + 1) / 2 + p + 1)](lbar);
          const double threej_r =
              // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
              threej_rs[static_cast<size_t>(static_cast<int>(lbar) +
                                            (r + 1) * 3 / 2 +
                                            6 * ((q + 1) / 2 + p + 1))]
                  .value()(lhat);
          const double threej_uv =
              // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
              threej_uvs[static_cast<size_t>((v + 1) / 2 + u + 1)](lbar);
          if (threej_r != 0.0 and threej_uv != 0.0 and threej_pq != 0.0) {
            for (int w = -1; w <= 1; w += 2) {
              const int mw = helpers::bv_to_m(src_bvs[0], w);
              if (mtildes[mtilde_indx] - mw == mhat and
                  lhat >= static_cast<size_t>(std::max(
                              abs(static_cast<int>(lbar) - 1),
                              std::max(abs(mr + mbars[mbar_indx]),
                                       abs(mw - mtildes[mtilde_indx])))) and
                  lhat <= lbar + 1) {
                const int m_src = mprime - mhat;
                const double sign_mtilde =
                    ((mtildes[mtilde_indx] - mbars[mbar_indx]) % 2 == 0 ? 1.0
                                                                        : -1.0);
                const double coeflhat = 0.5 * static_cast<double>(2 * lhat + 1);
                const double coeflbar = 0.5 * static_cast<double>(2 * lbar + 1);
                const std::complex<double> k_coefs =
                    helpers::bv_to_k(src_bvs[1], u) *
                    helpers::bv_to_k(dest_bvs[1], p) *
                    helpers::bv_to_k(src_bvs[2], v) *
                    helpers::bv_to_k(dest_bvs[2], q) *
                    helpers::bv_to_k(dest_bvs[0], r) *
                    helpers::bv_to_k(src_bvs[0], w);
                const std::complex<double> coef3j =
                    -coeflprime * coeflbar * coeflhat * sign_y * k_coefs;
                for (size_t l_dest = static_cast<size_t>(std::max(
                         {abs(static_cast<int>(lprime) -
                              static_cast<int>(lhat)),
                          abs(mhat - mprime), abs(m_dest), mcheck + mprime}));
                     l_dest <= std::min(lprime + lhat, ell_max); ++l_dest) {
                  const double threej_mhat_val = threej_mhat(l_dest);
                  const double threej_mcheck_val = threej_mcheck(l_dest);
                  const double threej_w =
                      // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
                      threej_ws[static_cast<size_t>(static_cast<int>(lbar) +
                                                    (w + 1) * 3 / 2 +
                                                    6 * ((v + 1) / 2 + u + 1))]
                          .value()(lhat);
                  if (threej_mhat_val != 0.0 and threej_mcheck_val != 0.0 and
                      threej_w != 0.0) {
                    const double sign_lhat =
                        ((lprime + l_dest + lhat) % 2 == 0 ? 1.0 : -1.0);
                    const std::complex<double> correction =
                        coef3j * threej_pq * threej_uv * threej_r * threej_w *
                        threej_mhat_val * threej_mcheck_val * sign_lhat *
                        sign_mtilde * symm_factor;
                    if (m_src > 0) {
                      // Main term, first term in Eq. (18)
                      // ReRe
                      if (correction.imag() == 0) {
                        iter_src.set(l_dest, static_cast<size_t>(m_src),
                                     SpherepackIterator::CoefficientArray::a);
                        iter_dest.set(l_dest, static_cast<size_t>(m_dest),
                                      SpherepackIterator::CoefficientArray::a);
                        add_element(correction.real());

                        // ImIm
                        iter_src.set(l_dest, static_cast<size_t>(m_src),
                                     SpherepackIterator::CoefficientArray::b);
                        iter_dest.set(l_dest, static_cast<size_t>(m_dest),
                                      SpherepackIterator::CoefficientArray::b);
                        add_element(correction.real());
                      } else {
                        // ReIm
                        iter_src.set(l_dest, static_cast<size_t>(m_src),
                                     SpherepackIterator::CoefficientArray::b);
                        iter_dest.set(l_dest, static_cast<size_t>(m_dest),
                                      SpherepackIterator::CoefficientArray::a);
                        add_element(-correction.imag());

                        // ImRe
                        iter_src.set(l_dest, static_cast<size_t>(m_src),
                                     SpherepackIterator::CoefficientArray::a);
                        iter_dest.set(l_dest, static_cast<size_t>(m_dest),
                                      SpherepackIterator::CoefficientArray::b);
                        add_element(correction.imag());
                      }
                    } else {
                      // Second term in Eq. (18)
                      const double sign = (m_src % 2 == 0 ? 1.0 : -1.0);
                      if (correction.imag() == 0) {
                        // ReRe
                        iter_src.set(l_dest, static_cast<size_t>(-m_src),
                                     SpherepackIterator::CoefficientArray::a);
                        iter_dest.set(l_dest, static_cast<size_t>(m_dest),
                                      SpherepackIterator::CoefficientArray::a);
                        add_element(sign * correction.real());

                        // ImIm
                        iter_src.set(l_dest, static_cast<size_t>(-m_src),
                                     SpherepackIterator::CoefficientArray::b);
                        iter_dest.set(l_dest, static_cast<size_t>(m_dest),
                                      SpherepackIterator::CoefficientArray::b);
                        add_element(-sign * correction.real());
                      } else {
                        // ReIm
                        iter_src.set(l_dest, static_cast<size_t>(-m_src),
                                     SpherepackIterator::CoefficientArray::b);
                        iter_dest.set(l_dest, static_cast<size_t>(m_dest),
                                      SpherepackIterator::CoefficientArray::a);
                        add_element(sign * correction.imag());

                        // ImRe
                        iter_src.set(l_dest, static_cast<size_t>(-m_src),
                                     SpherepackIterator::CoefficientArray::a);
                        iter_dest.set(l_dest, static_cast<size_t>(m_dest),
                                      SpherepackIterator::CoefficientArray::b);
                        add_element(sign * correction.imag());
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
}  // namespace

template <typename TensorStructure, typename SparseMatrixType>
void fill_filter(const gsl::not_null<SparseMatrixType*> matrix,
                 const size_t ell_max, const size_t number_of_ell_modes_to_kill,
                 const std::optional<size_t> half_power) {
  static constexpr size_t num_independent_components = TensorStructure::size();
  static constexpr size_t rank = TensorStructure::rank();
  static constexpr double alpha = 36.0;

  static_assert(rank >= 0 and rank < 4, "Implemented only for ranks 0,1,2,3");

  const auto lcutplus =
      static_cast<size_t>(ell_max - number_of_ell_modes_to_kill);
  const size_t lcutminus =
      half_power.has_value()
          ? size_t(std::ceil(static_cast<double>(lcutplus + 1) *
                             pow(std::numeric_limits<double>::epsilon() / alpha,
                                 1.0 / (2.0 * double(half_power.value())))))
          : lcutplus + 1;

  SpherepackIterator iter_src(ell_max, ell_max, 1, false);
  SpherepackIterator iter_dest(ell_max, ell_max, 1, false);
  SparseMatrixFiller filler(square(num_independent_components) *
                                iter_src.spherepack_array_size() *
                                iter_dest.spherepack_array_size(),
                            true, 1.0);

  // Special case for rank 0, which is so much simpler than all the
  // other ranks because there is actually no tensorylm
  // transformation, no symmetry considerations, and no mixing of
  // real/imaginary parts, just a filter based on ell.
  if constexpr (rank == 0) {
    for (iter_src.reset(); iter_src; ++iter_src) {
      const size_t ell = iter_src.l();
      if (ell >= lcutminus) {
        const double coefl =
            half_power.has_value() and ell <= lcutplus
                ? (1.0 -
                   exp(-alpha *
                       integer_pow(double(ell) / double(lcutplus + 1),
                                   2 * static_cast<int>(half_power.value()))))
                : 1.0;
        filler.add(-coefl, iter_src(), iter_src());
      }
    }
    filler.fill(matrix);
    return;
  }

  // We allocate space for some (not all) of the 3J symbols here, so
  // that they can be used and re-used later inside the inner loops.
  std::vector<WignerThreeJ> threej_pqs;  // NOLINT(misc-const-correctness)
  std::vector<int> mbars;                // NOLINT(misc-const-correctness)
  std::vector<WignerThreeJ> threej_uvs;  // NOLINT(misc-const-correctness)
  std::vector<int> mtildes;              // NOLINT(misc-const-correctness)
  if constexpr (rank > 1) {
    threej_pqs.reserve(4);
    mbars.reserve(4);
    threej_uvs.reserve(4);
    mtildes.reserve(4);
  } else {
    // For rank 1 we don't need threej_pqs, threej_uvs, mtildes, or mbars but we
    // need to declare them anyway for scoping; they will just be unused. We
    // will compute the rank-1 3J symbols below, and below we will also compute
    // additional 3J symbols for higher ranks.
    (void)threej_pqs;
    (void)mbars;
    (void)threej_uvs;
    (void)mtildes;
  }

  // threej_rs and threej_ws contain std::optionals
  // because some of the Wigner 3Js are not well-defined
  // (because |m| > ell).
  // The ones without values are never used or referenced.
  // It is possible to compute fewer than 24
  // threej_rs and threej_ws, by pushing_back only those
  // that have values, but then it is difficult to index
  // the threej_rs and threej_ws.  So keeping 24 of
  // them and making them std::optionals is easier.
  // NOLINTNEXTLINE(misc-const-correctness)
  std::vector<std::optional<WignerThreeJ>> threej_rs;
  // NOLINTNEXTLINE(misc-const-correctness)
  std::vector<std::optional<WignerThreeJ>> threej_ws;
  if constexpr (rank > 2) {
    threej_rs.reserve(24);
    threej_ws.reserve(24);
  } else {
    // Unneeded except for rank 3.
    (void)threej_rs;
    (void)threej_ws;
  }

  for (size_t dest_comp_index = 0; dest_comp_index < num_independent_components;
       dest_comp_index++) {
    static constexpr auto tensor_index_list =
        TensorStructure::storage_to_tensor_index();

    const auto dest_indices = tensor_index_list[dest_comp_index];
    const auto dest_bvs = helpers::to_cart_basis_vector(dest_indices);

    if constexpr (rank > 1) {
      // threej_pqs is (as a function of p and q) the second 3J symbol
      // in Eq. (22) for rank 2, and the first 3J symbol in Eq. (24) for
      // rank 3.  It is not used for rank 1.
      //
      // mbars are the values of mbar for rank 2 and 3, which are uniquely
      // determined by p and q because of the symmetries of threej_pqs.
      threej_pqs.clear();
      mbars.clear();
      for (int p = -1; p <= 1; p += 2) {
        for (int q = -1; q <= 1; q += 2) {
          mbars.push_back(helpers::bv_to_m(dest_bvs[rank - 2], p) +
                          helpers::bv_to_m(dest_bvs[rank - 1], q));
          threej_pqs.emplace_back(1, helpers::bv_to_m(dest_bvs[rank - 2], p), 1,
                                  helpers::bv_to_m(dest_bvs[rank - 1], q));
        }
      }
    }

    for (size_t src_comp_index = 0; src_comp_index < num_independent_components;
         src_comp_index++) {
      const auto src_indices = tensor_index_list[src_comp_index];
      const auto src_bvs = helpers::to_cart_basis_vector(src_indices);
      const size_t src_multiplicity =
          TensorStructure::multiplicity(src_comp_index);

      const double sign_y =
          alg::accumulate(src_bvs, 1.0, [](const double acc, const auto bv) {
            return acc * (bv == helpers::BasisVector::y ? -1.0 : 1.0);
          });

      if constexpr (rank > 1) {
        // threej_uvs is (as a function of u and v) the last 3J symbol
        // in Eq. (22) for rank 2, and the second 3J symbol in Eq. (24) for
        // rank 3.  It is not used for rank 1.
        //
        // mtildes are the values of mbar for rank 2 and 3, which are uniquely
        // determined by u and v because of the symmetries of threej_uvs.
        threej_uvs.clear();
        mtildes.clear();
        for (int u = -1; u <= 1; u += 2) {
          for (int v = -1; v <= 1; v += 2) {
            mtildes.push_back(-(helpers::bv_to_m(src_bvs[rank - 2], u) +
                                helpers::bv_to_m(src_bvs[rank - 1], v)));
            threej_uvs.emplace_back(1, helpers::bv_to_m(src_bvs[rank - 2], u),
                                    1, helpers::bv_to_m(src_bvs[rank - 1], v));
          }
        }
      }

      // threej_rs and threej_ws are (as a function of r, lbar, mbar,
      // and mtilde) the last two 3J symbols in Eq. (24) for rank 3.
      // They are not used for ranks 1 and 2.
      if constexpr (rank > 2) {
        threej_rs.clear();
        for (const int mbar : mbars) {
          for (int r = -1; r <= 1; r += 2) {
            const int mr = helpers::bv_to_m(dest_bvs[0], r);
            for (size_t lbar = 0; lbar <= 2; ++lbar) {
              if (abs(mbar) <= static_cast<int>(lbar)) {
                threej_rs.emplace_back(WignerThreeJ(1, mr, lbar, mbar));
              } else {
                threej_rs.emplace_back(std::nullopt);
              }
            }
          }
        }
        threej_ws.clear();
        for (const int mtilde : mtildes) {
          for (int w = -1; w <= 1; w += 2) {
            const int mw = helpers::bv_to_m(src_bvs[0], w);
            for (size_t lbar = 0; lbar <= 2; ++lbar) {
              if (abs(mtilde) <= static_cast<int>(lbar)) {
                threej_ws.emplace_back(WignerThreeJ(1, mw, lbar, -mtilde));
              } else {
                threej_ws.emplace_back(std::nullopt);
              }
            }
          }
        }
      }

      for (size_t lprime = lcutminus; lprime <= ell_max + rank; ++lprime) {
        // coeflprime is the factor (2 lprime+1) g(lprime)/2 that appears
        // for all ranks.
        const double coeflprime =
            half_power.has_value() and lprime <= lcutplus
                ? 0.5 * static_cast<double>(2 * lprime + 1) *
                      (1.0 - exp(-alpha *
                                 integer_pow(
                                     double(lprime) / double(lcutplus + 1),
                                     2 * static_cast<int>(half_power.value()))))
                : 0.5 * static_cast<double>(2 * lprime + 1);
        for (int mprime = -static_cast<int>(lprime);
             mprime <= static_cast<int>(lprime); ++mprime) {
          // Here is where the formulas differ for different ranks.
          if constexpr (rank == 1) {
            (void)src_multiplicity;  // Unused variable for rank 1.
            for (int p = -1; p <= 1; p += 2) {
              const int mdest = mprime + helpers::bv_to_m(dest_bvs[0], p);
              if (mdest >= 0) {
                // Fill only nonnegative m, since that is all we store.

                // threej_p is the last 3J symbol in Eq. (20).
                WignerThreeJ threej_p(lprime, mprime, 1,
                                      helpers::bv_to_m(dest_bvs[0], p));
                for (int j = -1; j <= 1; j += 2) {
                  // threej_j is the first 3J symbol in Eq. (20).
                  WignerThreeJ threej_j(lprime, mprime, 1,
                                        helpers::bv_to_m(src_bvs[0], j));
                  const int msrc = mprime + helpers::bv_to_m(src_bvs[0], j);
                  // coefjp is all the factors other than the 3J symbols
                  // that appears in Eq. (20).
                  std::complex<double> coefjp =
                      -coeflprime * helpers::bv_to_k(src_bvs[0], j) *
                      helpers::bv_to_k(dest_bvs[0], p) * sign_y;
                  if ((mdest + msrc) % 2 != 0) {
                    coefjp *= -1.0;
                  }
                  inner_loops_one(filler, iter_src, iter_dest, src_comp_index,
                                  dest_comp_index, ell_max, mdest, msrc, coefjp,
                                  threej_j, threej_p);
                }
              }
            }
          } else if constexpr (rank == 2) {
            for (size_t lbar = 0; lbar <= 2; ++lbar) {
              const double symm_factor =
                  helpers::get_symm_factor<typename TensorStructure::symmetry>(
                      src_multiplicity, lbar);
              if (symm_factor != 0.0) {
                const double coeflbar = 0.5 * static_cast<double>(2 * lbar + 1);
                for (int mbar = std::max(-static_cast<int>(lbar), -mprime);
                     mbar <= static_cast<int>(lbar); ++mbar) {
                  // The first 3J term in Eq. (22)
                  WignerThreeJ threej_lbar(lprime, mprime, lbar, mbar);
                  for (int mtilde = -static_cast<int>(lbar);
                       mtilde <= static_cast<int>(lbar); ++mtilde) {
                    // The penultimate 3J term in Eq. (22)
                    WignerThreeJ threej_ltilde(lprime, -mprime, lbar, mtilde);
                    inner_loops_two(filler, iter_src, iter_dest, src_comp_index,
                                    dest_comp_index, ell_max, lprime, mprime,
                                    lbar, mbar, mtilde, coeflbar, coeflprime,
                                    threej_lbar, threej_ltilde, mbars, mtildes,
                                    src_bvs, dest_bvs, symm_factor, sign_y,
                                    threej_pqs, threej_uvs);
                  }
                }
              }
            }
          } else if constexpr (rank == 3) {
            for (size_t lhat = 0; lhat <= 3; ++lhat) {
              for (int mhat = -static_cast<int>(lhat);
                   mhat <= static_cast<int>(lhat); ++mhat) {
                // The fourth 3J term in Eq. (24)
                WignerThreeJ threej_mhat(lprime, -mprime, lhat, mhat);
                for (int mcheck = -static_cast<int>(lhat);
                     mcheck <= static_cast<int>(lhat); ++mcheck) {
                  // The third 3J term in Eq. (24)
                  WignerThreeJ threej_mcheck(lprime, mprime, lhat, mcheck);
                  size_t mbar_indx = 0;
                  for (int p = -1; p <= 1; p += 2) {
                    for (int q = -1; q <= 1; q += 2, ++mbar_indx) {
                      for (int r = -1; r <= 1; r += 2) {
                        const int mr = helpers::bv_to_m(dest_bvs[0], r);
                        if (mcheck == mr + mbars[mbar_indx] and
                            mprime + mcheck >= 0) {
                          inner_loops_three<typename TensorStructure::symmetry>(
                              filler, iter_src, iter_dest, src_comp_index,
                              dest_comp_index, ell_max, lprime, coeflprime,
                              mprime, lhat, mhat, threej_mhat, mcheck,
                              threej_mcheck, mbar_indx, p, q, r, mr, mbars,
                              mtildes, src_bvs, dest_bvs, src_multiplicity,
                              threej_pqs, threej_uvs, threej_ws, threej_rs,
                              sign_y);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  filler.fill(matrix);
}

// Explicit instantiations
#define TSTRUCT(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template void                                                               \
          fill_filter<typename TSTRUCT(data) < DataVector, 3>::structure >    \
      (gsl::not_null<SimpleSparseMatrix*> matrix, size_t ell_max,             \
       size_t number_of_ell_modes_to_kill, std::optional<size_t> half_power); \
  template void                                                               \
          fill_filter<typename TSTRUCT(data) < DataVector, 3>::structure >    \
      (gsl::not_null<blaze::CompressedMatrix<double, blaze::rowMajor>*>       \
           matrix,                                                            \
       size_t ell_max, size_t number_of_ell_modes_to_kill,                    \
       std::optional<size_t> half_power);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (tnsr::i, tnsr::ii, tnsr::ij, tnsr::ijk, tnsr::ijj))

#undef INSTANTIATE
#undef TSTRUCT

// Instantiation for scalars.
template void fill_filter<typename Scalar<DataVector>::structure>(
    gsl::not_null<SimpleSparseMatrix*> matrix, size_t ell_max,
    size_t number_of_ell_modes_to_kill, std::optional<size_t> half_power);

}  // namespace ylm::TensorYlm
