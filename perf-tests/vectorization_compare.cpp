#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::high_resolution_clock;

template <class Func>
double measure_seconds(Func&& fn) {
  const auto start = Clock::now();
  fn();
  const auto stop = Clock::now();
  return std::chrono::duration<double>(stop - start).count();
}

double checksum(const std::vector<double>& v) {
  return std::accumulate(v.begin(), v.end(), 0.0);
}

void init_arrays(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c,
                 int n) {
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const int idx = j * n + i;
      a[idx] = std::sin(0.001 * static_cast<double>(i + j));
      b[idx] = std::cos(0.0015 * static_cast<double>(i + 2 * j));
      c[idx] = 0.0;
    }
  }
}

double run_openmp(int n, int reps) {
  std::vector<double> a(n * n), b(n * n), c(n * n);
  init_arrays(a, b, c, n);

  const double elapsed = measure_seconds([&]() {
    for (int rep = 0; rep < reps; ++rep) {
#pragma omp parallel for schedule(static)
      for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
          const int idx = j * n + i;
          c[idx] += 0.75 * a[idx] + 0.25 * b[idx];
        }
      }
    }
  });

  std::cout << "openmp_checksum=" << std::setprecision(15) << checksum(c) << '\n';
  return elapsed;
}

template <class Layout, class IterateOuter, class IterateInner>
double run_kokkos_layout(const std::string& label, int n, int reps) {
  using View2D = Kokkos::View<double**, Layout, Kokkos::DefaultExecutionSpace>;

  View2D a("a", n, n), b("b", n, n), c("c", n, n);

  Kokkos::parallel_for(
      "init", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, n}),
      KOKKOS_LAMBDA(const int j, const int i) {
        a(j, i) = sin(0.001 * static_cast<double>(i + j));
        b(j, i) = cos(0.0015 * static_cast<double>(i + 2 * j));
        c(j, i) = 0.0;
      });
  Kokkos::fence();

  const double elapsed = measure_seconds([&]() {
    for (int rep = 0; rep < reps; ++rep) {
      Kokkos::parallel_for(
          label,
          Kokkos::MDRangePolicy<Kokkos::Rank<2, IterateOuter, IterateInner>>({0, 0},
                                                                               {n, n}),
          KOKKOS_LAMBDA(const int j, const int i) {
            c(j, i) += 0.75 * a(j, i) + 0.25 * b(j, i);
          });
    }
    Kokkos::fence();
  });

  auto c_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c);
  double sum = 0.0;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      sum += c_host(j, i);
    }
  }
  std::cout << label << "_checksum=" << std::setprecision(15) << sum << '\n';
  return elapsed;
}

double run_kokkos_simd(int n, int reps) {
  using simd_t = Kokkos::Experimental::native_simd<double>;
  constexpr int lanes = simd_t::size();

  std::vector<double> a(n * n), b(n * n), c(n * n);
  init_arrays(a, b, c, n);

  const int total = n * n;
  const int simd_end = (total / lanes) * lanes;

  const double elapsed = measure_seconds([&]() {
    for (int rep = 0; rep < reps; ++rep) {
#pragma omp parallel for schedule(static)
      for (int idx = 0; idx < simd_end; idx += lanes) {
        simd_t av, bv, cv;
        av.copy_from(a.data() + idx, Kokkos::Experimental::simd_flag_default);
        bv.copy_from(b.data() + idx, Kokkos::Experimental::simd_flag_default);
        cv.copy_from(c.data() + idx, Kokkos::Experimental::simd_flag_default);
        cv += 0.75 * av + 0.25 * bv;
        cv.copy_to(c.data() + idx, Kokkos::Experimental::simd_flag_default);
      }
      for (int idx = simd_end; idx < total; ++idx) {
        c[idx] += 0.75 * a[idx] + 0.25 * b[idx];
      }
    }
  });

  std::cout << "kokkos_simd_checksum=" << std::setprecision(15) << checksum(c) << '\n';
  return elapsed;
}

}  // namespace

int main(int argc, char* argv[]) {
  int n = 2048;
  int reps = 80;
  if (argc > 1) n = std::atoi(argv[1]);
  if (argc > 2) reps = std::atoi(argv[2]);

  Kokkos::initialize(argc, argv);
  {
    std::cout << "N=" << n << ", reps=" << reps << '\n';

    const double t_openmp = run_openmp(n, reps);
    const double t_right =
        run_kokkos_layout<Kokkos::LayoutRight, Kokkos::Iterate::Right, Kokkos::Iterate::Right>(
            "kokkos_layout_right_rr", n, reps);
    const double t_left_bad =
        run_kokkos_layout<Kokkos::LayoutLeft, Kokkos::Iterate::Right, Kokkos::Iterate::Right>(
            "kokkos_layout_left_rr", n, reps);
    const double t_left_good =
        run_kokkos_layout<Kokkos::LayoutLeft, Kokkos::Iterate::Left, Kokkos::Iterate::Left>(
            "kokkos_layout_left_ll", n, reps);
    const double t_simd = run_kokkos_simd(n, reps);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "openmp_time_s=" << t_openmp << '\n';
    std::cout << "kokkos_layout_right_rr_time_s=" << t_right
              << " speedup_vs_openmp=" << t_openmp / t_right << '\n';
    std::cout << "kokkos_layout_left_rr_time_s=" << t_left_bad
              << " speedup_vs_openmp=" << t_openmp / t_left_bad << '\n';
    std::cout << "kokkos_layout_left_ll_time_s=" << t_left_good
              << " speedup_vs_openmp=" << t_openmp / t_left_good << '\n';
    std::cout << "kokkos_simd_time_s=" << t_simd << " speedup_vs_openmp=" << t_openmp / t_simd
              << '\n';
  }
  Kokkos::finalize();

  return 0;
}
