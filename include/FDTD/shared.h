#pragma once

#include <vector>
#include <cstring>
#include <functional>
#include <cstdlib>
#include <memory>
#include <omp.h>

namespace FDTD_openmp {

template <class T, size_t Alignment = 32>
class AlignedNUMA_Allocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    template <class U>
    struct rebind {
        using other = AlignedNUMA_Allocator<U, Alignment>;
    };

    constexpr AlignedNUMA_Allocator() noexcept = default;
    constexpr AlignedNUMA_Allocator(const AlignedNUMA_Allocator&) noexcept = default;
    
    template <class U>
    constexpr AlignedNUMA_Allocator(const AlignedNUMA_Allocator<U, Alignment>&) noexcept {}

    pointer allocate(size_t n) {
        if (n == 0) return nullptr;

        const size_t size = n * sizeof(T);
        void* ptr = nullptr;
        
        if (posix_memalign(&ptr, Alignment, size) != 0) {
            throw std::bad_alloc();
        }

        //ptr = malloc(size);
        //if (ptr == nullptr) {
        //    throw std::bad_alloc();
        //}

        const int num_threads = omp_get_max_threads();
        if (num_threads > 1 && size >= Alignment * num_threads) {
            const size_t block_size = (size / num_threads) & ~(Alignment-1);
#pragma omp parallel for
            for (int thr = 0; thr < num_threads; ++thr) {
                size_t offset = thr * block_size;
                size_t bytes = (thr == num_threads - 1) ? (size - offset) : block_size;
                std::memset(static_cast<char*>(ptr) + offset, 0, bytes);
            }
        } else {
            std::memset(ptr, 0, size);
        }

        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_t) noexcept {
        free(p);
    }

    bool operator==(const AlignedNUMA_Allocator&) const { return true; }
    bool operator!=(const AlignedNUMA_Allocator&) const { return false; }
};

    using Field = std::vector<double, AlignedNUMA_Allocator<double>>;
    using TimeField = std::vector<Field, AlignedNUMA_Allocator<Field>>;
    using Function = std::function<int(int, int, int)>;
}

