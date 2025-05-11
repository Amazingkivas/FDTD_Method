#pragma once

#include <vector>
#include <cstring>
#include <functional>
#include <cstdlib>
#include <memory>
#include <omp.h>

namespace FDTD_openmp {

template <class T>
class no_init_allocator
{
public:
    typedef T value_type;

    no_init_allocator() noexcept {}
    template <class U>
        no_init_allocator(const no_init_allocator<U>&) noexcept {}
    T* allocate(std::size_t n)
        {return static_cast<T*>(::operator new(n * sizeof(T)));}
    void deallocate(T* p, std::size_t) noexcept
        {::operator delete(static_cast<void*>(p));}
    template <class U>
        void construct(U*) noexcept
        {
            static_assert(std::is_trivially_default_constructible<U>::value,
            "This allocator can only be used with trivally default constructible types");
        }
    template <class U, class A0, class... Args>
        void construct(U* up, A0&& a0, Args&&... args) noexcept
        {
            ::new(up) U(std::forward<A0>(a0), std::forward<Args>(args)...);
        }
};

    using Field = std::vector<double, no_init_allocator<double>>;
    using TimeField = std::vector<Field, no_init_allocator<Field>>;
    using Function = std::function<int(int, int, int)>;
}

