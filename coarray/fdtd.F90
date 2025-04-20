program fdtd_coarray_optimized
    use iso_fortran_env
    use omp_lib
    implicit none

    ! Parameters
    integer, parameter :: Ni = 512, Nj = 512, Nk = 512
    integer, parameter :: num_iterations = 25
    real, parameter :: C = 3e10, PI = 3.14159265358
    real, parameter :: dx = C, dy = C, dz = C, dt = 0.2
    real, parameter :: coef_B_dx = C * dt / (2 * dx), coef_B_dy = C * dt / (2 * dy), coef_B_dz = C * dt / (2 * dz)
    real, parameter :: coef_E_dx = C * dt / dx, coef_E_dy = C * dt / dy, coef_E_dz = C * dt / dz
    real, parameter :: coef_J = 4*PI*dt

    ! Field arrays (optimized layout for vectorization)
    real, allocatable :: E(:,:,:,:)[:]   ! (3, Ni, Nj, k_local) - Ex,Ey,Ez components
    real, allocatable :: B(:,:,:,:)[:]   ! (3, Ni, Nj, k_local) - Bx,By,Bz components
    real, allocatable :: Jx(:,:,:)[:]    ! Current density

    ! Local variables
    integer :: this_img, total_imgs
    integer :: k_start, k_end, k_local
    integer :: i, j, k, t, idx
    integer(int64) :: start_time, end_time, rate
    real :: elapsed_time

    ! Coarray initialization
    this_img = this_image()
    total_imgs = num_images()

    ! Domain decomposition
    call init_decomposition()

    ! Allocate arrays with optimized layout
    allocate(E(3, Ni, Nj, k_local)[*])
    allocate(B(3, Ni, Nj, k_local)[*])
    allocate(Jx(Ni, Nj, k_local)[*])

    ! Initialize fields
    call init_fields()

    ! Timing and output
    if (this_img == 1) then
        print '(A, I0)', 'Running on ', total_imgs, ' images'
        call system_clock(count_rate=rate)
        call system_clock(count=start_time)
    end if

    ! Main FDTD loop
    do t = 1, num_iterations
        if (this_img == 1) print '(A, I3)', 'Iteration ', t

        call init_currents()
        sync all

        call update_B_field()
        sync all

        call update_E_field(t)
        sync all

        call update_B_field()
        sync all
    end do

    ! Final timing and cleanup
    if (this_img == 1) then
        call system_clock(count=end_time)
        elapsed_time = real(end_time - start_time)/real(rate)
        print '(A, F0.2, A)', 'Total execution time: ', elapsed_time, ' seconds'
    end if

    deallocate(E, B, Jx)

contains
    !===============================================================
    subroutine init_decomposition
        integer :: remainder, offset

        remainder = mod(Nk, total_imgs)
        k_local = Nk / total_imgs

        if (this_img <= remainder) then
            k_local = k_local + 1
            offset = (this_img - 1)*k_local
        else
            offset = remainder*(k_local + 1) + (this_img - remainder - 1)*k_local
        end if

        k_start = offset + 1
        k_end = offset + k_local
    end subroutine init_decomposition

    !===============================================================
    subroutine init_fields
        !DEC$ VECTOR ALIGNED
        E = 0.0
        B = 0.0
    end subroutine init_fields

    !===============================================================
    subroutine init_currents
        !DEC$ VECTOR ALIGNED
        do k = 1, k_local
            do j = 1, Nj
                do i = 1, Ni
                    Jx(i,j,k) = sin(2*PI*i/Ni) * cos(2*PI*j/Nj) * cos(2*PI*(k_start + k - 1)/Nk)
                end do
            end do
        end do
    end subroutine init_currents

    !===============================================================
    subroutine update_B_field
        integer :: ip, jp, kp

        !$OMP SIMD COLLAPSE(2)
        do k = 1, k_local
            do j = 1, Nj
                !$OMP SIMD
                do i = 1, Ni
                    ip = merge(i+1, 1, i < Ni)
                    jp = merge(j+1, 1, j < Nj)
                    kp = merge(k+1, 1, k < k_local)

                    ! Update Bx component
                    B(1,i,j,k) = B(1,i,j,k) + coef_B_dz * (E(2,i,j,kp) - E(2,i,j,k)) - &
                                                coef_B_dy * (E(3,i,jp,k) - E(3,i,j,k))

                    ! Update By component
                    B(2,i,j,k) = B(2,i,j,k) + coef_B_dx * (E(3,ip,j,k) - E(3,i,j,k)) - &
                                                coef_B_dz * (E(1,i,j,kp) - E(1,i,j,k))

                    ! Update Bz component
                    B(3,i,j,k) = B(3,i,j,k) + coef_B_dy * (E(1,i,jp,k) - E(1,i,j,k)) - &
                                                coef_B_dx * (E(2,ip,j,k) - E(2,i,j,k))
                end do
            end do
        end do
    end subroutine update_B_field

    !===============================================================
    subroutine update_E_field(t)
        integer, intent(in) :: t
        integer :: im, jm, km

        !$OMP SIMD COLLAPSE(2)
        do k = 1, k_local
            do j = 1, Nj
                !$OMP SIMD
                do i = 1, Ni
                    im = merge(i-1, Ni, i > 1)
                    jm = merge(j-1, Nj, j > 1)
                    km = merge(k-1, k_local, k > 1)

                    ! Update Ex component
                    E(1,i,j,k) = E(1,i,j,k) - coef_J * Jx(i,j,k) + &
                                coef_E_dy * (B(3,i,j,k) - B(3,i,jm,k)) - &
                                coef_E_dz * (B(2,i,j,k) - B(2,i,j,km))

                    ! Update Ey component
                    E(2,i,j,k) = E(2,i,j,k) - coef_J * Jx(i,j,k) + &
                                coef_E_dz * (B(1,i,j,k) - B(1,i,j,km)) - &
                                coef_E_dx * (B(3,i,j,k) - B(3,im,j,k))

                    ! Update Ez component
                    E(3,i,j,k) = E(3,i,j,k) - coef_J * Jx(i,j,k) + &
                                coef_E_dx * (B(2,i,j,k) - B(2,im,j,k)) - &
                                coef_E_dy * (B(1,i,j,k) - B(1,i,jm,k))
                end do
            end do
        end do
    end subroutine update_E_field

end program fdtd_coarray_optimized
