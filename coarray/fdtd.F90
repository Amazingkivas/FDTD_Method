program fdtd_coarray_optimized
    use iso_fortran_env
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

    real, allocatable :: next_Ex(:,:)[:], next_Ey(:,:)[:], pred_Bx(:,:)[:], pred_By(:,:)[:]

    ! Local variables
    integer :: this_img, total_imgs, next_img, pred_img
    integer :: k_start, k_end, k_local
    integer :: idx, t
    integer(int64) :: start_time, end_time, rate
    integer :: start_i, start_j, start_k, max_i, max_j, max_k
    real :: elapsed_time
    real :: Tx, Ty, Tz, TT, bnd
    
    ! Coarray initialization
    this_img = this_image()
    total_imgs = num_images()
    next_img = merge(this_image() + 1, 1, this_image() < num_images())
    pred_img = merge(this_image() - 1, num_images(), this_image() > 1)

    TT = 8.0
    Tx = 4 * C
    Ty = 4 * C
    Tz = 4 * C

    bnd = Ni / 2.0 * dx

    start_i = floor((-Tx/4.0 + bnd) / dx) + 1
    start_j = floor((-Ty/4.0 + bnd) / dy) + 1
    start_k = floor((-Tz/4.0 + bnd) / dz) + 1

    max_i = floor((Tx/4.0 + bnd) / dx) + 1
    max_j = floor((Ty/4.0 + bnd) / dy) + 1
    max_k = floor((Tz/4.0 + bnd) / dz) + 1
    
    ! Domain decomposition
    call init_decomposition()
    
    ! Allocate arrays with optimized layout
    allocate(E(3, Ni, Nj, k_local)[*])
    allocate(B(3, Ni, Nj, k_local)[*])
    allocate(Jx(Ni, Nj, k_local)[*])
    
    allocate(next_Ex(Ni, Nj)[*])
    allocate(next_Ey(Ni, Nj)[*])
    allocate(pred_Bx(Ni, Nj)[*])
    allocate(pred_By(Ni, Nj)[*])
    
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

        if (this_img == 1) then
            call init_currents(t)
        end if
        sync all

        call update_B_field()
        sync all
        
        call update_E_field()
        sync all

        call update_B_field()
        sync all

    end do

    sync all
    if ((this_image() == 1)) then !.and. (Nk <= 16)) then
        call print_full_E_slice()
    end if
    
    sync all
    ! Final timing and cleanup
    if (this_img == 1) then
        call system_clock(count=end_time)
        elapsed_time = real(end_time - start_time)/real(rate)
        print '(A, F0.2, A)', 'Total execution time: ', elapsed_time, ' seconds'
    end if
    
    deallocate(E, B, Jx)
    deallocate(next_Ex, next_Ey, pred_Bx, pred_By)

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
        E = 0.0
        B = 0.0
        Jx = 0.0
    end subroutine init_fields

    !===============================================================
    subroutine init_currents(this_t)
        integer, intent(in) :: this_t
        integer :: i, j, k
        integer :: this_cur_img

        do k = start_k, max_k
            do j = start_j, max_j
                do i = start_i, max_i
                    this_cur_img = ceiling(real(k) / real(k_local))
                    Jx(i,j,k - (this_cur_img - 1) * k_local)[this_cur_img] = (sin(2.0 * PI * this_t * dt / TT)) &
                                * (cos(2.0 * PI * (i-1) * dx / Tx)**2) &
                                * (cos(2.0 * PI * (j-1) * dy / Ty)**2) &
                                * (cos(2.0 * PI * (k-1) * dz / Tz)**2)
                end do
            end do
        end do
    end subroutine init_currents

    !===============================================================
    subroutine update_B_field
        integer :: ip, jp, kp
        integer :: i, j, k

        next_Ex(:,:) = E(1, :, :, 1)[next_img]
        next_Ey(:,:) = E(2, :, :, 1)[next_img]
        sync all

        do k = 1, k_local - 1
            do j = 1, Nj
                do i = 1, Ni
                    ip = merge(i+1, 1, i < Ni)
                    jp = merge(j+1, 1, j < Nj)
	                    
                    ! Update Bx component
                    B(1,i,j,k) = B(1,i,j,k) + coef_B_dz * (E(2,i,j,k+1) - E(2,i,j,k)) - &
                                                coef_B_dy * (E(3,i,jp,k) - E(3,i,j,k))
                    
                    ! Update By component
                    B(2,i,j,k) = B(2,i,j,k) + coef_B_dx * (E(3,ip,j,k) - E(3,i,j,k)) - &
                                                coef_B_dz * (E(1,i,j,k+1) - E(1,i,j,k))
                    
                    ! Update Bz component
                    B(3,i,j,k) = B(3,i,j,k) + coef_B_dy * (E(1,i,jp,k) - E(1,i,j,k)) - &
                                                coef_B_dx * (E(2,ip,j,k) - E(2,i,j,k))
                end do
            end do
        end do
        do j = 1, Nj
            do i = 1, Ni
                ip = merge(i+1, 1, i < Ni)
                jp = merge(j+1, 1, j < Nj)

                ! Update Bx component
                B(1,i,j,k_local) = B(1,i,j,k_local) + coef_B_dz * (next_Ey(i,j) - E(2,i,j,k_local)) - &
                                            coef_B_dy * (E(3,i,jp,k_local) - E(3,i,j,k_local))

                ! Update By component
                B(2,i,j,k_local) = B(2,i,j,k_local) + coef_B_dx * (E(3,ip,j,k_local) - E(3,i,j,k)) - &
                                            coef_B_dz * (next_Ex(i,j) - E(1,i,j,k_local))

                ! Update Bz component
                B(3,i,j,k_local) = B(3,i,j,k_local) + coef_B_dy * (E(1,i,jp,k_local) - E(1,i,j,k_local)) - &
                                            coef_B_dx * (E(2,ip,j,k_local) - E(2,i,j,k_local))
            end do
        end do

    end subroutine update_B_field

    !===============================================================
    subroutine update_E_field
        integer :: im, jm, km
        integer :: i, j, k

        pred_Bx(:,:) = B(1, :, :, k_local)[pred_img]
        pred_By(:,:) = B(2, :, :, k_local)[pred_img]
        sync all

        do j = 1, Nj
            do i = 1, Ni
                im = merge(i-1, Ni, i > 1)
                jm = merge(j-1, Nj, j > 1)

                ! Update Ex component
                E(1,i,j,1) = E(1,i,j,1) - coef_J * Jx(i,j,1) + &
                            coef_E_dy * (B(3,i,j,1) - B(3,i,jm,1)) - &
                            coef_E_dz * (B(2,i,j,1) - pred_By(i,j))

                ! Update Ey component
                E(2,i,j,1) = E(2,i,j,1) - coef_J * Jx(i,j,1) + &
                            coef_E_dz * (B(1,i,j,1) - pred_Bx(i,j)) - &
                            coef_E_dx * (B(3,i,j,1) - B(3,im,j,1))

                ! Update Ez component
                E(3,i,j,1) = E(3,i,j,1) - coef_J * Jx(i,j,1) + &
                            coef_E_dx * (B(2,i,j,1) - B(2,im,j,1)) - &
                            coef_E_dy * (B(1,i,j,1) - B(1,i,jm,1))
            end do
        end do
        do k = 2, k_local
            do j = 1, Nj
                do i = 1, Ni
                    im = merge(i-1, Ni, i > 1)
                    jm = merge(j-1, Nj, j > 1)
                    
                    ! Update Ex component
                    E(1,i,j,k) = E(1,i,j,k) - coef_J * Jx(i,j,k) + &
                                coef_E_dy * (B(3,i,j,k) - B(3,i,jm,k)) - &
                                coef_E_dz * (B(2,i,j,k) - B(2,i,j,k-1))
                    
                    ! Update Ey component
                    E(2,i,j,k) = E(2,i,j,k) - coef_J * Jx(i,j,k) + &
                                coef_E_dz * (B(1,i,j,k) - B(1,i,j,k-1)) - &
                                coef_E_dx * (B(3,i,j,k) - B(3,im,j,k))
                    
                    ! Update Ez component
                    E(3,i,j,k) = E(3,i,j,k) - coef_J * Jx(i,j,k) + &
                                coef_E_dx * (B(2,i,j,k) - B(2,im,j,k)) - &
                                coef_E_dy * (B(1,i,j,k) - B(1,i,jm,k))
                end do
            end do
        end do
    end subroutine update_E_field

    !===============================================================
subroutine print_full_E_slice()
    integer :: img, ik, ij
    do ik = Ni/2-5, Ni/2+5
        do ij = Nj/2-5, Nj/2+5
            img = ceiling(real(Nk/2+1) / real(k_local))
            write(*, '(F12.6)', advance='no')  E(1,ik,ij,Nk/2+1 - (img - 1) * k_local)[img]
!            write(*, '(F12.6)', advance='no') next_Ex(ik, ij)
        end do
        print *
    end do

end subroutine print_full_E_slice
end program fdtd_coarray_optimized
