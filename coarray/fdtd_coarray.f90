program fdtd_coarray
    use iso_fortran_env
    implicit none

    integer, parameter :: Ni = 512, Nj = 512, Nk = 512
    integer, parameter :: num_iterations = 5
    real, parameter :: C = 3e10, PI = 3.14159265358
    real, parameter :: dx = C, dy = C, dz = C, dt = 0.2
    real, parameter :: coef_B_dx = C * dt / (2 * dx), coef_B_dy = C * dt / (2 * dy), coef_B_dz = C * dt / (2 * dz)
    real, parameter :: coef_E_dx = C * dt / dx, coef_E_dy = C * dt / dy, coef_E_dz = C * dt / dz
    real, parameter :: coef_J = 4*PI*dt
    real :: center_ex
    integer :: k_slice = 0

    real, allocatable :: Ex(:)[:], Ey(:)[:], Ez(:)[:]
    real, allocatable :: Bx(:)[:], By(:)[:], Bz(:)[:]
    real, allocatable :: Jx(:,:,:)[:]
    integer :: target_img, local_k, size, index
    integer :: this_img, total_imgs, i, j, k, t
    integer :: k_start, k_end, k_local
    character(len=20) :: time_str
    integer(int64) :: start_time, end_time, rate
    real :: elapsed_time

    this_img = this_image()
    total_imgs = num_images()

    call init_decomposition()
    size = Ni * Nj * k_local

    allocate(Ex(size)[*])
    allocate(Ey(size)[*])
    allocate(Ez(size)[*])
    allocate(Bx(size)[*])
    allocate(By(size)[*])
    allocate(Bz(size)[*])
    allocate(Jx(Ni, Nj, k_local)[*])

    call init_fields()
    call init_currents()
    sync all

    if (this_img == 1) then
        print '(A, I0)',  'images: ', total_imgs
        call system_clock(count_rate=rate)
        call system_clock(count=start_time)
    end if
    if (this_img == 1) then
        call get_current_time(time_str)

        target_img = (1 - 1) / (Nk / num_images()) + 1
        local_k = mod(1 - 1, (Nk / num_images())) + 1
    end if
    sync all

    do t = 1, num_iterations
	if (this_img == 1) then
	    print '(A, I3, A, A, A, F10.6)', 'Iteration ', t
	end if

        call update_B_field()
        sync all

        call update_E_field(t)
        sync all

        call update_B_field()
        sync all
    end do

    !call print_Bx_slice(num_iterations-1)

    if (this_img == 1) then
        call system_clock(count=end_time)
        elapsed_time = real(end_time - start_time)/real(rate)
        print '(A, F0.2, A)', 'Total execution time: ', elapsed_time, ' seconds'
    end if


    deallocate(Ex, Ey, Ez, Bx, By, Bz, Jx)

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
        Ex = 0.0; Ey = 0.0; Ez = 0.0
        Bx = 0.0; By = 0.0; Bz = 0.0
    end subroutine init_fields

    !===============================================================
    subroutine init_currents
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
        integer :: ip, jp, kp, index
        
        do k = 1, k_local
            do j = 1, Nj
                index = (j-1) * Ni + (k-1) * Ni * Nj
                do i = 1, Ni
                    index = i + index
                    ip = merge(i+1, 1, i < Ni)
                    jp = merge(j+1, 1, j < Nj)
                    kp = merge(k+1, 1, k < k_local)
                    
                    ip = ip + (j-1) * Ni + (k-1) * Ni * Nj
                    jp = i + (jp-1) * Ni + (k-1) * Ni * Nj
                    kp = i + (j-1) * Ni + (kp-1) * Ni * Nj

                    Bx(index) = Bx(index) + coef_B_dz * (Ey(kp) - Ey(index)) - coef_B_dy * (Ez(jp) - Ez(index))

                    By(index) = By(index) + coef_B_dx * (Ez(ip) - Ez(index)) - coef_B_dz * (Ex(kp) - Ex(index))

                    Bz(index) = Bz(index) + coef_B_dy * (Ex(jp) - Ex(index)) - coef_B_dx * (Ey(ip) - Ey(index))
                end do
            end do
        end do
    end subroutine update_B_field

    !===============================================================
    subroutine update_E_field(t)
        integer, intent(in) :: t
        integer :: im, jm, km, index
        real :: Jx_val

        do k = 1, k_local
            do j = 1, Nj
                index = (j-1) * Ni + (k-1) * Ni * Nj
                do i = 1, Ni
                    im = merge(i-1, Ni, i > 1)
                    jm = merge(j-1, Nj, j > 1)
                    km = merge(k-1, k_local, k > 1)

                    im = im + (j-1) * Ni + (k-1) * Ni * Nj
                    jm = i + (jm-1) * Ni + (k-1) * Ni * Nj
                    km = i + (j-1) * Ni + (km-1) * Ni * Nj

                    Jx_val = coef_J * Jx(i,j,k)

                    Ex(index) = Ex(index) - Jx_val + coef_E_dy * (Bz(index) - Bz(jm)) - coef_E_dz * (By(index) - By(km))

                    Ey(index) = Ey(index) - Jx_val + coef_E_dz * (Bx(index) - Bx(km)) - coef_E_dx * (Bz(index) - Bz(im))

                    Ez(index) = Ez(index) - Jx_val + coef_E_dx * (By(index) - By(im)) - coef_E_dy * (Bx(index) - Bx(jm))
                end do
            end do
        end do
    end subroutine update_E_field

    !===============================================================
    subroutine get_current_time(time_str)
        character(len=*), intent(out) :: time_str
        integer :: values(8)
        call date_and_time(values=values)
        write(time_str, '(I4.4,"-",I2.2,"-",I2.2," ",I2.2,":",I2.2,":",I2.2)') &
            values(1:3), values(5:7)
    end subroutine get_current_time

    subroutine print_Bx_slice(iteration)
        integer, intent(in) :: iteration
        integer :: k_print, i, j, idx
        character(len=100) :: filename

        k_print = k_local / 2
        if (k_slice == 0) k_slice = k_print
        write(filename, '(A,I0,A,I0,A,I0,A)') 'Bx_slice_img_', this_img, &
                                            '_iter_', iteration, &
                                            '_k_', k_start + k_slice - 1, &
                                            '.dat'

        open(unit=20, file=filename, status='replace', action='write')

        write(20, '(A,I0,A,I0,A,I0)') '# Slice Bx at k=', k_start + k_slice - 1, &
                                    ' (local k=', k_slice, ') on image ', this_img
        write(20, '(A)') '# i, j, Bx_value'

        do j = 1, Nj
            do i = 1, Ni
                idx = i + (j-1)*Ni + (k_slice-1)*Ni*Nj
                write(20, '(I5,I5,E15.6)') i, j, Bx(idx)
            end do
            write(20, *)
        end do

        close(20)

        sync all
        if (this_img == 1) then
            print *, 'Bx slices written for iteration ', iteration
        end if
    end subroutine print_Bx_slice

end program fdtd_coarray
