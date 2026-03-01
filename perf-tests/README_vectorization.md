# Vectorization comparison micro-benchmark

Этот файл добавляет простой и воспроизводимый тест, чтобы сравнить:

1. OpenMP (обычный двойной цикл).
2. Kokkos `MDRangePolicy` + `LayoutRight`.
3. Kokkos `MDRangePolicy` + `LayoutLeft` (две стратегии обхода индексов).
4. «Ручную» SIMD-векторизацию через `Kokkos::Experimental::native_simd`.

## Сборка

```bash
mkdir -p build
cd build
cmake ..
cmake --build . --target vectorization_compare -j
```

## Запуск

```bash
./bin/vectorization_compare 2048 80
```

Аргументы:
- `N` — размер двумерной сетки `N x N`.
- `reps` — число повторов ядра.

## Что именно сравнивается

Вычисление для каждой ячейки:

```text
c = c + 0.75 * a + 0.25 * b
```

Это минимальный memory-bound kernel без сложной физики, чтобы фокус был на доступе в память и векторизации.

## Как интерпретировать

- `LayoutRight` обычно лучше, когда внутренний индекс (`i`) идёт по contiguous памяти.
- Для `LayoutLeft` выгоднее делать «левый» порядок итерации policy (`Iterate::Left`), иначе легко получить strided access.
- Вариант с `native_simd` показывает, что происходит при явном использовании SIMD-регистров.

## Как проверить причины слабой автo-векторизации в Kokkos

В этом окружении submodules не подтянулись из-за сетевых ограничений (доступ к GitHub запрещён), поэтому напрямую посмотреть исходники backend OpenMP не удалось.

После инициализации submodules можно проверить следующие места в Kokkos:

```bash
rg -n "pragma omp|omp simd|ivdep|unroll" 3rdparty/kokkos/core/src/OpenMP 3rdparty/kokkos/core/src/impl
rg -n "KOKKOS_ENABLE_PRAGMA" 3rdparty/kokkos
rg -n "struct ParallelFor" 3rdparty/kokkos/core/src/OpenMP
```

Практически полезно дополнительно собирать с отчётом векторизации:

```bash
cmake .. -DCMAKE_CXX_FLAGS="-O3 -fopenmp -march=native -fopt-info-vec-optimized -fopt-info-vec-missed"
cmake --build . --target vectorization_compare -j
```

И сравнить, какие циклы компилятор реально векторизует в:
- OpenMP-версии;
- Kokkos `MDRangePolicy` с разными layout/iterate;
- SIMD-версии (где векторизация явная).
