static char help[] = "Merge vectors test";

#include <petscao.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petsctime.h>

int main(int argc, char** args){
    Vec v;
    Vec q1, q2, r, or;
    VecScatter scatter;
    PetscInt N = 22963;
    PetscInt n2 = 575;
    PetscInt n1 = N - n2;
    PetscMPIInt size, rank;
    PetscViewer fd;
    PetscInt low, high;
    PetscScalar val;
    PetscInt l_n1, l_n2;
    PetscInt *idx, *q2_idx;
    PetscScalar *q1_array, *q2_array, *v_array;
    PetscInt nlocal;
    PetscInt *idx_to, *idx_from;
    IS to, from;
    int i;

    PetscInitialize(&argc, &args, (char*) 0, help);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    PetscPrintf(PETSC_COMM_WORLD, "mpi size: %d\n", size);

    PetscViewerBinaryOpen(PETSC_COMM_SELF, "./data/order.dat", FILE_MODE_READ, &fd);
    VecCreate(PETSC_COMM_SELF, &v);
    VecSetType(v, VECSEQ);
    VecLoad(v, fd);
    VecShift(v, -1);

    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, n1, &q1);
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, n2, &q2);
    
    VecGetOwnershipRange(q1, &low, &high);
    for(i = low; i < high; i++)
        VecSetValue(q1, i, i, INSERT_VALUES);
    VecAssemblyBegin(q1);
    VecAssemblyEnd(q1);

    VecGetOwnershipRange(q2, &low, &high);
    for(i = low; i < high; i++)
        VecSetValue(q2, i, (i+n1), INSERT_VALUES);
    VecAssemblyBegin(q2);
    VecAssemblyEnd(q2);
    //VecView(q2, PETSC_VIEWER_STDOUT_WORLD);

    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &r);
    VecGetOwnershipRange(q1, &low, &high);
    VecGetLocalSize(q1, &l_n1);
    PetscMalloc1(l_n1, &idx);
    for(i = 0; i < l_n1; i++) idx[i] = low + i;
    VecGetArray(q1, &q1_array);
    VecSetValues(r, l_n1, idx, q1_array, INSERT_VALUES);
    VecAssemblyBegin(r);
    VecAssemblyEnd(r);
    VecRestoreArray(q1, &q1_array);
    PetscFree(idx);

    VecGetOwnershipRange(q2, &low, &high);
    VecGetLocalSize(q2, &l_n2);
    PetscMalloc1(l_n2, &q2_idx);
    for(i = 0; i < l_n2; i++){
        q2_idx[i] = low + i + n1;
    }
    VecGetArray(q2, &q2_array);
    VecSetValues(r, l_n2, q2_idx, q2_array, INSERT_VALUES);
    VecAssemblyBegin(r);
    VecAssemblyEnd(r);
    VecRestoreArray(q2, &q2_array);
    PetscFree(q2_idx);

    //VecView(r, PETSC_VIEWER_STDOUT_WORLD);
    VecGetLocalSize(r, &nlocal);
    PetscMalloc1(nlocal, &idx_to);
    PetscMalloc1(nlocal, &idx_from);
    VecGetOwnershipRange(r, &low, &high);
    VecGetArray(v, &v_array);
    for(i = 0; i < nlocal; i++){
        idx_to[i] = (PetscInt) v_array[low + i];
        idx_from[i] = (PetscInt) low + i;
    }
    VecRestoreArray(v, &v_array);

    ISCreateGeneral(PETSC_COMM_SELF, nlocal, idx_from, PETSC_COPY_VALUES, &to);
    ISCreateGeneral(PETSC_COMM_SELF, nlocal, idx_to, PETSC_COPY_VALUES, &from);

    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &or);
    VecScatterCreate(r, from, or, to, &scatter);
    VecScatterBegin(scatter, r, or, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(scatter, r, or, INSERT_VALUES, SCATTER_FORWARD);
    
    VecView(or, PETSC_VIEWER_STDOUT_WORLD);
    PetscFree(idx_to);
    PetscFree(idx);


    PetscFinalize();

    return 0;
}
