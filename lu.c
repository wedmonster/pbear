

static char help[] = "LU test version 01";

#include <petscao.h>
#include <petscvec.h>
#include <petscsys.h>
#include <petsctime.h>

int main(int argc, char** args){
    PetscErrorCode err;


    PetscMPIInt rank, size;

    // Initialize PETSC and MPI
    err = PetscInitialize(&argc, &args, (char*) 0, help); CHKERRQ(err);
    err = MPI_Comm_size(PETSC_COMM_WORLD, &size); CHKERRQ(err);
    err = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(err);
    err = PetscPrintf(PETSC_COMM_WORLD, "mpi size: %d\n", size); CHKERRQ(err);


    // Set a toy matrix (4, 4) in aij format
    Mat mat;
    PetscInt n = 4;
    MatCreate(PETSC_COMM_WORLD, &mat);
    MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, n, n);
    MatSetType(mat, MATAIJ);
    MatSetUp(mat);

    PetscInt Ii = 1, J = 1, v = 1;
    MatSetValues(mat, 1, &Ii, 1, &J, &v, INSERT_VALUES);

    MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);

    MatView(mat, PETSC_VIEWER_STDOUT_WORLD);

    MatDestroy(&mat);
    PetscFinalize();
    return 0;
}
