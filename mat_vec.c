static char help[] = "Read a matrix from a binary file";

#include <petscmat.h>
#include <petscksp.h>
#include <petscsys.h>
#include <petsctime.h>

#undef __FUNCT__
#define __FUNCT__ "main"

int main(int argc, char **args){
    Mat A;
    Vec b, y;
    char Ain[PETSC_MAX_PATH_LEN];
    PetscErrorCode ierr;
    PetscBool flg_A;
    PetscViewer fd;
    PetscScalar ysum;
    int m, n;
    PetscMPIInt rank, size;
    PetscLogDouble tic, toc;

    PetscInitialize(&argc, &args, (char*) 0, help);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

    ierr = PetscOptionsGetString(PETSC_NULL, "-Ain", Ain, PETSC_MAX_PATH_LEN, &flg_A); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\nRead a matrix from a binary file\n"); CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, Ain, FILE_MODE_READ, &fd); CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD, &A);
    ierr = MatSetFromOptions(A);
    ierr = MatLoad(A, fd); CHKERRQ(ierr);

    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    //ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);
    //ierr = MatView(A, PETSC_VIEWER_DRAW_WORLD);
    
    //ierr = MatGetLocalSize(A, &m, &n);
    //ierr = PetscPrintf(PETSC_COMM_SELF, "%d %d\n", m, n);

    ierr = MatGetSize(A, &m, &n);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "%d %d\n", m, n);

    ierr = VecCreate(PETSC_COMM_WORLD, &b); CHKERRQ(ierr);
    ierr = VecSetSizes(b, PETSC_DECIDE, n); CHKERRQ(ierr);
    ierr = VecSetFromOptions(b); CHKERRQ(ierr);
    ierr = VecSet(b, 1); CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD, &y); CHKERRQ(ierr);
    ierr = VecSetSizes(y, PETSC_DECIDE, n); CHKERRQ(ierr); 
    ierr = VecSetFromOptions(y); CHKERRQ(ierr); 
    //ierr = VecView(b, PETSC_VIEWER_STDOUT_WORLD);


    //ierr = PetscGetCPUTime(&tic);
    ierr = PetscTime(&tic);
    for(int i = 0; i < 1000; i++){
        ierr = MatMult(A, b, y); CHKERRQ(ierr);
    }
    ierr = PetscTime(&toc);
    //ierr = PetscGetCPUTime(&toc);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "%f sec\n", toc-tic);
    //ierr = VecView(y, PETSC_VIEWER_STDOUT_WORLD);

    VecSum(y, &ysum);
    PetscPrintf(PETSC_COMM_WORLD, "%d\n", (int)ysum);

    ierr = MatDestroy(&A); CHKERRQ(ierr);
    ierr = PetscFinalize();


    return 0;
}
