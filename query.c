static char help[] = "Bear Query version 01";

#include <petscao.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petsctime.h>

PetscErrorCode loadMat(const char *path, Mat* A, MPI_Comm comm, PetscViewer* fd){
    PetscErrorCode err;
    err = PetscViewerBinaryOpen(comm, path, FILE_MODE_READ, fd); CHKERRQ(err);
    err = MatCreate(comm, A); CHKERRQ(err);
    err = MatSetFromOptions(*A); CHKERRQ(err);
    err = MatLoad(*A, *fd); CHKERRQ(err);
    err = PetscViewerDestroy(fd); CHKERRQ(err);
    return err;
}

PetscErrorCode loadVec(const char *path, Vec* v, MPI_Comm comm, PetscViewer* fd){
    PetscErrorCode err;
    err = PetscViewerBinaryOpen(comm, path, FILE_MODE_READ, fd); CHKERRQ(err);
    err = VecCreate(comm, v); CHKERRQ(err);
    err = VecSetFromOptions(*v); CHKERRQ(err);
    err = VecLoad(*v, *fd); CHKERRQ(err);
    err = PetscViewerDestroy(fd); CHKERRQ(err);
    return err;
}

PetscErrorCode checkMat(const char *label, Mat A){
    int m, n;
    PetscErrorCode err;
    err = MatGetSize(A, &m, &n); CHKERRQ(err);
    err = PetscPrintf(PETSC_COMM_WORLD, "%s's size: %d x %d\n", label, m, n); CHKERRQ(err);
    return err;
}

PetscErrorCode checkVec(const char *label, Vec v){
    int m;
    PetscErrorCode err;
    err = VecGetSize(v, &m); CHKERRQ(err);
    err = PetscPrintf(PETSC_COMM_WORLD, "%s's size: %d x 1\n", label, m); CHKERRQ(err);
    return err;
}  

int main(int argc, char** args){
    PetscErrorCode err;
    PetscViewer fd = NULL;
    Mat invL1 = NULL, invU1 = NULL, invL2 = NULL, invU2 = NULL, H12 = NULL, H21 = NULL;
    Vec order = NULL;
    PetscMPIInt rank, size;

    // Initialize PETSC and MPI
    err = PetscInitialize(&argc, &args, (char*) 0, help); CHKERRQ(err);
    err = MPI_Comm_size(PETSC_COMM_WORLD, &size); CHKERRQ(err);
    err = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(err);
    err = PetscPrintf(PETSC_COMM_WORLD, "mpi size: %d\n", size); CHKERRQ(err); 

    // Read matrices and an ordering vector
    err = PetscPrintf(PETSC_COMM_WORLD, "Read inputs (invL1, invU1, invL2, invU2, H12, H21, order)\n"); CHKERRQ(err);
    err = loadMat("./data/invL1.dat", &invL1, PETSC_COMM_WORLD, &fd); CHKERRQ(err);
    err = loadMat("./data/invU1.dat", &invU1, PETSC_COMM_WORLD, &fd); CHKERRQ(err);
    err = loadMat("./data/invL2.dat", &invL2, PETSC_COMM_WORLD, &fd); CHKERRQ(err);
    err = loadMat("./data/invU2.dat", &invU2, PETSC_COMM_WORLD, &fd); CHKERRQ(err);
    err = loadMat("./data/H12.dat", &H12, PETSC_COMM_WORLD, &fd); CHKERRQ(err);
    err = loadMat("./data/H21.dat", &H21, PETSC_COMM_WORLD, &fd); CHKERRQ(err);
    err = loadVec("./data/order.dat", &order, PETSC_COMM_SELF, &fd); CHKERRQ(err); //all processes must have this vector for ordering the result vector.
    
    // Check input
    err = checkMat("invL1", invL1); CHKERRQ(err);
    err = checkMat("invU1", invU1); CHKERRQ(err);
    err = checkMat("invL2", invL2); CHKERRQ(err);
    err = checkMat("invU2", invU2); CHKERRQ(err);
    err = checkMat("H12", H12); CHKERRQ(err);
    err = checkMat("H21", H21); CHKERRQ(err);
    err = checkVec("order", order); CHKERRQ(err);

    // Destory matrices and vectors
    err = MatDestroy(&invL1); CHKERRQ(err);
    err = MatDestroy(&invU1); CHKERRQ(err);
    err = MatDestroy(&invL2); CHKERRQ(err);
    err = MatDestroy(&invU2); CHKERRQ(err);
    err = MatDestroy(&H12); CHKERRQ(err);
    err = MatDestroy(&H21); CHKERRQ(err);
    err = VecDestroy(&order); CHKERRQ(err);

    // Finalize
    err = PetscFinalize(); CHKERRQ(err);
    return 0;
}
