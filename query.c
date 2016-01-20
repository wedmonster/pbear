static char help[] = "Bear Query version 01";

#include <petscao.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petsctime.h>

PetscErrorCode loadMat(const char *path, Mat* A, MPI_Comm comm, MatType type, PetscViewer* fd){
    PetscErrorCode err;
    err = PetscViewerBinaryOpen(comm, path, FILE_MODE_READ, fd); CHKERRQ(err);
    err = MatCreate(comm, A); CHKERRQ(err);
    err = MatSetType(*A, type); CHKERRQ(err);
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

PetscErrorCode printVecSum(Vec v){
    PetscErrorCode err;
    PetscScalar sum;
    err = VecSum(v, &sum); CHKERRQ(err);
    err = PetscPrintf(PETSC_COMM_WORLD, "%lf\n", sum);
    return err;
}
PetscErrorCode printMatInfo(const char *label, Mat m){
    PetscErrorCode err;
    MatInfo info;
    err = MatGetInfo(m, MAT_GLOBAL_SUM, &info); CHKERRQ(err);
    err = PetscPrintf(PETSC_COMM_WORLD, "%s nnz: %f\n", label, info.nz_used); CHKERRQ(err);
    return err;
}

PetscErrorCode VecMerge(Vec r1, Vec r2, Vec r){
    PetscErrorCode err;
    PetscInt low, high, n1_local, n2_local;
    PetscInt *n1_idx, *n2_idx;
    PetscScalar *r1_array, *r2_array;
    PetscInt n1;
    int i;

    //copy r1 to r
    err = VecGetSize(r1, &n1); CHKERRQ(err);
    err = VecGetOwnershipRange(r1, &low, &high); CHKERRQ(err);
    err = VecGetLocalSize(r1, &n1_local); CHKERRQ(err);
    err = PetscMalloc1(n1_local, &n1_idx); CHKERRQ(err);
    for(i = 0; i < n1_local; i++) 
        n1_idx[i] = low + i;
    err = VecGetArray(r1, &r1_array); CHKERRQ(err);
    err = VecSetValues(r, n1_local, n1_idx, r1_array, INSERT_VALUES); CHKERRQ(err);
    err = VecAssemblyBegin(r); CHKERRQ(err);
    err = VecAssemblyEnd(r); CHKERRQ(err);
    err = VecRestoreArray(r1, &r1_array); CHKERRQ(err);
    err = PetscFree(n1_idx); CHKERRQ(err);

    err = VecGetOwnershipRange(r2, &low, &high); CHKERRQ(err);
    err = VecGetLocalSize(r2, &n2_local); CHKERRQ(err);
    err = PetscMalloc1(n2_local, &n2_idx); CHKERRQ(err);
    for(i = 0; i < n2_local; i++) 
        n2_idx[i] = low + i + n1;
    err = VecGetArray(r2, &r2_array); CHKERRQ(err);
    err = VecSetValues(r, n2_local, n2_idx, r2_array, INSERT_VALUES); CHKERRQ(err);

    err = VecAssemblyBegin(r); CHKERRQ(err);
    err = VecAssemblyEnd(r); CHKERRQ(err);

    //err = VecRestoreArray(r1, &r1_array); CHKERRQ(err);
    err = VecRestoreArray(r2, &r2_array); CHKERRQ(err);
    err = PetscFree(n1_idx); CHKERRQ(err);
    //err = PetscFree(n2_idx); CHKERRQ(err);

    return err;
}

PetscErrorCode VecReorder(Vec r, Vec order, Vec or){
    PetscErrorCode err;
    PetscInt n_local, low, high;
    PetscInt *to_idx, *from_idx;
    PetscScalar *o_array;
    IS to, from;
    VecScatter scatter;
    int i;

    err = VecGetLocalSize(r, &n_local); CHKERRQ(err);
    err = PetscMalloc1(n_local, &to_idx); CHKERRQ(err);
    err = PetscMalloc1(n_local, &from_idx); CHKERRQ(err);
    err = VecGetOwnershipRange(r, &low, &high); CHKERRQ(err);
    err = VecGetArray(order, &o_array);
    for(i = 0; i < n_local; i++){
        to_idx[i] = (PetscInt) o_array[low + i];
        from_idx[i] = (PetscInt) low + i;
    }
    err = VecRestoreArray(order, &o_array);

    err = ISCreateGeneral(PETSC_COMM_SELF, n_local, from_idx, PETSC_OWN_POINTER, &to); CHKERRQ(err);
    err = ISCreateGeneral(PETSC_COMM_SELF, n_local, from_idx, PETSC_OWN_POINTER, &from); CHKERRQ(err);

    err = VecScatterCreate(r, from, or, to, &scatter); CHKERRQ(err);
    err = VecScatterBegin(scatter, r, or, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(err);
    err = VecScatterEnd(scatter, r, or, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(err);

    err = PetscFree(to_idx); CHKERRQ(err);
    err = PetscFree(from_idx); CHKERRQ(err);

    return err;
}

PetscErrorCode BearQueryMat(PetscInt s, PetscScalar c, Mat invL1, Mat invU1, Mat invL2, Mat invU2, Mat H12, Mat H21, Vec order){
    PetscErrorCode err;
    PetscInt n1, n2, n, M, N;
    PetscInt oseed;
    PetscScalar val, one = 1.0;
    PetscMPIInt size;
    PetscLogDouble tic, toc;
    Mat r = NULL;
    Mat r1 = NULL, q1 = NULL, t1_1 = NULL, t1_2 = NULL, t1_3 = NULL, t1_4 = NULL, t1_5 = NULL; // dimension: n1
    Mat r2 = NULL, q2 = NULL, q_tilda = NULL, t2_1 = NULL, t2_2 = NULL, t2_3 = NULL; // dimension: n2_idx
    Vec vr=NULL, vr1=NULL, vr2=NULL;
    PetscInt col = 0;

    err = MPI_Comm_size(PETSC_COMM_WORLD, &size); CHKERRQ(err);

    err = MatGetSize(H12, &n1, &n2); CHKERRQ(err);
    n = n1 + n2;
    err = PetscPrintf(PETSC_COMM_WORLD, "n1: %d, n2: %d\n", n1, n2); CHKERRQ(err);


    err = MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, 1, n,  size, 1, NULL, 1, NULL, &r); CHKERRQ(err);
    err = MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, 1, n1, size, 1, NULL, 1, NULL, &q1); CHKERRQ(err);
    err = MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, 1, n2, size, 1, NULL, 1, NULL, &q2); CHKERRQ(err);
    //    err = MatCreate(PETSC_COMM_WORLD, &q2); CHKERRQ(err);
    //    err = MatSetSizes(q2, PETSC_DECIDE, PETSC_DECIDE, n2, 1); CHKERRQ(err);
    //    err = MatSetType(q2, MATAIJ); CHKERRQ(err);
    //    err = MatSetUp(q2);


    s = s - 1; // shift -1 for zero-based index
    err = VecGetValues(order, 1, &s, &val); CHKERRQ(err);
    oseed = (PetscInt) val;
    //    err = PetscPrintf(PETSC_COMM_WORLD, "Given seed: %d, Reorered seed: %d (0 ~ n-1)\n", s, oseed); CHKERRQ(err);

    if(oseed < n1){
        //err = MatSetValues(q1, 1, &oseed, 1, &col, &one, INSERT_VALUES); CHKERRQ(err);
        err = MatSetValue(q1, oseed, col, one, INSERT_VALUES); CHKERRQ(err);
    }else{
        oseed = oseed - n1;
        //err = MatSetValues(q2, 1, &oseed, 1, &col, &one, INSERT_VALUES); CHKERRQ(err);
        err = MatSetValue(q2, oseed, col, one, INSERT_VALUES); CHKERRQ(err);
        //err = printVecSum(q2);
    }
    err = MatAssemblyBegin(q1, MAT_FINAL_ASSEMBLY); CHKERRQ(err);
    err = MatAssemblyEnd(q1, MAT_FINAL_ASSEMBLY); CHKERRQ(err);
    err = MatAssemblyBegin(q2, MAT_FINAL_ASSEMBLY); CHKERRQ(err);
    err = MatAssemblyEnd(q2, MAT_FINAL_ASSEMBLY); CHKERRQ(err);

    err = printMatInfo("q1", q1);
    err = printMatInfo("q2", q2);
    //err = MatView(q1, PETSC_VIEWER_STDOUT_WORLD);
    //err = MatView(q2, PETSC_VIEWER_STDOUT_WORLD);


    err = MatDuplicate(q1, MAT_DO_NOT_COPY_VALUES, &r1); CHKERRQ(err);
    err = MatDuplicate(q1, MAT_DO_NOT_COPY_VALUES, &t1_1); CHKERRQ(err);
    err = MatDuplicate(q1, MAT_DO_NOT_COPY_VALUES, &t1_2); CHKERRQ(err);
    err = MatDuplicate(q1, MAT_DO_NOT_COPY_VALUES, &t1_3); CHKERRQ(err);
    err = MatDuplicate(q1, MAT_DO_NOT_COPY_VALUES, &t1_4); CHKERRQ(err);
    err = MatDuplicate(q1, MAT_DO_NOT_COPY_VALUES, &t1_5); CHKERRQ(err);

    err = MatDuplicate(q2, MAT_DO_NOT_COPY_VALUES, &r2); CHKERRQ(err);
    err = MatDuplicate(q2, MAT_DO_NOT_COPY_VALUES, &q_tilda); CHKERRQ(err);
    err = MatDuplicate(q2, MAT_DO_NOT_COPY_VALUES, &t2_1); CHKERRQ(err);
    err = MatDuplicate(q2, MAT_DO_NOT_COPY_VALUES, &t2_2); CHKERRQ(err);
    err = MatDuplicate(q2, MAT_DO_NOT_COPY_VALUES, &t2_3); CHKERRQ(err);

    err = MatMatMult(invL1, q1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &t1_1); CHKERRQ(err);
    err = MatMatMult(invU1, t1_1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &t1_2); CHKERRQ(err);
    err = MatMatMult(H21, t1_2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &t2_1); CHKERRQ(err);
    err = MatScale(t2_1, -1.0); CHKERRQ(err);
    err = MatAXPY(t2_1, 1.0, q2, DIFFERENT_NONZERO_PATTERN); CHKERRQ(err);
    //MatView(t1_1, PETSC_VIEWER_STDOUT_WORLD);
    err = PetscTime(&tic);
    err = MatMatMult(invL2, t2_1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &t2_2); CHKERRQ(err);
    err = MatMatMult(invU2, t2_2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &r2); CHKERRQ(err);
    err = PetscTime(&toc);
    err = PetscPrintf(PETSC_COMM_WORLD, "running time: %f sec\n", toc-tic); CHKERRQ(err);
    
    err = MatMatMult(H12, r2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &t1_3); CHKERRQ(err);
    err = MatScale(t1_3, -1.0); CHKERRQ(err);
    err = MatAXPY(t1_3, 1.0, q1, DIFFERENT_NONZERO_PATTERN); CHKERRQ(err);
    err = MatMatMult(invL1, t1_3, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &t1_5); CHKERRQ(err);
    err = MatMatMult(invU1, t1_5, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &r1); CHKERRQ(err);

    //MatView(r1, PETSC_VIEWER_STDOUT_WORLD);
    MatGetSize(r1, &M, &N);
    PetscPrintf(PETSC_COMM_WORLD, "%d %d\n", M, N);

    err = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, n1, &vr1);
    err = MatGetColumnVector(r1, vr1, 0);

    err = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, n2, &vr2);
    err = MatGetColumnVector(r2, vr2, 0);


    err = printMatInfo("r2", r2);
    /*

    // Start matrix-vec multiplications
    err = MatMult(invU2, t2_2, r2); CHKERRQ(err);

    err = MatMult(H12, r2, t1_3); CHKERRQ(err);
    err = VecAXPBYPCZ(t1_4, 1.0, -1.0, 0.0, q1, t1_3); CHKERRQ(err);
    err = MatMult(invL1, t1_4, t1_5); CHKERRQ(err);
    err = MatMult(invU1, t1_5, r1); CHKERRQ(err);
    //err = printVecSum(r1); 

    //err = VecView(r2, PETSC_VIEWER_STDOUT_WORLD);

    // Concatenate r1 and r2
    err = VecMerge(r1, r2, r); CHKERRQ(err);
    err = VecScale(r, c); CHKERRQ(err);

    //err = VecView(r, PETSC_VIEWER_STDOUT_WORLD);

    //err = VecDuplicate(r, &or); CHKERRQ(err);
    err = VecReorder(r, order, or); CHKERRQ(err);
    //err = VecView(or, PETSC_VIEWER_STDOUT_WORLD);
    */

    err = MatDestroy(&r); CHKERRQ(err);
    err = MatDestroy(&r1); CHKERRQ(err);
    err = MatDestroy(&q1); CHKERRQ(err);
    err = MatDestroy(&t1_1); CHKERRQ(err);
    err = MatDestroy(&t1_2); CHKERRQ(err);
    err = MatDestroy(&t1_3); CHKERRQ(err);
    err = MatDestroy(&t1_4); CHKERRQ(err);
    err = MatDestroy(&t1_5); CHKERRQ(err);

    err = MatDestroy(&r2); CHKERRQ(err);
    err = MatDestroy(&q2); CHKERRQ(err);
    err = MatDestroy(&q_tilda); CHKERRQ(err);
    err = MatDestroy(&t2_1); CHKERRQ(err);
    err = MatDestroy(&t2_2); CHKERRQ(err);
    err = MatDestroy(&t2_3); CHKERRQ(err);

    return err;
}
PetscErrorCode BearQuery(PetscInt s, PetscScalar c, Mat invL1, Mat invU1, Mat invL2, Mat invU2, Mat H12, Mat H21, Vec order, Vec or){
    PetscErrorCode err;
    PetscInt n1, n2, n;
    PetscInt oseed;
    PetscScalar val, one = 1.0;
    Vec r = NULL;
    Vec r1 = NULL, q1 = NULL, t1_1 = NULL, t1_2 = NULL, t1_3 = NULL, t1_4 = NULL, t1_5 = NULL; // dimension: n1
    Vec r2 = NULL, q2 = NULL, q_tilda = NULL, t2_1 = NULL, t2_2 = NULL, t2_3 = NULL; // dimension: n2


    err = MatGetSize(H12, &n1, &n2); CHKERRQ(err);
    n = n1 + n2;
    //    err = PetscPrintf(PETSC_COMM_WORLD, "n1: %d, n2: %d\n", n1, n2); CHKERRQ(err);

    err = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, n, &r); CHKERRQ(err);
    err = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, n1, &q1); CHKERRQ(err);
    err = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, n2, &q2); CHKERRQ(err);
    err = VecSet(q1, 0); CHKERRQ(err);
    err = VecSet(q2, 0); CHKERRQ(err);

    s = s - 1; // shift -1 for zero-based index
    err = VecGetValues(order, 1, &s, &val); CHKERRQ(err);
    oseed = (PetscInt) val;
    //    err = PetscPrintf(PETSC_COMM_WORLD, "Given seed: %d, Reorered seed: %d (0 ~ n-1)\n", s, oseed); CHKERRQ(err);

    if(oseed < n1){
        err = VecSetValues(q1, 1, &oseed, &one, INSERT_VALUES); CHKERRQ(err);
    }else{
        oseed = oseed - n1;
        err = VecSetValues(q2, 1, &oseed, &one, INSERT_VALUES); CHKERRQ(err);
        //err = printVecSum(q2);
    }
    err = VecAssemblyBegin(q1); CHKERRQ(err);
    err = VecAssemblyBegin(q2); CHKERRQ(err);
    err = VecAssemblyEnd(q1); CHKERRQ(err);
    err = VecAssemblyEnd(q2); CHKERRQ(err);

    err = VecDuplicate(q1, &r1); CHKERRQ(err);
    err = VecDuplicate(q1, &t1_1); CHKERRQ(err);
    err = VecDuplicate(q1, &t1_2); CHKERRQ(err);
    err = VecDuplicate(q1, &t1_3); CHKERRQ(err);
    err = VecDuplicate(q1, &t1_4); CHKERRQ(err);
    err = VecDuplicate(q1, &t1_5); CHKERRQ(err);

    err = VecDuplicate(q2, &r2); CHKERRQ(err);
    err = VecDuplicate(q2, &q_tilda); CHKERRQ(err);
    err = VecDuplicate(q2, &t2_1); CHKERRQ(err);
    err = VecDuplicate(q2, &t2_2); CHKERRQ(err);
    err = VecDuplicate(q2, &t2_3); CHKERRQ(err);

    // Start matrix-vec multiplications
    err = MatMult(invL1, q1, t1_1); CHKERRQ(err);
    err = MatMult(invU1, t1_1, t1_2); CHKERRQ(err);
    err = MatMult(H21, t1_2, t2_1); CHKERRQ(err);
    err = VecAXPBYPCZ(q_tilda, 1.0, -1.0, 0.0, q2, t2_1); CHKERRQ(err);
    err = MatMult(invL2, q_tilda, t2_2); CHKERRQ(err);
    err = MatMult(invU2, t2_2, r2); CHKERRQ(err);

    err = MatMult(H12, r2, t1_3); CHKERRQ(err);
    err = VecAXPBYPCZ(t1_4, 1.0, -1.0, 0.0, q1, t1_3); CHKERRQ(err);
    err = MatMult(invL1, t1_4, t1_5); CHKERRQ(err);
    err = MatMult(invU1, t1_5, r1); CHKERRQ(err);
    //err = printVecSum(r1); 

    //err = VecView(r2, PETSC_VIEWER_STDOUT_WORLD);

    // Concatenate r1 and r2
    err = VecMerge(r1, r2, r); CHKERRQ(err);
    err = VecScale(r, c); CHKERRQ(err);

    //err = VecView(r, PETSC_VIEWER_STDOUT_WORLD);

    //err = VecDuplicate(r, &or); CHKERRQ(err);
    err = VecReorder(r, order, or); CHKERRQ(err);
    //err = VecView(or, PETSC_VIEWER_STDOUT_WORLD);


    err = VecDestroy(&r); CHKERRQ(err);
    err = VecDestroy(&r1); CHKERRQ(err);
    err = VecDestroy(&q1); CHKERRQ(err);
    err = VecDestroy(&t1_1); CHKERRQ(err);
    err = VecDestroy(&t1_2); CHKERRQ(err);
    err = VecDestroy(&t1_3); CHKERRQ(err);
    err = VecDestroy(&t1_4); CHKERRQ(err);
    err = VecDestroy(&t1_5); CHKERRQ(err);

    err = VecDestroy(&r2); CHKERRQ(err);
    err = VecDestroy(&q2); CHKERRQ(err);
    err = VecDestroy(&q_tilda); CHKERRQ(err);
    err = VecDestroy(&t2_1); CHKERRQ(err);
    err = VecDestroy(&t2_2); CHKERRQ(err);
    err = VecDestroy(&t2_3); CHKERRQ(err);

    return err;
}

int main(int argc, char** args){
    PetscErrorCode err;
    PetscViewer fd = NULL;
    Mat invL1 = NULL, invU1 = NULL, invL2 = NULL, invU2 = NULL, H12 = NULL, H21 = NULL;
    Vec order = NULL, r = NULL; //, or = NULL; //dimension: n: n1 + n2
    Vec seeds = NULL;
    //    Vec r1 = NULL, q1 = NULL, t1_1 = NULL, t1_2 = NULL, t1_3 = NULL, t1_4 = NULL, t1_5 = NULL; // dimension: n1
    //    Vec r2 = NULL, q2 = NULL, q_tilda = NULL, t2_1 = NULL, t2_2 = NULL, t2_3 = NULL; // dimension: n2
    PetscRandom rand;

    PetscLogDouble tic, toc, total_time, time;
    PetscInt n, i;
    PetscMPIInt rank, size;
    PetscInt seed;
    PetscScalar c, val;
    PetscInt QN = 100;

    // Initialize PETSC and MPI
    err = PetscInitialize(&argc, &args, (char*) 0, help); CHKERRQ(err);
    err = MPI_Comm_size(PETSC_COMM_WORLD, &size); CHKERRQ(err);
    err = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(err);
    err = PetscPrintf(PETSC_COMM_WORLD, "mpi size: %d\n", size); CHKERRQ(err); 

    // Read matrices and an ordering vector
    err = PetscPrintf(PETSC_COMM_WORLD, "Read inputs (invL1, invU1, invL2, invU2, H12, H21, order)\n"); CHKERRQ(err);

    err = loadMat("./data/invL1.dat", &invL1, PETSC_COMM_WORLD, MATMPIAIJ, &fd); CHKERRQ(err);
    err = checkMat("invL1", invL1); CHKERRQ(err);
    
    err = loadMat("./data/invU1.dat", &invU1, PETSC_COMM_WORLD, MATMPIAIJ, &fd); CHKERRQ(err);
    err = checkMat("invU1", invU1); CHKERRQ(err);
    
    err = loadMat("./data/invL2.dat", &invL2, PETSC_COMM_WORLD, MATMPIAIJ, &fd); CHKERRQ(err);
    err = checkMat("invL2", invL2); CHKERRQ(err);
    
    err = loadMat("./data/invU2.dat", &invU2, PETSC_COMM_WORLD, MATMPIAIJ, &fd); CHKERRQ(err);
    err = checkMat("invU2", invU2); CHKERRQ(err);
    
    err = loadMat("./data/H12.dat", &H12, PETSC_COMM_WORLD, MATMPIAIJ, &fd); CHKERRQ(err);
    err = checkMat("H12", H12); CHKERRQ(err);
    
    err = loadMat("./data/H21.dat", &H21, PETSC_COMM_WORLD, MATMPIAIJ, &fd); CHKERRQ(err);
    err = checkMat("H21", H21); CHKERRQ(err);

    err = loadVec("./data/order.dat", &order, PETSC_COMM_SELF, &fd); CHKERRQ(err); //all processes must have this vector for ordering the result vector.
    err = checkVec("order", order); CHKERRQ(err);

    // shift -1 for zero-based index
    err = VecShift(order, -1); CHKERRQ(err);
    err = VecGetSize(order, &n); CHKERRQ(err);

    seed = 5;
    c = 0.05;
    err = PetscTime(&tic); CHKERRQ(err);

    err = BearQueryMat(seed, c, invL1, invU1, invL2, invU2, H12, H21, order); CHKERRQ(err);
    err = PetscTime(&toc); CHKERRQ(err);
    time = toc - tic;
    err = PetscPrintf(PETSC_COMM_WORLD, "running time: %f sec\n", time); CHKERRQ(err);

    /* 100 times querying
       err = VecCreateSeq(PETSC_COMM_SELF, QN, &seeds); CHKERRQ(err);
       err = VecSetFromOptions(seeds); CHKERRQ(err); 
       err = PetscRandomCreate(PETSC_COMM_WORLD, &rand); CHKERRQ(err);
       err = PetscRandomSetSeed(rand, 100); CHKERRQ(err);
       err = PetscRandomSetInterval(rand, (PetscScalar) 0, (PetscScalar) n); CHKERRQ(err);
       err = PetscRandomSetFromOptions(rand); CHKERRQ(err);
       err = VecSetRandom(seeds, rand); CHKERRQ(err);
       err = PetscRandomDestroy(&rand); CHKERRQ(err);

       seed = 5; //seed is give by user on one-based index
       c = 0.05;

       i = 0;

       err = VecDuplicate(order, &r); CHKERRQ(err);
       for(i = 0; i < QN; i++){
       err = VecGetValues(seeds, 1, &i, &val);
       seed = (PetscInt) val;
    //err = PetscPrintf(PETSC_COMM_SELF, "rank: %d, seed: %d\n", rank, seed);
    err = PetscTime(&tic); CHKERRQ(err);
    err = BearQuery(seed, c, invL1, invU1, invL2, invU2, H12, H21, order, r); CHKERRQ(err);
    err = PetscTime(&toc); CHKERRQ(err);
    time = toc - tic;
    err = PetscPrintf(PETSC_COMM_WORLD, "running time: %f sec\n", time); CHKERRQ(err);
    total_time += time;
    }
    err = PetscPrintf(PETSC_COMM_WORLD, "average running time: %f sec\n", total_time/QN); CHKERRQ(err);
    err = VecDestroy(&r);
    */

    /*    err = MatGetSize(H12, &n1, &n2); CHKERRQ(err);
          n = n1 + n2;
          err = PetscPrintf(PETSC_COMM_WORLD, "n1: %d, n2: %d\n", n1, n2); CHKERRQ(err);

          err = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, n, &r); CHKERRQ(err);
          err = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, n1, &q1); CHKERRQ(err);
          err = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, n2, &q2); CHKERRQ(err);
          err = VecSet(q1, 0); CHKERRQ(err);
          err = VecSet(q2, 0); CHKERRQ(err);

          seed = seed - 1; // shift -1 for zero-based index
          err = VecGetValues(order, 1, &seed, &val); CHKERRQ(err);
          oseed = (PetscInt) val;
          err = PetscPrintf(PETSC_COMM_WORLD, "Given seed: %d, Reorered seed: %d (0 ~ n-1)\n", seed, oseed); CHKERRQ(err);

          if(oseed < n1){
          err = VecSetValues(q1, 1, &oseed, &one, INSERT_VALUES); CHKERRQ(err);
          }else{
          oseed = oseed - n1;
          err = VecSetValues(q2, 1, &oseed, &one, INSERT_VALUES); CHKERRQ(err);
    //err = printVecSum(q2);
    }
    err = VecAssemblyBegin(q1); CHKERRQ(err);
    err = VecAssemblyBegin(q2); CHKERRQ(err);
    err = VecAssemblyEnd(q1); CHKERRQ(err);
    err = VecAssemblyEnd(q2); CHKERRQ(err);

    err = VecDuplicate(q1, &r1); CHKERRQ(err);
    err = VecDuplicate(q1, &t1_1); CHKERRQ(err);
    err = VecDuplicate(q1, &t1_2); CHKERRQ(err);
    err = VecDuplicate(q1, &t1_3); CHKERRQ(err);
    err = VecDuplicate(q1, &t1_4); CHKERRQ(err);
    err = VecDuplicate(q1, &t1_5); CHKERRQ(err);

    err = VecDuplicate(q2, &r2); CHKERRQ(err);
    err = VecDuplicate(q2, &q_tilda); CHKERRQ(err);
    err = VecDuplicate(q2, &t2_1); CHKERRQ(err);
    err = VecDuplicate(q2, &t2_2); CHKERRQ(err);
    err = VecDuplicate(q2, &t2_3); CHKERRQ(err);

    // Start matrix-vec multiplications
    err = MatMult(invL1, q1, t1_1); CHKERRQ(err);
    err = MatMult(invU1, t1_1, t1_2); CHKERRQ(err);
    err = MatMult(H21, t1_2, t2_1); CHKERRQ(err);
    err = VecAXPBYPCZ(q_tilda, 1.0, -1.0, 0.0, q2, t2_1); CHKERRQ(err);
    err = MatMult(invL2, q_tilda, t2_2); CHKERRQ(err);
    err = MatMult(invU2, t2_2, r2); CHKERRQ(err);

    err = MatMult(H12, r2, t1_3); CHKERRQ(err);
    err = VecAXPBYPCZ(t1_4, 1.0, -1.0, 0.0, q1, t1_3); CHKERRQ(err);
    err = MatMult(invL1, t1_4, t1_5); CHKERRQ(err);
    err = MatMult(invU1, t1_5, r1); CHKERRQ(err);
    //err = printVecSum(r1); 

    //err = VecView(r2, PETSC_VIEWER_STDOUT_WORLD);

    // Concatenate r1 and r2
    err = VecMerge(r1, r2, r); CHKERRQ(err);
    err = VecScale(r, c); CHKERRQ(err);

    //err = VecView(r, PETSC_VIEWER_STDOUT_WORLD);

    err = VecDuplicate(r, &or); CHKERRQ(err);
    err = VecReorder(r, order, or); CHKERRQ(err);
    //err = VecView(or, PETSC_VIEWER_STDOUT_WORLD);*/

    // Destory matrices and vectors
    err = MatDestroy(&invL1); CHKERRQ(err);
    err = MatDestroy(&invU1); CHKERRQ(err);
    err = MatDestroy(&invL2); CHKERRQ(err);
    err = MatDestroy(&invU2); CHKERRQ(err);
    err = MatDestroy(&H12); CHKERRQ(err);
    err = MatDestroy(&H21); CHKERRQ(err);
    err = VecDestroy(&order); CHKERRQ(err);
    err = VecDestroy(&r); CHKERRQ(err);
    err = VecDestroy(&seeds); CHKERRQ(err);
    //err = VecDestroy(&or); CHKERRQ(err);

    /*    err = VecDestroy(&r1); CHKERRQ(err);
          err = VecDestroy(&q1); CHKERRQ(err);
          err = VecDestroy(&t1_1); CHKERRQ(err);
          err = VecDestroy(&t1_2); CHKERRQ(err);
          err = VecDestroy(&t1_3); CHKERRQ(err);
          err = VecDestroy(&t1_4); CHKERRQ(err);
          err = VecDestroy(&t1_5); CHKERRQ(err);

          err = VecDestroy(&r2); CHKERRQ(err);
          err = VecDestroy(&q2); CHKERRQ(err);
          err = VecDestroy(&q_tilda); CHKERRQ(err);
          err = VecDestroy(&t2_1); CHKERRQ(err);
          err = VecDestroy(&t2_2); CHKERRQ(err);
          err = VecDestroy(&t2_3); CHKERRQ(err);*/

    // Finalize
    err = PetscFinalize(); CHKERRQ(err);
    return 0;
}
