include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

query: query.o chkopts
	-${CLINKER} -o query query.o ${PETSC_LIB} ${PETSC_MAT_LIB} ${PETSC_KSP_LIB}
	${RM} query.o
	mpiexec -np 2 ./query

