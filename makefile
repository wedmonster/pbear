include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

query: query.o chkopts
	-${CLINKER} -o query query.o ${PETSC_LIB} ${PETSC_MAT_LIB} ${PETSC_KSP_LIB}
	${RM} query.o
	mpiexec -np 10 ./query

