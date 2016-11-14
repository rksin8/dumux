# additional macros
include(AddGstatFileLinks)
include(AddInputFileLinks)
include(DumuxDoxygen)
include(DumuxTestMacros)

find_package(GLPK)
find_package(Gstat)
find_package(Gnuplot)
set(HAVE_GNUPLOT ${GNUPLOT_FOUND})

find_package(NLOPT)
find_package(Valgrind)
find_package(PTScotch)
