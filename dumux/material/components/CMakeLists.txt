add_subdirectory(iapws)

install(FILES
air.hh
ammonia.hh
base.hh
benzene.hh
brine.hh
calcite.hh
calciumion.hh
cao.hh
cao2h2.hh
carbonateion.hh
ch4.hh
chlorideion.hh
co2.hh
co2tablereader.hh
co2tables.inc
componenttraits.hh
constant.hh
gas.hh
glucose.hh
granite.hh
h2.hh
h2o.hh
heavyoil.hh
ion.hh
liquid.hh
mesitylene.hh
n2.hh
nacl.hh
o2.hh
simpleh2o.hh
sodiumion.hh
solid.hh
tabulatedcomponent.hh
trichloroethene.hh
urea.hh
xylene.hh
DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dumux/material/components)
