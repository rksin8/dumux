add_subdirectory(box)
add_subdirectory(cellcentered)
add_subdirectory(fem)
add_subdirectory(projection)
add_subdirectory(staggered)

install(FILES
basegridgeometry.hh
box.hh
ccmpfa.hh
cctpfa.hh
checkoverlapsize.hh
elementsolution.hh
evalgradients.hh
evalsolution.hh
extrusion.hh
fluxstencil.hh
functionspacebasis.hh
fvgridvariables.hh
fvproperties.hh
localview.hh
method.hh
rotationpolicy.hh
rotationsymmetricgridgeometrytraits.hh
rotationsymmetricscv.hh
rotationsymmetricscvf.hh
scvandscvfiterators.hh
staggered.hh
subcontrolvolumebase.hh
subcontrolvolumefacebase.hh
DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dumux/discretization)
