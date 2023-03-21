
# This package depends on the network simplex implementation by Nicolas Bonnel (link: https://github.com/nbonneel/network_simplex)
# We have to complie the cooresponding library.
const path_to_makefile = abspath(joinpath(@__DIR__, "../src/emd_bonneel"))

try run(`which gcc`)
    run(`make -C $path_to_makefile`)    
  catch _
    @warn "cc compiler not found. Package not built successfully"
  end