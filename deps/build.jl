
# Compile the Boneel OT-Code

const path_to_makefile = abspath(joinpath(@__DIR__, "../src/emd_bonneel"))

try run(`which gcc`)
    run(`make -C $path_to_makefile`)    
  catch _
    @warn "cc compiler not found. Package not built successfully"
  end