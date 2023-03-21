module MMSpaces

include( joinpath(@__DIR__,"LBs+Tests.jl"))
include( joinpath(@__DIR__,"GW.jl"))

export sort_thrust!, emd_bonneel, emd_bonneel_with_plan, flb, Euc_flb, slb, Euc_slb, tlb, Euc_tlb, FLB_Test, SLB_Test, TLB_Test, gromov_wasserstein

end
