function ask_float(prompt::String; must_exist_dir::Union{Nothing,String}=nothing,paraname::Union{Nothing,String}=nothing)
    while true
        print(prompt)
        s = readline()
        try
            x = parse(Float64, s)
            if must_exist_dir === nothing || paraname !== nothing && isdir(joinpath(must_exist_dir, paraname * "=$(x)"))
                return x
            elseif paraname === nothing
                println("You must provide 'paraname' if 'must_exist_dir' is given. i.e. try 'a' for α, etc.")
            elseif !isdir(joinpath(must_exist_dir, paraname * "=$(x)"))
                println("No such directory: $(joinpath(must_exist_dir, paraname * "=$(x)"))")
            end
        catch e
            println("Invalid input: $(e)")
        end
    end
end

function select_parameters(input_dir)
    # Ask for α
    α = ask_float("Alpha Value?: "; must_exist_dir=input_dir,paraname = "a")

    # Ask for β
    β_dir = joinpath(input_dir, "a=$(α)")
    β = ask_float("Beta Value?: "; must_exist_dir=β_dir,paraname = "b")

    # Ask for γ
    global γ_dir = joinpath(β_dir, "b=$(β)")
    γ = ask_float("Noise 'γ' Value?: "; must_exist_dir=γ_dir,paraname = "g")
    γ_path = joinpath(γ_dir, "g=$(γ)")
    potential_pend_data = filter(f -> contains(f, "pendulum"), readdir(γ_path; join=true))
    if isdir(γ_path)
        println("Directories for α=$(α), β=$(β), γ=$(γ) exist! Let's run...\n")
        if !isempty(potential_pend_data)
            println("Pendulum data detected, proceeding...\n")
            return (α=α, β=β, γ=γ)
        else
            println("No pendulum data detected, please re-enter parameters.\n")
            return select_parameters(input_dir)
        end
    else
        println("No such directories for α=$(α), β=$(β), γ=$(γ)")
    end
end
