using Graphs
using Plots

mutable struct Node
    p::Float64 # pressure
    x::Tuple{Float64,Float64} # (x,y)
end

mutable struct Edge
    D::Float64 # Diffusion
    Q::Float64 # Flow Rate
    L::Float64 # Length
    a::Float64 # Risk factor
end

Base.broadcastable(x::Node) = Ref(x)
Base.broadcastable(x::Edge) = Ref(x)

function init(nx::Int, ny::Int, bound::Float64, r::Float64)
    g = Graphs.SimpleGraphs.grid([nx, ny])

    g_node = vertices(g) |> collect
    g_edge = edges(g) |> collect

    node = Vector{Node}(undef, length(g_node))
    edge = Array{Edge,2}(undef, length(g_node), length(g_node))

    p_init = 0
    for i in 1:nx*ny
        x = (i - 1) % nx + 1
        y = div(i - 1, nx) + 1
        node[i] = Node(p_init, (x + r * randn(), y + r * randn()))
    end

    big_num = 10000000.0
    edge = reshape([Edge(0.0, 0.0, big_num, 0) for _ in 1:length(edge)], size(edge))
    for e in g_edge
        l = (node[e.src].x .- node[e.dst].x) .^ 2 |> sum |> sqrt
        if (node[e.src].x[2] + node[e.dst].x[2]) / 2 > bound
            a = 2
        else
            a = 1
        end
        edge[e.src, e.dst] = Edge(1.0, 0, l, a)
        edge[e.dst, e.src] = Edge(1.0, 0, l, a)
    end

    return g_node, g_edge, node, edge
end


function plot_world(g_node, g_edge, node, edge)
    coors = getfield.(node, 2)
    p = scatter(getindex.(coors, 1), getindex.(coors, 2), markeralpha=getfield.(node, :p) / maximum(getfield.(node, :p)), markersize=3, markerstrokewidth=0, lengend=:none, aspect_ratio=1)
    colortable = [:red, :blue]
    for e in g_edge
        x_src = node[e.src].x
        x_dst = node[e.dst].x
        c = begin
            if edge[e.src, e.dst].a == 2
                colortable[2]
            else
                colortable[1]
            end
        end
        plot!(p, [x_src[1], x_dst[1]], [x_src[2], x_dst[2]], legend=:none, aspect_ratio=1, linecolor=c, linewidth=edge[e.src, e.dst].D)
    end
    p
end

function update!(g_node, g_edge, node, edge, nx, ny, Q_inlet, dt)
    A = Array{Float64,2}(undef, length(node), length(node))
    for (j, n) in enumerate(node)
        t = 0
        for i in 1:length(node)
            if i != j
                A[i, j] = edge[i, j].D / edge[i, j].L
                t -= A[i, j]
            end
        end
        A[j, j] = t
    end

    A[end, 1:end] = vcat(zeros(nx * ny - 1), 1)
    B = vcat(Q_inlet, zeros(nx * ny - 2), 0)
    P = A \ B
    setfield!.(node, :p, P)

    for e in g_edge
        e1 = @view edge[e.src, e.dst]
        e2 = @view edge[e.dst, e.src]
        e1[1].Q = e1[1].D / e1[1].L * (node[e.src].p - node[e.dst].p)
        e2[1].Q = e2[1].D / e2[1].L * (node[e.dst].p - node[e.src].p)
    end

    for e in g_edge
        e1 = @view edge[e.src, e.dst]
        e2 = @view edge[e.dst, e.src]
        e1[1].D += (abs(e1[1].Q) - e1[1].a * e1[1].D) * dt
        e2[1].D += (abs(e2[1].Q) - e2[1].a * e2[1].D) * dt
    end
end

function main(nx, ny, bound, n, dt, snap)
    g_node, g_edge, node, edge = init(nx, ny, bound, 0.2)
    g_node′, g_edge′, node′, edge′ = deepcopy(g_node), deepcopy(g_edge), deepcopy(node), deepcopy(edge)
    p_init = plot_world(g_node, g_edge, node, edge)
    p_init′ = plot_world(g_node′, g_edge′, node′, edge′)
    p_update = []
    p_update′ = []
    for t in 1:n
        for _ in 1:Int(1 / dt)
            update!(g_node, g_edge, node, edge, nx, ny, -10 - 5 * sin(2 * pi * t / n), dt)
            update!(g_node′, g_edge′, node′, edge′, nx, ny, -10, dt)
        end
        push!(p_update, plot_world(g_node, g_edge, node, edge))
        push!(p_update′, plot_world(g_node′, g_edge′, node′, edge′))
    end
    p_init, p_update, p_init′, p_update′
end

@time p_init, p_update, p_init′, p_update′ = main(40, 30, 10.0, 15, 0.2, 1)
plots = vcat(p_init, p_update)
plots′ = vcat(p_init′, p_update′)
plot_result = plot(plots..., layout=(4, 4), size=(2400, 1600))
plot_result′ = plot(plots′..., layout=(4, 4), size=(2400, 1600))
savefig(plot_result, "plot_sin_16_0_2.png")
savefig(plot_result′, "plot_flat_16_0_2.png")
plot(plots[end], plots′[end])