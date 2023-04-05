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
    p = scatter(getindex.(coors, 1), getindex.(coors, 2), markeralpha=getfield.(node, :p) / maximum(getfield.(node, :p)), markersize=1, markerstrokewidth=0, lengend=:none, aspect_ratio=1)
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
        plot!(p, [x_src[1], x_dst[1]], [x_src[2], x_dst[2]], legend=:none, aspect_ratio=1, linecolor=c, linewidth=edge[e.src, e.dst].D / 2)
    end
    p
end

function update!(g_node, g_edge, node, edge, nx, ny, dt=0.1, time=10)
    step = Int(time / dt)
    for _ in 1:step
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
        B = vcat(-10, zeros(nx * ny - 2), 0)
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
end

function main(nx, ny, bound, dt, time, n)
    g_node, g_edge, node, edge = init(nx, ny, bound, 0.2)
    p_init = plot_world(g_node, g_edge, node, edge)
    p_update = []
    for _ in 1:n
        update!(g_node, g_edge, node, edge, nx, ny, dt, time)
        push!(p_update, plot_world(g_node, g_edge, node, edge))
    end
    p_init, p_update
end

@time p_init, p_update = main(32, 20, 10.0, 0.1, 10, 8)
plots = vcat(p_init, p_update)
plot(plots..., layout=(3, 3))
