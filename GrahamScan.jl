using DataFrames

"""
This code is an implementation of the Graham scan algorithm to find the
largest bounding convex polygon from a collection of points.
"""

# %% Graham Scan Function, pointcollection is Nx2
function graham(pointcollection)
    # Find Lowest point for boundary
    N = size(pointcollection)[1]
    points = DataFrame(pointcollection, :auto)
    hull = []
    interior = []
    minindex = findmin(points[:,2])[2]
    push!(hull,Vector(points[minindex,:]))
    delete!(points,minindex)

    # Calculate all angles and sort counterclockwise
    relVectors = Matrix(points)-transpose(repeat(hull[1],1,N-1))
    angles = atan.(relVectors[:,2],relVectors[:,1])
    insertcols!(points, :angles => angles)
    sort!(points,:angles)

    push!(hull, Vector(points[1,1:2]))
    delete!(points, 1)

    #Graham algorithm
    while nrow(points) > 0
        refVector1 = hull[end]-hull[end-1]
        testpoint = Vector(points[1,1:2])
        refVector2 = testpoint-hull[end]

        angle1 = (atan(refVector1[2],refVector1[1])+2*pi)%(2*pi)
        angle2 = (atan(refVector2[2],refVector2[1])+2*pi)%(2*pi)
        diff = (angle2-angle1)%pi

        if diff >= 0
            push!(hull, testpoint)
            delete!(points,1)

        else
            push!(interior,pop!(hull))
        end
    end

    # Convert points to output formats
    hullpoints = zeros(length(hull),2)
    inpoints = zeros(length(interior),2)

    for i in 1:1:length(hull)
        hullpoints[i,:] = hull[i]
    end

    for i in 1:1:length(interior)
        inpoints[i,:] = interior[i]
    end

    boundary = vcat(hullpoints, transpose(hullpoints[1,:]))

    output = Dict("Points" => pointcollection,"Hull" => hullpoints, "Interior" => inpoints, "Boundary" => boundary)
end
