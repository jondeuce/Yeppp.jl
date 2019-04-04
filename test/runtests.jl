using Yeppp
using Test
import LinearAlgebra

for T in (Float32, Float64)
    for vecfun in (v -> Vector{T}(v),
                   v -> view(Vector{T}(v), 100:700),
                   v -> reshape(Vector{T}(v), 50,20),
                   v -> view(reshape(Vector{T}(v), 50,20), 10:50, 5),
                   v -> SubArray(reshape(Vector{T}(v), 50,20), (10:50, 5)))
        x = vecfun(rand(1000))
        y = vecfun(rand(1000))
        z = vecfun(randn(1000))
        res = vecfun(rand(1000))

        @test Yeppp.add!(res, x, y) == x .+ y
        @test Yeppp.multiply!(res, x, y) == x .* y
        @test Yeppp.subtract!(res, x, y) == x .- y
        @test Yeppp.multiply(x, y) == x .* y
        @test Yeppp.subtract(x, y) == x .- y
        @test Yeppp.min!(res, x, y) == min.(x, y)
        @test Yeppp.max!(res, x, y) == max.(x, y)
        @test Yeppp.min(x, y) == min.(x, y)
        @test Yeppp.max(x, y) == max.(x, y)

        @test Yeppp.sum(x) ≈ sum(x)
        @test Yeppp.sumabs(x) ≈ sum(abs, x)
        @test Yeppp.sumabs2(x) ≈ sum(abs2, x)
        if ndims(x) > 1
            @test_throws MethodError Yeppp.dot(x, y)
        else
            @test Yeppp.dot(x, y) ≈ LinearAlgebra.dot(x, y)
        end
    end
end

for T in (Float32, Float64)
    for vecfun in (v -> view(Vector{T}(v), 1:2:500),
                   v -> view(reshape(Vector{T}(v), 50,20), 10:2:50, 5),
                   v -> view(reshape(Vector{T}(v), 50,20), 5, :),
                   v -> SubArray(reshape(Vector{T}(v), 50,20), (10:50, 5:10)),
                   v -> SubArray(reshape(Vector{T}(v), 50,20), (10:2:50, 1)))
        x = vecfun(rand(1000))
        y = vecfun(rand(1000))
        z = vecfun(randn(1000))
        res = vecfun(rand(1000))

        @test_throws AssertionError Yeppp.add!(res, x, y)
        @test_throws AssertionError Yeppp.multiply!(res, x, y)
        @test_throws AssertionError Yeppp.subtract!(res, x, y)
        @test_throws AssertionError Yeppp.multiply(x, y)
        @test_throws AssertionError Yeppp.subtract(x, y)
        @test_throws AssertionError Yeppp.min!(res, x, y)
        @test_throws AssertionError Yeppp.max!(res, x, y)
        @test_throws AssertionError Yeppp.min(x, y)
        @test_throws AssertionError Yeppp.max(x, y)

        @test_throws AssertionError Yeppp.sum(x)
        @test_throws AssertionError Yeppp.sumabs(x)
        @test_throws AssertionError Yeppp.sumabs2(x)
        if ndims(x) > 1
            @test_throws MethodError Yeppp.dot(x, y)
        else
            @test_throws AssertionError Yeppp.dot(x, y)
        end
    end
end

for vecfun in (v -> v,
               v -> view(v, 100:700),
               v -> reshape(v, 50,20),
               v -> view(reshape(v, 50,20), 10:50, 5),
               v -> SubArray(reshape(v, 50,20), (10:50, 5)))
    x = vecfun(rand(1000))
    y = vecfun(rand(1000))
    z = vecfun(randn(1000))
    res = vecfun(rand(1000))

    @test Yeppp.exp!(res, x) ≈ exp.(x)
    @test Yeppp.log!(res, x) ≈ log.(x)
    @test Yeppp.sin!(res, x) ≈ sin.(x)
    @test Yeppp.cos!(res, x) ≈ cos.(x)
    @test Yeppp.tan!(res, x) ≈ tan.(x)
end

for vecfun in (v -> view(v, 1:2:500),
               v -> view(reshape(v, 50,20), 10:2:50, 5),
               v -> view(reshape(v, 50,20), 5, :),
               v -> SubArray(reshape(v, 50,20), (10:50, 5:10)),
               v -> SubArray(reshape(v, 50,20), (10:2:50, 1)))
    x = vecfun(rand(1000))
    y = vecfun(rand(1000))
    z = vecfun(randn(1000))
    res = vecfun(rand(1000))

    @test_throws AssertionError Yeppp.exp!(res, x)
    @test_throws AssertionError Yeppp.log!(res, x)
    @test_throws AssertionError Yeppp.sin!(res, x)
    @test_throws AssertionError Yeppp.cos!(res, x)
    @test_throws AssertionError Yeppp.tan!(res, x)
end
