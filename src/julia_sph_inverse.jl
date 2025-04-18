using FastTransforms
using JSON

function encode_complex(C)
    return map(x -> Dict("__complex__" => true, "real" => real(x), "imag" => imag(x)), C)
end

# Read JSON input from stdin
json_input = readline(stdin)

# Convert JSON to Julia array
data = JSON.parse(json_input)

# Extract the real and imaginary parts
real_part = data["real"]
imag_part = data["imag"]

complex_array = real_part .+ imag_part*im
complex_array = reduce(vcat, complex_array')

# Perform the Fourier transform
F = sph2fourier(complex_array)

# Convert to JSON and print
println(JSON.json(encode_complex(permutedims(F))))