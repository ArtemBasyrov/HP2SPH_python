using FastTransforms
using JSON

function healpy_index(l, m)
    return Int(l*(l+1)/2 + m + 1)
end

function transform2HealpixConvention(C)
    # Convert the input array to the Healpix convention
    lmax = size(C, 1)-1
    alm_arr = Array{ComplexF64}(undef, Int((lmax+1)*(lmax+2)/2))

    for l in 0:lmax
        alm_arr[healpy_index(l,  0)] = C[l+1, 1] # m=0
        for m in 1:l
            alm_arr[healpy_index(l,  m)] = C[l-m+1, 2*m   +1] # m>0
            #alm_arr[healpy_index(l, -m)] = C[l-m+1, 2*m-1 +1] # m<0 negative m are not stored in healpy
        end
    end

    return alm_arr
end

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

real_part = hcat(real_part...)
imag_part = hcat(imag_part...)

# Convert to a complex array
complex_array = real_part .+ imag_part*im
complex_array = transpose(complex_array)

# Perform the Fourier transform
C = fourier2sph(complex_array)
#alm = transform2HealpixConvention(C)

# Convert to JSON and print
#println(JSON.json(alm))
println(JSON.json(encode_complex(C)))
