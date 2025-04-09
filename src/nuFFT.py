import jax
import jax.numpy as jnp
import numpy as np
import finufft
from scipy.sparse.linalg import cg, LinearOperator

from .data_interpolation import create_latitude_array


def cg_nufft_forward(x, f_samples):
    # Get dimensions
    n_trans = f_samples.shape[0]  # = 4*nside
    M_samples = f_samples.shape[1]
    N_modes = n_trans+1  # Assuming square system (adjust if needed)

    # Precompute NUFFT plans with batch processing (n_trans transforms)
    plan_forward = finufft.Plan(2, (N_modes,), n_trans=n_trans, isign=1, dtype=np.complex128)
    plan_adjoint = finufft.Plan(1, (N_modes,), n_trans=n_trans, isign=-1, dtype=np.complex128)

    # Set nonuniform points (same for all transforms)
    plan_forward.setpts(x)
    plan_adjoint.setpts(x)

    # Reshape helpers
    def vec_to_mat_hat(vec):
        return vec.reshape(n_trans, N_modes)
    
    def vec_to_mat_samples(vec):
        return vec.reshape(n_trans, M_samples)
    
    def mat_to_vec(mat):
        return mat.ravel()

    # Define NUFFT operators with batch processing
    def forward_op(f_hat_vec):
        """A @ f_hat for all transforms (batched)"""
        f_hat_mat = vec_to_mat_hat(f_hat_vec)
        out = np.zeros((n_trans, M_samples), dtype=np.complex128)
        plan_forward.execute(f_hat_mat, out)
        return mat_to_vec(out/ np.sqrt(np.max([N_modes, M_samples])) )

    def adjoint_op(f_samples_vec):
        """A^H @ f_samples for all transforms (batched)"""
        f_samples_mat = vec_to_mat_samples(f_samples_vec)
        out = np.zeros((n_trans, N_modes), dtype=np.complex128)
        plan_adjoint.execute(f_samples_mat, out)
        return mat_to_vec(out / np.sqrt(np.max([N_modes, M_samples])) )
    
    def calc_rhs(f_sample_init):
        """A^H @ f_samples for all transforms (batched) initial"""
        out = np.zeros((n_trans, N_modes), dtype=np.complex128)
        plan_adjoint.execute(f_sample_init, out)
        return mat_to_vec(out / np.sqrt(np.max([N_modes, M_samples])) )

    # Solve (A^H A) f_hat = A^H f_samples using CG (batched)
    def apply_AHA(f_hat_vec):
        return adjoint_op(forward_op(f_hat_vec))

    # Reshape RHS and initial guess
    rhs = calc_rhs(f_samples)  # Flatten input data
    f_hat_guess = np.zeros(n_trans * N_modes, dtype=np.complex128)

    # Linear operator for CG
    A = LinearOperator(
        shape=(n_trans * N_modes, n_trans * N_modes),
        matvec=apply_AHA,
        dtype=np.complex128
    )

    # Run CG
    f_hat_recon_flat, info = cg(
        A,
        rhs,
        x0=f_hat_guess,
        rtol=1e-6,
        maxiter=100
    )

    # Reshape back to (4*nside, N_modes)
    return f_hat_recon_flat.reshape(n_trans, N_modes).T, info


def cg_nufft_backward(x, f_hat):
    """
    Solve for nonuniform samples f_samples from Fourier modes f_hat using CG.
    Solves: (A A^H) f_samples = A f_hat
    """
    # Get dimensions
    n_trans = f_hat.shape[0]  # = 4*nside (assuming f_hat shape [n_trans, N_modes])
    N_modes = f_hat.shape[1]
    M_samples = len(x)
    nside = n_trans // 4

    # Precompute NUFFT plans (same as forward code)
    plan_forward = finufft.Plan(2, (N_modes,), n_trans=n_trans, isign=1, dtype=np.complex128)
    plan_adjoint = finufft.Plan(1, (N_modes,), n_trans=n_trans, isign=-1, dtype=np.complex128)
    plan_forward.setpts(x)
    plan_adjoint.setpts(x)

    # Reshape helpers (adjusted for backward problem)
    def vec_to_mat_samples(vec):
        return vec.reshape(n_trans, M_samples)
    
    def vec_to_mat_hat(vec):
        return vec.reshape(n_trans, N_modes)
    
    def mat_to_vec(mat):
        return mat.ravel()

    # Define operators for (A A^H)
    def adjoint_op(f_samples_vec):
        """A^H @ f_samples (NUFFT Type 1)"""
        f_samples_mat = vec_to_mat_samples(f_samples_vec)
        out = np.zeros((n_trans, N_modes), dtype=np.complex128)
        plan_adjoint.execute(f_samples_mat, out)
        return mat_to_vec(out / np.sqrt(np.max([N_modes, M_samples])))

    def forward_op(f_hat_vec):
        """A @ f_hat (NUFFT Type 2)"""
        f_hat_mat = vec_to_mat_hat(f_hat_vec)
        out = np.zeros((n_trans, M_samples), dtype=np.complex128)
        plan_forward.execute(f_hat_mat, out)
        return mat_to_vec(out / np.sqrt(np.max([N_modes, M_samples])))

    def apply_AAH(f_samples_vec):
        """Compute A (A^H f_samples) = (A A^H) f_samples"""
        return forward_op(adjoint_op(f_samples_vec))

    # Right-hand side: A @ f_hat (flattened)
    rhs = forward_op(f_hat.ravel())  # Input f_hat shape [N_modes, n_trans]

    # Linear operator
    A = LinearOperator(
        shape=(n_trans * M_samples, n_trans * M_samples),
        matvec=apply_AAH,
        dtype=np.complex128
    )

    # Initial guess
    f_samples_guess = np.zeros(n_trans * M_samples, dtype=np.complex128)

    # Run CG
    f_samples_recon_flat, info = cg(
        A,
        rhs,
        x0=f_samples_guess,
        rtol=1e-6,
        maxiter=100
    )

    # Reshape to (n_trans, M_samples)
    return f_samples_recon_flat.reshape(n_trans, M_samples).T, info


def apply_nuFFT(mp: jnp.array) -> jnp.array: 
    nside = mp.shape[1] // 4

    # Calculate the latitude values for the upsampled map
    latitudes = create_latitude_array(nside)
    DFT_upsampled_lat = np.zeros(len(latitudes)*2+2)
    DFT_upsampled_lat[0] = 90
    DFT_upsampled_lat[1:len(latitudes)+1] = latitudes
    DFT_upsampled_lat[len(latitudes)+1] = -90
    DFT_upsampled_lat[len(latitudes)+2:] = -180 + latitudes

    # transform to [-pi, pi) range
    DFT_upsampled_lat = DFT_upsampled_lat * np.pi / 180 + np.pi/2

    # applying the CG nufft solver
    fft_lat, info = cg_nufft_forward(DFT_upsampled_lat, mp.T.copy())

    # output a warning if the solver didn't converge
    if info != 0:
        print("CG solver didn't converge!")

    return fft_lat


def inverse_nuFFT(fft_lat: jnp.array) -> jnp.array:
    """
    Perform inverse NUFFT (Type-2) to reconstruct signal at non-uniform latitudes.

    Parameters:
    - fft_lat (jnp.array): Fourier coefficients from uniform frequency space.

    Returns:
    - jnp.array: Reconstructed signal at non-uniform latitude samples.
    """
    nside = fft_lat.shape[1] // 4
    
    # Calculate the latitude values for the upsampled map
    latitudes = create_latitude_array(nside)
    DFT_upsampled_lat = np.zeros(len(latitudes)*2+2)
    DFT_upsampled_lat[0] = 90
    DFT_upsampled_lat[1:len(latitudes)+1] = latitudes
    DFT_upsampled_lat[len(latitudes)+1] = -90
    DFT_upsampled_lat[len(latitudes)+2:] = -180 + latitudes

    # transform to [-pi, pi) range
    DFT_upsampled_lat = DFT_upsampled_lat * np.pi / 180 + np.pi/2

    
    # applying the CG nufft solver
    fft_lat = np.array(fft_lat)
    mp_reconstructed, info = cg_nufft_backward(DFT_upsampled_lat, fft_lat.T.copy())

    # output a warning if the solver didn't converge
    if info != 0:
        print("CG solver didn't converge!")

    return mp_reconstructed


