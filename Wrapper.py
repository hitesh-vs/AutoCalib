import cv2
import numpy as np
import glob
import pickle
import os
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def detect_checkerboard_corners(directory_path, board_size=(9,6)):
    """
    Detect checkerboard corners in all JPG images in a directory.
    
    Args:
        directory_path (str): Path to the directory containing the images
        board_size (tuple): Number of inner corners (width, height) in the checkerboard
        
    Returns:
        tuple: Lists of object points and image points
    """
    imgpoints = []  # 2D points in image plane
    
    # Normalize directory path
    directory_path = os.path.normpath(directory_path)
    
    # Get list of all JPG images in the directory
    search_pattern = os.path.join(directory_path, '*.jpg')
    images = glob.glob(search_pattern)
    
    if not images:
        raise ValueError(f"No JPG images found in {directory_path}")
    
    print(f"Found {len(images)} JPG images")
    
    # Process each image
    for idx, fname in enumerate(images):
        # Normalize the file path
        fname = os.path.normpath(fname)
        print(f"\nProcessing image {idx + 1}/{len(images)}: {fname}")
            
        # Read image
        img = cv2.imread(fname)
        if img is None:
            print(f"Error: Could not read image {fname}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, board_size, 
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        # If found, add object points and image points
        if ret:
           
            imgpoints.append(corners)
            
            # Draw and display the corners
            cv2.drawChessboardCorners(img, board_size, corners, ret)
            
            # Save the annotated image
            base_name = os.path.basename(fname)
            output_path = f'detected_corners_{base_name}'
            cv2.imwrite(output_path, img)
            print(f"Success: Corners detected and saved to {output_path}")
        else:
            print(f"Warning: Failed to find corners in {fname}")
            # Save the image with a different prefix to check it
            base_name = os.path.basename(fname)
            debug_path = f'failed_detection_{base_name}'
            cv2.imwrite(debug_path, img)
            print(f"Saved failing image to {debug_path} for inspection")
    
    return imgpoints


def world_points():
    '''
    Returns the world points for the checkerboard pattern

    '''
    src_img = cv2.imread("src_checker.png")
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    board_size = (9, 6)
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)
    
    if not ret:
        raise ValueError("Could not find checkerboard corners in source image!")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    src_points = refined_corners.reshape(-1, 2)
    objpoints = np.hstack((src_points, np.zeros((src_points.shape[0], 1))))  # Z=0 for planar pattern

    return objpoints

def get_homography(imgpoints, objpoints):
    '''
    Compute homography matrix from image points and object points

    '''
    H_matrices = []
    for i in range(len(imgpoints)):
        H, mask = cv2.findHomography(objpoints, imgpoints[i][:, :2], method=cv2.RANSAC)
        H_matrices.append(H)
    return H_matrices

def compute_v_ij(H):
    """ Compute v_ij constraints from a homography matrix H. """
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]

    v_12 = np.array([
        h1[0] * h2[0], h1[0] * h2[1] + h1[1] * h2[0], h1[1] * h2[1],
        h1[2] * h2[0] + h1[0] * h2[2], h1[2] * h2[1] + h1[1] * h2[2], h1[2] * h2[2]
    ])

    v_11 = np.array([
        h1[0] * h1[0], h1[0] * h1[1] + h1[1] * h1[0], h1[1] * h1[1],
        h1[2] * h1[0] + h1[0] * h1[2], h1[2] * h1[1] + h1[1] * h1[2], h1[2] * h1[2]
    ])

    v_22 = np.array([
        h2[0] * h2[0], h2[0] * h2[1] + h2[1] * h2[0], h2[1] * h2[1],
        h2[2] * h2[0] + h2[0] * h2[2], h2[2] * h2[1] + h2[1] * h2[2], h2[2] * h2[2]
    ])

    return v_12, (v_11 - v_22)

def estimate_intrinsic_parameters(H_matrices):
    """ Compute the camera intrinsic matrix A from multiple homographies. """
    V = []

    for H in H_matrices:
        v_12, v_diff = compute_v_ij(H)
        V.append(v_12)
        V.append(v_diff)

    V = np.array(V)  # Shape: (2n, 6)

    # Solve Vb = 0 using SVD (smallest eigenvector)
    _, _, vh = np.linalg.svd(V)
    b = vh[-1, :]  # Last row of V^T (smallest singular value)

    # Construct matrix B = A^(-T) A^(-1)
    B = np.array([
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]]
    ])

    # Compute intrinsic parameters from B
    v0 = (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]) / (B[0, 0] * B[1, 1] - B[0, 1] ** 2)
    lambda_ = B[2, 2] - (B[0, 2] ** 2 + v0 * (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2])) / B[0, 0]
    alpha = np.sqrt(lambda_ / B[0, 0])
    beta = np.sqrt(lambda_ * B[0, 0] / (B[0, 0] * B[1, 1] - B[0, 1] ** 2))
    gamma = -B[0, 1] * alpha ** 2 * beta / lambda_
    u0 = gamma * v0 / beta - B[0, 2] * alpha ** 2 / lambda_

    # Intrinsic matrix A
    A = np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1]
    ])

    return A

def estimate_extrinsic_parameters(H_matrices, A):
    """ Compute the camera extrinsic matrix R and t from multiple homographies. """
    R_matrices = []
    t_vectors = []

    for H in H_matrices:
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]

        # Compute the scaling factor
        lambda_ = 1 / np.linalg.norm(np.dot(np.linalg.inv(A), h1))

        r1 = lambda_ * np.dot(np.linalg.inv(A), h1)
        r2 = lambda_ * np.dot(np.linalg.inv(A), h2)
        r3 = np.cross(r1, r2)

        # Ensure the determinant is positive
        if np.linalg.det([r1, r2, r3]) < 0:
            r1, r2 = -r1, -r2

        # Compute the translation vector
        t = lambda_ * np.dot(np.linalg.inv(A), h3)

        R = np.column_stack((r1, r2, r3))
        R_matrices.append(R)
        t_vectors.append(t)

    return R_matrices, t_vectors

def compute_Rt(A,H_matrices):
    extrinsic = []
    for h in H_matrices:
        h1,h2,h3 = h.T # get the column vectors

        K_inv = np.linalg.inv(A)
        lamda = 1/np.linalg.norm(K_inv.dot(h1),ord =2 )
        r1 = lamda*K_inv.dot(h1)
        r2 = lamda*K_inv.dot(h2)
        r3 = np.cross(r1,r2)
        t = lamda*K_inv.dot(h3)
        RT = np.vstack((r1, r2, r3, t)).T
        extrinsic.append(RT)
    return extrinsic

def reprojection_error(params, objpoints, imgpoints):
    """
    Compute the reprojection error.

    Args:
        params: Flattened parameter vector (intrinsics + extrinsics)
        objpoints: 3D object points (N,3)
        imgpoints: Observed 2D image points (N,2)

    Returns:
        Residuals: Flattened reprojection error
    """
    # Extract parameters
    fx, gamma, fy, cx, cy = params[:5]  # Intrinsic parameters
    k1,k2 = params[5:7]
    R = params[7:8]  # Rotation vector
    t = params[8:9]  # Translation vector

    # Reconstruct intrinsic matrix
    A = np.array([[fx, gamma, cx], [0, fy, cy], [0, 0, 1]])

    # Convert rotation vector to matrix
    #R, _ = cv2.Rodrigues(rvec)

    # Compute projected points
    projected_pts = project_points(A, R, t, k1,k2, objpoints)

    # Compute residuals (differences between detected and projected points)
    point_errors = np.sqrt((np.sum((imgpoints - projected_pts)**2, axis=1)))
    
    # Compute mean geometric error
    mean_error = np.mean(point_errors)

    return mean_error

def project_points(A, R, t, k1, k2, objpoints):
    """
    Project 3D object points onto the 2D image plane using the camera model.

    Args:
        A: Intrinsic matrix (3x3)
        R: Rotation matrix (3x3)
        t: Translation vector (3x1) - can be list or numpy array
        k1, k2: Radial distortion coefficients
        objpoints: 3D object points in shape (N,3), where each row is [x,y,z]

    Returns:
        projected_pts_distorted: 2D projected points with distortion (N,2)
    """
    # Convert inputs to numpy arrays if they aren't already
    t = np.array(t).reshape(3, 1)
    objpoints = np.array(objpoints)
    
    # Convert points to shape (3,N) for matrix operations
    objpoints = objpoints.T  # Now shape (3,N)
    N = objpoints.shape[1]
    
    # Apply extrinsic transformation: world to camera
    cam_pts = R @ objpoints + t  # Shape (3,N)
    cam_pts_new = np.squeeze(cam_pts)
    # Normalize homogeneous coordinates
    x = cam_pts_new[0, :] / cam_pts_new[2, :]  # Shape (N,)
    y = cam_pts_new[1, :] / cam_pts_new[2, :]  # Shape (N,)
    
    # Project using intrinsic matrix
    img_pts = A @ cam_pts  # Shape (3,N)
    img_pts = np.squeeze(img_pts)  # Remove singleton dimension
    img_pts = img_pts[:2, :] / img_pts[2, :]  # Normalize homogeneous coordinates
    
    # Apply radial distortion
    r2 = x**2 + y**2
    radial_distortion = k1 * r2 + k2 * r2**2

    # u and v 
    u = img_pts[0, :]
    v = img_pts[1, :]
    
    u0 = A[0, 2]
    v0 = A[1, 2]

    u_distorted = u + (u-u0) * radial_distortion
    v_distorted = v + (v-v0) * radial_distortion
    
    # Stack x and y coordinates
    projected_pts_distorted = np.vstack((u_distorted, v_distorted)).T  # Shape (N,2)
    
    return projected_pts_distorted

def pack_parameters(A, k1, k2):
    """
    Pack all parameters into a single vector for optimization.
    Convert rotation matrices to Rodrigues vectors.
    """
    params = []
    # Intrinsic parameters
    params.extend([A[0, 0], A[0,1], A[1, 1], A[0, 2], A[1, 2]])  # fx, fy, cx, cy
    # Distortion coefficients
    params.extend([k1, k2])
    # Extrinsic parameters for each image
    # for R, t in zip(R_matrices, t_vectors):
    #     # Convert rotation matrix to Rodrigues vector (3 parameters)
    #     rvec, _ = cv2.Rodrigues(R)
    #     params.extend(rvec.flatten())  # 3 parameters for rotation
    #     params.extend(t.flatten())     # 3 parameters for translation
    
    return np.array(params)

def unpack_parameters(params):
    """
    Unpack parameters vector into individual components.
    Convert Rodrigues vectors back to rotation matrices.
    """
    # Intrinsic parameters
    fx, gamma, fy, cx, cy = params[:5]
    A = np.array([[fx, gamma, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    
    # Distortion coefficients
    k1, k2 = params[5:7]
    
    return A, k1, k2

def total_reprojection_error(params, objpoints, imgpoints):
    """
    Compute total reprojection error across all images.
    """
    num_images = len(imgpoints)
    A, k1, k2 = unpack_parameters(params)
    
    errors = []
    total_error = 0
    
    for i in range(num_images):
        # Project points using current parameters
        projected_pts = project_points(A, R_matrices[i], t_vectors[i], k1, k2, objpoints)
        
       # Compute L2 norm for each point
        diff = imgpoints[i] - projected_pts
        point_errors = (diff).flatten()
        errors.append(point_errors)
    
    return np.concatenate(errors)
    #return np.sqrt(total_error)


def optimize_parameters(A_initial, objpoints, imgpoints):
    """
    Optimize camera parameters using least squares.
    """
    # Initial parameter vector
    initial_params = pack_parameters(A_initial, 0, 0)
    
    # Perform optimization
    result = least_squares(
        total_reprojection_error,
        initial_params,
        args=(objpoints, imgpoints),
        method='lm',
        verbose=2,
        #xtol=1e-10,  # Terminate if change in x is less than this
        #x_scale='jac'  # Scale parameters automatically
    )
    
    # Unpack optimized parameters
    A_opt, k1_opt, k2_opt = unpack_parameters(
        result.x
    )
    
    return A_opt, k1_opt, k2_opt, result

if '__main__' == __name__:
    # Detect checkerboard corners
    imgs_path = "Calibration_Imgs/Calibration_Imgs" # Path to the directory containing the images used for calibration
    imgpoints = detect_checkerboard_corners(imgs_path)
    objpoints = world_points()

    # Compute homography matrices
    H_matrices = get_homography(imgpoints, objpoints)

    # Estimate intrinsic parameters
    A = estimate_intrinsic_parameters(H_matrices)

    # Estimate extrinsic parameters
    R_matrices, t_vectors = estimate_extrinsic_parameters(H_matrices, A)

    # Compute R and t matrices
    #RT = compute_Rt(A,H_matrices)
    #print(f'R_matrices', R_matrices)
    #print(f't_vectors', t_vectors)

    # Remove singletons
    for i in range(len(imgpoints)):
        imgpoints[i] = np.squeeze(imgpoints[i])

    print(f'imgpoints[0].shape', imgpoints[0].shape)

    # Get Reprojection error
    projection_errors = []
    for i in range(len(imgpoints)):
        initial_params = [A[0, 0], A[0,1], A[1, 1], A[0, 2], A[1, 2], 0, 0, R_matrices[i], t_vectors[i]]
        projection_error = reprojection_error(initial_params, objpoints, imgpoints[i])
        projection_errors.append(projection_error)
        print(f"Projection error img{i+1}:", projection_error)
    print(f"Mean",np.mean(projection_errors))

    print(f'Intrinsic matrix A before opti:\n{A}')

    # Optimize parameters
    A_opt, k1_opt, k2_opt, result = optimize_parameters(
        A, objpoints, imgpoints
    )
    
    # Print results
    print("\nOptimization Results:")
    initial_error = np.mean(total_reprojection_error(
        pack_parameters(A, 0, 0), 
        objpoints, imgpoints
    ))
    print(f"Initial mean reprojection error: {initial_error:.4f} pixels")
    print(f"Final mean reprojection error: {np.mean(result.fun):.4f} pixels")
    print("\nOptimized Parameters:")
    print(f"Intrinsic matrix:\n{A_opt}")
    print(f"Distortion coefficients (k1, k2): {k1_opt:.6f}, {k2_opt:.6f}")

    # Compute final projection error
    final_projection_errors = []
    for i in range(len(imgpoints)):
        final_params = [A_opt[0, 0],A_opt[0,1], A_opt[1, 1], A_opt[0, 2], A_opt[1, 2], k1_opt, k2_opt, R_matrices[i], t_vectors[i]]
        final_projection_error = reprojection_error(final_params, objpoints, imgpoints[i])
        final_projection_errors.append(final_projection_error)
        print(f"Opt Projection error img{i+1}:", final_projection_error)
    print(f"Opt Mean",np.mean(final_projection_errors))

    # Get list of all JPG images in the directory
    directory_path = "Calibration_Imgs/Calibration_Imgs"
    search_pattern = os.path.join(directory_path, '*.jpg')
    images = sorted(glob.glob(search_pattern))


    for i in range(len(imgpoints)):
        reprojected_points = project_points(A_opt, R_matrices[i], t_vectors[i], k1_opt, k2_opt, objpoints)

        # Load and process image
        image = cv2.imread(images[i])
        if image is None:
            print(f"Warning: Could not read image {images[i]}")
            continue
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(image_rgb)
        
        # Plot original detected points in green
        #plt.scatter(imgpoints[i][:, 0], imgpoints[i][:, 1], 
                  # c='g', marker='o', label='Detected Points', alpha=0.5)
        
        # Plot reprojected points in red
        plt.scatter(reprojected_points[:, 0], reprojected_points[:, 1], 
                   c='r', marker='x', label='Reprojected Points')
        
        plt.title(f"Image {i+1}")
        plt.legend()
        
        # Save the figure
        output_filename = f'reprojected_img{i+1}.png'
        plt.savefig(output_filename)
        plt.close()  # Close the figure to free memory

    #print(f"Opt Mean", np.mean(final_projection_errors))
    print(f"Saved reprojection visualizations for {len(images)} images")
