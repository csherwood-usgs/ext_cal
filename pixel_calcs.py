import numpy as np
import yaml

def yaml2dict(yamlfile):
    """ Import contents of a YAML file as a dict

    Args:
        yamlfile (str): YAML file to read
    Returns:
        dict interpreted from YAML file
    """
    dictname = None
    with open(yamlfile, "r") as infile:
        try:
            dictname = yaml.safe_load(infile)
        except yaml.YAMLerror as exc:
            print(exc)
    return dictname
    

def undistort_UV( Ud, Vd, intrinsics, tolerance = 1e-5 ):
    """
    Correct distorted pixel coordinates according to intrinsic lens characteristics

    Input:
        Ud, Vd - Pixel locations from image (np.arrays)
        intrinsics - lens calibration coeffients in CalTech format (dict)
        tolerance - desired accuracy in iterative refinement (optional, scalar)
    Returns:
        U, V - Pixle locations corrected for lens characteristics

    Converted from the Matlab version written by Brittany Bruder
    """
    # Section 1: Define Coefficients
    NU=intrinsics['NU']
    NV=intrinsics['NV']
    c0U=intrinsics['c0U']
    c0V=intrinsics['c0V']
    fx=intrinsics['fx']
    fy=intrinsics['fy']
    d1=intrinsics['d1']
    d2=intrinsics['d2']
    d3=intrinsics['d3']
    t1=intrinsics['t1']
    t2=intrinsics['t2']

    # Section 2: Provide first guess for dx, dy, and fr using distorted x,y
    # Calculate Distorted camera coordinates x,y, and r
    xd = (Ud-c0U)/fx
    yd = (Vd-c0V)/fy
    rd = np.sqrt(xd*xd + yd*yd)
    r2d = rd*rd

    # Calculate first guess for aggregate coefficients
    fr1 = 1 + d1*r2d + d2*r2d*r2d + d3*r2d*r2d*r2d
    dx1=2*t1*xd*yd + t2*(r2d+2*xd*xd)
    dy1=t1*(r2d+2*yd*yd) + 2*t2*xd*yd

    # Section 3: Calculate undistorted X and Y using first guess
    # Work backwards lines 57-58 in Matlab version of distortUV.
    x= (xd-dx1)/fr1
    y= (yd-dy1)/fr1

    # Section 4: Iterate on solution until difference for all values is <.001%
    # (in my initial coding, most number of iterations was five)
    within_tolerance = False
    nit = 0
    max_nit = 9
    while not within_tolerance and nit < max_nit:
        nit+=1
        # Calculate new coefficients
        rn= np.sqrt(x*x + y*y)
        r2n=rn*rn
        frn = 1 + d1*r2n + d2*r2n*r2n + d3*r2n*r2n*r2n
        dxn=2*t1*x*y + t2*(r2n+2*x*x)
        dyn=t1*(r2n+2*y*y) + 2*t2*x*y
        
        # Determine percent change from fr,dx,and dy calculated with distorted
        # values
        chk1=np.max(np.abs((fr1-frn)/fr1))
        chk2=np.max(np.abs((dx1-dxn)/dx1))
        chk3=np.max(np.abs((dy1-dyn)/dy1))

        within_tolerance = chk1<=tolerance and chk2<=tolerance and chk3<=tolerance
        
        # Calculate new x,y for next iteration
        x= (xd-dxn)/frn
        y= (yd-dyn)/frn
        
        # Set the new coeffcients as previous solution for next iteration
        fr1=frn
        dx1=dxn
        dy1=dyn
    
    # Section 5: Convert x and y to U V
    U = x*fx + c0U
    V = y*fy + c0V
    return U, V


def distort_UV(U, V, intrinsics):
        """Distorts pixel locations to match camera distortion.

        Notes:
            - Derived from distortCaltech.m from Coastal Imaging Research Network - Oregon State University
            - Originally derived from Caltech lens distortion manuals

            - This code is adapted from B. Bruder's Matlab version
        Arguments:
            U, V (np.ndarray) - Coordinates of each pixel (undistorted).
            intrinsics - dict containing Cal Tech intrincis calibration params

        Returns:
            Ud, Vd (np.ndarray) - U and V arrays distorted to match camera distortion.
        """
        # calculate normalized image coordinates
        # - undistorted U-V to x-y space (normalized image coordinates)
        # - translated image center divded by focal length in pixels
        NU=intrinsics['NU']
        NV=intrinsics['NV']
        c0U=intrinsics['c0U']
        c0V=intrinsics['c0V']
        fx=intrinsics['fx']
        fy=intrinsics['fy']
        d1=intrinsics['d1']
        d2=intrinsics['d2']
        d3=intrinsics['d3']
        t1=intrinsics['t1']
        t2=intrinsics['t2']

        x = (U - c0U) / fx
        y = (V - c0V) / fy

        # distortion found based on large format units
        r2 = x*x + y*y
        fr = 1 + d1*r2 + d2*r2*r2 + d3*r2*r2*r2

        # get values for dx and dy at grid locations
        # - use linear in x + y direction
        # - this method will extrapolate, so we mask out values beyond
        #   source grid with nans to match matlab version
        
        # Tangential distortion
        dx=2*t1*x*y + t2*(r2+2*x*x)
        dy=t1*(r2+2*y*y) + 2*t2*x*y

        #  Apply correction, answer in chip pixel units
        xd = x*fr + dx
        yd = y*fr + dy
        Ud = xd*fx+c0U
        Vd = yd*fy+c0V

        # Deleted the part that masks out-of-bounds points

        return (Ud, Vd) 


def angle2R(azimuth, tilt, swing):
    """Assembles and returns a rotation matrix R from azimuth, tilt, and swing (roll)

    Notes:
        - derived from angles2R.m by Costal Imaging Research Network and Oregon State University
        - From p 612 of Wolf, 1983

    Arguments:
        azimuth (float): azimuth
        tilt (float): tilt
        swith (float): swing

    Returns:
        R (np.ndarray): Rotation matrix
    """
    a = azimuth
    t = tilt
    s = swing
    R = np.zeros((3, 3))

    R[0, 0] = np.cos(a) * np.cos(s) + np.sin(a) * np.cos(t) * np.sin(s)
    R[0, 1] = -np.cos(s) * np.sin(a) + np.sin(s) * np.cos(t) * np.cos(a)
    R[0, 2] = np.sin(s) * np.sin(t)

    R[1, 0] = -np.sin(s) * np.cos(a) + np.cos(s) * np.cos(t) * np.sin(a)
    R[1, 1] = np.sin(s) * np.sin(a) + np.cos(s) * np.cos(t) * np.cos(a)
    R[1, 2] = np.cos(s) * np.sin(t)

    R[2, 0] = np.sin(t) * np.sin(a)
    R[2, 1] = np.sin(t) * np.cos(a)
    R[2, 2] = -np.cos(t)

    return R

def assembleP( intrinsics, extrinsics ):

    # K: intrinsic matrix, puts image in pixel units of the specific camera
    # note...there is a sign difference in this matrix, compared with the Matlab version
    K = np.array([
        [intrinsics['fx'],               0., intrinsics['c0U']],
         [0.,             -intrinsics['fy'], intrinsics['c0V']],
         [0.,                            0.,                1.]
         ])

    # R: rotation matrix, puts image in camera orientation
    R = angle2R(extrinsics['a'], extrinsics['t'], extrinsics['r'])

    # I: identify matrix augmented by camera center, puts image in camera coordinates
    IC = np.vstack((
        np.eye(3),
        -np.array([extrinsics['x'], extrinsics['y'], extrinsics['z']])
    )).T

    KR = np.matmul(K, R)
    P = np.matmul(KR, IC)

    # Make the matrix homogenous, methods use homogenous coordinates for easier math
    # - normalize to make last element equal 1
    P = P/P[-1, -1]

    return P, K, R, IC
    


intrinsics = yaml2dict( '2021-02-25_CACO02_C1_IO.yml' )
extrinsics = yaml2dict( '2021-03-17_CACO02_C1_EO.yml')
P, K, R, IC = assembleP( intrinsics, extrinsics )
print(R)
print(K)
print(IC)
print(P)

# check roundtrip from U, V to Ud, Vd, and back
if False:
    Ui = np.random.uniform(.001, .999, 100)*intrinsics['NU']
    Vi = np.random.uniform(0.001, .999, 100)*intrinsics['NV']
    U, V = undistort_UV(Ui, Vi, intrinsics)

    Ud, Vd = distort_UV( U, V, intrinsics)
    madU = np.mean(np.abs(Ud-Ui))
    madV = np.mean(np.abs(Vd-Vi))

    print(madU, madV)



