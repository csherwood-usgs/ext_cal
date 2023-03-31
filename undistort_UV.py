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
    

def undistortUV( Ud, Vd, intrinsics, tolerance = 1e-5 ):
    """
    Correct distorted pixel coordinates according to intrinsic lens characteristics

    Input:
        Ud, Vd - Pixel locations from image
        intrinsics - dict containing

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

    # Section 4: Iterate on solution Until difference for all values is <.001%
    # Initiate Variables for While Loop
    chk1=1
    chk2=1
    chk3=1
    within_tolerance = False
    nit = 0
    while not within_tolerance:
        nit+=1
        # Calculate new coefficients
        rn= np.sqrt(x*x + y*y)
        r2n=rn*rn
        frn = 1 + d1*r2n + d2*r2n*r2n + d3*r2n*r2n*r2n
        dxn=2*t1*x*y + t2*(r2n+2*x*x)
        dyn=t1*(r2n+2*y*y) + 2*t2*x*y
        
        # Determine percent change from fr,dx,and dy calculated with distorted
        # values
        chk1=np.abs((fr1-frn)/fr1)
        chk2=np.abs((dx1-dxn)/dx1)
        chk3=np.abs((dy1-dyn)/dy1)

        within_tolerance = chk1<=tolerance and chk2<=tolerance and chk3<=tolerance
        
        # Calculate new x,y for next iteration
        x= (xd-dxn)/frn
        y= (yd-dyn)/frn
        
        # Set the new coeffcients as previous solution for next iteration
        fr1=frn
        dx1=dxn
        dy1=dyn
    print('nit =', nit)    
    
    # Section 5: Convert x and y to U V
    U = x*fx + c0U
    V = y*fy + c0V
    return U, V

intrinsics = yaml2dict( '2021-02-25_CACO02_C2_IO.yml' )
print(intrinsics)
U, V = undistortUV(2000, 2000, intrinsics)
print(U, V)



