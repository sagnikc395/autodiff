import math

def z(x,y,nograd=True):
    v1 = x
    v2 = y
    v3 = v1 + 2 * v2
    v4 = v1 * v2
    v5 = v3 ** 2
    v6 = math.sin(v4)
    v7 = v5 * v6

    if nograd:
        return v7 

    dv1 = 1
    dv2 = 0
    dv3 = dv1 + 2 * dv2
    dv4 = v1 * dv2 + v2 * dv1
    dv5 = 2* v3 * dv3
    dv6 = math.cos(v4) * dv4
    dv7 = v5 * dv6 + v6 * dv5
    dz_dx = dv7

    dv1 = 0
    dv2 = 1
    dv3 = dv1 + 2 * dv2
    dv4 = v1 * dv2 + v2 * dv1
    dv5 = 2* v3 * dv3
    dv6 = math.cos(v4) * dv4
    dv7 = v5 * dv6 + v6 * dv5
    dz_dy = dv7

    return v7, dz_dx, dz_dy     


if __name__ == '__main__':
    x_val = 2.0
    y_val = 3.0

    print(f"The value of the function at ({x_val}, {y_val}): {z(x_val,y_val)}")
    print(f"Partial Derivative df/dx at ({x_val}, {y_val}): {z(x_val,y_val,False)[1]}")
    print(f"Partial Derivative df/dy at ({x_val}, {y_val}): {z(x_val,y_val,False)[0]}")


    
