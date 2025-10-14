import math

def f(x,y):
    return math.sin(x*y) * (x+2*y) ** 2

def df_dx(x,y):
    a = 2 * (x+2*y) * math.sin(x*y)
    b = y * math.cos(x*y) *(x+2*y) ** 2
    return a+b

def df_dy(x,y):
    a = 4 * (x+2 *y) * math.sin(x*y)
    b = x* math.cos(x*y) *(x+2*y) ** 2
    return a+b

if __name__ == '__main__':
    x_val = 2.0
    y_val = 3.0
    print(f"f({x_val}, {y_val}) = {f(x_val,y_val)}")
    print(f"df/dx({x_val}, {y_val}) = {df_dx(x_val,y_val)}")
    print(f"df/dy({x_val}, {y_val}) = {df_dy(x_val,y_val)}")

    
    
